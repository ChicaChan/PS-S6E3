# Usage:
# 1) Kaggle / full run:
#    python hybrid_search.py \
#      --config-path hybrid_config_v1.json \
#      --output-dir /kaggle/working/phase12_rank_hybrid_v1
#
# 2) Local smoke:
#    python hybrid_search.py \
#      --config-path hybrid_config_smoke.json \
#      --output-dir .artifacts/smoke_phase12_rank_hybrid/output

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_REMOTE_CONFIG: dict[str, Any] = {
    "seed": 42,
    "id_col": "id",
    "target_col": "Churn",
    "inputs": {
        "reference_blend": {
            "name": "phase9_blend_best_v1",
            "oof_path": "/kaggle/input/ps-s6e3-phase11-hybrid-inputs/ref_oof_blend_opt.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase11-hybrid-inputs/ref_submission_blend_opt.csv",
        },
        "stack_best": {
            "name": "phase10_stack_best_v1",
            "oof_path": "/kaggle/input/ps-s6e3-phase11-hybrid-inputs/phase10_oof_stack_best.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase11-hybrid-inputs/phase10_submission_stack_best.csv",
        },
    },
    "candidates": [
        {"name": "rank_stack0.155", "blend_space": "rank", "stack_weight": 0.155},
        {"name": "rank_stack0.160", "blend_space": "rank", "stack_weight": 0.160},
        {"name": "rank_stack0.1625", "blend_space": "rank", "stack_weight": 0.1625},
        {"name": "rank_stack0.165", "blend_space": "rank", "stack_weight": 0.165},
        {"name": "rank_stack0.1675", "blend_space": "rank", "stack_weight": 0.1675},
        {"name": "rank_stack0.170", "blend_space": "rank", "stack_weight": 0.170},
        {"name": "rank_stack0.1725", "blend_space": "rank", "stack_weight": 0.1725},
        {"name": "rank_stack0.175", "blend_space": "rank", "stack_weight": 0.175},
        {"name": "rank_stack0.1775", "blend_space": "rank", "stack_weight": 0.1775},
        {"name": "rank_stack0.180", "blend_space": "rank", "stack_weight": 0.180},
        {"name": "rank_stack0.1825", "blend_space": "rank", "stack_weight": 0.1825},
        {"name": "rank_stack0.185", "blend_space": "rank", "stack_weight": 0.185},
        {"name": "rank_stack0.190", "blend_space": "rank", "stack_weight": 0.190},
        {"name": "rank_stack0.195", "blend_space": "rank", "stack_weight": 0.195},
        {"name": "rank_stack0.200", "blend_space": "rank", "stack_weight": 0.200},
    ],
}

REMOTE_INPUT_PREFIX = "/kaggle/input/ps-s6e3-phase11-hybrid-inputs"
REMOTE_REQUIRED_FILES = (
    "ref_oof_blend_opt.csv",
    "ref_submission_blend_opt.csv",
    "phase10_oof_stack_best.csv",
    "phase10_submission_stack_best.csv",
)


def read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def detect_remote_input_root() -> Path | None:
    explicit_root = Path(REMOTE_INPUT_PREFIX)
    if explicit_root.exists():
        return explicit_root

    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None

    candidate_dirs = [kaggle_input]
    candidate_dirs.extend(path for path in kaggle_input.rglob("*") if path.is_dir())

    best_dir: Path | None = None
    best_score = 0
    for candidate in candidate_dirs:
        score = sum((candidate / name).exists() for name in REMOTE_REQUIRED_FILES)
        if score > best_score:
            best_score = score
            best_dir = candidate

    if best_dir is None or best_score == 0:
        return None
    return best_dir


def build_remote_config(input_root: Path) -> dict[str, Any]:
    payload = json.loads(json.dumps(DEFAULT_REMOTE_CONFIG))
    root_text = str(input_root).replace("\\", "/")

    def rewrite(node: Any) -> Any:
        if isinstance(node, dict):
            return {key: rewrite(value) for key, value in node.items()}
        if isinstance(node, list):
            return [rewrite(item) for item in node]
        if isinstance(node, str):
            return node.replace(REMOTE_INPUT_PREFIX, root_text)
        return node

    return rewrite(payload)


def resolve_path(base_dir: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def infer_prediction_column(df: pd.DataFrame, target_col: str) -> str:
    candidates = ["oof_prediction", "prediction", target_col, "Churn", "pred"]
    for cand in candidates:
        if cand in df.columns:
            return cand
    if len(df.columns) < 2:
        raise ValueError("Prediction file must have at least 2 columns.")
    return df.columns[1]


def infer_target_column(df: pd.DataFrame, target_col: str) -> str:
    candidates = ["target_binary", "target", target_col, "label", "y_true"]
    for cand in candidates:
        if cand in df.columns:
            return cand
    raise ValueError("Cannot infer target column from OOF file.")


def validate_prediction_array(values: np.ndarray, label: str) -> None:
    arr = np.asarray(values, dtype=np.float64)
    if np.isnan(arr).any():
        raise ValueError(f"{label} contains NaN values.")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{label} must be in [0, 1].")


def validate_submission_df(df: pd.DataFrame, id_col: str, target_col: str) -> None:
    if list(df.columns) != [id_col, target_col]:
        raise ValueError(f"Submission columns mismatch: {list(df.columns)}")
    if df[id_col].isna().any():
        raise ValueError(f"Submission column '{id_col}' contains NaN.")
    validate_prediction_array(df[target_col].to_numpy(dtype=np.float64), f"submission.{target_col}")


def rank_percentiles(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def align_by_id(df: pd.DataFrame, expected_ids: pd.Series, id_col: str) -> pd.DataFrame:
    if df[id_col].equals(expected_ids):
        return df
    return df.set_index(id_col).loc[expected_ids.values].reset_index()


def load_pair_predictions(
    config: dict[str, Any],
    base_dir: Path,
) -> tuple[np.ndarray, pd.Series, np.ndarray, np.ndarray, pd.Series, np.ndarray, np.ndarray, str, str]:
    id_col = str(config.get("id_col", "id"))
    target_col = str(config.get("target_col", "Churn"))
    inputs = config.get("inputs", {})
    if "reference_blend" not in inputs or "stack_best" not in inputs:
        raise ValueError("Config 'inputs' must contain 'reference_blend' and 'stack_best'.")

    ref_cfg = inputs["reference_blend"]
    stack_cfg = inputs["stack_best"]

    ref_oof_path = resolve_path(base_dir, str(ref_cfg["oof_path"]))
    ref_sub_path = resolve_path(base_dir, str(ref_cfg["submission_path"]))
    stack_oof_path = resolve_path(base_dir, str(stack_cfg["oof_path"]))
    stack_sub_path = resolve_path(base_dir, str(stack_cfg["submission_path"]))

    for path in [ref_oof_path, ref_sub_path, stack_oof_path, stack_sub_path]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    ref_oof_df = pd.read_csv(ref_oof_path)
    ref_sub_df = pd.read_csv(ref_sub_path)
    stack_oof_df = pd.read_csv(stack_oof_path)
    stack_sub_df = pd.read_csv(stack_sub_path)

    for label, df in [
        ("reference_oof", ref_oof_df),
        ("reference_submission", ref_sub_df),
        ("stack_oof", stack_oof_df),
        ("stack_submission", stack_sub_df),
    ]:
        if id_col not in df.columns:
            raise ValueError(f"{label} missing id column '{id_col}'.")

    ref_pred_col_oof = infer_prediction_column(ref_oof_df, target_col)
    ref_pred_col_sub = infer_prediction_column(ref_sub_df, target_col)
    stack_pred_col_oof = infer_prediction_column(stack_oof_df, target_col)
    stack_pred_col_sub = infer_prediction_column(stack_sub_df, target_col)
    ref_target_col = infer_target_column(ref_oof_df, target_col)
    stack_target_col = infer_target_column(stack_oof_df, target_col)

    train_ids = ref_oof_df[id_col].copy()
    test_ids = ref_sub_df[id_col].copy()
    stack_oof_df = align_by_id(stack_oof_df, train_ids, id_col)
    stack_sub_df = align_by_id(stack_sub_df, test_ids, id_col)

    y_true = ref_oof_df[ref_target_col].to_numpy(dtype=np.int32)
    y_stack = stack_oof_df[stack_target_col].to_numpy(dtype=np.int32)
    if not np.array_equal(y_true, y_stack):
        raise ValueError("Target mismatch between reference blend and stack OOF files.")

    ref_oof = ref_oof_df[ref_pred_col_oof].to_numpy(dtype=np.float64)
    ref_sub = ref_sub_df[ref_pred_col_sub].to_numpy(dtype=np.float64)
    stack_oof = stack_oof_df[stack_pred_col_oof].to_numpy(dtype=np.float64)
    stack_sub = stack_sub_df[stack_pred_col_sub].to_numpy(dtype=np.float64)

    validate_prediction_array(ref_oof, "reference_blend.oof")
    validate_prediction_array(ref_sub, "reference_blend.submission")
    validate_prediction_array(stack_oof, "stack_best.oof")
    validate_prediction_array(stack_sub, "stack_best.submission")
    validate_submission_df(ref_sub_df[[id_col, ref_pred_col_sub]].rename(columns={ref_pred_col_sub: target_col}), id_col, target_col)
    validate_submission_df(stack_sub_df[[id_col, stack_pred_col_sub]].rename(columns={stack_pred_col_sub: target_col}), id_col, target_col)

    reference_name = str(ref_cfg.get("name", "reference_blend"))
    stack_name = str(stack_cfg.get("name", "stack_best"))
    return y_true, train_ids, ref_oof, stack_oof, test_ids, ref_sub, stack_sub, reference_name, stack_name


def build_candidate_prediction(
    ref_values: np.ndarray,
    stack_values: np.ndarray,
    stack_weight: float,
    label: str,
) -> np.ndarray:
    ref_weight = 1.0 - stack_weight
    blended = ref_weight * ref_values + stack_weight * stack_values
    validate_prediction_array(blended, label)
    return blended


def iter_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    explicit_candidates = config.get("candidates")
    if explicit_candidates:
        return list(explicit_candidates)

    blend_spaces = list(config.get("blend_spaces", ["prob"]))
    stack_weights = [float(item) for item in config.get("stack_weights", [])]
    if not stack_weights:
        raise ValueError("Config must provide 'stack_weights' when 'candidates' is absent.")

    candidates: list[dict[str, Any]] = []
    for blend_space in blend_spaces:
        for stack_weight in stack_weights:
            candidates.append(
                {
                    "name": f"{blend_space}_stack{stack_weight:.3f}",
                    "blend_space": blend_space,
                    "stack_weight": stack_weight,
                }
            )
    return candidates


def evaluate_candidates(
    config: dict[str, Any],
    y_true: np.ndarray,
    ref_oof: np.ndarray,
    stack_oof: np.ndarray,
    ref_sub: np.ndarray,
    stack_sub: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    reference_auc = float(roc_auc_score(y_true, ref_oof))
    stack_auc = float(roc_auc_score(y_true, stack_oof))
    baselines = {
        "reference_oof_auc": reference_auc,
        "stack_oof_auc": stack_auc,
    }
    cached_spaces = {
        "prob": {
            "oof": (ref_oof, stack_oof),
            "submission": (ref_sub, stack_sub),
        },
        "rank": {
            # Rank features are reused across every candidate, so cache once.
            "oof": (rank_percentiles(ref_oof), rank_percentiles(stack_oof)),
            "submission": (rank_percentiles(ref_sub), rank_percentiles(stack_sub)),
        },
    }

    candidate_rows: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    for candidate in iter_candidates(config):
        blend_space = str(candidate["blend_space"])
        stack_weight = float(candidate["stack_weight"])
        ref_weight = 1.0 - stack_weight
        if blend_space not in cached_spaces:
            raise ValueError(f"Unsupported blend_space: {blend_space}")

        ref_oof_values, stack_oof_values = cached_spaces[blend_space]["oof"]
        ref_sub_values, stack_sub_values = cached_spaces[blend_space]["submission"]

        oof_pred = build_candidate_prediction(
            ref_oof_values,
            stack_oof_values,
            stack_weight,
            f"candidate.{blend_space}.oof.{stack_weight:.4f}",
        )
        submission_pred = build_candidate_prediction(
            ref_sub_values,
            stack_sub_values,
            stack_weight,
            f"candidate.{blend_space}.submission.{stack_weight:.4f}",
        )
        overall_auc = float(roc_auc_score(y_true, oof_pred))

        row = {
            "name": str(candidate.get("name", f"{blend_space}_stack{stack_weight:.3f}")),
            "blend_space": blend_space,
            "stack_weight": stack_weight,
            "reference_weight": ref_weight,
            "overall_auc": overall_auc,
            "delta_vs_reference": overall_auc - reference_auc,
            "delta_vs_stack": overall_auc - stack_auc,
        }
        candidate_rows.append(row)

        payload = {
            **row,
            "oof_pred": oof_pred,
            "submission_pred": submission_pred,
        }
        if best_candidate is None or payload["overall_auc"] > best_candidate["overall_auc"]:
            best_candidate = payload

    if best_candidate is None:
        raise RuntimeError("No hybrid candidate finished successfully.")

    candidate_rows.sort(
        key=lambda item: (
            -float(item["overall_auc"]),
            -float(item["stack_weight"]),
        )
    )
    return baselines, candidate_rows, best_candidate


def save_outputs(
    output_dir: Path,
    y_true: np.ndarray,
    train_ids: pd.Series,
    test_ids: pd.Series,
    target_col: str,
    id_col: str,
    reference_name: str,
    stack_name: str,
    baselines: dict[str, float],
    candidate_rows: list[dict[str, Any]],
    best_candidate: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    best_oof_path = output_dir / "oof_hybrid_best.csv"
    best_submission_path = output_dir / "submission_hybrid_best.csv"
    candidate_summary_path = output_dir / "candidate_summary.json"
    report_path = output_dir / "hybrid_report.json"

    pd.DataFrame(
        {
            id_col: train_ids,
            "target_binary": y_true,
            "oof_prediction": best_candidate["oof_pred"],
        }
    ).to_csv(best_oof_path, index=False)

    pd.DataFrame(
        {
            id_col: test_ids,
            target_col: best_candidate["submission_pred"],
        }
    ).to_csv(best_submission_path, index=False)

    candidate_summary_path.write_text(
        json.dumps(candidate_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "reference_name": reference_name,
        "stack_name": stack_name,
        "reference_oof_auc": baselines["reference_oof_auc"],
        "stack_oof_auc": baselines["stack_oof_auc"],
        "best_candidate": {
            key: value
            for key, value in best_candidate.items()
            if key not in {"oof_pred", "submission_pred"}
        },
        "beats_reference": bool(best_candidate["overall_auc"] > baselines["reference_oof_auc"]),
        "beats_stack": bool(best_candidate["overall_auc"] > baselines["stack_oof_auc"]),
        "output_files": {
            "oof_best": str(best_oof_path),
            "submission_best": str(best_submission_path),
            "candidate_summary": str(candidate_summary_path),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pairwise stack-blend hybrid search for PS-S6E3.")
    parser.add_argument("--config-path", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    remote_input_root = detect_remote_input_root()

    if args.config_path is not None:
        config = read_config(args.config_path)
        base_dir = args.config_path.resolve().parent
    elif remote_input_root is not None:
        config = build_remote_config(remote_input_root)
        base_dir = script_dir
    else:
        raise ValueError("Missing --config-path and Kaggle remote inputs were not detected.")

    output_dir = args.output_dir
    if output_dir is None:
        if remote_input_root is not None:
            output_dir = Path("/kaggle/working/phase12_rank_hybrid_v1")
        else:
            raise ValueError("Missing --output-dir for local execution.")

    id_col = str(config.get("id_col", "id"))
    target_col = str(config.get("target_col", "Churn"))
    (
        y_true,
        train_ids,
        ref_oof,
        stack_oof,
        test_ids,
        ref_sub,
        stack_sub,
        reference_name,
        stack_name,
    ) = load_pair_predictions(config=config, base_dir=base_dir)

    baselines, candidate_rows, best_candidate = evaluate_candidates(
        config=config,
        y_true=y_true,
        ref_oof=ref_oof,
        stack_oof=stack_oof,
        ref_sub=ref_sub,
        stack_sub=stack_sub,
    )
    save_outputs(
        output_dir=output_dir,
        y_true=y_true,
        train_ids=train_ids,
        test_ids=test_ids,
        target_col=target_col,
        id_col=id_col,
        reference_name=reference_name,
        stack_name=stack_name,
        baselines=baselines,
        candidate_rows=candidate_rows,
        best_candidate=best_candidate,
    )

    print("Hybrid search completed.")
    print(f"Reference '{reference_name}' OOF AUC: {baselines['reference_oof_auc']:.6f}")
    print(f"Stack '{stack_name}' OOF AUC: {baselines['stack_oof_auc']:.6f}")
    print(
        "Best candidate: "
        f"{best_candidate['name']} | AUC={best_candidate['overall_auc']:.6f} | "
        f"stack_weight={best_candidate['stack_weight']:.3f}"
    )


if __name__ == "__main__":
    main()
