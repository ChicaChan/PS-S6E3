# Usage:
# 1) Kaggle / full run:
#    python hybrid_plus_realmlp_search.py \
#      --config-path phase13_config_v1.json \
#      --output-dir /kaggle/working/phase13_hybrid_plus_realmlp_v1
#
# 2) Local smoke:
#    python hybrid_plus_realmlp_search.py \
#      --config-path phase13_config_smoke.json \
#      --output-dir .artifacts/smoke_phase13_hybrid_plus_realmlp/output

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
        "base_hybrid": {
            "name": "phase12_rank_hybrid_best_v1",
            "oof_path": "/kaggle/input/ps-s6e3-phase13-realmllp-inputs/phase12_oof_hybrid_best.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase13-realmllp-inputs/phase12_submission_hybrid_best.csv",
        },
        "realmlp": {
            "name": "phase9_realmlp_v2",
            "oof_path": "/kaggle/input/ps-s6e3-phase13-realmllp-inputs/phase9_oof_realmlp.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase13-realmllp-inputs/phase9_submission_realmlp.csv",
        },
    },
    "candidates": [
        {"name": "rank_realmlp0.0050", "blend_space": "rank", "realmlp_weight": 0.0050},
        {"name": "rank_realmlp0.0075", "blend_space": "rank", "realmlp_weight": 0.0075},
        {"name": "rank_realmlp0.0100", "blend_space": "rank", "realmlp_weight": 0.0100},
        {"name": "rank_realmlp0.0125", "blend_space": "rank", "realmlp_weight": 0.0125},
        {"name": "rank_realmlp0.0150", "blend_space": "rank", "realmlp_weight": 0.0150},
        {"name": "rank_realmlp0.0175", "blend_space": "rank", "realmlp_weight": 0.0175},
        {"name": "rank_realmlp0.0200", "blend_space": "rank", "realmlp_weight": 0.0200},
        {"name": "rank_realmlp0.0225", "blend_space": "rank", "realmlp_weight": 0.0225},
        {"name": "rank_realmlp0.0250", "blend_space": "rank", "realmlp_weight": 0.0250},
        {"name": "rank_realmlp0.0275", "blend_space": "rank", "realmlp_weight": 0.0275},
        {"name": "rank_realmlp0.0300", "blend_space": "rank", "realmlp_weight": 0.0300},
        {"name": "prob_realmlp0.0050", "blend_space": "prob", "realmlp_weight": 0.0050},
        {"name": "prob_realmlp0.0100", "blend_space": "prob", "realmlp_weight": 0.0100},
        {"name": "prob_realmlp0.0150", "blend_space": "prob", "realmlp_weight": 0.0150},
        {"name": "prob_realmlp0.0200", "blend_space": "prob", "realmlp_weight": 0.0200},
    ],
}

REMOTE_INPUT_PREFIX = "/kaggle/input/ps-s6e3-phase13-realmllp-inputs"
REMOTE_REQUIRED_FILES = (
    "phase12_oof_hybrid_best.csv",
    "phase12_submission_hybrid_best.csv",
    "phase9_oof_realmlp.csv",
    "phase9_submission_realmlp.csv",
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


def load_predictions(
    config: dict[str, Any],
    base_dir: Path,
) -> tuple[np.ndarray, pd.Series, np.ndarray, np.ndarray, pd.Series, np.ndarray, np.ndarray, str, str]:
    id_col = str(config.get("id_col", "id"))
    target_col = str(config.get("target_col", "Churn"))
    inputs = config.get("inputs", {})
    if "base_hybrid" not in inputs or "realmlp" not in inputs:
        raise ValueError("Config 'inputs' must contain 'base_hybrid' and 'realmlp'.")

    base_cfg = inputs["base_hybrid"]
    realm_cfg = inputs["realmlp"]

    base_oof_path = resolve_path(base_dir, str(base_cfg["oof_path"]))
    base_sub_path = resolve_path(base_dir, str(base_cfg["submission_path"]))
    realm_oof_path = resolve_path(base_dir, str(realm_cfg["oof_path"]))
    realm_sub_path = resolve_path(base_dir, str(realm_cfg["submission_path"]))

    for path in [base_oof_path, base_sub_path, realm_oof_path, realm_sub_path]:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    base_oof_df = pd.read_csv(base_oof_path)
    base_sub_df = pd.read_csv(base_sub_path)
    realm_oof_df = pd.read_csv(realm_oof_path)
    realm_sub_df = pd.read_csv(realm_sub_path)

    for label, df in [
        ("base_oof", base_oof_df),
        ("base_submission", base_sub_df),
        ("realmlp_oof", realm_oof_df),
        ("realmlp_submission", realm_sub_df),
    ]:
        if id_col not in df.columns:
            raise ValueError(f"{label} missing id column '{id_col}'.")

    base_pred_col_oof = infer_prediction_column(base_oof_df, target_col)
    base_pred_col_sub = infer_prediction_column(base_sub_df, target_col)
    realm_pred_col_oof = infer_prediction_column(realm_oof_df, target_col)
    realm_pred_col_sub = infer_prediction_column(realm_sub_df, target_col)
    base_target_col = infer_target_column(base_oof_df, target_col)
    realm_target_col = infer_target_column(realm_oof_df, target_col)

    train_ids = base_oof_df[id_col].copy()
    test_ids = base_sub_df[id_col].copy()
    realm_oof_df = align_by_id(realm_oof_df, train_ids, id_col)
    realm_sub_df = align_by_id(realm_sub_df, test_ids, id_col)

    y_true = base_oof_df[base_target_col].to_numpy(dtype=np.int32)
    y_realm = realm_oof_df[realm_target_col].to_numpy(dtype=np.int32)
    if not np.array_equal(y_true, y_realm):
        raise ValueError("Target mismatch between base hybrid and RealMLP OOF files.")

    base_oof = base_oof_df[base_pred_col_oof].to_numpy(dtype=np.float64)
    base_sub = base_sub_df[base_pred_col_sub].to_numpy(dtype=np.float64)
    realm_oof = realm_oof_df[realm_pred_col_oof].to_numpy(dtype=np.float64)
    realm_sub = realm_sub_df[realm_pred_col_sub].to_numpy(dtype=np.float64)

    validate_prediction_array(base_oof, "base_hybrid.oof")
    validate_prediction_array(base_sub, "base_hybrid.submission")
    validate_prediction_array(realm_oof, "realmlp.oof")
    validate_prediction_array(realm_sub, "realmlp.submission")
    validate_submission_df(
        base_sub_df[[id_col, base_pred_col_sub]].rename(columns={base_pred_col_sub: target_col}),
        id_col,
        target_col,
    )
    validate_submission_df(
        realm_sub_df[[id_col, realm_pred_col_sub]].rename(columns={realm_pred_col_sub: target_col}),
        id_col,
        target_col,
    )

    base_name = str(base_cfg.get("name", "base_hybrid"))
    realm_name = str(realm_cfg.get("name", "realmlp"))
    return y_true, train_ids, base_oof, realm_oof, test_ids, base_sub, realm_sub, base_name, realm_name


def build_candidate_prediction(
    base_values: np.ndarray,
    realm_values: np.ndarray,
    realmlp_weight: float,
    label: str,
) -> np.ndarray:
    base_weight = 1.0 - realmlp_weight
    blended = base_weight * base_values + realmlp_weight * realm_values
    validate_prediction_array(blended, label)
    return blended


def iter_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    explicit_candidates = config.get("candidates")
    if explicit_candidates:
        return list(explicit_candidates)
    raise ValueError("Config must provide explicit 'candidates'.")


def evaluate_candidates(
    y_true: np.ndarray,
    base_oof: np.ndarray,
    realm_oof: np.ndarray,
    base_sub: np.ndarray,
    realm_sub: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    base_auc = float(roc_auc_score(y_true, base_oof))
    realm_auc = float(roc_auc_score(y_true, realm_oof))
    baselines = {
        "base_hybrid_oof_auc": base_auc,
        "realmlp_oof_auc": realm_auc,
    }

    cached_spaces = {
        "prob": {
            "oof": (base_oof, realm_oof),
            "submission": (base_sub, realm_sub),
        },
        "rank": {
            "oof": (rank_percentiles(base_oof), rank_percentiles(realm_oof)),
            "submission": (rank_percentiles(base_sub), rank_percentiles(realm_sub)),
        },
    }

    candidate_rows: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    for candidate in iter_candidates(config):
        blend_space = str(candidate["blend_space"])
        realmlp_weight = float(candidate["realmlp_weight"])
        base_weight = 1.0 - realmlp_weight
        if blend_space not in cached_spaces:
            raise ValueError(f"Unsupported blend_space: {blend_space}")

        base_oof_values, realm_oof_values = cached_spaces[blend_space]["oof"]
        base_sub_values, realm_sub_values = cached_spaces[blend_space]["submission"]
        oof_pred = build_candidate_prediction(
            base_oof_values,
            realm_oof_values,
            realmlp_weight,
            f"candidate.{blend_space}.oof.{realmlp_weight:.4f}",
        )
        submission_pred = build_candidate_prediction(
            base_sub_values,
            realm_sub_values,
            realmlp_weight,
            f"candidate.{blend_space}.submission.{realmlp_weight:.4f}",
        )
        overall_auc = float(roc_auc_score(y_true, oof_pred))

        row = {
            "name": str(candidate.get("name", f"{blend_space}_realmlp{realmlp_weight:.4f}")),
            "blend_space": blend_space,
            "realmlp_weight": realmlp_weight,
            "base_weight": base_weight,
            "overall_auc": overall_auc,
            "delta_vs_base": overall_auc - base_auc,
            "delta_vs_realmlp": overall_auc - realm_auc,
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
        raise RuntimeError("No phase13 candidate finished successfully.")

    candidate_rows.sort(
        key=lambda item: (
            -float(item["overall_auc"]),
            -float(item["realmlp_weight"]),
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
    base_name: str,
    realm_name: str,
    baselines: dict[str, float],
    candidate_rows: list[dict[str, Any]],
    best_candidate: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    best_oof_path = output_dir / "oof_phase13_best.csv"
    best_submission_path = output_dir / "submission_phase13_best.csv"
    candidate_summary_path = output_dir / "candidate_summary.json"
    report_path = output_dir / "phase13_report.json"

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
        "base_name": base_name,
        "realmlp_name": realm_name,
        "base_hybrid_oof_auc": baselines["base_hybrid_oof_auc"],
        "realmlp_oof_auc": baselines["realmlp_oof_auc"],
        "best_candidate": {
            key: value
            for key, value in best_candidate.items()
            if key not in {"oof_pred", "submission_pred"}
        },
        "beats_base": bool(best_candidate["overall_auc"] > baselines["base_hybrid_oof_auc"]),
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
    parser = argparse.ArgumentParser(description="Run phase13 hybrid plus RealMLP search for PS-S6E3.")
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
            output_dir = Path("/kaggle/working/phase13_hybrid_plus_realmlp_v1")
        else:
            raise ValueError("Missing --output-dir for local execution.")

    id_col = str(config.get("id_col", "id"))
    target_col = str(config.get("target_col", "Churn"))
    (
        y_true,
        train_ids,
        base_oof,
        realm_oof,
        test_ids,
        base_sub,
        realm_sub,
        base_name,
        realm_name,
    ) = load_predictions(config=config, base_dir=base_dir)

    baselines, candidate_rows, best_candidate = evaluate_candidates(
        y_true=y_true,
        base_oof=base_oof,
        realm_oof=realm_oof,
        base_sub=base_sub,
        realm_sub=realm_sub,
        config=config,
    )
    save_outputs(
        output_dir=output_dir,
        y_true=y_true,
        train_ids=train_ids,
        test_ids=test_ids,
        target_col=target_col,
        id_col=id_col,
        base_name=base_name,
        realm_name=realm_name,
        baselines=baselines,
        candidate_rows=candidate_rows,
        best_candidate=best_candidate,
    )

    print("Phase13 search completed.")
    print(f"Base '{base_name}' OOF AUC: {baselines['base_hybrid_oof_auc']:.6f}")
    print(f"RealMLP '{realm_name}' OOF AUC: {baselines['realmlp_oof_auc']:.6f}")
    print(
        "Best candidate: "
        f"{best_candidate['name']} | AUC={best_candidate['overall_auc']:.6f} | "
        f"realmlp_weight={best_candidate['realmlp_weight']:.4f}"
    )


if __name__ == "__main__":
    main()
