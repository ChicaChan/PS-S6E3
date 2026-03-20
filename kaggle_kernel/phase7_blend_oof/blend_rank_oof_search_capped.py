# Usage:
# 1) Blend with per-model max_weight caps:
#    python blend_rank_oof_search_capped.py \
#      --config-path local_blend_config_phase9_realmlp_candidates.json \
#      --sample-submission-path /kaggle/input/competitions/playground-series-s6e3/sample_submission.csv \
#      --output-dir output_phase9_realmlp_candidates
#
# 2) Local run:
#    python blend_rank_oof_search_capped.py \
#      --config-path local_blend_config_phase9_realmlp_candidates.json \
#      --sample-submission-path D:/workplace/kaggle/PS-S6E3/sample_submission.csv \
#      --output-dir output_phase9_realmlp_candidates

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


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


def rank_percentiles(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("weights must be 1-D")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("weights sum must be positive")
    return arr / total


def project_weights_with_caps(raw_weights: np.ndarray, max_weights: np.ndarray) -> np.ndarray:
    weights = normalize_weights(raw_weights)
    caps = np.asarray(max_weights, dtype=np.float64)
    if weights.shape != caps.shape:
        raise ValueError("weights and caps shape mismatch")

    if np.any(caps <= 0) or float(caps.sum()) < 1.0:
        raise ValueError("Invalid caps: each cap must be > 0 and total caps must be >= 1.")

    # Iteratively fix overweight capped models and redistribute residual mass.
    fixed = np.zeros_like(weights, dtype=bool)
    while True:
        violation = (weights > caps + 1e-12) & (~fixed)
        if not violation.any():
            break

        fixed = fixed | violation
        weights[violation] = caps[violation]

        residual = 1.0 - float(weights[fixed].sum())
        if residual < 0:
            residual = 0.0

        free_idx = np.where(~fixed)[0]
        if len(free_idx) == 0:
            break

        free_raw = raw_weights[free_idx].astype(np.float64)
        if float(free_raw.sum()) <= 0:
            free_raw = np.full(len(free_idx), 1.0 / len(free_idx), dtype=np.float64)
        else:
            free_raw = free_raw / float(free_raw.sum())
        weights[free_idx] = free_raw * residual

    # Final safety pass.
    weights = np.minimum(weights, caps)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Projected weights sum to zero.")

    weights /= total
    overweight = weights > caps + 1e-9
    if overweight.any():
        # conservative fallback: fill capped models first, then residual equally to free models
        weights = np.minimum(weights, caps)
        deficit = 1.0 - float(weights.sum())
        free_idx = np.where(caps - weights > 1e-9)[0]
        if len(free_idx) == 0:
            return normalize_weights(weights)
        room = caps[free_idx] - weights[free_idx]
        room = room / float(room.sum())
        weights[free_idx] += deficit * room
        weights = np.minimum(weights, caps)
        weights = normalize_weights(weights)
    return weights


def blend_prob(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.average(matrix, axis=1, weights=normalize_weights(weights))
    return np.clip(out, 0.0, 1.0)


def blend_rank(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    rank_matrix = np.column_stack([rank_percentiles(matrix[:, i]) for i in range(matrix.shape[1])])
    out = np.average(rank_matrix, axis=1, weights=normalize_weights(weights))
    return np.clip(out, 0.0, 1.0)


def resolve_path(base_dir: Path, path_text: str) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_model_predictions(
    config: dict[str, Any],
    config_path: Path,
    sample_submission_path: Path,
    id_col: str,
    target_col: str,
) -> tuple[np.ndarray, pd.Series, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    models = config.get("models", [])
    if not models:
        raise ValueError("Config models is empty.")

    base_dir = config_path.parent
    sample_df = pd.read_csv(sample_submission_path)
    if id_col not in sample_df.columns:
        raise ValueError(f"Sample submission missing id column: {id_col}")
    sample_ids = sample_df[id_col].values

    base_oof_df = None
    y = None
    oof_matrix_list: list[np.ndarray] = []
    sub_matrix_list: list[np.ndarray] = []
    names: list[str] = []
    caps: list[float] = []

    for entry in models:
        name = str(entry["name"])
        caps.append(float(entry.get("max_weight", 1.0)))
        oof_path = resolve_path(base_dir, str(entry["oof_path"]))
        sub_path = resolve_path(base_dir, str(entry["submission_path"]))

        oof_df = pd.read_csv(oof_path)
        sub_df = pd.read_csv(sub_path)

        if id_col not in oof_df.columns:
            raise ValueError(f"OOF file missing id column: {oof_path}")
        if id_col not in sub_df.columns:
            raise ValueError(f"Submission file missing id column: {sub_path}")

        if base_oof_df is None:
            base_oof_df = oof_df[[id_col]].copy()
            target_name = infer_target_column(oof_df, target_col)
            y = oof_df[target_name].to_numpy(dtype=np.int32)

        pred_col_oof = infer_prediction_column(oof_df, target_col)
        pred_col_sub = infer_prediction_column(sub_df, target_col)

        aligned_oof = oof_df.set_index(id_col).loc[base_oof_df[id_col].values, pred_col_oof].to_numpy(dtype=np.float64)
        aligned_sub = sub_df.set_index(id_col).loc[sample_ids, pred_col_sub].to_numpy(dtype=np.float64)

        if np.isnan(aligned_oof).any() or np.isnan(aligned_sub).any():
            raise ValueError(f"NaN detected in predictions for model: {name}")

        oof_matrix_list.append(aligned_oof)
        sub_matrix_list.append(aligned_sub)
        names.append(name)

    if y is None or base_oof_df is None:
        raise RuntimeError("Failed to load OOF target.")

    oof_matrix = np.column_stack(oof_matrix_list)
    sub_matrix = np.column_stack(sub_matrix_list)
    return y, base_oof_df[id_col], oof_matrix, np.asarray(caps, dtype=np.float64), names, pd.DataFrame({id_col: sample_ids, target_col: 0.0})


def filter_by_corr(
    oof_matrix: np.ndarray,
    names: list[str],
    y: np.ndarray,
    corr_threshold: float,
) -> tuple[list[int], dict[str, Any]]:
    aucs = [float(roc_auc_score(y, oof_matrix[:, i])) for i in range(oof_matrix.shape[1])]
    order = sorted(range(len(names)), key=lambda i: aucs[i], reverse=True)

    selected: list[int] = []
    dropped: list[dict[str, Any]] = []

    for idx in order:
        if not selected:
            selected.append(idx)
            continue

        corr_vals = [abs(float(np.corrcoef(oof_matrix[:, idx], oof_matrix[:, j])[0, 1])) for j in selected]
        max_corr = max(corr_vals)
        if max_corr <= corr_threshold:
            selected.append(idx)
        else:
            dropped.append(
                {
                    "name": names[idx],
                    "auc": aucs[idx],
                    "max_abs_corr_with_selected": max_corr,
                }
            )

    stats = {
        "individual_auc": {names[i]: aucs[i] for i in range(len(names))},
        "selected_names": [names[i] for i in selected],
        "dropped": dropped,
    }
    return selected, stats


def search_best_weights(
    oof_matrix: np.ndarray,
    y: np.ndarray,
    max_weights: np.ndarray,
    n_trials: int,
    seed: int,
) -> tuple[np.ndarray, str, float]:
    rng = np.random.default_rng(seed)
    n_models = oof_matrix.shape[1]

    best_auc = -1.0
    best_weights = normalize_weights(np.minimum(np.full(n_models, 1.0 / n_models), max_weights))
    best_method = "rank"

    for _ in range(max(1, n_trials)):
        raw = rng.dirichlet(np.ones(n_models, dtype=np.float64))
        weights = project_weights_with_caps(raw, max_weights)

        pred_rank = blend_rank(oof_matrix, weights)
        auc_rank = float(roc_auc_score(y, pred_rank))
        if auc_rank > best_auc:
            best_auc = auc_rank
            best_weights = weights.copy()
            best_method = "rank"

        pred_prob = blend_prob(oof_matrix, weights)
        auc_prob = float(roc_auc_score(y, pred_prob))
        if auc_prob > best_auc:
            best_auc = auc_prob
            best_weights = weights.copy()
            best_method = "prob"

    return best_weights, best_method, best_auc


def run_blend(
    config_path: Path,
    sample_submission_path: Path,
    output_dir: Path,
    id_col: str,
    target_col: str,
) -> None:
    cfg = read_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    corr_threshold = float(cfg.get("corr_threshold", 0.995))
    n_trials = int(cfg.get("n_trials", 3000))
    seed = int(cfg.get("seed", 42))

    y, oof_ids, oof_matrix_all, caps_all, names_all, _ = load_model_predictions(
        config=cfg,
        config_path=config_path,
        sample_submission_path=sample_submission_path,
        id_col=id_col,
        target_col=target_col,
    )

    selected_idx, filter_stats = filter_by_corr(
        oof_matrix=oof_matrix_all,
        names=names_all,
        y=y,
        corr_threshold=corr_threshold,
    )

    sel_oof = oof_matrix_all[:, selected_idx]
    sel_caps = caps_all[selected_idx]
    sel_names = [names_all[i] for i in selected_idx]

    if sel_oof.shape[1] == 0:
        raise RuntimeError("No models left after correlation filter.")

    eq_raw = np.full(sel_oof.shape[1], 1.0 / sel_oof.shape[1], dtype=np.float64)
    eq_weights = project_weights_with_caps(eq_raw, sel_caps)
    eq_oof = blend_rank(sel_oof, eq_weights)
    eq_auc = float(roc_auc_score(y, eq_oof))

    best_weights, best_method, best_auc = search_best_weights(
        oof_matrix=sel_oof,
        y=y,
        max_weights=sel_caps,
        n_trials=n_trials,
        seed=seed,
    )

    base_dir = config_path.parent
    sample_df = pd.read_csv(sample_submission_path)
    sample_ids = sample_df[id_col].values

    selected_sub_preds: list[np.ndarray] = []
    for idx in selected_idx:
        entry = cfg["models"][idx]
        sub_path = resolve_path(base_dir, str(entry["submission_path"]))
        sub_df = pd.read_csv(sub_path)
        pred_col = infer_prediction_column(sub_df, target_col)
        pred = sub_df.set_index(id_col).loc[sample_ids, pred_col].to_numpy(dtype=np.float64)
        selected_sub_preds.append(pred)

    sub_matrix = np.column_stack(selected_sub_preds)

    eq_sub_pred = blend_rank(sub_matrix, eq_weights)
    if best_method == "rank":
        best_sub_pred = blend_rank(sub_matrix, best_weights)
        best_oof = blend_rank(sel_oof, best_weights)
    else:
        best_sub_pred = blend_prob(sub_matrix, best_weights)
        best_oof = blend_prob(sel_oof, best_weights)

    submission_eq = sample_df[[id_col]].copy()
    submission_eq[target_col] = eq_sub_pred.astype(np.float32)

    submission_opt = sample_df[[id_col]].copy()
    submission_opt[target_col] = best_sub_pred.astype(np.float32)

    oof_eq_df = pd.DataFrame(
        {
            id_col: oof_ids.values,
            "target_binary": y,
            "oof_prediction": eq_oof.astype(np.float32),
        }
    )
    oof_opt_df = pd.DataFrame(
        {
            id_col: oof_ids.values,
            "target_binary": y,
            "oof_prediction": best_oof.astype(np.float32),
        }
    )

    eq_sub_path = output_dir / "submission_blend_eq.csv"
    opt_sub_path = output_dir / "submission_blend_opt.csv"
    eq_oof_path = output_dir / "oof_blend_eq.csv"
    opt_oof_path = output_dir / "oof_blend_opt.csv"
    report_path = output_dir / "blend_report.json"

    submission_eq.to_csv(eq_sub_path, index=False)
    submission_opt.to_csv(opt_sub_path, index=False)
    oof_eq_df.to_csv(eq_oof_path, index=False)
    oof_opt_df.to_csv(opt_oof_path, index=False)

    report = {
        "corr_threshold": corr_threshold,
        "n_trials": n_trials,
        "selected_models": sel_names,
        "selected_model_caps": {name: float(cap) for name, cap in zip(sel_names, sel_caps)},
        "equal_rank_oof_auc": eq_auc,
        "equal_rank_weights": {name: float(w) for name, w in zip(sel_names, eq_weights)},
        "best_method": best_method,
        "best_oof_auc": best_auc,
        "best_weights": {name: float(w) for name, w in zip(sel_names, best_weights)},
        "filter_stats": filter_stats,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {eq_sub_path}")
    print(f"Saved: {opt_sub_path}")
    print(f"Saved: {eq_oof_path}")
    print(f"Saved: {opt_oof_path}")
    print(f"Saved: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PS-S6E3 OOF-constrained capped blend search")
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--sample-submission-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=False, default=Path("/kaggle/working"))
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--target-col", type=str, default="Churn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_blend(
        config_path=args.config_path,
        sample_submission_path=args.sample_submission_path,
        output_dir=args.output_dir,
        id_col=args.id_col,
        target_col=args.target_col,
    )


if __name__ == "__main__":
    main()
