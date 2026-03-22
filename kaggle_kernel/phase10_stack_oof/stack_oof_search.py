# Usage:
# 1) Kaggle / full run:
#    python stack_oof_search.py \
#      --config-path stack_config_v1.json \
#      --output-dir /kaggle/working/phase10_stack_v1
#
# 2) Local smoke:
#    python stack_oof_search.py \
#      --config-path stack_config_smoke.json \
#      --output-dir .artifacts/smoke_phase10_stack/output

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_REMOTE_CONFIG: dict[str, Any] = {
    "seed": 42,
    "id_col": "id",
    "target_col": "Churn",
    "n_meta_folds": 5,
    "min_models": 2,
    "allow_missing_models": False,
    "models": [
        {
            "name": "phase8_cat_v1",
            "oof_path": "/kaggle/input/ps-s6e3-phase10-stack-inputs/phase8_oof_cat.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase10-stack-inputs/phase8_submission_cat.csv",
        },
        {
            "name": "phase6_cat_v1",
            "oof_path": "/kaggle/input/ps-s6e3-phase10-stack-inputs/phase6_oof_cat.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase10-stack-inputs/phase6_submission_cat.csv",
        },
        {
            "name": "phase6_lgbm_v1",
            "oof_path": "/kaggle/input/ps-s6e3-phase10-stack-inputs/phase6_oof_lgbm.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase10-stack-inputs/phase6_submission_lgbm.csv",
        },
        {
            "name": "phase2_fe_v1",
            "oof_path": "/kaggle/input/ps-s6e3-phase10-stack-inputs/phase2_oof_predictions.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase10-stack-inputs/phase2_submission.csv",
        },
        {
            "name": "phase3_ridge_xgb_v1",
            "oof_path": "/kaggle/input/ps-s6e3-phase10-stack-inputs/phase3_oof_predictions.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase10-stack-inputs/phase3_submission.csv",
        },
        {
            "name": "phase9_realmlp_v2",
            "oof_path": "/kaggle/input/ps-s6e3-phase10-stack-inputs/phase9_oof_realmlp.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase10-stack-inputs/phase9_submission_realmlp.csv",
        },
    ],
    "reference_blend": {
        "name": "phase9_blend_best_v1",
        "oof_path": "/kaggle/input/ps-s6e3-phase10-stack-inputs/ref_oof_blend_opt.csv",
        "submission_path": "/kaggle/input/ps-s6e3-phase10-stack-inputs/ref_submission_blend_opt.csv",
    },
    "candidate_sets": [
        {
            "name": "all_core",
            "models": [
                "phase8_cat_v1",
                "phase6_cat_v1",
                "phase6_lgbm_v1",
                "phase2_fe_v1",
                "phase3_ridge_xgb_v1",
                "phase9_realmlp_v2",
            ],
        },
        {
            "name": "drop_realmlp",
            "models": [
                "phase8_cat_v1",
                "phase6_cat_v1",
                "phase6_lgbm_v1",
                "phase2_fe_v1",
                "phase3_ridge_xgb_v1",
            ],
        },
        {
            "name": "tree_plus_diversity",
            "models": [
                "phase8_cat_v1",
                "phase6_cat_v1",
                "phase6_lgbm_v1",
                "phase9_realmlp_v2",
            ],
        },
    ],
    "feature_modes": ["raw", "raw_rank", "raw_rank_logit"],
    "meta_models": [
        {
            "name": "logreg_l2_c1",
            "type": "logreg",
            "params": {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"},
        },
        {
            "name": "logreg_l2_c0p25",
            "type": "logreg",
            "params": {"C": 0.25, "max_iter": 1000, "solver": "lbfgs"},
        },
        {
            "name": "ridge_a1",
            "type": "ridge",
            "params": {"alpha": 1.0},
        },
        {
            "name": "xgb_meta_v1",
            "type": "xgb",
            "params": {
                "n_estimators": 500,
                "learning_rate": 0.03,
                "max_depth": 3,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        },
    ],
}

REMOTE_INPUT_PREFIX = "/kaggle/input/ps-s6e3-phase10-stack-inputs"
REMOTE_REQUIRED_FILES = (
    "phase8_oof_cat.csv",
    "phase6_oof_cat.csv",
    "phase6_oof_lgbm.csv",
    "phase2_oof_predictions.csv",
    "phase3_oof_predictions.csv",
    "phase9_oof_realmlp.csv",
    "ref_oof_blend_opt.csv",
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


def resolve_n_splits(y: np.ndarray, desired: int) -> int:
    class_counts = np.bincount(y.astype(np.int32))
    positive_counts = class_counts[class_counts > 0]
    min_class_count = int(positive_counts.min())
    n_splits = min(desired, min_class_count)
    if n_splits < 2:
        raise ValueError("Not enough samples per class to run at least 2 folds.")
    return int(n_splits)


def rank_percentiles(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def safe_logit(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


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


def load_model_predictions(
    config: dict[str, Any],
    config_path: Path,
) -> tuple[np.ndarray, pd.Series, np.ndarray, pd.Series, np.ndarray, list[str], list[dict[str, str]]]:
    id_col = str(config.get("id_col", "id"))
    target_col = str(config.get("target_col", "Churn"))
    allow_missing_models = bool(config.get("allow_missing_models", False))
    min_models = int(config.get("min_models", 2))
    models = config.get("models", [])
    if not models:
        raise ValueError("Config 'models' is empty.")

    base_dir = config_path.parent
    base_train_ids: pd.Series | None = None
    base_test_ids: pd.Series | None = None
    y_true: np.ndarray | None = None
    oof_matrix_list: list[np.ndarray] = []
    sub_matrix_list: list[np.ndarray] = []
    names: list[str] = []
    skipped: list[dict[str, str]] = []

    for entry in models:
        if not bool(entry.get("enabled", True)):
            continue

        name = str(entry["name"])
        oof_path = resolve_path(base_dir, str(entry["oof_path"]))
        sub_path = resolve_path(base_dir, str(entry["submission_path"]))

        if not oof_path.exists() or not sub_path.exists():
            message = {
                "name": name,
                "reason": f"missing input files: oof={oof_path.exists()} submission={sub_path.exists()}",
            }
            if allow_missing_models:
                skipped.append(message)
                continue
            raise FileNotFoundError(message["reason"])

        oof_df = pd.read_csv(oof_path)
        sub_df = pd.read_csv(sub_path)

        if id_col not in oof_df.columns:
            raise ValueError(f"OOF file missing id column '{id_col}': {oof_path}")
        if id_col not in sub_df.columns:
            raise ValueError(f"Submission file missing id column '{id_col}': {sub_path}")

        pred_col_oof = infer_prediction_column(oof_df, target_col)
        pred_col_sub = infer_prediction_column(sub_df, target_col)

        if base_train_ids is None:
            target_name = infer_target_column(oof_df, target_col)
            base_train_ids = oof_df[id_col].copy()
            base_test_ids = sub_df[id_col].copy()
            y_true = oof_df[target_name].to_numpy(dtype=np.int32)
        else:
            if not oof_df[id_col].equals(base_train_ids):
                oof_df = oof_df.set_index(id_col).loc[base_train_ids.values].reset_index()
            if not sub_df[id_col].equals(base_test_ids):
                sub_df = sub_df.set_index(id_col).loc[base_test_ids.values].reset_index()
            target_name = infer_target_column(oof_df, target_col)
            aligned_target = oof_df[target_name].to_numpy(dtype=np.int32)
            if not np.array_equal(aligned_target, y_true):
                raise ValueError(f"Target mismatch detected in OOF file: {oof_path}")

        aligned_oof = oof_df[pred_col_oof].to_numpy(dtype=np.float64)
        aligned_sub = sub_df[pred_col_sub].to_numpy(dtype=np.float64)
        validate_prediction_array(aligned_oof, f"{name}.oof")
        validate_prediction_array(aligned_sub, f"{name}.submission")

        oof_matrix_list.append(aligned_oof)
        sub_matrix_list.append(aligned_sub)
        names.append(name)

    if y_true is None or base_train_ids is None or base_test_ids is None:
        raise RuntimeError("Failed to load any model predictions.")
    if len(names) < min_models:
        raise ValueError(f"Available models {len(names)} is below min_models={min_models}")

    oof_matrix = np.column_stack(oof_matrix_list)
    sub_matrix = np.column_stack(sub_matrix_list)
    return y_true, base_train_ids, oof_matrix, base_test_ids, sub_matrix, names, skipped


def load_reference_predictions(
    reference_cfg: dict[str, Any] | None,
    config_path: Path,
    train_ids: pd.Series,
    target_col: str,
    id_col: str,
) -> tuple[str | None, np.ndarray | None]:
    if not reference_cfg:
        return None, None

    base_dir = config_path.parent
    name = str(reference_cfg["name"])
    oof_path = resolve_path(base_dir, str(reference_cfg["oof_path"]))

    oof_df = pd.read_csv(oof_path).set_index(id_col).loc[train_ids.values].reset_index()

    oof_pred_col = infer_prediction_column(oof_df, target_col)
    ref_oof = oof_df[oof_pred_col].to_numpy(dtype=np.float64)
    validate_prediction_array(ref_oof, f"{name}.oof")
    return name, ref_oof


def resolve_candidate_indices(candidate_set: dict[str, Any], names: list[str]) -> list[int]:
    wanted = [str(item) for item in candidate_set.get("models", [])]
    if not wanted:
        return list(range(len(names)))

    missing = [item for item in wanted if item not in names]
    if missing:
        raise ValueError(f"Candidate set '{candidate_set['name']}' references missing models: {missing}")
    return [names.index(item) for item in wanted]


def precompute_feature_bank(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    feature_modes: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    feature_bank = {
        "raw": {
            "train": train_matrix,
            "test": test_matrix,
        }
    }
    requested = set(feature_modes)

    if "raw_rank" in requested or "raw_rank_logit" in requested:
        feature_bank["rank"] = {
            "train": np.column_stack([rank_percentiles(train_matrix[:, i]) for i in range(train_matrix.shape[1])]),
            "test": np.column_stack([rank_percentiles(test_matrix[:, i]) for i in range(test_matrix.shape[1])]),
        }

    if "raw_rank_logit" in requested:
        feature_bank["logit"] = {
            "train": safe_logit(train_matrix),
            "test": safe_logit(test_matrix),
        }

    return feature_bank


def build_meta_features(
    feature_bank: dict[str, dict[str, np.ndarray]],
    indices: list[int],
    feature_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = [feature_bank["raw"]["train"][:, indices]]
    test_parts: list[np.ndarray] = [feature_bank["raw"]["test"][:, indices]]

    if feature_mode in {"raw_rank", "raw_rank_logit"}:
        train_parts.append(feature_bank["rank"]["train"][:, indices])
        test_parts.append(feature_bank["rank"]["test"][:, indices])

    if feature_mode == "raw_rank_logit":
        train_parts.append(feature_bank["logit"]["train"][:, indices])
        test_parts.append(feature_bank["logit"]["test"][:, indices])

    if feature_mode not in {"raw", "raw_rank", "raw_rank_logit"}:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    return np.column_stack(train_parts), np.column_stack(test_parts)


def make_meta_model(model_spec: dict[str, Any], seed: int) -> Any | None:
    model_type = str(model_spec["type"]).lower()
    params = dict(model_spec.get("params", {}))

    if model_type == "logreg":
        params.setdefault("C", 1.0)
        params.setdefault("max_iter", 1000)
        params.setdefault("solver", "lbfgs")
        params.setdefault("random_state", seed)
        return make_pipeline(StandardScaler(), LogisticRegression(**params))

    if model_type == "ridge":
        params.setdefault("alpha", 1.0)
        return make_pipeline(StandardScaler(), Ridge(**params))

    if model_type == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return None
        default_params: dict[str, Any] = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": seed,
            "tree_method": "hist",
            "verbosity": 0
        }
        default_params.update(params)
        return XGBClassifier(**default_params)

    raise ValueError(f"Unsupported meta model type: {model_type}")


def predict_meta_scores(model: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        pred = model.predict_proba(features)[:, 1]
    else:
        pred = model.predict(features)
    return np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)


def run_meta_cv(
    train_features: np.ndarray,
    test_features: np.ndarray,
    y_true: np.ndarray,
    model_spec: dict[str, Any],
    n_meta_folds: int,
    seed: int,
) -> dict[str, Any]:
    meta_model = make_meta_model(model_spec, seed)
    if meta_model is None:
        return {
            "status": "skipped",
            "reason": f"meta model '{model_spec['name']}' unavailable in current environment"
        }

    n_splits = resolve_n_splits(y_true, n_meta_folds)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros(len(y_true), dtype=np.float64)
    fold_aucs: list[float] = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(train_features, y_true), start=1):
        fold_model = make_meta_model(model_spec, seed + fold_idx)
        if fold_model is None:
            raise RuntimeError(f"Failed to instantiate fold model: {model_spec['name']}")
        fold_model.fit(train_features[train_idx], y_true[train_idx])
        valid_pred = predict_meta_scores(fold_model, train_features[valid_idx])
        oof_pred[valid_idx] = valid_pred
        fold_auc = float(roc_auc_score(y_true[valid_idx], valid_pred))
        fold_aucs.append(fold_auc)

    overall_auc = float(roc_auc_score(y_true, oof_pred))
    full_model = make_meta_model(model_spec, seed + 10_000)
    if full_model is None:
        raise RuntimeError(f"Failed to instantiate full model: {model_spec['name']}")
    full_model.fit(train_features, y_true)
    test_pred = predict_meta_scores(full_model, test_features)

    return {
        "status": "ok",
        "oof_pred": oof_pred,
        "test_pred": test_pred,
        "overall_auc": overall_auc,
        "fold_aucs": fold_aucs,
        "fold_auc_mean": float(np.mean(fold_aucs)),
        "fold_auc_std": float(np.std(fold_aucs)),
        "n_meta_folds": n_splits
    }


def save_outputs(
    output_dir: Path,
    best_candidate: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    train_ids: pd.Series,
    test_ids: pd.Series,
    y_true: np.ndarray,
    target_col: str,
    id_col: str,
    reference_name: str | None,
    reference_auc: float | None,
    available_names: list[str],
    skipped_models: list[dict[str, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    best_oof_path = output_dir / "oof_stack_best.csv"
    best_submission_path = output_dir / "submission_stack_best.csv"
    candidate_summary_path = output_dir / "candidate_summary.json"
    stack_report_path = output_dir / "stack_report.json"

    oof_df = pd.DataFrame(
        {
            id_col: train_ids.values,
            "target_binary": y_true.astype(np.int32),
            "oof_prediction": best_candidate["oof_pred"]
        }
    )
    submission_df = pd.DataFrame(
        {
            id_col: test_ids.values,
            target_col: best_candidate["test_pred"]
        }
    )
    validate_submission_df(submission_df, id_col=id_col, target_col=target_col)

    oof_df.to_csv(best_oof_path, index=False)
    submission_df.to_csv(best_submission_path, index=False)
    candidate_summary_path.write_text(
        json.dumps(candidate_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stack_report = {
        "reference_name": reference_name,
        "reference_oof_auc": reference_auc,
        "available_models": available_names,
        "skipped_models": skipped_models,
        "best_candidate": {
            key: value
            for key, value in best_candidate.items()
            if key not in {"oof_pred", "test_pred"}
        },
        "beats_reference": None if reference_auc is None else bool(best_candidate["overall_auc"] > reference_auc),
        "output_files": {
            "oof_best": str(best_oof_path),
            "submission_best": str(best_submission_path),
            "candidate_summary": str(candidate_summary_path)
        }
    }
    stack_report_path.write_text(
        json.dumps(stack_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OOF-based stacking search for PS-S6E3.")
    parser.add_argument("--config-path", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kaggle_input_root = detect_remote_input_root()
    if args.config_path is not None:
        config = read_config(args.config_path)
    elif kaggle_input_root is not None:
        config = build_remote_config(kaggle_input_root)
    else:
        raise ValueError("Missing --config-path and Kaggle remote inputs were not detected.")

    output_dir = args.output_dir
    if output_dir is None:
        if kaggle_input_root is not None:
            output_dir = Path("/kaggle/working/phase10_stack_v1")
        else:
            raise ValueError("Missing --output-dir for local execution.")

    seed = int(config.get("seed", 42))
    target_col = str(config.get("target_col", "Churn"))
    id_col = str(config.get("id_col", "id"))
    n_meta_folds = int(config.get("n_meta_folds", 5))

    y_true, train_ids, oof_matrix, test_ids, sub_matrix, names, skipped_models = load_model_predictions(
        config=config,
        config_path=args.config_path or Path.cwd(),
    )
    reference_name, reference_oof = load_reference_predictions(
        reference_cfg=config.get("reference_blend"),
        config_path=args.config_path or Path.cwd(),
        train_ids=train_ids,
        target_col=target_col,
        id_col=id_col,
    )
    reference_auc = None if reference_oof is None else float(roc_auc_score(y_true, reference_oof))

    candidate_rows: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None

    candidate_sets = config.get("candidate_sets", [{"name": "all_models", "models": names}])
    feature_modes = [str(item) for item in config.get("feature_modes", ["raw"])]
    feature_bank = precompute_feature_bank(
        train_matrix=oof_matrix,
        test_matrix=sub_matrix,
        feature_modes=feature_modes,
    )
    meta_models = config.get("meta_models", [])
    if not meta_models:
        raise ValueError("Config 'meta_models' is empty.")

    for candidate_set in candidate_sets:
        candidate_set_name = str(candidate_set["name"])
        indices = resolve_candidate_indices(candidate_set, names)
        selected_names = [names[i] for i in indices]

        for feature_mode in feature_modes:
            train_features, test_features = build_meta_features(feature_bank, indices, feature_mode)

            for model_spec in meta_models:
                result = run_meta_cv(
                    train_features=train_features,
                    test_features=test_features,
                    y_true=y_true,
                    model_spec=model_spec,
                    n_meta_folds=n_meta_folds,
                    seed=seed,
                )

                row = {
                    "candidate_set": candidate_set_name,
                    "feature_mode": feature_mode,
                    "meta_model": str(model_spec["name"]),
                    "selected_models": selected_names,
                    "n_selected_models": len(selected_names),
                    "reference_oof_auc": reference_auc,
                    "status": result["status"]
                }

                if result["status"] != "ok":
                    row["reason"] = result["reason"]
                    candidate_rows.append(row)
                    continue

                row.update(
                    {
                        "overall_auc": result["overall_auc"],
                        "fold_auc_mean": result["fold_auc_mean"],
                        "fold_auc_std": result["fold_auc_std"],
                        "n_meta_folds": result["n_meta_folds"],
                        "beats_reference": None if reference_auc is None else bool(result["overall_auc"] > reference_auc)
                    }
                )
                candidate_rows.append(row)

                candidate_payload = {
                    **row,
                    "fold_aucs": result["fold_aucs"],
                    "oof_pred": result["oof_pred"],
                    "test_pred": result["test_pred"]
                }

                if best_candidate is None:
                    best_candidate = candidate_payload
                    continue

                current_key = (
                    candidate_payload["overall_auc"],
                    -candidate_payload["fold_auc_std"],
                    -candidate_payload["n_selected_models"],
                )
                best_key = (
                    best_candidate["overall_auc"],
                    -best_candidate["fold_auc_std"],
                    -best_candidate["n_selected_models"],
                )
                if current_key > best_key:
                    best_candidate = candidate_payload

    if best_candidate is None:
        raise RuntimeError("No stacking candidate finished successfully.")

    candidate_rows.sort(
        key=lambda item: (
            -float(item.get("overall_auc", -1.0)),
            float(item.get("fold_auc_std", 999.0)),
            int(item.get("n_selected_models", 999)),
        )
    )
    save_outputs(
        output_dir=output_dir,
        best_candidate=best_candidate,
        candidate_rows=candidate_rows,
        train_ids=train_ids,
        test_ids=test_ids,
        y_true=y_true,
        target_col=target_col,
        id_col=id_col,
        reference_name=reference_name,
        reference_auc=reference_auc,
        available_names=names,
        skipped_models=skipped_models,
    )

    print("Stacking search completed.")
    print(f"Available models: {names}")
    if skipped_models:
        print(f"Skipped models: {skipped_models}")
    if reference_auc is not None and reference_name is not None:
        print(f"Reference '{reference_name}' OOF AUC: {reference_auc:.6f}")
    print(
        "Best candidate: "
        f"{best_candidate['candidate_set']} | {best_candidate['feature_mode']} | "
        f"{best_candidate['meta_model']} | OOF AUC={best_candidate['overall_auc']:.6f}"
    )


if __name__ == "__main__":
    main()
