# Usage:
# 1) Kaggle / full run:
#    python stack_pipeline_search.py \
#      --config-path stack_pipeline_config_v1.json \
#      --output-dir /kaggle/working/phase15_stack_plus_orig_xgb_v1
#
# 2) Local smoke:
#    python stack_pipeline_search.py \
#      --config-path stack_pipeline_config_smoke.json \
#      --output-dir .artifacts/smoke_phase15_stack/output

from __future__ import annotations

import argparse
import json
from itertools import combinations
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
    "run_name": "phase15_stack_plus_orig_xgb_v1",
    "seed": 42,
    "id_col": "id",
    "target_col": "Churn",
    "n_meta_folds": 5,
    "min_models": 3,
    "allow_missing_models": False,
    "anchor_model_name": "phase10_stack_best_v1",
    "reference_model_name": "phase10_stack_best_v1",
    "models": [
        {
            "name": "phase13_hybrid_best_v1",
            "oof_path": "/kaggle/input/ps-s6e3-phase15-stack-inputs/phase13_oof_phase13_best.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase15-stack-inputs/phase13_submission_phase13_best.csv",
        },
        {
            "name": "phase10_stack_best_v1",
            "oof_path": "/kaggle/input/ps-s6e3-phase15-stack-inputs/phase10_oof_stack_best.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase15-stack-inputs/phase10_submission_stack_best.csv",
        },
        {
            "name": "phase8_cat_v1",
            "oof_path": "/kaggle/input/ps-s6e3-phase15-stack-inputs/phase8_oof_cat.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase15-stack-inputs/phase8_submission_cat.csv",
        },
        {
            "name": "phase9_realmlp_v2",
            "oof_path": "/kaggle/input/ps-s6e3-phase15-stack-inputs/phase9_oof_realmlp.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase15-stack-inputs/phase9_submission_realmlp.csv",
        },
        {
            "name": "phase15_orig_fe_xgb_v1",
            "oof_path": "/kaggle/input/ps-s6e3-phase15-stack-inputs/phase15_oof_orig_fe_xgb.csv",
            "submission_path": "/kaggle/input/ps-s6e3-phase15-stack-inputs/phase15_submission_orig_fe_xgb.csv",
        },
    ],
    "candidate_sets": [
        {
            "name": "phase15_core",
            "models": [
                "phase15_orig_fe_xgb_v1",
                "phase10_stack_best_v1",
                "phase13_hybrid_best_v1",
            ],
        },
        {
            "name": "phase15_plus_diversity",
            "models": [
                "phase15_orig_fe_xgb_v1",
                "phase13_hybrid_best_v1",
                "phase10_stack_best_v1",
                "phase9_realmlp_v2",
            ],
        },
        {
            "name": "phase15_plus_cat",
            "models": [
                "phase15_orig_fe_xgb_v1",
                "phase13_hybrid_best_v1",
                "phase10_stack_best_v1",
                "phase8_cat_v1",
            ],
        },
        {
            "name": "phase15_full_pool",
            "models": [
                "phase15_orig_fe_xgb_v1",
                "phase13_hybrid_best_v1",
                "phase10_stack_best_v1",
                "phase8_cat_v1",
                "phase9_realmlp_v2",
            ],
        },
    ],
    "feature_packs": [
        {
            "name": "raw_only",
            "blocks": ["raw"],
        },
        {
            "name": "raw_rank_stats",
            "blocks": ["raw", "rank", "raw_stats", "rank_stats"],
        },
        {
            "name": "raw_rank_anchor",
            "blocks": [
                "raw",
                "rank",
                "raw_stats",
                "rank_stats",
                "raw_anchor_gaps",
                "raw_anchor_abs_gaps",
                "rank_anchor_gaps",
            ],
        },
        {
            "name": "raw_rank_pairwise",
            "blocks": [
                "raw",
                "rank",
                "raw_stats",
                "rank_stats",
                "pairwise_raw_absdiff",
                "pairwise_rank_absdiff",
            ],
        },
        {
            "name": "full_linear",
            "blocks": [
                "raw",
                "rank",
                "logit",
                "raw_stats",
                "rank_stats",
                "raw_anchor_gaps",
                "raw_anchor_abs_gaps",
                "rank_anchor_gaps",
                "logit_anchor_gaps",
                "pairwise_raw_absdiff",
                "pairwise_rank_absdiff",
            ],
        },
        {
            "name": "full_linear_no_rank_pairwise",
            "blocks": [
                "raw",
                "rank",
                "logit",
                "raw_stats",
                "rank_stats",
                "raw_anchor_gaps",
                "raw_anchor_abs_gaps",
                "rank_anchor_gaps",
                "logit_anchor_gaps",
                "pairwise_raw_absdiff",
            ],
        },
        {
            "name": "full_linear_no_logit_anchor",
            "blocks": [
                "raw",
                "rank",
                "logit",
                "raw_stats",
                "rank_stats",
                "raw_anchor_gaps",
                "raw_anchor_abs_gaps",
                "rank_anchor_gaps",
                "pairwise_raw_absdiff",
                "pairwise_rank_absdiff",
            ],
        },
        {
            "name": "anchor_compact",
            "blocks": [
                "raw",
                "rank",
                "logit",
                "raw_stats",
                "rank_stats",
                "raw_anchor_gaps",
                "raw_anchor_abs_gaps",
                "rank_anchor_gaps",
                "pairwise_raw_absdiff",
            ],
        },
    ],
    "meta_models": [
        {
            "name": "logreg_newtoncg_c0p15",
            "type": "logreg",
            "params": {"C": 0.15, "max_iter": 1000, "solver": "newton-cg"},
        },
        {
            "name": "logreg_newtoncg_c0p20",
            "type": "logreg",
            "params": {"C": 0.20, "max_iter": 1000, "solver": "newton-cg"},
        },
        {
            "name": "logreg_newtoncg_c0p25",
            "type": "logreg",
            "params": {"C": 0.25, "max_iter": 1000, "solver": "newton-cg"},
        },
        {
            "name": "logreg_newtoncg_c0p30",
            "type": "logreg",
            "params": {"C": 0.30, "max_iter": 1000, "solver": "newton-cg"},
        },
        {
            "name": "logreg_newtoncg_c0p35",
            "type": "logreg",
            "params": {"C": 0.35, "max_iter": 1000, "solver": "newton-cg"},
        },
        {
            "name": "logreg_lbfgs_c0p25",
            "type": "logreg",
            "params": {"C": 0.25, "max_iter": 1000, "solver": "lbfgs"},
        },
    ],
}

REMOTE_INPUT_PREFIX = "/kaggle/input/ps-s6e3-phase15-stack-inputs"
REMOTE_REQUIRED_FILES = (
    "phase13_oof_phase13_best.csv",
    "phase13_submission_phase13_best.csv",
    "phase10_oof_stack_best.csv",
    "phase10_submission_stack_best.csv",
    "phase8_oof_cat.csv",
    "phase8_submission_cat.csv",
    "phase9_oof_realmlp.csv",
    "phase9_submission_realmlp.csv",
    "phase15_oof_orig_fe_xgb.csv",
    "phase15_submission_orig_fe_xgb.csv",
)
SUPPORTED_FEATURE_BLOCKS = {
    "raw",
    "rank",
    "logit",
    "raw_stats",
    "rank_stats",
    "raw_anchor_gaps",
    "raw_anchor_abs_gaps",
    "rank_anchor_gaps",
    "logit_anchor_gaps",
    "pairwise_raw_absdiff",
    "pairwise_rank_absdiff",
}


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


def resolve_base_dir(config_source: Path) -> Path:
    return config_source if config_source.is_dir() else config_source.parent


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
    return str(df.columns[1])


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


def align_by_id(df: pd.DataFrame, expected_ids: pd.Series, id_col: str) -> pd.DataFrame:
    if df[id_col].equals(expected_ids):
        return df
    return df.set_index(id_col).loc[expected_ids.values].reset_index()


def load_model_predictions(
    config: dict[str, Any],
    config_source: Path,
) -> tuple[np.ndarray, pd.Series, np.ndarray, pd.Series, np.ndarray, list[str], list[dict[str, str]]]:
    id_col = str(config.get("id_col", "id"))
    target_col = str(config.get("target_col", "Churn"))
    allow_missing_models = bool(config.get("allow_missing_models", False))
    min_models = int(config.get("min_models", 2))
    models = config.get("models", [])
    if not models:
        raise ValueError("Config 'models' is empty.")

    base_dir = resolve_base_dir(config_source)
    y_true: np.ndarray | None = None
    base_train_ids: pd.Series | None = None
    base_test_ids: pd.Series | None = None
    oof_matrix_list: list[np.ndarray] = []
    sub_matrix_list: list[np.ndarray] = []
    names: list[str] = []
    skipped: list[dict[str, str]] = []

    for model_cfg in models:
        name = str(model_cfg["name"])
        oof_path = resolve_path(base_dir, str(model_cfg["oof_path"]))
        sub_path = resolve_path(base_dir, str(model_cfg["submission_path"]))

        missing_paths = [str(path) for path in [oof_path, sub_path] if not path.exists()]
        if missing_paths:
            if allow_missing_models:
                skipped.append({"name": name, "reason": f"missing files: {missing_paths}"})
                continue
            raise FileNotFoundError(f"Model '{name}' missing files: {missing_paths}")

        oof_df = pd.read_csv(oof_path)
        sub_df = pd.read_csv(sub_path)
        if id_col not in oof_df.columns or id_col not in sub_df.columns:
            raise ValueError(f"Model '{name}' is missing id column '{id_col}'.")

        oof_pred_col = infer_prediction_column(oof_df, target_col)
        sub_pred_col = infer_prediction_column(sub_df, target_col)
        local_target_col = infer_target_column(oof_df, target_col)

        if y_true is None:
            y_true = oof_df[local_target_col].to_numpy(dtype=np.int32)
            base_train_ids = oof_df[id_col].copy()
            base_test_ids = sub_df[id_col].copy()
        else:
            oof_df = align_by_id(oof_df, base_train_ids, id_col)
            sub_df = align_by_id(sub_df, base_test_ids, id_col)
            local_y = oof_df[local_target_col].to_numpy(dtype=np.int32)
            if not np.array_equal(y_true, local_y):
                raise ValueError(f"Target mismatch detected for model '{name}'.")

        oof_pred = oof_df[oof_pred_col].to_numpy(dtype=np.float64)
        sub_pred = sub_df[sub_pred_col].to_numpy(dtype=np.float64)
        validate_prediction_array(oof_pred, f"{name}.oof")
        validate_prediction_array(sub_pred, f"{name}.submission")
        validate_submission_df(
            sub_df[[id_col, sub_pred_col]].rename(columns={sub_pred_col: target_col}),
            id_col=id_col,
            target_col=target_col,
        )

        oof_matrix_list.append(oof_pred)
        sub_matrix_list.append(sub_pred)
        names.append(name)

    if y_true is None or base_train_ids is None or base_test_ids is None:
        raise RuntimeError("No valid models were loaded.")
    if len(names) < min_models:
        raise ValueError(f"Available models {len(names)} is below min_models={min_models}")

    oof_matrix = np.column_stack(oof_matrix_list)
    sub_matrix = np.column_stack(sub_matrix_list)
    return y_true, base_train_ids, oof_matrix, base_test_ids, sub_matrix, names, skipped


def get_feature_packs(config: dict[str, Any]) -> list[dict[str, Any]]:
    feature_packs = config.get("feature_packs", [])
    if not feature_packs:
        raise ValueError("Config 'feature_packs' is empty.")

    normalized: list[dict[str, Any]] = []
    for item in feature_packs:
        name = str(item["name"])
        blocks = [str(block) for block in item.get("blocks", [])]
        if not blocks:
            raise ValueError(f"Feature pack '{name}' has no blocks.")
        unknown_blocks = [block for block in blocks if block not in SUPPORTED_FEATURE_BLOCKS]
        if unknown_blocks:
            raise ValueError(f"Feature pack '{name}' contains unsupported blocks: {unknown_blocks}")
        normalized.append({"name": name, "blocks": blocks})
    return normalized


def resolve_candidate_indices(candidate_set: dict[str, Any], names: list[str], min_models: int) -> list[int]:
    wanted = [str(item) for item in candidate_set.get("models", [])]
    if not wanted:
        indices = list(range(len(names)))
    else:
        missing = [item for item in wanted if item not in names]
        if missing:
            raise ValueError(f"Candidate set '{candidate_set['name']}' references missing models: {missing}")
        indices = [names.index(item) for item in wanted]

    if len(indices) < min_models:
        raise ValueError(f"Candidate set '{candidate_set['name']}' has only {len(indices)} models.")
    return indices


def precompute_feature_bank(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    requested_blocks: set[str],
) -> dict[str, dict[str, np.ndarray]]:
    feature_bank = {
        "raw": {
            "train": train_matrix.astype(np.float64),
            "test": test_matrix.astype(np.float64),
        }
    }

    need_rank = any("rank" in block for block in requested_blocks)
    need_logit = "logit" in requested_blocks or "logit_anchor_gaps" in requested_blocks

    if need_rank:
        feature_bank["rank"] = {
            "train": np.column_stack([rank_percentiles(train_matrix[:, i]) for i in range(train_matrix.shape[1])]),
            "test": np.column_stack([rank_percentiles(test_matrix[:, i]) for i in range(test_matrix.shape[1])]),
        }

    if need_logit:
        feature_bank["logit"] = {
            "train": safe_logit(train_matrix),
            "test": safe_logit(test_matrix),
        }

    return feature_bank


def empty_feature_matrix(n_rows: int) -> np.ndarray:
    return np.empty((n_rows, 0), dtype=np.float64)


def reduce_stats_features(matrix: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.mean(matrix, axis=1),
            np.std(matrix, axis=1),
            np.min(matrix, axis=1),
            np.max(matrix, axis=1),
            np.median(matrix, axis=1),
            np.max(matrix, axis=1) - np.min(matrix, axis=1),
        ]
    )


def pairwise_absdiff_features(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[1] < 2:
        return empty_feature_matrix(matrix.shape[0])
    diffs = [np.abs(matrix[:, left] - matrix[:, right]) for left, right in combinations(range(matrix.shape[1]), 2)]
    return np.column_stack(diffs)


def select_columns(matrix: np.ndarray, positions: list[int]) -> np.ndarray:
    if not positions:
        return empty_feature_matrix(matrix.shape[0])
    return matrix[:, positions]


def combine_feature_parts(parts: list[np.ndarray]) -> np.ndarray:
    kept = [part for part in parts if part.size > 0]
    if not kept:
        raise ValueError("Feature pack resolved to zero feature columns.")
    return np.column_stack(kept)


def build_meta_features(
    feature_bank: dict[str, dict[str, np.ndarray]],
    indices: list[int],
    names: list[str],
    blocks: list[str],
    anchor_model_name: str | None,
) -> tuple[np.ndarray, np.ndarray, str]:
    raw_train = feature_bank["raw"]["train"][:, indices]
    raw_test = feature_bank["raw"]["test"][:, indices]
    selected_names = [names[i] for i in indices]
    resolved_anchor_name = anchor_model_name if anchor_model_name in names else selected_names[0]
    anchor_global_idx = names.index(resolved_anchor_name)

    positions_without_anchor = [
        local_idx
        for local_idx, model_name in enumerate(selected_names)
        if model_name != resolved_anchor_name
    ]

    part_lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "raw": (raw_train, raw_test),
        "raw_stats": (reduce_stats_features(raw_train), reduce_stats_features(raw_test)),
        "pairwise_raw_absdiff": (
            pairwise_absdiff_features(raw_train),
            pairwise_absdiff_features(raw_test),
        ),
    }

    if "rank" in feature_bank:
        rank_train = feature_bank["rank"]["train"][:, indices]
        rank_test = feature_bank["rank"]["test"][:, indices]
        anchor_rank_train = feature_bank["rank"]["train"][:, anchor_global_idx][:, None]
        anchor_rank_test = feature_bank["rank"]["test"][:, anchor_global_idx][:, None]
        part_lookup["rank"] = (rank_train, rank_test)
        part_lookup["rank_stats"] = (reduce_stats_features(rank_train), reduce_stats_features(rank_test))
        part_lookup["rank_anchor_gaps"] = (
            select_columns(rank_train, positions_without_anchor) - anchor_rank_train,
            select_columns(rank_test, positions_without_anchor) - anchor_rank_test,
        )
        part_lookup["pairwise_rank_absdiff"] = (
            pairwise_absdiff_features(rank_train),
            pairwise_absdiff_features(rank_test),
        )

    anchor_raw_train = feature_bank["raw"]["train"][:, anchor_global_idx][:, None]
    anchor_raw_test = feature_bank["raw"]["test"][:, anchor_global_idx][:, None]
    part_lookup["raw_anchor_gaps"] = (
        select_columns(raw_train, positions_without_anchor) - anchor_raw_train,
        select_columns(raw_test, positions_without_anchor) - anchor_raw_test,
    )
    part_lookup["raw_anchor_abs_gaps"] = (
        np.abs(select_columns(raw_train, positions_without_anchor) - anchor_raw_train),
        np.abs(select_columns(raw_test, positions_without_anchor) - anchor_raw_test),
    )

    if "logit" in feature_bank:
        logit_train = feature_bank["logit"]["train"][:, indices]
        logit_test = feature_bank["logit"]["test"][:, indices]
        anchor_logit_train = feature_bank["logit"]["train"][:, anchor_global_idx][:, None]
        anchor_logit_test = feature_bank["logit"]["test"][:, anchor_global_idx][:, None]
        part_lookup["logit"] = (logit_train, logit_test)
        part_lookup["logit_anchor_gaps"] = (
            select_columns(logit_train, positions_without_anchor) - anchor_logit_train,
            select_columns(logit_test, positions_without_anchor) - anchor_logit_test,
        )

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for block in blocks:
        if block not in part_lookup:
            raise ValueError(f"Feature block '{block}' was requested but not precomputed.")
        train_part, test_part = part_lookup[block]
        train_parts.append(train_part)
        test_parts.append(test_part)

    return combine_feature_parts(train_parts), combine_feature_parts(test_parts), resolved_anchor_name


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
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 2,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": seed,
            "tree_method": "hist",
            "verbosity": 0,
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
            "reason": f"meta model '{model_spec['name']}' unavailable in current environment",
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
        fold_aucs.append(float(roc_auc_score(y_true[valid_idx], valid_pred)))

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
        "n_meta_folds": n_splits,
    }


def resolve_reference_auc(
    reference_model_name: str | None,
    names: list[str],
    oof_matrix: np.ndarray,
    y_true: np.ndarray,
) -> tuple[str | None, float | None]:
    if not reference_model_name:
        return None, None
    if reference_model_name not in names:
        raise ValueError(f"Reference model '{reference_model_name}' is not available.")
    ref_idx = names.index(reference_model_name)
    reference_auc = float(roc_auc_score(y_true, oof_matrix[:, ref_idx]))
    return reference_model_name, reference_auc


def reset_text_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def log_message(log_path: Path, message: str) -> None:
    print(message, flush=True)
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(message + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_best_snapshot(path: Path, payload: dict[str, Any]) -> None:
    snapshot = {
        key: value
        for key, value in payload.items()
        if key not in {"oof_pred", "test_pred", "fold_aucs"}
    }
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")


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

    best_oof_path = output_dir / "oof_stack_pipeline_best.csv"
    best_submission_path = output_dir / "submission_stack_pipeline_best.csv"
    candidate_summary_path = output_dir / "candidate_summary.json"
    stack_report_path = output_dir / "stack_pipeline_report.json"

    oof_df = pd.DataFrame(
        {
            id_col: train_ids.values,
            "target_binary": y_true.astype(np.int32),
            "oof_prediction": best_candidate["oof_pred"],
        }
    )
    submission_df = pd.DataFrame(
        {
            id_col: test_ids.values,
            target_col: best_candidate["test_pred"],
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
        "reference_model_name": reference_name,
        "reference_oof_auc": reference_auc,
        "available_models": available_names,
        "skipped_models": skipped_models,
        "best_candidate": {
            key: value
            for key, value in best_candidate.items()
            if key not in {"oof_pred", "test_pred", "fold_aucs"}
        },
        "beats_reference": None if reference_auc is None else bool(best_candidate["overall_auc"] > reference_auc),
        "output_files": {
            "oof_best": str(best_oof_path),
            "submission_best": str(best_submission_path),
            "candidate_summary": str(candidate_summary_path),
            "progress_log": str(output_dir / "progress.log"),
            "candidate_progress": str(output_dir / "candidate_progress.jsonl"),
            "best_snapshot": str(output_dir / "best_candidate_snapshot.json"),
        },
    }
    stack_report_path.write_text(
        json.dumps(stack_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stronger stacking pipeline search for PS-S6E3.")
    parser.add_argument("--config-path", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kaggle_input_root = detect_remote_input_root()
    if args.config_path is not None:
        config = read_config(args.config_path)
        config_source = args.config_path
    elif kaggle_input_root is not None:
        config = build_remote_config(kaggle_input_root)
        config_source = Path.cwd()
    else:
        raise ValueError("Missing --config-path and Kaggle remote inputs were not detected.")

    output_dir = args.output_dir
    if output_dir is None:
        if kaggle_input_root is not None:
            output_dir = Path("/kaggle/working") / str(config.get("run_name", "phase14_stronger_stack_pipeline_v1"))
        else:
            raise ValueError("Missing --output-dir for local execution.")
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_log_path = output_dir / "progress.log"
    candidate_progress_path = output_dir / "candidate_progress.jsonl"
    best_snapshot_path = output_dir / "best_candidate_snapshot.json"
    reset_text_file(progress_log_path)
    reset_text_file(candidate_progress_path)
    if best_snapshot_path.exists():
        best_snapshot_path.unlink()

    seed = int(config.get("seed", 42))
    target_col = str(config.get("target_col", "Churn"))
    id_col = str(config.get("id_col", "id"))
    n_meta_folds = int(config.get("n_meta_folds", 5))
    min_models = int(config.get("min_models", 2))
    anchor_model_name = config.get("anchor_model_name")
    if anchor_model_name is not None:
        anchor_model_name = str(anchor_model_name)
    reference_model_name = config.get("reference_model_name")
    if reference_model_name is not None:
        reference_model_name = str(reference_model_name)

    y_true, train_ids, oof_matrix, test_ids, sub_matrix, names, skipped_models = load_model_predictions(
        config=config,
        config_source=config_source,
    )
    reference_name, reference_auc = resolve_reference_auc(
        reference_model_name=reference_model_name,
        names=names,
        oof_matrix=oof_matrix,
        y_true=y_true,
    )

    candidate_sets = config.get("candidate_sets", [{"name": "all_models", "models": names}])
    feature_packs = get_feature_packs(config)
    requested_blocks = {block for pack in feature_packs for block in pack["blocks"]}
    feature_bank = precompute_feature_bank(
        train_matrix=oof_matrix,
        test_matrix=sub_matrix,
        requested_blocks=requested_blocks,
    )
    meta_models = config.get("meta_models", [])
    if not meta_models:
        raise ValueError("Config 'meta_models' is empty.")
    total_candidates = len(candidate_sets) * len(feature_packs) * len(meta_models)

    candidate_rows: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    candidate_idx = 0

    log_message(
        progress_log_path,
        "Starting stack pipeline search: "
        f"candidate_sets={len(candidate_sets)}, feature_packs={len(feature_packs)}, "
        f"meta_models={len(meta_models)}, total_candidates={total_candidates}",
    )

    for candidate_set in candidate_sets:
        candidate_set_name = str(candidate_set["name"])
        indices = resolve_candidate_indices(candidate_set, names, min_models)
        selected_names = [names[i] for i in indices]

        for feature_pack in feature_packs:
            feature_pack_name = str(feature_pack["name"])
            blocks = [str(block) for block in feature_pack["blocks"]]
            train_features, test_features, resolved_anchor_name = build_meta_features(
                feature_bank=feature_bank,
                indices=indices,
                names=names,
                blocks=blocks,
                anchor_model_name=anchor_model_name,
            )

            for model_spec in meta_models:
                candidate_idx += 1
                candidate_label = (
                    f"[{candidate_idx}/{total_candidates}] {candidate_set_name} | "
                    f"{feature_pack_name} | {model_spec['name']} | features={train_features.shape[1]}"
                )
                log_message(progress_log_path, f"Running {candidate_label}")
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
                    "feature_pack": feature_pack_name,
                    "feature_blocks": blocks,
                    "meta_model": str(model_spec["name"]),
                    "selected_models": selected_names,
                    "n_selected_models": len(selected_names),
                    "n_features": int(train_features.shape[1]),
                    "anchor_model_name": resolved_anchor_name,
                    "reference_oof_auc": reference_auc,
                    "status": result["status"],
                }

                if result["status"] != "ok":
                    row["reason"] = result["reason"]
                    candidate_rows.append(row)
                    append_jsonl(candidate_progress_path, row)
                    log_message(progress_log_path, f"Skipped {candidate_label}: {row['reason']}")
                    continue

                row.update(
                    {
                        "overall_auc": result["overall_auc"],
                        "fold_auc_mean": result["fold_auc_mean"],
                        "fold_auc_std": result["fold_auc_std"],
                        "n_meta_folds": result["n_meta_folds"],
                        "beats_reference": None if reference_auc is None else bool(result["overall_auc"] > reference_auc),
                    }
                )
                candidate_rows.append(row)
                append_jsonl(candidate_progress_path, row)
                finished_message = f"Finished {candidate_label} | auc={row['overall_auc']:.9f}"
                if reference_auc is not None:
                    finished_message += f" | delta={row['overall_auc'] - reference_auc:+.9f}"
                log_message(progress_log_path, finished_message)

                candidate_payload = {
                    **row,
                    "fold_aucs": result["fold_aucs"],
                    "oof_pred": result["oof_pred"],
                    "test_pred": result["test_pred"],
                }

                if best_candidate is None:
                    best_candidate = candidate_payload
                    write_best_snapshot(best_snapshot_path, best_candidate)
                    log_message(
                        progress_log_path,
                        "New best candidate: "
                        f"{best_candidate['candidate_set']} | {best_candidate['feature_pack']} | "
                        f"{best_candidate['meta_model']} | auc={best_candidate['overall_auc']:.9f}",
                    )
                    continue

                current_key = (
                    candidate_payload["overall_auc"],
                    -candidate_payload["fold_auc_std"],
                    -candidate_payload["n_features"],
                    -candidate_payload["n_selected_models"],
                )
                best_key = (
                    best_candidate["overall_auc"],
                    -best_candidate["fold_auc_std"],
                    -best_candidate["n_features"],
                    -best_candidate["n_selected_models"],
                )
                if current_key > best_key:
                    best_candidate = candidate_payload
                    write_best_snapshot(best_snapshot_path, best_candidate)
                    log_message(
                        progress_log_path,
                        "New best candidate: "
                        f"{best_candidate['candidate_set']} | {best_candidate['feature_pack']} | "
                        f"{best_candidate['meta_model']} | auc={best_candidate['overall_auc']:.9f}",
                    )

    if best_candidate is None:
        raise RuntimeError("No stacking pipeline candidate finished successfully.")

    candidate_rows.sort(
        key=lambda item: (
            -float(item.get("overall_auc", -1.0)),
            float(item.get("fold_auc_std", 999.0)),
            int(item.get("n_features", 999999)),
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

    print("Stack pipeline search completed.")
    print(f"Available models: {names}")
    if skipped_models:
        print(f"Skipped models: {skipped_models}")
    if reference_auc is not None and reference_name is not None:
        print(f"Reference '{reference_name}' OOF AUC: {reference_auc:.6f}")
    print(
        "Best candidate: "
        f"{best_candidate['candidate_set']} | {best_candidate['feature_pack']} | "
        f"{best_candidate['meta_model']} | OOF AUC={best_candidate['overall_auc']:.6f}"
    )


if __name__ == "__main__":
    main()
