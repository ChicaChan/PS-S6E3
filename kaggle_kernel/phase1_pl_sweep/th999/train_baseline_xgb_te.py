# Usage:
# 1) Kaggle remote training:
#    python src/remote/train_baseline_xgb_te.py \
#      --train-path /kaggle/input/competitions/playground-series-s6e3/train.csv \
#      --test-path /kaggle/input/competitions/playground-series-s6e3/test.csv \
#      --sample-submission-path /kaggle/input/competitions/playground-series-s6e3/sample_submission.csv \
#      --config-path src/remote/config_baseline.json \
#      --output-dir /kaggle/working/outputs/baseline
#
# 2) Local smoke mode (small sampled csv):
#    python src/remote/train_baseline_xgb_te.py \
#      --train-path .artifacts/smoke/input/train.csv \
#      --test-path .artifacts/smoke/input/test.csv \
#      --sample-submission-path .artifacts/smoke/input/sample_submission.csv \
#      --config-path .artifacts/smoke/input/smoke_config.json \
#      --output-dir .artifacts/smoke/output

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


DEFAULT_CONFIG: dict[str, Any] = {
    "target_col": "Churn",
    "id_col": "id",
    "positive_label": "Yes",
    "negative_label": "No",
    "seed": 42,
    "n_folds": 5,
    "inner_folds": 5,
    "use_pseudo_label": True,
    "pseudo_label_threshold": 0.999,
    "min_pseudo_label_count": 500,
    "xgb_params": {
        "n_estimators": 50000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.05,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "early_stopping_rounds": 300,
        "enable_categorical": True,
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 0,
    },
}


def read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()
    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    merged = DEFAULT_CONFIG.copy()
    merged.update({k: v for k, v in loaded.items() if k != "xgb_params"})
    merged["xgb_params"] = DEFAULT_CONFIG["xgb_params"].copy()
    merged["xgb_params"].update(loaded.get("xgb_params", {}))
    return merged


def to_binary_target(series: pd.Series, pos_label: str, neg_label: str) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        values = series.to_numpy()
        uniq = set(np.unique(values))
        if uniq.issubset({0, 1}):
            return values.astype(np.int32)
    mapped = series.astype(str).str.strip().replace({pos_label: "1", neg_label: "0"})
    if mapped.isin(["0", "1"]).all():
        return mapped.astype(np.int32).to_numpy()
    lowered = series.astype(str).str.strip().str.lower()
    mapped2 = lowered.replace({"yes": "1", "no": "0", "true": "1", "false": "0"})
    if mapped2.isin(["0", "1"]).all():
        return mapped2.astype(np.int32).to_numpy()
    raise ValueError("Target column cannot be converted to binary labels.")


def resolve_n_splits(y: np.ndarray, desired: int) -> int:
    class_counts = np.bincount(y)
    positive_counts = class_counts[class_counts > 0]
    min_class_count = int(positive_counts.min())
    n_splits = min(desired, min_class_count)
    if n_splits < 2:
        raise ValueError("Not enough samples per class to run at least 2 folds.")
    return int(n_splits)


def detect_feature_columns(
    train_df: pd.DataFrame, id_col: str, target_col: str
) -> tuple[list[str], list[str], list[str]]:
    feature_cols = [c for c in train_df.columns if c not in {id_col, target_col}]

    def is_categorical_like_dtype(dtype: Any) -> bool:
        return (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_categorical_dtype(dtype)
        )

    cat_cols = [
        c
        for c in feature_cols
        if is_categorical_like_dtype(train_df[c].dtype)
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, num_cols, cat_cols


def cast_numeric_with_train_median(
    x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame, num_cols: list[str]
) -> None:
    for col in num_cols:
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce")
        x_val[col] = pd.to_numeric(x_val[col], errors="coerce")
        x_test[col] = pd.to_numeric(x_test[col], errors="coerce")
        fill_value = float(x_train[col].median()) if x_train[col].notna().any() else 0.0
        x_train[col] = x_train[col].fillna(fill_value).astype(np.float32)
        x_val[col] = x_val[col].fillna(fill_value).astype(np.float32)
        x_test[col] = x_test[col].fillna(fill_value).astype(np.float32)


def cast_categories_consistently(
    x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame, cat_cols: list[str]
) -> None:
    for col in cat_cols:
        train_s = x_train[col].astype("object").where(x_train[col].notna(), "__MISSING__").astype(str)
        val_s = x_val[col].astype("object").where(x_val[col].notna(), "__MISSING__").astype(str)
        test_s = x_test[col].astype("object").where(x_test[col].notna(), "__MISSING__").astype(str)
        categories = pd.Index(pd.concat([train_s, val_s, test_s], axis=0).unique(), dtype="object")
        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        x_train[col] = train_s.astype(dtype)
        x_val[col] = val_s.astype(dtype)
        x_test[col] = test_s.astype(dtype)


def build_target_mean_features(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    cat_cols: list[str],
    inner_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_mean = float(np.mean(y_train))
    inner_n_splits = resolve_n_splits(y_train, inner_folds)
    inner_skf = StratifiedKFold(n_splits=inner_n_splits, shuffle=True, random_state=seed)
    inner_splits = list(inner_skf.split(np.zeros(len(x_train)), y_train))
    y_series = pd.Series(y_train)

    te_train = pd.DataFrame(index=x_train.index)
    te_val = pd.DataFrame(index=x_val.index)
    te_test = pd.DataFrame(index=x_test.index)

    for col in cat_cols:
        col_name = f"TE_MEAN_{col}"
        tr_values = x_train[col].astype("string").fillna("__MISSING__")
        val_values = x_val[col].astype("string").fillna("__MISSING__")
        test_values = x_test[col].astype("string").fillna("__MISSING__")

        oof_encoded = np.full(len(x_train), global_mean, dtype=np.float32)

        for in_tr_idx, in_va_idx in inner_splits:
            mean_map = y_series.iloc[in_tr_idx].groupby(tr_values.iloc[in_tr_idx], sort=False).mean()
            mapped = tr_values.iloc[in_va_idx].map(mean_map).fillna(global_mean).to_numpy(dtype=np.float32)
            oof_encoded[in_va_idx] = mapped

        full_mean_map = y_series.groupby(tr_values, sort=False).mean()

        te_train[col_name] = oof_encoded
        te_val[col_name] = val_values.map(full_mean_map).fillna(global_mean).astype(np.float32)
        te_test[col_name] = test_values.map(full_mean_map).fillna(global_mean).astype(np.float32)

    return te_train, te_val, te_test


def fit_fold_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    xgb_params: dict[str, Any],
) -> tuple[XGBClassifier, np.ndarray, float]:
    model = XGBClassifier(**xgb_params)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    val_pred = model.predict_proba(x_val)[:, 1].astype(np.float32)
    val_auc = float(roc_auc_score(y_val, val_pred))
    return model, val_pred, val_auc


def maybe_upgrade_with_pseudo_label(
    model: XGBClassifier,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    x_test: pd.DataFrame,
    xgb_params: dict[str, Any],
    threshold: float,
    min_count: int,
) -> tuple[XGBClassifier, np.ndarray, float, dict[str, Any]]:
    base_val_pred = model.predict_proba(x_val)[:, 1].astype(np.float32)
    base_auc = float(roc_auc_score(y_val, base_val_pred))

    test_pred = model.predict_proba(x_test)[:, 1].astype(np.float32)
    confident_mask = (test_pred >= threshold) | (test_pred <= (1.0 - threshold))
    confident_count = int(confident_mask.sum())

    info = {
        "pseudo_label_attempted": True,
        "base_auc": base_auc,
        "pseudo_auc": base_auc,
        "pseudo_label_count": confident_count,
        "pseudo_used": False,
    }

    if confident_count < min_count:
        return model, base_val_pred, base_auc, info

    x_pl = pd.concat([x_train, x_test.loc[confident_mask]], axis=0, ignore_index=True)
    y_pl = np.concatenate([y_train, (test_pred[confident_mask] >= 0.5).astype(np.int32)])

    model_pl = XGBClassifier(**xgb_params)
    model_pl.fit(x_pl, y_pl, eval_set=[(x_val, y_val)], verbose=False)
    pseudo_val_pred = model_pl.predict_proba(x_val)[:, 1].astype(np.float32)
    pseudo_auc = float(roc_auc_score(y_val, pseudo_val_pred))

    info["pseudo_auc"] = pseudo_auc
    info["pseudo_used"] = pseudo_auc > base_auc
    if pseudo_auc > base_auc:
        return model_pl, pseudo_val_pred, pseudo_auc, info
    return model, base_val_pred, base_auc, info


def run_pipeline(
    train_path: Path,
    test_path: Path,
    sample_submission_path: Path,
    config_path: Path,
    output_dir: Path,
) -> None:
    config = read_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_col = config["target_col"]
    id_col = config["id_col"]
    seed = int(config["seed"])

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub_df = pd.read_csv(sample_submission_path)

    if target_col not in train_df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if id_col not in train_df.columns or id_col not in test_df.columns:
        raise ValueError(f"Missing id column: {id_col}")

    y = to_binary_target(
        train_df[target_col],
        pos_label=config["positive_label"],
        neg_label=config["negative_label"],
    )
    feature_cols, num_cols, cat_cols = detect_feature_columns(train_df, id_col, target_col)

    xgb_params = config["xgb_params"].copy()
    xgb_params["random_state"] = seed
    if "n_jobs" not in xgb_params:
        xgb_params["n_jobs"] = -1

    n_splits = resolve_n_splits(y, int(config["n_folds"]))
    outer_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_pred = np.zeros(len(train_df), dtype=np.float32)
    test_pred = np.zeros(len(test_df), dtype=np.float32)
    fold_records: list[dict[str, Any]] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(outer_skf.split(np.zeros(len(train_df)), y), start=1):
        x_train = train_df.iloc[tr_idx][feature_cols].reset_index(drop=True).copy()
        x_val = train_df.iloc[va_idx][feature_cols].reset_index(drop=True).copy()
        x_test = test_df[feature_cols].reset_index(drop=True).copy()
        y_train = y[tr_idx]
        y_val = y[va_idx]

        cast_numeric_with_train_median(x_train, x_val, x_test, num_cols)
        cast_categories_consistently(x_train, x_val, x_test, cat_cols)

        te_train, te_val, te_test = build_target_mean_features(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            x_test=x_test,
            cat_cols=cat_cols,
            inner_folds=int(config["inner_folds"]),
            seed=seed + fold_idx,
        )

        x_train_all = pd.concat([x_train, te_train], axis=1)
        x_val_all = pd.concat([x_val, te_val], axis=1)
        x_test_all = pd.concat([x_test, te_test], axis=1)

        model, val_fold_pred, val_auc = fit_fold_model(
            x_train=x_train_all,
            y_train=y_train,
            x_val=x_val_all,
            y_val=y_val,
            xgb_params=xgb_params,
        )

        pseudo_info = {
            "pseudo_label_attempted": False,
            "base_auc": val_auc,
            "pseudo_auc": val_auc,
            "pseudo_label_count": 0,
            "pseudo_used": False,
        }
        if bool(config["use_pseudo_label"]):
            model, val_fold_pred, val_auc, pseudo_info = maybe_upgrade_with_pseudo_label(
                model=model,
                x_train=x_train_all,
                y_train=y_train,
                x_val=x_val_all,
                y_val=y_val,
                x_test=x_test_all,
                xgb_params=xgb_params,
                threshold=float(config["pseudo_label_threshold"]),
                min_count=int(config["min_pseudo_label_count"]),
            )

        test_fold_pred = model.predict_proba(x_test_all)[:, 1].astype(np.float32)
        oof_pred[va_idx] = val_fold_pred
        test_pred += test_fold_pred / n_splits

        fold_records.append(
            {
                "fold": fold_idx,
                "fold_auc": float(val_auc),
                **pseudo_info,
            }
        )
        print(
            f"Fold {fold_idx}/{n_splits} AUC={val_auc:.6f}, "
            f"pseudo_used={pseudo_info['pseudo_used']}, pseudo_count={pseudo_info['pseudo_label_count']}"
        )

    overall_auc = float(roc_auc_score(y, oof_pred))
    print(f"OOF AUC={overall_auc:.6f}")

    submission_df = sample_sub_df.copy()
    if id_col not in submission_df.columns:
        submission_df[id_col] = test_df[id_col].values
    submission_df[target_col] = test_pred.astype(np.float32)
    submission_df = submission_df[[id_col, target_col]]

    oof_df = pd.DataFrame(
        {
            id_col: train_df[id_col].values,
            "target_binary": y.astype(np.int32),
            "oof_prediction": oof_pred.astype(np.float32),
        }
    )

    metrics = {
        "overall_auc": overall_auc,
        "folds": fold_records,
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_features_total": int(len(feature_cols)),
        "n_numeric_features": int(len(num_cols)),
        "n_categorical_features": int(len(cat_cols)),
        "config": {
            "n_folds": n_splits,
            "inner_folds": int(config["inner_folds"]),
            "seed": seed,
            "use_pseudo_label": bool(config["use_pseudo_label"]),
            "pseudo_label_threshold": float(config["pseudo_label_threshold"]),
            "min_pseudo_label_count": int(config["min_pseudo_label_count"]),
            "xgb_params": xgb_params,
        },
    }

    submission_path = output_dir / "submission.csv"
    oof_path = output_dir / "oof_predictions.csv"
    metrics_path = output_dir / "cv_metrics.json"

    submission_df.to_csv(submission_path, index=False)
    oof_df.to_csv(oof_path, index=False)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {submission_path}")
    print(f"Saved: {oof_path}")
    print(f"Saved: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PS-S6E3 baseline trainer (XGB + leak-free TE)")
    parser.add_argument(
        "--train-path",
        type=Path,
        required=False,
        default=Path("/kaggle/input/playground-series-s6e3/train.csv"),
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        required=False,
        default=Path("/kaggle/input/playground-series-s6e3/test.csv"),
    )
    parser.add_argument(
        "--sample-submission-path",
        type=Path,
        required=False,
        default=Path("/kaggle/input/playground-series-s6e3/sample_submission.csv"),
    )
    parser.add_argument("--config-path", type=Path, required=False, default=Path("config_baseline.json"))
    parser.add_argument("--output-dir", type=Path, required=False, default=Path("/kaggle/working"))
    return parser.parse_args()


def auto_find_input_file(name: str) -> Path:
    root = Path("/kaggle/input")
    if root.exists():
        direct = root / name
        if direct.exists():
            return direct
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot locate '{name}' under /kaggle/input")


def main() -> None:
    args = parse_args()
    train_path = args.train_path if args.train_path.exists() else auto_find_input_file("train.csv")
    test_path = args.test_path if args.test_path.exists() else auto_find_input_file("test.csv")
    sample_path = (
        args.sample_submission_path
        if args.sample_submission_path.exists()
        else auto_find_input_file("sample_submission.csv")
    )
    run_pipeline(
        train_path=train_path,
        test_path=test_path,
        sample_submission_path=sample_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

