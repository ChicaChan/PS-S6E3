# Usage:
# 1) Kaggle remote training:
#    python train_phase16_catboost_orig_transfer.py \
#      --train-path /kaggle/input/playground-series-s6e3/train.csv \
#      --test-path /kaggle/input/playground-series-s6e3/test.csv \
#      --sample-submission-path /kaggle/input/playground-series-s6e3/sample_submission.csv \
#      --orig-path /kaggle/input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv \
#      --config-path config_phase16_catboost_orig_transfer.json \
#      --output-dir /kaggle/working/phase16_catboost_orig_transfer_v1
#
# 2) Local smoke mode:
#    python train_phase16_catboost_orig_transfer.py \
#      --train-path .artifacts/smoke_phase16_catboost_orig_transfer/input/train.csv \
#      --test-path .artifacts/smoke_phase16_catboost_orig_transfer/input/test.csv \
#      --sample-submission-path .artifacts/smoke_phase16_catboost_orig_transfer/input/sample_submission.csv \
#      --config-path config_phase16_catboost_orig_transfer_smoke.json \
#      --output-dir .artifacts/smoke_phase16_catboost_orig_transfer/output

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


DEFAULT_TOP_CATS = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
]

DEFAULT_ORIG_SINGLE_COLS = [
    "Contract",
    "PaymentMethod",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "OnlineBackup",
    "DeviceProtection",
    "PaperlessBilling",
    "StreamingMovies",
    "StreamingTV",
    "Partner",
    "Dependents",
]

SHORT_NAMES = {
    "InternetService": "IS",
    "Contract": "C",
    "PaymentMethod": "PM",
    "MonthlyCharges": "MC",
    "TotalCharges": "TC",
    "tenure": "T",
}

DEFAULT_CONFIG: dict[str, Any] = {
    "run_name": "phase16_catboost_orig_transfer_v5_default_screen",
    "target_col": "Churn",
    "id_col": "id",
    "positive_label": "Yes",
    "negative_label": "No",
    "seed": 42,
    "n_folds": 2,
    "inner_folds": 2,
    "max_te_categories": 800,
    "orig_data_path": "",
    "top_cats_for_ngram": ["Contract", "InternetService", "PaymentMethod"],
    "orig_single_mode": "selected",
    "orig_single_cols": ["Contract", "PaymentMethod", "InternetService", "OnlineSecurity"],
    "enable_orig_cross": False,
    "enable_orig_support_features": False,
    "enable_orig_logit_features": False,
    "enable_pctrank_orig": True,
    "enable_pctrank_churn_gap": True,
    "enable_zscore_orig": False,
    "conditional_rank_group_cols": ["InternetService"],
    "conditional_rank_value_col": "TotalCharges",
    "conditional_residual_specs": [],
    "frequency_numeric_cols": ["TotalCharges"],
    "frequency_categorical_cols": ["Contract"],
    "quantile_distance_cols": ["TotalCharges"],
    "quantile_levels": [0.5],
    "cat_params": {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "learning_rate": 0.08,
        "depth": 4,
        "l2_leaf_reg": 3.0,
        "n_estimators": 1200,
        "random_seed": 42,
        "verbose": 0,
        "task_type": "GPU",
        "use_best_model": True,
        "od_type": "Iter",
        "od_wait": 40,
    },
}


def read_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        print(f"Config file not found, using DEFAULT_CONFIG: {config_path}")
        return DEFAULT_CONFIG.copy()
    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    print(f"Loaded config from: {config_path}")
    merged = DEFAULT_CONFIG.copy()
    merged.update({k: v for k, v in loaded.items() if k != "cat_params"})
    merged["cat_params"] = DEFAULT_CONFIG["cat_params"].copy()
    merged["cat_params"].update(loaded.get("cat_params", {}))
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


def is_categorical_like_dtype(dtype: Any) -> bool:
    return (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def detect_feature_columns(train_df: pd.DataFrame, id_col: str, target_col: str) -> tuple[list[str], list[str], list[str]]:
    feature_cols = [c for c in train_df.columns if c not in {id_col, target_col}]
    cat_cols = [c for c in feature_cols if is_categorical_like_dtype(train_df[c].dtype)]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, num_cols, cat_cols


def cast_numeric_with_train_median(x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame, num_cols: list[str]) -> None:
    for col in num_cols:
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce")
        x_val[col] = pd.to_numeric(x_val[col], errors="coerce")
        x_test[col] = pd.to_numeric(x_test[col], errors="coerce")
        fill_value = float(x_train[col].median()) if x_train[col].notna().any() else 0.0
        x_train[col] = x_train[col].fillna(fill_value).astype(np.float32)
        x_val[col] = x_val[col].fillna(fill_value).astype(np.float32)
        x_test[col] = x_test[col].fillna(fill_value).astype(np.float32)


def cast_categories_consistently(x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame, cat_cols: list[str]) -> None:
    for col in cat_cols:
        tr = x_train[col].astype("object").where(x_train[col].notna(), "__MISSING__").astype(str)
        va = x_val[col].astype("object").where(x_val[col].notna(), "__MISSING__").astype(str)
        te = x_test[col].astype("object").where(x_test[col].notna(), "__MISSING__").astype(str)
        categories = pd.Index(pd.concat([tr, va, te], axis=0).unique(), dtype="object")
        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        x_train[col] = tr.astype(dtype)
        x_val[col] = va.astype(dtype)
        x_test[col] = te.astype(dtype)


def build_target_mean_features(x_train: pd.DataFrame, y_train: np.ndarray, x_val: pd.DataFrame, x_test: pd.DataFrame, te_cols: list[str], inner_folds: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_mean = float(np.mean(y_train))
    te_train = pd.DataFrame(index=x_train.index)
    te_val = pd.DataFrame(index=x_val.index)
    te_test = pd.DataFrame(index=x_test.index)
    if not te_cols:
        return te_train, te_val, te_test

    inner_n_splits = resolve_n_splits(y_train, inner_folds)
    inner_skf = StratifiedKFold(n_splits=inner_n_splits, shuffle=True, random_state=seed)
    inner_splits = list(inner_skf.split(np.zeros(len(x_train)), y_train))
    y_series = pd.Series(y_train)

    for col in te_cols:
        col_name = f"TE_MEAN_{col}"
        tr_values = x_train[col].astype("string").fillna("__MISSING__")
        va_values = x_val[col].astype("string").fillna("__MISSING__")
        te_values = x_test[col].astype("string").fillna("__MISSING__")
        oof_encoded = np.full(len(x_train), global_mean, dtype=np.float32)

        for in_tr_idx, in_va_idx in inner_splits:
            mean_map = y_series.iloc[in_tr_idx].groupby(tr_values.iloc[in_tr_idx], sort=False).mean()
            mapped = tr_values.iloc[in_va_idx].map(mean_map).fillna(global_mean).to_numpy(dtype=np.float32)
            oof_encoded[in_va_idx] = mapped

        full_mean_map = y_series.groupby(tr_values, sort=False).mean()
        te_train[col_name] = oof_encoded
        te_val[col_name] = va_values.map(full_mean_map).fillna(global_mean).astype(np.float32)
        te_test[col_name] = te_values.map(full_mean_map).fillna(global_mean).astype(np.float32)

    return te_train, te_val, te_test


def normalize_string_category(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype("object").where(df[col].notna(), "__MISSING__").astype(str)


def short_feature_name(name: str) -> str:
    return SHORT_NAMES.get(name, name.replace(" ", "").replace("-", "_"))


def safe_logit(values: pd.Series | np.ndarray, eps: float = 1e-5) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def pctrank_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if reference.size == 0:
        return np.full(values.shape[0], 0.5, dtype=np.float32)
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values, side="left") / max(ref_sorted.size, 1)).astype(np.float32)


def zscore_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if reference.size == 0:
        return np.zeros(values.shape[0], dtype=np.float32)
    mean_val = float(np.mean(reference))
    std_val = float(np.std(reference))
    if std_val == 0.0:
        return np.zeros(values.shape[0], dtype=np.float32)
    return ((values - mean_val) / std_val).astype(np.float32)


def build_digit_features(series: pd.Series, prefix: str) -> pd.DataFrame:
    numeric = pd.to_numeric(series, errors="coerce")
    missing = numeric.isna().astype(np.int8)
    scaled = np.floor(numeric.fillna(0).abs()).astype(np.int64)
    text = scaled.astype(str)
    out = pd.DataFrame(index=series.index)
    out[f"{prefix}_num_digits"] = text.str.len().astype(np.float32)
    out[f"{prefix}_first_digit"] = text.str[0].fillna("0").astype(np.int16).astype(np.float32)
    out[f"{prefix}_second_digit"] = text.str[1].fillna("0").astype(np.int16).astype(np.float32)
    out[f"{prefix}_last_digit"] = text.str[-1].fillna("0").astype(np.int16).astype(np.float32)
    out[f"{prefix}_is_missing"] = missing.astype(np.float32)
    return out


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if {"tenure", "MonthlyCharges", "TotalCharges"}.issubset(out.columns):
        out["charges_deviation"] = (out["TotalCharges"] - out["tenure"] * out["MonthlyCharges"]).astype(np.float32)
        out["charges_deviation_abs"] = np.abs(out["charges_deviation"]).astype(np.float32)
        out["monthly_to_total_ratio"] = (out["MonthlyCharges"] / (out["TotalCharges"] + 1.0)).astype(np.float32)
        out["avg_monthly_charges"] = (out["TotalCharges"] / (out["tenure"] + 1.0)).astype(np.float32)
        out["tenure_mod10"] = (out["tenure"] % 10).astype(np.float32)
        out["tenure_mod12"] = (out["tenure"] % 12).astype(np.float32)
        out["is_new_customer"] = (out["tenure"].fillna(0) <= 1).astype(np.float32)
        out["charges_dev_zeroish"] = (np.abs(out["charges_deviation"].fillna(0)) <= 1.0).astype(np.float32)
        out["charges_dev_sign"] = np.sign(out["charges_deviation"].fillna(0)).astype(np.float32)
        for prefix, col in [("tenure", "tenure"), ("mc", "MonthlyCharges"), ("tc", "TotalCharges")]:
            out = pd.concat([out, build_digit_features(out[col], prefix)], axis=1)

    service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    existing_service_cols = [c for c in service_cols if c in out.columns]
    if existing_service_cols:
        out["service_count"] = (out[existing_service_cols] == "Yes").sum(axis=1).astype(np.float32)
        out["has_service_bundle"] = (out["service_count"] >= 3).astype(np.float32)

    if "InternetService" in out.columns:
        out["has_internet"] = (out["InternetService"].astype(str) != "No").astype(np.float32)
    if "PhoneService" in out.columns:
        out["has_phone"] = (out["PhoneService"].astype(str) == "Yes").astype(np.float32)
    if "Contract" in out.columns:
        out["is_month_to_month"] = (out["Contract"].astype(str) == "Month-to-month").astype(np.float32)

    if {"Contract", "InternetService"}.issubset(out.columns):
        out["cross_contract_internet"] = out["Contract"].astype(str) + "__" + out["InternetService"].astype(str)
    if {"PaymentMethod", "Contract"}.issubset(out.columns):
        out["cross_payment_contract"] = out["PaymentMethod"].astype(str) + "__" + out["Contract"].astype(str)

    for col in ["tenure", "MonthlyCharges", "TotalCharges", "service_count", "tenure_mod12"]:
        if col in out.columns:
            rounded = pd.to_numeric(out[col], errors="coerce").round(3)
            out[f"CAT_{col}"] = rounded.astype(str).replace("nan", "__MISSING__")
    return out


def add_ngram_features(df: pd.DataFrame, top_cats: list[str]) -> pd.DataFrame:
    out = df.copy()
    usable = [c for c in top_cats if c in out.columns]
    for c1, c2 in itertools.combinations(usable, 2):
        out[f"BG_{c1}_{c2}"] = normalize_string_category(out, c1) + "_" + normalize_string_category(out, c2)
    for c1, c2, c3 in itertools.combinations(usable[:4], 3):
        out[f"TG_{c1}_{c2}_{c3}"] = normalize_string_category(out, c1) + "_" + normalize_string_category(out, c2) + "_" + normalize_string_category(out, c3)
    return out


def load_original_reference(train_df: pd.DataFrame, y: np.ndarray, target_col: str, config: dict[str, Any], cli_orig_path: Path | None) -> pd.DataFrame:
    candidate_paths: list[Path] = []
    if cli_orig_path is not None:
        candidate_paths.append(cli_orig_path)
    cfg_orig = str(config.get("orig_data_path", "")).strip()
    if cfg_orig:
        candidate_paths.append(Path(cfg_orig))
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        candidate_paths.extend(sorted(kaggle_root.rglob("WA_Fn-UseC_-Telco-Customer-Churn.csv")))

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            ref = pd.read_csv(path)
            if "customerID" in ref.columns:
                ref = ref.drop(columns=["customerID"])
            if target_col not in ref.columns:
                continue
            ref[target_col] = to_binary_target(ref[target_col], pos_label=str(config["positive_label"]), neg_label=str(config["negative_label"]))
            if "TotalCharges" in ref.columns:
                ref["TotalCharges"] = pd.to_numeric(ref["TotalCharges"], errors="coerce")
                ref["TotalCharges"] = ref["TotalCharges"].fillna(ref["TotalCharges"].median())
            print(f"Using original reference data: {path}")
            return ref
        except Exception:
            continue

    fallback = train_df.copy()
    fallback[target_col] = y
    print("Original reference data not found. Fallback to train split as reference.")
    return fallback

def add_frequency_features(train_df: pd.DataFrame, test_df: pd.DataFrame, orig_df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    train = train_df.copy()
    test = test_df.copy()
    created = 0
    train_extra: dict[str, pd.Series] = {}
    test_extra: dict[str, pd.Series] = {}

    for col in [str(x) for x in config.get("frequency_numeric_cols", [])]:
        if col not in train.columns or col not in test.columns or col not in orig_df.columns:
            continue
        train_key = pd.to_numeric(train[col], errors="coerce").round(3).astype(str).replace("nan", "__MISSING__")
        test_key = pd.to_numeric(test[col], errors="coerce").round(3).astype(str).replace("nan", "__MISSING__")
        orig_key = pd.to_numeric(orig_df[col], errors="coerce").round(3).astype(str).replace("nan", "__MISSING__")
        freq = pd.concat([train_key, test_key, orig_key], axis=0).value_counts(normalize=True)
        feature_name = f"FREQ_{short_feature_name(col)}"
        train_extra[feature_name] = train_key.map(freq).fillna(0.0).astype(np.float32)
        test_extra[feature_name] = test_key.map(freq).fillna(0.0).astype(np.float32)
        created += 1

    for col in [str(x) for x in config.get("frequency_categorical_cols", [])]:
        if col not in train.columns or col not in test.columns or col not in orig_df.columns:
            continue
        train_key = normalize_string_category(train, col)
        test_key = normalize_string_category(test, col)
        orig_key = normalize_string_category(orig_df, col)
        freq = pd.concat([train_key, test_key, orig_key], axis=0).value_counts(normalize=True)
        feature_name = f"FREQ_{short_feature_name(col)}"
        train_extra[feature_name] = train_key.map(freq).fillna(0.0).astype(np.float32)
        test_extra[feature_name] = test_key.map(freq).fillna(0.0).astype(np.float32)
        created += 1

    if train_extra:
        train = pd.concat([train, pd.DataFrame(train_extra, index=train.index)], axis=1).copy()
        test = pd.concat([test, pd.DataFrame(test_extra, index=test.index)], axis=1).copy()

    return train, test, created


def build_smoothed_target_mapping(orig_df: pd.DataFrame, target_col: str, key_series: pd.Series, global_mean: float, smoothing: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    grouped = orig_df.groupby(key_series, sort=False)[target_col].agg(["mean", "count"])
    smoothed = ((grouped["mean"] * grouped["count"]) + global_mean * smoothing) / (grouped["count"] + smoothing)
    count_map = grouped["count"].astype(np.float32)
    support_map = (grouped["count"] / max(len(orig_df), 1)).astype(np.float32)
    return smoothed.astype(np.float32), count_map, support_map


def add_orig_signal_features(train_df: pd.DataFrame, test_df: pd.DataFrame, orig_df: pd.DataFrame, target_col: str, top_cats: list[str], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train = train_df.copy()
    test = test_df.copy()
    global_mean = float(np.mean(orig_df[target_col].to_numpy(dtype=np.float32)))
    smoothing = 30.0
    created = {"freq": 0, "orig_single": 0, "orig_cross": 0, "orig_dist": 0, "orig_conditional": 0, "orig_quantile": 0}
    train_extra: dict[str, pd.Series | np.ndarray] = {}
    test_extra: dict[str, pd.Series | np.ndarray] = {}

    train, test, created["freq"] = add_frequency_features(train, test, orig_df, config)

    single_mode = str(config.get("orig_single_mode", "selected")).strip().lower()
    if single_mode == "top_cats":
        single_cols = [c for c in top_cats if c in train.columns and c in test.columns and c in orig_df.columns]
    elif single_mode == "all_categorical":
        single_cols = [c for c in train.columns if c != target_col and c in test.columns and c in orig_df.columns and is_categorical_like_dtype(train[c].dtype)]
    else:
        single_cols = [c for c in config.get("orig_single_cols", DEFAULT_ORIG_SINGLE_COLS) if c in train.columns and c in test.columns and c in orig_df.columns]

    enable_support = bool(config.get("enable_orig_support_features", True))
    enable_logit = bool(config.get("enable_orig_logit_features", True))
    for col in single_cols:
        orig_key = normalize_string_category(orig_df, col)
        rate_map, count_map, support_map = build_smoothed_target_mapping(orig_df, target_col, orig_key, global_mean, smoothing)
        train_key = normalize_string_category(train, col)
        test_key = normalize_string_category(test, col)
        base_name = f"ORIG_rate_{col}"
        train_rate = train_key.map(rate_map).fillna(global_mean).astype(np.float32)
        test_rate = test_key.map(rate_map).fillna(global_mean).astype(np.float32)
        train_extra[base_name] = train_rate
        test_extra[base_name] = test_rate
        train_extra[f"ORIG_delta_{col}"] = (train_rate - global_mean).astype(np.float32)
        test_extra[f"ORIG_delta_{col}"] = (test_rate - global_mean).astype(np.float32)
        if enable_logit:
            train_extra[f"ORIG_logit_{col}"] = safe_logit(train_rate)
            test_extra[f"ORIG_logit_{col}"] = safe_logit(test_rate)
        if enable_support:
            train_extra[f"ORIG_count_{col}"] = train_key.map(count_map).fillna(0.0).astype(np.float32)
            test_extra[f"ORIG_count_{col}"] = test_key.map(count_map).fillna(0.0).astype(np.float32)
            train_extra[f"ORIG_support_{col}"] = train_key.map(support_map).fillna(0.0).astype(np.float32)
            test_extra[f"ORIG_support_{col}"] = test_key.map(support_map).fillna(0.0).astype(np.float32)
        created["orig_single"] += 1

    if bool(config.get("enable_orig_cross", True)):
        cross_cols = [c for c in top_cats[:5] if c in train.columns and c in test.columns and c in orig_df.columns]
        for c1, c2 in itertools.combinations(cross_cols, 2):
            orig_key = normalize_string_category(orig_df, c1) + "__" + normalize_string_category(orig_df, c2)
            rate_map, count_map, _ = build_smoothed_target_mapping(orig_df, target_col, orig_key, global_mean, smoothing)
            train_key = normalize_string_category(train, c1) + "__" + normalize_string_category(train, c2)
            test_key = normalize_string_category(test, c1) + "__" + normalize_string_category(test, c2)
            name = f"ORIG_rate_{c1}_{c2}"
            train_extra[name] = train_key.map(rate_map).fillna(global_mean).astype(np.float32)
            test_extra[name] = test_key.map(rate_map).fillna(global_mean).astype(np.float32)
            train_extra[f"ORIG_count_{c1}_{c2}"] = train_key.map(count_map).fillna(0.0).astype(np.float32)
            test_extra[f"ORIG_count_{c1}_{c2}"] = test_key.map(count_map).fillna(0.0).astype(np.float32)
            created["orig_cross"] += 1

    enable_pctrank_orig = bool(config.get("enable_pctrank_orig", True))
    enable_pctrank_gap = bool(config.get("enable_pctrank_churn_gap", True))
    enable_zscore_orig = bool(config.get("enable_zscore_orig", True))
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col not in train.columns or col not in orig_df.columns:
            continue
        orig_all = pd.to_numeric(orig_df[col], errors="coerce").dropna().to_numpy(dtype=np.float32)
        orig_pos = pd.to_numeric(orig_df.loc[orig_df[target_col] == 1, col], errors="coerce").dropna().to_numpy(dtype=np.float32)
        orig_neg = pd.to_numeric(orig_df.loc[orig_df[target_col] == 0, col], errors="coerce").dropna().to_numpy(dtype=np.float32)
        if orig_all.size == 0:
            continue
        fill_value = float(np.nanmedian(orig_all))
        tr_vals = pd.to_numeric(train[col], errors="coerce").fillna(fill_value).to_numpy(dtype=np.float32)
        te_vals = pd.to_numeric(test[col], errors="coerce").fillna(fill_value).to_numpy(dtype=np.float32)
        short = short_feature_name(col)

        if enable_pctrank_orig:
            train_extra[f"pctrank_orig_{short}"] = pctrank_against(tr_vals, orig_all)
            test_extra[f"pctrank_orig_{short}"] = pctrank_against(te_vals, orig_all)
            train_extra[f"pctrank_ch_{short}"] = pctrank_against(tr_vals, orig_pos)
            test_extra[f"pctrank_ch_{short}"] = pctrank_against(te_vals, orig_pos)
            train_extra[f"pctrank_nc_{short}"] = pctrank_against(tr_vals, orig_neg)
            test_extra[f"pctrank_nc_{short}"] = pctrank_against(te_vals, orig_neg)
            created["orig_dist"] += 3

        if enable_pctrank_gap:
            train_extra[f"pctrank_gap_{short}"] = (pctrank_against(tr_vals, orig_pos) - pctrank_against(tr_vals, orig_neg)).astype(np.float32)
            test_extra[f"pctrank_gap_{short}"] = (pctrank_against(te_vals, orig_pos) - pctrank_against(te_vals, orig_neg)).astype(np.float32)
            created["orig_dist"] += 1

        if enable_zscore_orig:
            train_extra[f"zscore_orig_{short}"] = zscore_against(tr_vals, orig_all)
            test_extra[f"zscore_orig_{short}"] = zscore_against(te_vals, orig_all)
            train_extra[f"zscore_gap_{short}"] = (zscore_against(tr_vals, orig_pos) - zscore_against(tr_vals, orig_neg)).astype(np.float32)
            test_extra[f"zscore_gap_{short}"] = (zscore_against(te_vals, orig_pos) - zscore_against(te_vals, orig_neg)).astype(np.float32)
            created["orig_dist"] += 2

        train_extra[f"median_gap_{short}"] = (tr_vals - np.median(orig_all)).astype(np.float32)
        test_extra[f"median_gap_{short}"] = (te_vals - np.median(orig_all)).astype(np.float32)
        created["orig_dist"] += 1

    value_col = str(config.get("conditional_rank_value_col", "TotalCharges"))
    group_cols = [str(x) for x in config.get("conditional_rank_group_cols", [])]
    if value_col in train.columns and value_col in test.columns and value_col in orig_df.columns:
        for group_col in group_cols:
            if group_col not in train.columns or group_col not in test.columns or group_col not in orig_df.columns:
                continue
            cond_tr = np.full(len(train), 0.5, dtype=np.float32)
            cond_te = np.full(len(test), 0.5, dtype=np.float32)
            orig_group = normalize_string_category(orig_df, group_col)
            train_group = normalize_string_category(train, group_col)
            test_group = normalize_string_category(test, group_col)

            for val in sorted(orig_group.unique()):
                ref = pd.to_numeric(orig_df.loc[orig_group == val, value_col], errors="coerce").dropna().to_numpy(dtype=np.float32)
                if ref.size == 0:
                    continue
                ref_fill = float(np.nanmedian(ref))
                mask_tr = train_group == val
                mask_te = test_group == val
                if bool(mask_tr.any()):
                    tr_vals = pd.to_numeric(train.loc[mask_tr, value_col], errors="coerce").fillna(ref_fill).to_numpy(dtype=np.float32)
                    cond_tr[mask_tr.to_numpy()] = pctrank_against(tr_vals, ref)
                if bool(mask_te.any()):
                    te_vals = pd.to_numeric(test.loc[mask_te, value_col], errors="coerce").fillna(ref_fill).to_numpy(dtype=np.float32)
                    cond_te[mask_te.to_numpy()] = pctrank_against(te_vals, ref)

            feature_name = f"cond_pctrank_{short_feature_name(group_col)}_{short_feature_name(value_col)}"
            train_extra[feature_name] = cond_tr
            test_extra[feature_name] = cond_te
            created["orig_conditional"] += 1

    for spec in config.get("conditional_residual_specs", []):
        group_col = str(spec.get("group_col", ""))
        value_col = str(spec.get("value_col", ""))
        if not group_col or not value_col:
            continue
        if group_col not in train.columns or group_col not in test.columns or group_col not in orig_df.columns:
            continue
        if value_col not in train.columns or value_col not in test.columns or value_col not in orig_df.columns:
            continue
        orig_key = normalize_string_category(orig_df, group_col)
        grouped = pd.DataFrame({"group": orig_key, "value": pd.to_numeric(orig_df[value_col], errors="coerce")}).dropna()
        if grouped.empty:
            continue

        mean_map = grouped.groupby("group", sort=False)["value"].mean()
        std_map = grouped.groupby("group", sort=False)["value"].std().fillna(0.0)
        train_key = normalize_string_category(train, group_col)
        test_key = normalize_string_category(test, group_col)
        default_median = float(grouped["value"].median())
        default_mean = float(grouped["value"].mean())
        default_std = float(grouped["value"].std() or 1.0)
        tr_vals = pd.to_numeric(train[value_col], errors="coerce").fillna(default_median).to_numpy(dtype=np.float32)
        te_vals = pd.to_numeric(test[value_col], errors="coerce").fillna(default_median).to_numpy(dtype=np.float32)
        tr_center = train_key.map(mean_map).fillna(default_mean).to_numpy(dtype=np.float32)
        te_center = test_key.map(mean_map).fillna(default_mean).to_numpy(dtype=np.float32)
        tr_scale = train_key.map(std_map).fillna(default_std).replace(0, 1.0).to_numpy(dtype=np.float32)
        te_scale = test_key.map(std_map).fillna(default_std).replace(0, 1.0).to_numpy(dtype=np.float32)
        base = f"resid_{short_feature_name(group_col)}_{short_feature_name(value_col)}"
        train_extra[base] = (tr_vals - tr_center).astype(np.float32)
        test_extra[base] = (te_vals - te_center).astype(np.float32)
        train_extra[f"z_{base}"] = ((tr_vals - tr_center) / tr_scale).astype(np.float32)
        test_extra[f"z_{base}"] = ((te_vals - te_center) / te_scale).astype(np.float32)
        created["orig_conditional"] += 2

    for col in [str(x) for x in config.get("quantile_distance_cols", [])]:
        if col not in train.columns or col not in test.columns or col not in orig_df.columns:
            continue
        orig_pos = pd.to_numeric(orig_df.loc[orig_df[target_col] == 1, col], errors="coerce").dropna().to_numpy(dtype=np.float32)
        orig_neg = pd.to_numeric(orig_df.loc[orig_df[target_col] == 0, col], errors="coerce").dropna().to_numpy(dtype=np.float32)
        if orig_pos.size == 0 or orig_neg.size == 0:
            continue
        fill_value = float(np.nanmedian(np.concatenate([orig_pos, orig_neg], axis=0)))
        tr_vals = pd.to_numeric(train[col], errors="coerce").fillna(fill_value).to_numpy(dtype=np.float32)
        te_vals = pd.to_numeric(test[col], errors="coerce").fillna(fill_value).to_numpy(dtype=np.float32)
        short = short_feature_name(col)
        for q in [float(x) for x in config.get("quantile_levels", [])]:
            q_tag = str(q).replace(".", "p")
            pos_q = float(np.quantile(orig_pos, q))
            neg_q = float(np.quantile(orig_neg, q))
            train_extra[f"qdist_ch_{short}_{q_tag}"] = np.abs(tr_vals - pos_q).astype(np.float32)
            test_extra[f"qdist_ch_{short}_{q_tag}"] = np.abs(te_vals - pos_q).astype(np.float32)
            train_extra[f"qdist_nc_{short}_{q_tag}"] = np.abs(tr_vals - neg_q).astype(np.float32)
            test_extra[f"qdist_nc_{short}_{q_tag}"] = np.abs(te_vals - neg_q).astype(np.float32)
            train_extra[f"qdist_gap_{short}_{q_tag}"] = (np.abs(tr_vals - pos_q) - np.abs(tr_vals - neg_q)).astype(np.float32)
            test_extra[f"qdist_gap_{short}_{q_tag}"] = (np.abs(te_vals - pos_q) - np.abs(te_vals - neg_q)).astype(np.float32)
            created["orig_quantile"] += 3

    if train_extra:
        train = pd.concat([train, pd.DataFrame(train_extra, index=train.index)], axis=1).copy()
        test = pd.concat([test, pd.DataFrame(test_extra, index=test.index)], axis=1).copy()

    return train, test, created

def train_catboost(train_df: pd.DataFrame, test_df: pd.DataFrame, y: np.ndarray, feature_cols: list[str], num_cols: list[str], cat_cols: list[str], id_col: str, target_col: str, config: dict[str, Any], created_signals: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    from catboost import CatBoostClassifier

    n_splits = resolve_n_splits(y, int(config["n_folds"]))
    outer_skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(config["seed"]))
    oof_pred = np.zeros(len(train_df), dtype=np.float32)
    test_pred = np.zeros(len(test_df), dtype=np.float32)
    fold_aucs: list[float] = []
    best_iterations: list[int | None] = []
    feature_importances: list[pd.DataFrame] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(outer_skf.split(np.zeros(len(train_df)), y), start=1):
        x_train = train_df.iloc[tr_idx][feature_cols].reset_index(drop=True).copy()
        x_val = train_df.iloc[va_idx][feature_cols].reset_index(drop=True).copy()
        x_test = test_df[feature_cols].reset_index(drop=True).copy()
        y_train = y[tr_idx]
        y_val = y[va_idx]

        cast_numeric_with_train_median(x_train, x_val, x_test, num_cols)
        cast_categories_consistently(x_train, x_val, x_test, cat_cols)

        te_cols = [c for c in cat_cols if x_train[c].nunique(dropna=False) <= int(config.get("max_te_categories", 2500))]
        te_train, te_val, te_test = build_target_mean_features(x_train, y_train, x_val, x_test, te_cols, int(config["inner_folds"]), int(config["seed"]) + fold_idx)
        x_train_all = pd.concat([x_train, te_train], axis=1)
        x_val_all = pd.concat([x_val, te_val], axis=1)
        x_test_all = pd.concat([x_test, te_test], axis=1)

        final_cat_cols = [c for c in x_train_all.columns if isinstance(x_train_all[c].dtype, pd.CategoricalDtype)]
        for col in final_cat_cols:
            x_train_all[col] = x_train_all[col].astype(str)
            x_val_all[col] = x_val_all[col].astype(str)
            x_test_all[col] = x_test_all[col].astype(str)

        model = CatBoostClassifier(**config["cat_params"])
        model.fit(x_train_all, y_train, eval_set=(x_val_all, y_val), cat_features=final_cat_cols, verbose=False)

        val_pred = model.predict_proba(x_val_all)[:, 1].astype(np.float32)
        test_fold_pred = model.predict_proba(x_test_all)[:, 1].astype(np.float32)
        fold_auc = float(roc_auc_score(y_val, val_pred))
        oof_pred[va_idx] = val_pred
        test_pred += test_fold_pred / n_splits
        fold_aucs.append(fold_auc)
        best_iter = model.get_best_iteration()
        best_iterations.append(None if best_iter is None or best_iter < 0 else int(best_iter))
        feature_importances.append(pd.DataFrame({"feature": x_train_all.columns, f"importance_fold_{fold_idx}": model.get_feature_importance().astype(np.float32)}))
        print(f"[PHASE16-CAT] Fold {fold_idx}/{n_splits} AUC={fold_auc:.6f}, te_cols={len(te_cols)}")

    oof_df = pd.DataFrame({id_col: train_df[id_col].values, "target_binary": y.astype(np.int32), "oof_prediction": oof_pred})
    sub_df = pd.DataFrame({id_col: test_df[id_col].values, target_col: test_pred})

    importance_df = feature_importances[0]
    for part in feature_importances[1:]:
        importance_df = importance_df.merge(part, on="feature", how="outer")
    fold_cols = [c for c in importance_df.columns if c.startswith("importance_fold_")]
    importance_df["importance_mean"] = importance_df[fold_cols].mean(axis=1)
    importance_df["importance_std"] = importance_df[fold_cols].std(axis=1).fillna(0.0)
    importance_df = importance_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    metrics = {
        "model": "phase16_catboost_orig_transfer",
        "overall_auc": float(roc_auc_score(y, oof_pred)),
        "fold_aucs": fold_aucs,
        "auc_mean": float(np.mean(fold_aucs)),
        "auc_std": float(np.std(fold_aucs)),
        "best_iterations": best_iterations,
        "n_features_total": int(len(feature_cols)),
        "n_numeric_features": int(len(num_cols)),
        "n_categorical_features": int(len(cat_cols)),
        "created_orig_signals": created_signals,
        "run_name": str(config.get("run_name", "phase16_catboost_orig_transfer_v1")),
        "cat_params": config["cat_params"],
    }
    return oof_df, sub_df, metrics, importance_df


def save_phase16_artifacts(output_dir: Path, oof_df: pd.DataFrame, sub_df: pd.DataFrame, metrics: dict[str, Any], feature_importance_df: pd.DataFrame, report: dict[str, Any], active_config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cv_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "active_config.json").write_text(json.dumps(active_config, ensure_ascii=False, indent=2), encoding="utf-8")
    oof_df.to_csv(output_dir / "oof_phase16_catboost_orig_transfer.csv", index=False)
    sub_df.to_csv(output_dir / "submission_phase16_catboost_orig_transfer.csv", index=False)
    oof_df.to_csv(output_dir / "oof.csv", index=False)
    sub_df.to_csv(output_dir / "submission.csv", index=False)
    feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)


def run_pipeline(train_path: Path, test_path: Path, sample_submission_path: Path, config_path: Path, output_dir: Path, orig_path: Path | None) -> None:
    config = read_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_col = str(config["target_col"])
    id_col = str(config["id_col"])
    top_cats = [str(x) for x in config.get("top_cats_for_ngram", DEFAULT_TOP_CATS)]

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    sample_sub_df = pd.read_csv(sample_submission_path)
    if target_col not in train_raw.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if id_col not in train_raw.columns or id_col not in test_raw.columns:
        raise ValueError(f"Missing id column: {id_col}")

    y = to_binary_target(train_raw[target_col], pos_label=str(config["positive_label"]), neg_label=str(config["negative_label"]))
    train_df = add_base_features(train_raw)
    test_df = add_base_features(test_raw)
    train_df = add_ngram_features(train_df, top_cats=top_cats)
    test_df = add_ngram_features(test_df, top_cats=top_cats)

    orig_df = load_original_reference(train_raw, y, target_col, config, orig_path)
    orig_df = add_base_features(orig_df)
    orig_df = add_ngram_features(orig_df, top_cats=top_cats)

    train_df, test_df, created_signals = add_orig_signal_features(train_df, test_df, orig_df, target_col, top_cats, config)
    feature_cols, num_cols, cat_cols = detect_feature_columns(train_df, id_col, target_col)
    oof_df, sub_df, metrics, feature_importance_df = train_catboost(train_df, test_df, y, feature_cols, num_cols, cat_cols, id_col, target_col, config, created_signals)

    sub_final = sample_sub_df[[id_col]].copy() if id_col in sample_sub_df.columns else pd.DataFrame({id_col: test_df[id_col].values})
    sub_final = sub_final.merge(sub_df[[id_col, target_col]], on=id_col, how="left")
    report = {
        "run_name": str(config.get("run_name", "phase16_catboost_orig_transfer_v1")),
        "overall_auc": float(metrics["overall_auc"]),
        "feature_count": int(len(feature_cols)),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "created_orig_signals": created_signals,
        "resolved_config_path": str(config_path),
        "top_features_preview": feature_importance_df.head(20).to_dict(orient="records"),
    }
    active_config = json.loads(json.dumps(config, ensure_ascii=False))
    active_config["_resolved_config_path"] = str(config_path)
    save_phase16_artifacts(output_dir, oof_df, sub_final, metrics, feature_importance_df, report, active_config)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PS-S6E3 Phase16 CatBoost original-transfer trainer")
    parser.add_argument("--train-path", type=Path, required=False, default=Path("/kaggle/input/playground-series-s6e3/train.csv"))
    parser.add_argument("--test-path", type=Path, required=False, default=Path("/kaggle/input/playground-series-s6e3/test.csv"))
    parser.add_argument("--sample-submission-path", type=Path, required=False, default=Path("/kaggle/input/playground-series-s6e3/sample_submission.csv"))
    parser.add_argument("--orig-path", type=Path, required=False, default=None)
    parser.add_argument("--config-path", type=Path, required=False, default=Path("config_phase16_catboost_orig_transfer.json"))
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


def resolve_config_path(config_path: Path) -> Path:
    if config_path.exists():
        return config_path
    if not config_path.is_absolute():
        candidate = Path(__file__).resolve().parent / config_path
        if candidate.exists():
            return candidate
    return config_path


def main() -> None:
    args = parse_args()
    train_path = args.train_path if args.train_path.exists() else auto_find_input_file("train.csv")
    test_path = args.test_path if args.test_path.exists() else auto_find_input_file("test.csv")
    sample_path = args.sample_submission_path if args.sample_submission_path.exists() else auto_find_input_file("sample_submission.csv")
    run_pipeline(
        train_path=train_path,
        test_path=test_path,
        sample_submission_path=sample_path,
        config_path=resolve_config_path(args.config_path),
        output_dir=args.output_dir,
        orig_path=args.orig_path,
    )


if __name__ == "__main__":
    main()
