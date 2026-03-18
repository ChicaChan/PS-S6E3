# %% cell 3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VER = 200

import numpy as np
import pandas as pd

import cudf
import cupy as cp
from cuml.preprocessing import TargetEncoder
from cuml.linear_model import LogisticRegression as cuLogReg

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# -----------------------------
# LOAD
# -----------------------------
train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e3/train.csv")
test  = pd.read_csv("/kaggle/input/competitions/playground-series-s6e3/test.csv")

# -----------------------------
# CONFIG
# -----------------------------
TARGET = "Churn"
DROP_COLS = ["customerID", "id"]
FEATURES = [c for c in train.columns if c not in DROP_COLS + [TARGET]]

N_SPLITS = 5
RANDOM_STATE = 42

# TargetEncoder internal folds (ONLY within outer-train)
TE_N_FOLDS = 5

# -----------------------------
# TARGET
# -----------------------------
y = (train[TARGET].astype(str).str.strip().str.lower() == "yes").astype(np.int32).values
y_all = y.astype(np.int32)
N = len(train)
n_feat = len(FEATURES)

kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ============================================================
# LABEL ENCODE (train+test together per column)
# ============================================================
def label_encode_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, cols):
    train_out = train_df.copy()
    test_out  = test_df.copy()

    for c in cols:
        tr = train_out[c]
        te = test_out[c]

        # treat missing as category
        tr_s = tr.astype("string").fillna("__MISSING__")
        te_s = te.astype("string").fillna("__MISSING__")

        # fit mapping on combined values (stable codes across train/test)
        all_vals = pd.concat([tr_s, te_s], axis=0)
        uniq = pd.Index(all_vals.unique())

        # map to int32 codes
        mapping = pd.Series(np.arange(len(uniq), dtype=np.int32), index=uniq)
        train_out[c] = tr_s.map(mapping).astype(np.int32)
        test_out[c]  = te_s.map(mapping).astype(np.int32)

    return train_out, test_out

train_enc, test_enc = label_encode_train_test(train[FEATURES], test[FEATURES], FEATURES)

# -----------------------------
# Move all features to GPU once
# -----------------------------
X_all_g  = cudf.DataFrame({f: cudf.Series(train_enc[f].values).astype("int32") for f in FEATURES})
X_test_g = cudf.DataFrame({f: cudf.Series(test_enc[f].values).astype("int32") for f in FEATURES})

# -----------------------------
# Allocate pair bookkeeping
# -----------------------------
n_pair = n_feat * (n_feat - 1) // 2  # C(n,2)
pair_cols = []
pair_names = []
for i in range(n_feat):
    for j in range(i + 1, n_feat):
        f1, f2 = FEATURES[i], FEATURES[j]
        pair_cols.append((f1, f2))
        pair_names.append(f"{f1}__x__{f2}")

print(f"Num raw features: {n_feat}")
print(f"Num pairs:        {n_pair}")
print(f"Outer folds:      {N_SPLITS}")
print(f"TE inner folds:   {TE_N_FOLDS} (inside each outer fold)")

# ============================================================
# META MODEL: LOGIT 3 (z + z^2 + z^3) utilities
# ============================================================
def _clip01(x, eps=1e-5):
    return np.clip(x, eps, 1.0 - eps)

def _logit(x, eps=1e-5):
    x = _clip01(x, eps)
    return np.log(x / (1.0 - x)).astype(np.float32)

def make_logit3_features(tr_m, va_m, te_m, eps=1e-5):
    z_tr = _logit(tr_m, eps=eps)
    z_va = _logit(va_m, eps=eps)
    z_te = _logit(te_m, eps=eps)
    X_tr = np.hstack([z_tr, z_tr**2, z_tr**3]).astype(np.float32)
    X_va = np.hstack([z_va, z_va**2, z_va**3]).astype(np.float32)
    X_te = np.hstack([z_te, z_te**2, z_te**3]).astype(np.float32)
    return X_tr, X_va, X_te

# -----------------------------
# Outputs for meta model OOF + test pred
# -----------------------------
oof_meta  = np.zeros(N, dtype=np.float32)
pred_meta = np.zeros(len(test), dtype=np.float32)
fold_aucs = []

print("\n================ LEAK-FREE OUTER CV (PAIR TE INSIDE FOLDS) ================\n")

# ============================================================
# OUTER CV LOOP:
# ============================================================
for fold, (tr_idx, va_idx) in enumerate(kf.split(np.zeros(N), y_all), 1):
    print("\n" + "=" * 90)
    print(f"[OUTER FOLD {fold:02d}/{N_SPLITS}] tr={len(tr_idx)} va={len(va_idx)}")
    print("=" * 90)

    tr_idx_g = cudf.Series(tr_idx)
    va_idx_g = cudf.Series(va_idx)

    X_tr_g = X_all_g.take(tr_idx_g)
    X_va_g = X_all_g.take(va_idx_g)

    y_tr = y_all[tr_idx].astype(np.int32)
    y_va = y_all[va_idx].astype(np.int32)
    y_tr_g = cudf.Series(y_tr)

    tr_pair = np.zeros((len(tr_idx), n_pair), dtype=np.float32)
    va_pair = np.zeros((len(va_idx), n_pair), dtype=np.float32)
    te_pair = np.zeros((len(test),  n_pair), dtype=np.float32)

    te = TargetEncoder(
        n_folds=TE_N_FOLDS,
        smooth=0,
        seed=RANDOM_STATE + fold,
        split_method="random",
        stat="mean",
        output_type="cupy",
    )

    for t, (f1, f2) in enumerate(pair_cols):
        if (t == 0) or ((t + 1) % 200 == 0) or (t + 1 == n_pair):
            print(f"  Pair {t+1:>6d}/{n_pair}: {f1} x {f2}")

        X_tr_ij_g = X_tr_g[[f1, f2]]
        X_va_ij_g = X_va_g[[f1, f2]]
        X_te_ij_g = X_test_g[[f1, f2]]

        tr_oof_cp = te.fit_transform(X_tr_ij_g, y_tr_g)
        tr_pair[:, t] = cp.asnumpy(tr_oof_cp).ravel().astype(np.float32)

        te.fit(X_tr_ij_g, y_tr_g)
        va_cp = te.transform(X_va_ij_g)
        te_cp = te.transform(X_te_ij_g)
        va_pair[:, t] = cp.asnumpy(va_cp).ravel().astype(np.float32)
        te_pair[:, t] = cp.asnumpy(te_cp).ravel().astype(np.float32)

    X_tr_raw, X_va_raw, X_te_raw = make_logit3_features(tr_pair, va_pair, te_pair, eps=1e-5)
    print("  Raw dim (pairs):", tr_pair.shape[1])
    print("  Logit3 dim:", X_tr_raw.shape[1])

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_va = scaler.transform(X_va_raw).astype(np.float32)
    X_te = scaler.transform(X_te_raw).astype(np.float32)

    meta = cuLogReg(
        penalty="l2",
        C=0.5,
        max_iter=4000,
        tol=1e-4,
        fit_intercept=True,
        verbose=0,
    )

    meta.fit(cp.asarray(X_tr), cp.asarray(y_tr))

    oof_va = cp.asnumpy(meta.predict_proba(cp.asarray(X_va))[:, 1]).astype(np.float32)
    oof_meta[va_idx] = oof_va

    fold_auc = roc_auc_score(y_va, oof_va)
    fold_aucs.append(float(fold_auc))
    print(f"  [logit3] Fold {fold:02d} AUC: {fold_auc:.6f}")

    pred_meta += cp.asnumpy(meta.predict_proba(cp.asarray(X_te))[:, 1]).astype(np.float32) / N_SPLITS

# -----------------------------
# FINAL METRICS
# -----------------------------
meta_auc = roc_auc_score(y_all, oof_meta)
print("\n================ FINAL ================\n")
print("[logit3] OOF AUC:", f"{meta_auc:.6f}")
print("[logit3] Fold AUCs:", [float(f"{a:.6f}") for a in fold_aucs])

# -----------------------------
# SAVE
# -----------------------------
np.save(f"oof_tepair_logit3_v{VER}.npy", oof_meta.astype(np.float32))
np.save(f"pred_tepair_logit3_v{VER}.npy", pred_meta.astype(np.float32))

print(f"\nSaved:")
print(f"  oof_tepair_logit3_v{VER}.npy")
print(f"  pred_tepair_logit3_v{VER}.npy")

# %% cell 5
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VER=102

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# -----------------------------
# CONFIG
# -----------------------------

TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e3/train.csv"
TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e3/test.csv"

TARGET = "Churn"
DROP_COLS = ["customerID", "id"]

N_SPLITS = 5
RANDOM_STATE = 42

# -----------------------------
# LOAD
# -----------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

assert TARGET in train_df.columns, f"Missing target column '{TARGET}' in train.csv"
assert TARGET not in test_df.columns, "test.csv should not contain the target column"

# Target -> {0,1}
y = (train_df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(np.int32).values

# Keep raw features
X_train_raw = train_df.drop(columns=[TARGET]).copy()
X_test_raw  = test_df.copy()

# -----------------------------
# PREPROCESS (TRAIN+TEST TOGETHER)
# -----------------------------
def preprocess_together(X_tr: pd.DataFrame, X_te: pd.DataFrame):
    X_tr = X_tr.copy()
    X_te = X_te.copy()

    # Drop id columns if present (apply to both)
    X_tr = X_tr.drop(columns=[c for c in DROP_COLS if c in X_tr.columns], errors="ignore")
    X_te = X_te.drop(columns=[c for c in DROP_COLS if c in X_te.columns], errors="ignore")

    # Combine so we define categories once
    n_tr = len(X_tr)
    X_all = pd.concat([X_tr, X_te], axis=0, ignore_index=True)

    # ---- numeric fixes (same) ----
    if "TotalCharges" in X_all.columns:
        X_all["TotalCharges"] = pd.to_numeric(X_all["TotalCharges"], errors="coerce").fillna(0.0).astype("float32")
    for c in ["tenure", "MonthlyCharges"]:
        if c in X_all.columns:
            X_all[c] = pd.to_numeric(X_all[c], errors="coerce")

    # ---- make shared categorical dtypes WITHOUT pandas string dtype ----
    # Columns to treat as categorical: all objects + any existing category cols
    cat_cols = set(X_all.select_dtypes(include=["object"]).columns.tolist())
    cat_cols |= set(X_all.select_dtypes(include=["category"]).columns.tolist())

    for c in sorted(cat_cols):
        # get values as object; convert non-nan values to python str to be consistent
        s = X_all[c]

        # If it's categorical already, convert to object first to rebuild cleanly
        if str(s.dtype) == "category":
            s = s.astype("object")

        # Build categories from ALL non-null values as python strings
        non_null = s[~pd.isna(s)]
        # robust str conversion for mixed types
        cats = pd.Index(non_null.map(lambda v: str(v)).unique(), dtype="object").sort_values()

        dtype = pd.CategoricalDtype(categories=cats, ordered=False)

        # Now cast: keep NaNs as NaN, otherwise str(v)
        X_all[c] = s.map(lambda v: np.nan if pd.isna(v) else str(v)).astype(dtype)

    # Split back
    X_tr2 = X_all.iloc[:n_tr].reset_index(drop=True)
    X_te2 = X_all.iloc[n_tr:].reset_index(drop=True)
    return X_tr2, X_te2

X_train, X_test = preprocess_together(X_train_raw, X_test_raw)

# NaN info
if X_train.isna().any().any() or X_test.isna().any().any():
    n_nan_tr = int(X_train.isna().sum().sum())
    n_nan_te = int(X_test.isna().sum().sum())
    print(f"Info: NaNs -> train: {n_nan_tr}, test: {n_nan_te}. XGBoost will handle NaNs natively.")

# -----------------------------
# MODEL
# -----------------------------
xgb_params = dict(
    n_estimators=100_000,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=5,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    enable_categorical=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    early_stopping_rounds=200,
    device="cuda",
)

# -----------------------------
# CV + OOF + TEST INFERENCE
# -----------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof = np.zeros(len(X_train), dtype=np.float32)
test_pred_sum = np.zeros(len(X_test), dtype=np.float32)
test_pred_folds = np.zeros((N_SPLITS, len(X_test)), dtype=np.float32)
fold_aucs = []

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y), 1):
    X_trn, X_val = X_train.iloc[trn_idx], X_train.iloc[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]

    model = XGBClassifier(**xgb_params)
    model.fit(
        X_trn,
        y_trn,
        eval_set=[(X_val, y_val)],
        verbose=500,
    )

    p_val = model.predict_proba(X_val)[:, 1].astype(np.float32)
    oof[val_idx] = p_val

    auc = roc_auc_score(y_val, p_val)
    fold_aucs.append(auc)
    print(f"Fold {fold}/{N_SPLITS} AUC = {auc:.6f}")

    p_test = model.predict_proba(X_test)[:, 1].astype(np.float32)
    test_pred_folds[fold - 1] = p_test
    test_pred_sum += p_test

oof_auc = roc_auc_score(y, oof)
print("-" * 60)
print(f"OOF AUC (all folds combined) = {oof_auc:.6f}")
print(f"Mean fold AUC = {np.mean(fold_aucs):.6f}  |  Std = {np.std(fold_aucs):.6f}")

test_pred = (test_pred_sum / N_SPLITS).astype(np.float32)

# -----------------------------
# SAVE
# -----------------------------
np.save(f"oof_pred_proba_v{VER}.npy", oof)
np.save(f"test_pred_proba_v{VER}.npy", test_pred)
#np.save(f"test_pred_proba_folds_v{VER}.npy", test_pred_folds)

print("\nSaved:")
print(f" - oof_pred_proba_v{VER}.npy")
print(f" - test_pred_proba_v{VER}.npy")
#print(f" - test_pred_proba_folds_v{VER}.npy")

# %% cell 7
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VER = 312

import numpy as np
import pandas as pd
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # PyTorch 2.x

# -----------------------------
# Repro (optional)
# -----------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

train = pd.read_csv("/kaggle/input/competitions/playground-series-s6e3/train.csv")
test  = pd.read_csv("/kaggle/input/competitions/playground-series-s6e3/test.csv")
print("train:", train.shape, "test:", test.shape)

# -----------------------------
# Config
# -----------------------------
TARGET_COL = "Churn"
FEATS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
         'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
         'MonthlyCharges', 'TotalCharges']

NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT_FEATS = [c for c in FEATS if c not in NUMS]

# Rare handling for numeric->categorical proxy
RARE_MIN_COUNT = 25

N_SPLITS = 5
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 10
BATCH_SIZE = 256

LR = 2.5e-5
PATIENCE = 10

# stronger reg (helps slow down / avoid sharp minima)
WEIGHT_DECAY = 3e-4

EMB_DROPOUT = 0.10
MLP_DROPOUT = 0.30
HIDDEN = (512, 256)

# LR schedule
WARMUP_EPOCHS = 1
ETA_MIN = LR * 0.05

# -----------------------------
# Helpers
# -----------------------------
def make_vocab_maps(train_df: pd.DataFrame, cols):
    """Build value->id maps per column using ONLY training fold data. Reserve 0 for UNK."""
    maps = {}
    sizes = {}
    for c in cols:
        uniq = pd.Series(train_df[c].values).astype(str).unique().tolist()
        v2i = {v: i + 1 for i, v in enumerate(uniq)}  # 1..K
        maps[c] = v2i
        sizes[c] = len(v2i) + 1  # + UNK(0)
    return maps, sizes

def encode_with_maps(df: pd.DataFrame, cols, maps):
    """Encode df[cols] into int64 array [N, C] using per-column maps. Unknown -> 0."""
    X = np.zeros((len(df), len(cols)), dtype=np.int64)
    for j, c in enumerate(cols):
        v2i = maps[c]
        s = pd.Series(df[c].values).astype(str).map(v2i).fillna(0).astype(np.int64).values
        X[:, j] = s
    return X

def emb_dim_from_card(card: int) -> int:
    """
    Avoid overfitting: don't use sqrt(card) for huge-card cols.
    This is a safer "tabular embedding" rule that still allows >16 dims.
    """
    d = int(round(1.8 * (card ** 0.25)))
    return int(np.clip(d, 4, 64))

def build_numeric_snapper(train_series: pd.Series, rare_min_count: int):
    """
    From train-fold only:
      - compute value counts
      - define frequent values (count >= rare_min_count)
      - mapping for any series: snap rare -> nearest frequent
      - rare flag: 1 if rare (or nan), 0 if frequent
    """
    s = pd.to_numeric(train_series, errors="coerce").astype(np.float32)
    vc = pd.Series(s).value_counts(dropna=False)

    frequent_vals = vc[vc >= rare_min_count].index.values
    frequent_vals = np.array([v for v in frequent_vals if pd.notna(v)], dtype=np.float32)

    if frequent_vals.size == 0:
        frequent_vals = np.array(pd.Series(s.dropna()).unique(), dtype=np.float32)

    frequent_vals = np.sort(frequent_vals)
    frequent_set = set(frequent_vals.tolist())

    def transform(series_any: pd.Series):
        x = pd.to_numeric(series_any, errors="coerce").astype(np.float32).values
        is_nan = np.isnan(x)

        # rare flag (nan => rare)
        is_rare = np.ones_like(x, dtype=np.int32)
        # vectorized-ish: membership check via python set (ok for 1D)
        for i, v in enumerate(x):
            if np.isnan(v):
                is_rare[i] = 1
            else:
                is_rare[i] = 0 if (float(v) in frequent_set) else 1

        x_snapped = x.copy()
        if frequent_vals.size > 0:
            idx_snap = np.where((~is_nan) & (is_rare == 1))[0]
            if idx_snap.size > 0:
                v = x[idx_snap]
                pos = np.searchsorted(frequent_vals, v)
                pos = np.clip(pos, 0, len(frequent_vals) - 1)
                left = np.clip(pos - 1, 0, len(frequent_vals) - 1)
                right = pos
                left_vals = frequent_vals[left]
                right_vals = frequent_vals[right]
                choose_right = (np.abs(v - right_vals) <= np.abs(v - left_vals))
                nearest = np.where(choose_right, right_vals, left_vals)
                x_snapped[idx_snap] = nearest.astype(np.float32)

        return x_snapped.astype(np.float32), is_rare.astype(np.int32)

    return transform

# -----------------------------
# Dataset
# -----------------------------
class TabMixDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None):
        self.Xc = torch.as_tensor(X_cat, dtype=torch.long)
        self.Xn = torch.as_tensor(X_num, dtype=torch.float32)
        self.y  = None if y is None else torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.Xc.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.Xc[idx], self.Xn[idx]
        return self.Xc[idx], self.Xn[idx], self.y[idx]

# -----------------------------
# Model
# -----------------------------
class EmbMLP_Mixed(nn.Module):
    def __init__(self, cat_cardinals, n_num, hidden=(256, 128), emb_dropout=0.1, mlp_dropout=0.2):
        super().__init__()
        self.n_cat = len(cat_cardinals)

        self.emb_layers = nn.ModuleList()
        emb_out_dim = 0
        for card in cat_cardinals:
            d = emb_dim_from_card(card)
            self.emb_layers.append(nn.Embedding(num_embeddings=card, embedding_dim=d))
            emb_out_dim += d

        self.emb_drop = nn.Dropout(emb_dropout)

        in_dim = emb_out_dim + n_num
        layers = []
        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),     # <-- helps generalization, slows "overfit drift"
                nn.ReLU(inplace=True),
                nn.Dropout(mlp_dropout),
            ]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)

        for emb in self.emb_layers:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, x_cat, x_num):
        embs = []
        for j, emb in enumerate(self.emb_layers):
            embs.append(emb(x_cat[:, j]))
        z = torch.cat(embs, dim=1) if len(embs) else None
        if z is None:
            z = x_num
        else:
            z = self.emb_drop(z)
            z = torch.cat([z, x_num], dim=1)
        logit = self.mlp(z).squeeze(1)
        return logit

# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    out = []
    for batch in loader:
        if len(batch) == 3:
            xc, xn, _ = batch
        else:
            xc, xn = batch
        xc = xc.to(DEVICE, non_blocking=True)
        xn = xn.to(DEVICE, non_blocking=True)
        logit = model(xc, xn)
        prob = torch.sigmoid(logit).detach().cpu().numpy()
        out.append(prob)
    return np.concatenate(out)

class SmoothBCE(nn.Module):
    def __init__(self, eps=0.02):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets * (1 - self.eps) + 0.5 * self.eps
        return nn.functional.binary_cross_entropy_with_logits(logits, targets)

def train_one_fold(Xc_tr, Xn_tr, y_tr, Xc_va, Xn_va, y_va, cat_cardinals):
    model = EmbMLP_Mixed(
        cat_cardinals,
        n_num=Xn_tr.shape[1],
        hidden=HIDDEN,
        emb_dropout=EMB_DROPOUT,
        mlp_dropout=MLP_DROPOUT
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = SmoothBCE(0.02)

    dl_tr = DataLoader(
        TabMixDataset(Xc_tr, Xn_tr, y_tr),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    dl_va = DataLoader(
        TabMixDataset(Xc_va, Xn_va, y_va),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Cosine AFTER warmup
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS), eta_min=ETA_MIN
    )

    best_auc = -1.0
    best_state = None
    bad = 0

    print(f"  -> starting training: n_tr={len(Xc_tr):,} n_va={len(Xc_va):,} batches/tr={len(dl_tr):,}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()

        # -------- warmup LR (linear ramp) --------
        if epoch <= WARMUP_EPOCHS:
            # ramp from 0.1*LR -> LR
            warm_lr = LR * (0.1 + 0.9 * (epoch / WARMUP_EPOCHS))
            for pg in opt.param_groups:
                pg["lr"] = warm_lr

        running_loss = 0.0
        n_seen = 0

        for xc, xn, yb in dl_tr:
            xc = xc.to(DEVICE, non_blocking=True)
            xn = xn.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logit = model(xc, xn)
            loss = loss_fn(logit, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = xc.size(0)
            running_loss += loss.item() * bs
            n_seen += bs

        # step cosine only after warmup
        if epoch > WARMUP_EPOCHS:
            sched.step()

        cur_lr = opt.param_groups[0]["lr"]
        train_loss = running_loss / max(1, n_seen)

        p_va = predict_proba(model, dl_va)
        auc = roc_auc_score(y_va, p_va)

        dt = time.time() - t0
        print(f"     epoch {epoch:03d} | lr {cur_lr:.2e} | loss {train_loss:.5f} | val_auc {auc:.6f} | {dt:.1f}s")

        if auc > best_auc + 1e-6:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"     early stopping (no improvement for {PATIENCE} epochs). best_auc={best_auc:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_auc

# -----------------------------
# CV + TEST inference
# -----------------------------
y = train[TARGET_COL].values
if y.dtype == object:
    y = pd.Series(y).map({"Yes": 1, "No": 0}).values
y = y.astype(np.float32)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof = np.zeros(len(train), dtype=np.float32)
pred_test = np.zeros(len(test), dtype=np.float32)
fold_aucs = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train)), y), 1):
    fold_t0 = time.time()
    print(f"\n================ Fold {fold}/{N_SPLITS} ================")

    tr_df = train.iloc[tr_idx].reset_index(drop=True)
    va_df = train.iloc[va_idx].reset_index(drop=True)

    # -------------------------
    # Build numeric categorical proxies (fold-safe)
    # -------------------------
    tr_num_cat = {}
    va_num_cat = {}
    te_num_cat = {}
    tr_rare = {}
    va_rare = {}
    te_rare = {}

    for col in NUMS:
        snapper = build_numeric_snapper(tr_df[col], rare_min_count=RARE_MIN_COUNT)
        tr_snap, tr_israre = snapper(tr_df[col])
        va_snap, va_israre = snapper(va_df[col])
        te_snap, te_israre = snapper(test[col])

        tr_num_cat[f"{col}__cat"] = tr_snap.astype(np.float32)
        va_num_cat[f"{col}__cat"] = va_snap.astype(np.float32)
        te_num_cat[f"{col}__cat"] = te_snap.astype(np.float32)

        tr_rare[f"{col}__is_rare"] = tr_israre.astype(np.int32)
        va_rare[f"{col}__is_rare"] = va_israre.astype(np.int32)
        te_rare[f"{col}__is_rare"] = te_israre.astype(np.int32)

    # Build fold dataframes for categorical encoding
    tr_cat_df = tr_df[CAT_FEATS].copy()
    va_cat_df = va_df[CAT_FEATS].copy()
    te_cat_df = test[CAT_FEATS].copy()

    # Add numeric categorical proxies + rare flags
    for col in NUMS:
        tr_cat_df[f"{col}__cat"] = pd.Series(tr_num_cat[f"{col}__cat"]).astype(str).values
        va_cat_df[f"{col}__cat"] = pd.Series(va_num_cat[f"{col}__cat"]).astype(str).values
        te_cat_df[f"{col}__cat"] = pd.Series(te_num_cat[f"{col}__cat"]).astype(str).values

        tr_cat_df[f"{col}__is_rare"] = tr_rare[f"{col}__is_rare"]
        va_cat_df[f"{col}__is_rare"] = va_rare[f"{col}__is_rare"]
        te_cat_df[f"{col}__is_rare"] = te_rare[f"{col}__is_rare"]

    CAT_ALL = list(tr_cat_df.columns)

    # ---- categorical vocab on TRAIN fold only ----
    maps, sizes = make_vocab_maps(tr_cat_df, CAT_ALL)
    cat_cardinals = [sizes[c] for c in CAT_ALL]

    Xc_tr = encode_with_maps(tr_cat_df, CAT_ALL, maps)
    Xc_va = encode_with_maps(va_cat_df, CAT_ALL, maps)
    Xc_te = encode_with_maps(te_cat_df, CAT_ALL, maps)

    # ---- numeric float: original NUMS standardized on TRAIN fold only ----
    Xn_tr_raw = tr_df[NUMS].astype(np.float32).values
    Xn_va_raw = va_df[NUMS].astype(np.float32).values
    Xn_te_raw = test[NUMS].astype(np.float32).values

    scaler = StandardScaler()
    Xn_tr = scaler.fit_transform(Xn_tr_raw).astype(np.float32)
    Xn_va = scaler.transform(Xn_va_raw).astype(np.float32)
    Xn_te = scaler.transform(Xn_te_raw).astype(np.float32)

    y_tr = y[tr_idx]
    y_va = y[va_idx]

    model, best_auc = train_one_fold(Xc_tr, Xn_tr, y_tr, Xc_va, Xn_va, y_va, cat_cardinals)
    fold_aucs.append(best_auc)

    # OOF
    dl_va = DataLoader(TabMixDataset(Xc_va, Xn_va, y_va), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    p_va = predict_proba(model, dl_va)
    oof[va_idx] = p_va

    # TEST
    dl_te = DataLoader(TabMixDataset(Xc_te, Xn_te, None), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    p_te = predict_proba(model, dl_te).astype(np.float32)
    pred_test += p_te / N_SPLITS

    print(f"[Fold {fold}] best val AUC: {best_auc:.6f} | fold time {time.time()-fold_t0:.1f}s")

cv_auc = roc_auc_score(y, oof)
print("\n====================")
print("Fold AUCs:", [f"{a:.6f}" for a in fold_aucs])
print(f"OOF CV AUC: {cv_auc:.6f}")
print("====================")

np.save(f"oof_nn_v{VER}.npy", oof.astype(np.float32))
np.save(f"pred_nn_v{VER}.npy", pred_test.astype(np.float32))
print(f"\nSaved:\n  oof_nn_v{VER}.npy\n  pred_nn_v{VER}.npy")

# %% cell 9
logreg_oof = np.load("oof_tepair_logit3_v200.npy")
xgb_oof = np.load("oof_pred_proba_v102.npy")
mlp_oof = np.load("oof_nn_v312.npy")

ensemble_oof = (logreg_oof+xgb_oof+mlp_oof)/3.
m = roc_auc_score(y, ensemble_oof)
print(f"Overall Ensemble CV AUC = {m}")

# %% cell 11
sub = pd.read_csv("/kaggle/input/competitions/playground-series-s6e3/sample_submission.csv")

logreg_pred = np.load("pred_tepair_logit3_v200.npy")
xgb_pred = np.load("test_pred_proba_v102.npy")
mlp_pred = np.load("pred_nn_v312.npy")
pred = (logreg_pred+xgb_pred+mlp_pred)/3.

sub['Churn'] = pred
sub.to_csv("submission_ensemble.csv",index=False)
sub.head()

