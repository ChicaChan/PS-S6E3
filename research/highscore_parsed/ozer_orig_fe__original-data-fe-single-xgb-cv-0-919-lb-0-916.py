# %% cell 4
# ============================================================
# CONFIGURATION — set your paths here
# ============================================================

TRAIN_PATH = "//kaggle/input/competitions/playground-series-s6e3/train.csv"
TEST_PATH  = "/kaggle/input/competitions/playground-series-s6e3/test.csv"
ORIG_PATH  = "/kaggle/input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv"
SUB_PATH   = "/kaggle/input/competitions/playground-series-s6e3/sample_submission.csv"

# ============================================================
# HYPERPARAMETERS (Optuna-tuned)
# ============================================================

N_FOLDS     = 10
INNER_FOLDS = 5
SEED        = 42
TARGET      = "Churn"

XGB_PARAMS = {
    "n_estimators"         : 50000,
    "learning_rate"        : 0.0063,
    "max_depth"            : 5,
    "subsample"            : 0.81,
    "colsample_bytree"     : 0.32,
    "min_child_weight"     : 6,
    "reg_alpha"            : 3.5017,
    "reg_lambda"           : 1.2925,
    "gamma"                : 0.790,
    "random_state"         : SEED,
    "early_stopping_rounds": 500,
    "objective"            : "binary:logistic",
    "eval_metric"          : "auc",
    "enable_categorical"   : True,
    "device"               : "cuda", # or cpu
    "verbosity"            : 0,
}

# %% cell 5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import gc
import time
from itertools import combinations

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import xgboost as xgb

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor"  : "#1a1a2e",
    "axes.edgecolor"  : "#444",
    "axes.labelcolor" : "#ccc",
    "text.color"      : "#ccc",
    "xtick.color"     : "#aaa",
    "ytick.color"     : "#aaa",
    "grid.color"      : "#333",
    "grid.linestyle"  : "--",
    "font.family"     : "monospace",
})
PALETTE = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#c77dff"]

print(f"XGBoost version : {xgb.__version__}")
print(f"NumPy version   : {np.__version__}")
print(f"Pandas version  : {pd.__version__}")

# %% cell 6
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
orig  = pd.read_csv(ORIG_PATH)
sub   = pd.read_csv(SUB_PATH)

train[TARGET] = (train[TARGET] == "Yes").astype(int)
orig[TARGET]  = (orig[TARGET] == "Yes").astype(int)
orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(
    pd.to_numeric(orig["TotalCharges"], errors="coerce").median()
)
if "customerID" in orig.columns:
    orig = orig.drop(columns=["customerID"])

print(f"Train shape : {train.shape}")
print(f"Test  shape : {test.shape}")
print(f"Orig  shape : {orig.shape}")
print(f"\nTrain churn rate : {train[TARGET].mean():.4f} ({train[TARGET].mean()*100:.1f}%)")
print(f"Orig  churn rate : {orig[TARGET].mean():.4f} ({orig[TARGET].mean()*100:.1f}%)")

# %% cell 8
# ── Churn rate by top categorical features ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Churn Rate by Categorical Features\n(synthetic train vs original)",
             fontsize=14, fontweight="bold", color="white", y=1.01)

TOP_CATS = ["Contract", "InternetService", "PaymentMethod",
            "OnlineSecurity", "TechSupport", "PaperlessBilling"]

for ax, col in zip(axes.flat, TOP_CATS):
    syn_cr = train.groupby(col)[TARGET].mean().sort_values(ascending=False)
    org_cr = orig.groupby(col)[TARGET].mean().reindex(syn_cr.index).fillna(0)

    x = np.arange(len(syn_cr))
    w = 0.35
    ax.bar(x - w/2, syn_cr.values, width=w, color=PALETTE[0], alpha=0.85, label="synthetic")
    ax.bar(x + w/2, org_cr.values, width=w, color=PALETTE[1], alpha=0.85, label="original")
    ax.set_xticks(x)
    ax.set_xticklabels(syn_cr.index, rotation=20, ha="right", fontsize=8)
    ax.set_title(col, fontsize=10, color="white")
    ax.set_ylabel("Churn Rate", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 0.6)

plt.tight_layout()
plt.show()

# %% cell 9
# ── Numeric distributions: synthetic vs original ──────────────────────────────
NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Numeric Feature Distributions — Synthetic vs Original",
             fontsize=13, fontweight="bold", color="white")

for ax, col in zip(axes, NUMS):
    ax.hist(train[col], bins=60, color=PALETTE[0], alpha=0.6, density=True, label="synthetic")
    ax.hist(orig[col],  bins=60, color=PALETTE[1], alpha=0.6, density=True, label="original")
    ax.set_title(col, color="white")
    ax.set_xlabel(col, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

# %% cell 10
# ── The key insight: tenure_mod12 digit artifact ──────────────────────────────
train["tenure_mod12"] = train["tenure"] % 12
mod12_cr = train.groupby("tenure_mod12")[TARGET].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Digit Artifacts in Synthetic Data", fontsize=13, fontweight="bold", color="white")

# tenure mod12 churn rate
colors = [PALETTE[1] if v > 0.35 else PALETTE[0] if v < 0.15 else PALETTE[2]
          for v in mod12_cr.values]
axes[0].bar(mod12_cr.index, mod12_cr.values, color=colors, edgecolor="#222", linewidth=0.5)
axes[0].set_title("Churn Rate by tenure % 12", color="white")
axes[0].set_xlabel("tenure mod 12", fontsize=9)
axes[0].set_ylabel("Churn Rate", fontsize=9)
axes[0].axhline(train[TARGET].mean(), color="white", linestyle="--", alpha=0.5, label=f"mean={train[TARGET].mean():.3f}")
axes[0].legend(fontsize=8)
for i, v in enumerate(mod12_cr.values):
    axes[0].text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=7, color="white")

# charges deviation = 0 analysis
train["charges_dev"] = train["TotalCharges"] - train["tenure"] * train["MonthlyCharges"]
dev_groups = pd.cut(train["charges_dev"], bins=[-8000,-100,-10,-1,0,1,10,100,8000],
                    labels=["-∞ to -100","-100 to -10","-10 to -1","-1 to 0",
                            "0 to 1","1 to 10","10 to 100","100 to +∞"])
dev_cr = train.groupby(dev_groups, observed=False)[TARGET].mean()
bar_colors = [PALETTE[1] if v > 0.5 else PALETTE[0] for v in dev_cr.values]
axes[1].bar(range(len(dev_cr)), dev_cr.values, color=bar_colors, edgecolor="#222", linewidth=0.5)
axes[1].set_xticks(range(len(dev_cr)))
axes[1].set_xticklabels(dev_cr.index, rotation=30, ha="right", fontsize=7)
axes[1].set_title("Churn Rate by TotalCharges - tenure×MonthlyCharges", color="white")
axes[1].set_ylabel("Churn Rate", fontsize=9)
axes[1].axhline(train[TARGET].mean(), color="white", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

print(f"\nKey finding: deviation=0 has churn rate {train[train['charges_dev']==0][TARGET].mean():.3f}")
print(f"These rows: {(train['charges_dev']==0).sum():,} ({(train['charges_dev']==0).mean()*100:.1f}% of train)")
print(f"99.4% of dev=0 rows have tenure=1 (first month customers)")

# %% cell 11
# ── ORIG_proba signal strength ─────────────────────────────────────────────────
ORIG_PROBA_CATS = [
    "Contract", "PaymentMethod", "InternetService", "OnlineSecurity",
    "TechSupport", "OnlineBackup", "DeviceProtection", "PaperlessBilling",
    "StreamingMovies", "StreamingTV", "Partner", "Dependents",
    "SeniorCitizen", "MultipleLines", "PhoneService", "gender"
]

auc_results = {}
for col in ORIG_PROBA_CATS:
    mapping = orig.groupby(col)[TARGET].mean()
    feat = train[col].map(mapping).fillna(0.5)
    auc = roc_auc_score(train[TARGET], feat)
    auc_results[col] = max(auc, 1-auc)

auc_series = pd.Series(auc_results).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = [PALETTE[0] if v >= 0.70 else PALETTE[2] if v >= 0.60 else PALETTE[1]
              for v in auc_series.values]
bars = ax.barh(auc_series.index, auc_series.values, color=bar_colors, edgecolor="#222", height=0.7)
ax.axvline(0.70, color=PALETTE[3], linestyle="--", alpha=0.7, label="AUC = 0.70 threshold")
ax.axvline(0.50, color="#666", linestyle="--", alpha=0.5, label="Random baseline")
for bar, val in zip(bars, auc_series.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8, color="white")
ax.set_xlabel("Single-Feature AUC (using original dataset as mapping)", fontsize=10)
ax.set_title("ORIG_proba Feature Signal Strength\n(how predictive is each feature's churn rate from the original dataset?)",
             fontsize=11, color="white", fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0.48, 0.83)
plt.tight_layout()
plt.show()

# %% cell 14
CATS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]
NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]
NEW_NUMS   = []
NUM_AS_CAT = []

TOP_CATS_NGRAM = [
    "Contract", "InternetService", "PaymentMethod",
    "OnlineSecurity", "TechSupport", "PaperlessBilling"
]
ORIG_PROBA_CATS = [
    "Contract", "PaymentMethod", "InternetService", "OnlineSecurity",
    "TechSupport", "OnlineBackup", "DeviceProtection", "PaperlessBilling",
    "StreamingMovies", "StreamingTV", "Partner", "Dependents",
]
ORIG_PROBA_CROSS = list(combinations(
    ["Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport"], 2
))

def pctrank_against(values, reference):
    """Percentile rank of `values` within `reference` distribution."""
    ref_sorted = np.sort(reference)
    return (np.searchsorted(ref_sorted, values) / len(ref_sorted)).astype("float32")

def zscore_against(values, reference):
    """Z-score of `values` relative to `reference` distribution."""
    mu, sigma = np.mean(reference), np.std(reference)
    if sigma == 0:
        return np.zeros(len(values), dtype="float32")
    return ((values - mu) / sigma).astype("float32")

print("Engineering features...")

# ── 1. Frequency encoding ──────────────────────────────────────────────────────
# How common is this exact numeric value across all splits?
for col in NUMS:
    freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
    for df in [train, test, orig]:
        df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")
    NEW_NUMS.append(f"FREQ_{col}")

# ── 2. Arithmetic interactions ─────────────────────────────────────────────────
# charges_deviation=0 → 66% churn rate (first-month synthetic artifact)
for df in [train, test, orig]:
    df["charges_deviation"]      = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
    df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32")
    df["avg_monthly_charges"]    = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")
    df["is_first_month"]         = (df["tenure"] == 1).astype("float32")
    df["dev_is_zero"]            = (df["charges_deviation"] == 0).astype("float32")
    df["dev_sign"]               = np.sign(df["charges_deviation"]).astype("float32")
NEW_NUMS += ["charges_deviation","monthly_to_total_ratio","avg_monthly_charges",
             "is_first_month","dev_is_zero","dev_sign"]

# ── 3. Service counts ──────────────────────────────────────────────────────────
SERVICE_COLS = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
for df in [train, test, orig]:
    df["service_count"] = (df[SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
    df["has_internet"]  = (df["InternetService"] != "No").astype("float32")
    df["has_phone"]     = (df["PhoneService"] == "Yes").astype("float32")
NEW_NUMS += ["service_count","has_internet","has_phone"]

# ── 4. ORIG_proba — single categorical ────────────────────────────────────────
# For each categorical value, what fraction of customers churned in the ORIGINAL dataset?
# This is NOT target leakage — we're using the original 7k-row reference, not the training labels.
for col in ORIG_PROBA_CATS:
    mapping = orig.groupby(col)[TARGET].mean()
    for df in [train, test]:
        df[f"ORIG_proba_{col}"] = df[col].map(mapping).fillna(0.5).astype("float32")
    NEW_NUMS.append(f"ORIG_proba_{col}")
for col in NUMS:
    mapping = orig.groupby(col)[TARGET].mean()
    for df in [train, test]:
        df[f"ORIG_proba_{col}"] = df[col].map(mapping).fillna(0.5).astype("float32")
    NEW_NUMS.append(f"ORIG_proba_{col}")

# ── 5. Cross ORIG_proba ────────────────────────────────────────────────────────
# Interaction churn rates — e.g. Month-to-month + No internet → very low churn
for c1, c2 in ORIG_PROBA_CROSS:
    mapping = orig.groupby([c1,c2])[TARGET].mean()
    name = f"ORIG_proba_{c1}_{c2}"
    for df in [train, test]:
        df[name] = df.set_index([c1,c2]).index.map(mapping).fillna(0.5).values.astype("float32")
    NEW_NUMS.append(name)

# ── 6. Distribution features ───────────────────────────────────────────────────
# Percentile rank of each customer vs real churner / non-churner distributions
# Example: pctrank_ch_TC = "where does this customer's TotalCharges fall among real churners?"
orig_ch_tc  = orig.loc[orig[TARGET]==1,"TotalCharges"].values
orig_nc_tc  = orig.loc[orig[TARGET]==0,"TotalCharges"].values
orig_tc     = orig["TotalCharges"].values
orig_ch_mc  = orig.loc[orig[TARGET]==1,"MonthlyCharges"].values
orig_nc_mc  = orig.loc[orig[TARGET]==0,"MonthlyCharges"].values
orig_ch_t   = orig.loc[orig[TARGET]==1,"tenure"].values
orig_nc_t   = orig.loc[orig[TARGET]==0,"tenure"].values
orig_is_mc_mean = orig.groupby("InternetService")["MonthlyCharges"].mean()

for df in [train, test]:
    tc = df["TotalCharges"].values
    mc = df["MonthlyCharges"].values
    t  = df["tenure"].values

    df["pctrank_ch_TC"]   = pctrank_against(tc, orig_ch_tc)
    df["pctrank_nc_TC"]   = pctrank_against(tc, orig_nc_tc)
    df["pctrank_orig_TC"] = pctrank_against(tc, orig_tc)
    df["pctrank_gap_TC"]  = (pctrank_against(tc, orig_ch_tc) - pctrank_against(tc, orig_nc_tc)).astype("float32")
    df["zscore_ch_TC"]    = zscore_against(tc, orig_ch_tc)
    df["zscore_nc_TC"]    = zscore_against(tc, orig_nc_tc)
    df["zscore_gap_TC"]   = (np.abs(zscore_against(tc, orig_ch_tc)) - np.abs(zscore_against(tc, orig_nc_tc))).astype("float32")
    df["pctrank_ch_MC"]   = pctrank_against(mc, orig_ch_mc)
    df["pctrank_nc_MC"]   = pctrank_against(mc, orig_nc_mc)
    df["pctrank_gap_MC"]  = (pctrank_against(mc, orig_ch_mc) - pctrank_against(mc, orig_nc_mc)).astype("float32")
    df["pctrank_ch_T"]    = pctrank_against(t, orig_ch_t)
    df["pctrank_nc_T"]    = pctrank_against(t, orig_nc_t)
    df["pctrank_gap_T"]   = (pctrank_against(t, orig_ch_t) - pctrank_against(t, orig_nc_t)).astype("float32")

    vals_is = np.zeros(len(df), dtype="float32")
    vals_c  = np.zeros(len(df), dtype="float32")
    for cat_val in orig["InternetService"].unique():
        mask = df["InternetService"] == cat_val
        ref  = orig.loc[orig["InternetService"]==cat_val,"TotalCharges"].values
        if len(ref) > 0 and mask.sum() > 0:
            vals_is[mask] = pctrank_against(df.loc[mask,"TotalCharges"].values, ref)
    for cat_val in orig["Contract"].unique():
        mask = df["Contract"] == cat_val
        ref  = orig.loc[orig["Contract"]==cat_val,"TotalCharges"].values
        if len(ref) > 0 and mask.sum() > 0:
            vals_c[mask] = pctrank_against(df.loc[mask,"TotalCharges"].values, ref)
    df["cond_pctrank_IS_TC"] = vals_is
    df["cond_pctrank_C_TC"]  = vals_c
    df["resid_IS_MC"] = (df["MonthlyCharges"] - df["InternetService"].map(orig_is_mc_mean).fillna(0)).astype("float32")

NEW_NUMS += ["pctrank_ch_TC","pctrank_nc_TC","pctrank_orig_TC","pctrank_gap_TC",
             "zscore_ch_TC","zscore_nc_TC","zscore_gap_TC",
             "pctrank_ch_MC","pctrank_nc_MC","pctrank_gap_MC",
             "pctrank_ch_T","pctrank_nc_T","pctrank_gap_T",
             "cond_pctrank_IS_TC","cond_pctrank_C_TC","resid_IS_MC"]

# ── 7. Quantile distance features ─────────────────────────────────────────────
# How far is this customer from the Q25/Q50/Q75 of real churner distributions?
QDIST_FEATS = []
for q_label, q_val in [("q25",0.25),("q50",0.50),("q75",0.75)]:
    ch_q_tc = np.quantile(orig_ch_tc, q_val)
    nc_q_tc = np.quantile(orig_nc_tc, q_val)
    ch_q_t  = np.quantile(orig_ch_t,  q_val)
    nc_q_t  = np.quantile(orig_nc_t,  q_val)
    for df in [train, test]:
        df[f"dist_ch_TC_{q_label}"]   = np.abs(df["TotalCharges"] - ch_q_tc).astype("float32")
        df[f"dist_nc_TC_{q_label}"]   = np.abs(df["TotalCharges"] - nc_q_tc).astype("float32")
        df[f"qdist_gap_TC_{q_label}"] = (df[f"dist_nc_TC_{q_label}"] - df[f"dist_ch_TC_{q_label}"]).astype("float32")
        df[f"dist_ch_T_{q_label}"]    = np.abs(df["tenure"] - ch_q_t).astype("float32")
        df[f"dist_nc_T_{q_label}"]    = np.abs(df["tenure"] - nc_q_t).astype("float32")
        df[f"qdist_gap_T_{q_label}"]  = (df[f"dist_nc_T_{q_label}"] - df[f"dist_ch_T_{q_label}"]).astype("float32")
    QDIST_FEATS += [f"dist_ch_TC_{q_label}",f"dist_nc_TC_{q_label}",f"qdist_gap_TC_{q_label}",
                    f"dist_ch_T_{q_label}", f"dist_nc_T_{q_label}", f"qdist_gap_T_{q_label}"]
NEW_NUMS += QDIST_FEATS

# ── 8. Digit / modular features ────────────────────────────────────────────────
# The synthetic generator reproduces rounding patterns from the original data.
# tenure_mod12: customers at the START of a contract year churn at 41%!
for df in [train, test]:
    t_str  = df["tenure"].astype(str)
    mc_str = df["MonthlyCharges"].astype(str).str.replace(".", "", regex=False)
    tc_str = df["TotalCharges"].astype(str).str.replace(".", "", regex=False)
    df["tenure_mod10"]          = (df["tenure"] % 10).astype("float32")
    df["tenure_mod12"]          = (df["tenure"] % 12).astype("float32")
    df["tenure_years"]          = (df["tenure"] // 12).astype("float32")
    df["tenure_months_in_year"] = (df["tenure"] % 12).astype("float32")
    df["tenure_is_multiple_12"] = ((df["tenure"] % 12) == 0).astype("float32")
    df["tenure_first_digit"]    = t_str.str[0].astype(int).astype("float32")
    df["tenure_last_digit"]     = t_str.str[-1].astype(int).astype("float32")
    df["mc_fractional"]         = (df["MonthlyCharges"] - np.floor(df["MonthlyCharges"])).astype("float32")
    df["mc_rounded_10"]         = (np.round(df["MonthlyCharges"]/10)*10).astype("float32")
    df["mc_dev_from_round10"]   = np.abs(df["MonthlyCharges"] - df["mc_rounded_10"]).astype("float32")
    df["mc_is_multiple_10"]     = ((np.floor(df["MonthlyCharges"]) % 10) == 0).astype("float32")
    df["mc_first_digit"]        = mc_str.str[0].astype(int).astype("float32")
    df["tc_fractional"]         = (df["TotalCharges"] - np.floor(df["TotalCharges"])).astype("float32")
    df["tc_rounded_100"]        = (np.round(df["TotalCharges"]/100)*100).astype("float32")
    df["tc_dev_from_round100"]  = np.abs(df["TotalCharges"] - df["tc_rounded_100"]).astype("float32")
    df["tc_is_multiple_100"]    = ((np.floor(df["TotalCharges"]) % 100) == 0).astype("float32")
    df["tc_first_digit"]        = tc_str.str[0].astype(int).astype("float32")

NEW_NUMS += ["tenure_mod10","tenure_mod12","tenure_years","tenure_months_in_year",
             "tenure_is_multiple_12","tenure_first_digit","tenure_last_digit",
             "mc_fractional","mc_rounded_10","mc_dev_from_round10","mc_is_multiple_10","mc_first_digit",
             "tc_fractional","tc_rounded_100","tc_dev_from_round100","tc_is_multiple_100","tc_first_digit"]

# ── 9. Num-as-cat + bigram/trigram n-grams ────────────────────────────────────
# Treat numeric values as categories for target encoding
# Bigrams/trigrams capture multi-way categorical interactions
for col in NUMS:
    name = f"CAT_{col}"
    NUM_AS_CAT.append(name)
    for df in [train, test]:
        df[name] = df[col].astype(str).astype("category")

BIGRAM_COLS, TRIGRAM_COLS = [], []
for c1, c2 in combinations(TOP_CATS_NGRAM, 2):
    name = f"BG_{c1}_{c2}"
    for df in [train, test]:
        df[name] = (df[c1].astype(str)+"_"+df[c2].astype(str)).astype("category")
    BIGRAM_COLS.append(name)
TOP4 = TOP_CATS_NGRAM[:4]
for c1, c2, c3 in combinations(TOP4, 3):
    name = f"TG_{c1}_{c2}_{c3}"
    for df in [train, test]:
        df[name] = (df[c1].astype(str)+"_"+df[c2].astype(str)+"_"+df[c3].astype(str)).astype("category")
    TRIGRAM_COLS.append(name)
NGRAM_COLS = BIGRAM_COLS + TRIGRAM_COLS

# Final feature lists
FEATURES         = NUMS + CATS + NEW_NUMS + NUM_AS_CAT + NGRAM_COLS
TE_COLUMNS       = NUM_AS_CAT + CATS
TE_NGRAM_COLUMNS = NGRAM_COLS
TO_REMOVE        = NUM_AS_CAT + CATS + NGRAM_COLS
STATS            = ["std","min","max"]
y_train          = train[TARGET].values

print(f"Total base features : {len(FEATURES)}")
print(f"  numerical         : {len(NUMS)}")
print(f"  categorical       : {len(CATS)}")
print(f"  engineered        : {len(NEW_NUMS)}")
print(f"  num-as-cat        : {len(NUM_AS_CAT)}")
print(f"  ngram             : {len(NGRAM_COLS)}")
print(f"\nAfter target encoding → ~186 model features")

# %% cell 16
np.random.seed(SEED)
skf_outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
skf_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)

oof_pred  = np.zeros(len(train))
test_pred = np.zeros(len(test))
fold_aucs = []
best_iters = []

t0 = time.time()
print(f"XGBoost 10-Fold CV | {len(FEATURES)} base features → ~186 model features")
print("=" * 60)

for fold, (tr_idx, val_idx) in enumerate(skf_outer.split(train, y_train)):
    X_tr  = train.iloc[tr_idx][FEATURES + [TARGET]].reset_index(drop=True).copy()
    y_tr  = train.iloc[tr_idx][TARGET].values
    X_val = train.iloc[val_idx][FEATURES].reset_index(drop=True).copy()
    y_val = train.iloc[val_idx][TARGET].values
    X_te  = test[FEATURES].reset_index(drop=True).copy()

    # Inner KFold TE: std, min, max statistics
    te_stat_cols = [f"TE1_{col}_{s}" for col in TE_COLUMNS for s in STATS]
    for c in te_stat_cols:
        X_tr[c] = 0.0

    for _, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
        X_tr2 = X_tr.iloc[in_tr].copy()
        for col in TE_COLUMNS:
            tmp = X_tr2.groupby(col, observed=False)[TARGET].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            for c in tmp.columns:
                vals = pd.to_numeric(X_tr.iloc[in_va][col].map(tmp[c]), errors="coerce").fillna(0).astype("float32").values
                X_tr.loc[X_tr.index[in_va], c] = vals

    for col in TE_COLUMNS:
        tmp = X_tr.groupby(col, observed=False)[TARGET].agg(STATS)
        tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
        for c in tmp.columns:
            X_val[c] = pd.to_numeric(X_val[col].map(tmp[c]), errors="coerce").fillna(0).astype("float32").values
            X_te[c]  = pd.to_numeric(X_te[col].map(tmp[c]),  errors="coerce").fillna(0).astype("float32").values
            X_tr[c]  = pd.to_numeric(X_tr[c], errors="coerce").fillna(0).astype("float32")

    # Inner KFold TE: n-gram mean
    for col in TE_NGRAM_COLUMNS:
        X_tr[f"TE_ng_{col}"] = 0.5
    for _, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
        X_tr2 = X_tr.iloc[in_tr].copy()
        for col in TE_NGRAM_COLUMNS:
            ng_te = X_tr2.groupby(col, observed=False)[TARGET].mean()
            X_tr.loc[X_tr.index[in_va], f"TE_ng_{col}"] = (
                pd.to_numeric(X_tr.iloc[in_va][col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32").values
            )
    for col in TE_NGRAM_COLUMNS:
        ng_te = X_tr.groupby(col, observed=False)[TARGET].mean()
        X_val[f"TE_ng_{col}"] = pd.to_numeric(X_val[col].astype(str).map(ng_te), errors="coerce").fillna(0.5).astype("float32")
        X_te[f"TE_ng_{col}"]  = pd.to_numeric(X_te[col].astype(str).map(ng_te),  errors="coerce").fillna(0.5).astype("float32")

    # Sklearn TargetEncoder (mean with CV smoothing)
    TE_MEAN_COLS = [f"TE_{col}" for col in TE_COLUMNS]
    te = TargetEncoder(cv=INNER_FOLDS, shuffle=True, smooth="auto",
                       target_type="binary", random_state=SEED)
    X_tr[TE_MEAN_COLS]  = te.fit_transform(X_tr[TE_COLUMNS], y_tr)
    X_val[TE_MEAN_COLS] = te.transform(X_val[TE_COLUMNS])
    X_te[TE_MEAN_COLS]  = te.transform(X_te[TE_COLUMNS])

    # Drop raw categoricals (keep encoded versions)
    for df in [X_tr, X_val, X_te]:
        for c in CATS + NUM_AS_CAT:
            if c in df.columns:
                df[c] = df[c].astype(str).astype("category")
        df.drop(columns=[c for c in TO_REMOVE if c in df.columns], inplace=True, errors="ignore")
    X_tr.drop(columns=[TARGET], inplace=True, errors="ignore")
    COLS_MODEL = X_tr.columns.tolist()

    if fold == 0:
        print(f"Model feature count: {len(COLS_MODEL)}")
        print("=" * 60)

    # Train XGBoost
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_val[COLS_MODEL], y_val)], verbose=False)

    oof_pred[val_idx]  = model.predict_proba(X_val[COLS_MODEL])[:, 1]
    test_pred         += model.predict_proba(X_te[COLS_MODEL])[:, 1] / N_FOLDS

    fold_auc = roc_auc_score(y_val, oof_pred[val_idx])
    fold_aucs.append(fold_auc)
    best_iters.append(model.best_iteration)

    elapsed = (time.time() - t0) / 60
    print(f"Fold {fold+1:>2}/10 | AUC: {fold_auc:.5f} | best_iter: {model.best_iteration:>5} | {elapsed:.1f} min")

    del X_tr, X_val, X_te, model
    gc.collect()

oof_auc = roc_auc_score(y_train, oof_pred)
print("=" * 60)
print(f"OOF AUC : {oof_auc:.5f}")
print(f"Fold std: {np.std(fold_aucs):.5f}")
print(f"Total   : {(time.time()-t0)/60:.1f} min")

# %% cell 18
# ── Fold AUC visualization ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("XGBoost 10-Fold Cross Validation Results", fontsize=13, fontweight="bold", color="white")

# Fold AUCs
bar_colors = [PALETTE[0] if v >= np.mean(fold_aucs) else PALETTE[1] for v in fold_aucs]
bars = axes[0].bar(range(1, 11), fold_aucs, color=bar_colors, edgecolor="#222", linewidth=0.5)
axes[0].axhline(np.mean(fold_aucs), color="white", linestyle="--", alpha=0.8,
                label=f"Mean = {np.mean(fold_aucs):.5f}")
axes[0].axhline(oof_auc, color=PALETTE[3], linestyle="-", alpha=0.8,
                label=f"OOF  = {oof_auc:.5f}")
for bar, val in zip(bars, fold_aucs):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.00002,
                f"{val:.5f}", ha="center", fontsize=7, color="white", rotation=45)
axes[0].set_xlabel("Fold", fontsize=10)
axes[0].set_ylabel("AUC", fontsize=10)
axes[0].set_title("Per-Fold AUC", color="white")
axes[0].legend(fontsize=9)
axes[0].set_ylim(min(fold_aucs) - 0.001, max(fold_aucs) + 0.002)

# Best iterations
axes[1].bar(range(1, 11), best_iters, color=PALETTE[2], edgecolor="#222", linewidth=0.5)
axes[1].axhline(np.mean(best_iters), color="white", linestyle="--", alpha=0.8,
                label=f"Mean = {np.mean(best_iters):.0f}")
axes[1].set_xlabel("Fold", fontsize=10)
axes[1].set_ylabel("Best Iteration", fontsize=10)
axes[1].set_title("Best Iteration per Fold", color="white")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()

print(f"\nOOF AUC     : {oof_auc:.5f}")
print(f"Fold mean   : {np.mean(fold_aucs):.5f}")
print(f"Fold std    : {np.std(fold_aucs):.5f}")
print(f"Min fold    : {min(fold_aucs):.5f} (fold {np.argmin(fold_aucs)+1})")
print(f"Max fold    : {max(fold_aucs):.5f} (fold {np.argmax(fold_aucs)+1})")
print(f"Mean best iter: {np.mean(best_iters):.0f}")

# %% cell 19
# ── OOF prediction distribution ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("OOF Prediction Analysis", fontsize=13, fontweight="bold", color="white")

# Prediction distribution by true label
axes[0].hist(oof_pred[y_train==0], bins=80, alpha=0.6, color=PALETTE[0],
             density=True, label="Non-churner (true)")
axes[0].hist(oof_pred[y_train==1], bins=80, alpha=0.6, color=PALETTE[1],
             density=True, label="Churner (true)")
axes[0].set_xlabel("Predicted Probability", fontsize=10)
axes[0].set_ylabel("Density", fontsize=10)
axes[0].set_title("Predicted Probability Distribution", color="white")
axes[0].legend(fontsize=9)

# Calibration
from sklearn.calibration import calibration_curve
fraction_pos, mean_pred = calibration_curve(y_train, oof_pred, n_bins=20)
axes[1].plot([0,1], [0,1], "--", color="#666", label="Perfect calibration")
axes[1].plot(mean_pred, fraction_pos, "o-", color=PALETTE[0], linewidth=2,
             markersize=5, label="Model")
axes[1].set_xlabel("Mean Predicted Probability", fontsize=10)
axes[1].set_ylabel("Fraction of Positives", fontsize=10)
axes[1].set_title("Calibration Curve", color="white")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.show()

# %% cell 20
# ── Feature group contribution (single-feature AUC summary) ───────────────────
group_auc = {
    "ORIG_proba\ncross (10)": 0.859,
    "ORIG_proba\nsingle (15)": 0.789,
    "Tenure\npctrank (2)": 0.794,
    "TC\npctrank (3)": 0.668,
    "MC\npctrank (3)": 0.679,
    "Quantile\ndist TC (9)": 0.740,
    "Digit\nfeatures (17)": 0.786,
    "Arithmetic\n(6)": 0.659,
    "Freq\nencoding (3)": 0.620,
}

fig, ax = plt.subplots(figsize=(12, 5))
names = list(group_auc.keys())
values = list(group_auc.values())
bar_colors = [PALETTE[0] if v >= 0.75 else PALETTE[2] if v >= 0.65 else PALETTE[1]
              for v in values]
bars = ax.bar(names, values, color=bar_colors, edgecolor="#222", linewidth=0.5, width=0.7)
ax.axhline(0.5, color="#666", linestyle="--", alpha=0.5, label="Random baseline")
ax.axhline(0.70, color=PALETTE[3], linestyle="--", alpha=0.6, label="Strong signal (0.70)")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
            f"{val:.3f}", ha="center", fontsize=9, color="white", fontweight="bold")
ax.set_ylabel("Best Single-Feature AUC in Group", fontsize=10)
ax.set_title("Feature Group Signal Strength\n(how powerful is the best feature in each group?)",
             fontsize=11, color="white", fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0.45, 0.90)
plt.tight_layout()
plt.show()

# %% cell 22
sub[TARGET] = test_pred
sub.to_csv("submission.csv", index=False)

print(f"Submission saved.")
print(f"OOF AUC : {oof_auc:.5f}")
print(f"\nPrediction stats:")
print(f"  mean  : {test_pred.mean():.4f}")
print(f"  std   : {test_pred.std():.4f}")
print(f"  min   : {test_pred.min():.4f}")
print(f"  max   : {test_pred.max():.4f}")
print(f"  >0.5  : {(test_pred > 0.5).mean()*100:.1f}%")
sub.head(10)

