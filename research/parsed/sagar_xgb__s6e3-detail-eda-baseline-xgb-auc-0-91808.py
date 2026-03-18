# %% cell 3
# cuDF accelerates pandas on GPU — load safely with fallback
try:
    %load_ext cudf.pandas
    print("cudf.pandas loaded ✅ — GPU-accelerated pandas active")
except Exception as e:
    print(f"cudf.pandas unavailable ({e.__class__.__name__}) — using standard pandas ✅")

# %% cell 4
import numpy as np
import pandas as pd
import warnings
import gc
import time

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

from sklearn.metrics import (roc_auc_score, roc_curve, auc,
                              precision_recall_curve, average_precision_score,
                              confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

import xgboost as xgb

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)

print("=" * 70)
print("  LIBRARIES IMPORTED ✅")
print("=" * 70)

# %% cell 5
rcParams['figure.figsize']     = (12, 6)
rcParams['axes.facecolor']    = '#f8f9fa'
rcParams['figure.facecolor']  = 'white'
rcParams['axes.grid']         = True
rcParams['grid.alpha']        = 0.3
rcParams['grid.color']        = '#cccccc'
rcParams['axes.edgecolor']    = '#333333'
rcParams['axes.linewidth']    = 1.2
rcParams['font.family']       = 'sans-serif'
rcParams['font.size']         = 11
rcParams['axes.labelsize']    = 12
rcParams['axes.titlesize']    = 14
rcParams['axes.titleweight']  = 'bold'
rcParams['xtick.labelsize']   = 10
rcParams['ytick.labelsize']   = 10
rcParams['legend.fontsize']   = 10
rcParams['legend.framealpha'] = 0.9

COLORS = ['#7B2CBF', '#9D4EDD', '#C77DFF', '#E0AAFF', '#5A189A', '#240046']

print("=" * 70)
print("  VISUALIZATION THEME CONFIGURED ✅")
print("=" * 70)

# %% cell 6
CONFIG = {
    'N_FOLDS'      : 5,
    'INNER_FOLDS'  : 5,
    'RANDOM_SEED'  : 42,
    'TARGET'       : 'Churn',
    'PSEUDO_LABELS': True,
    'TRES'         : 0.995,
}

XGB_PARAMS = {
    'n_estimators'         : 50000,
    'learning_rate'        : 0.05,
    'max_depth'            : 6,
    'subsample'            : 0.8,
    'colsample_bytree'     : 0.8,
    'min_child_weight'     : 5,
    'reg_alpha'            : 0.1,
    'reg_lambda'           : 1.0,
    'gamma'                : 0.05,
    'random_state'         : CONFIG['RANDOM_SEED'],
    'early_stopping_rounds': 500,
    'objective'            : 'binary:logistic',
    'eval_metric'          : 'auc',
    'enable_categorical'   : True,
    'device'               : 'cuda',
    'verbosity'            : 0,
}

print("=" * 70)
print("  CONFIGURATION:")
for k, v in CONFIG.items():
    print(f"  {k:<20} : {v}")
print()
print(f"  XGB early_stopping_rounds : {XGB_PARAMS['early_stopping_rounds']}")
print("=" * 70)
print("  ENVIRONMENT CONFIGURED ✅")
print("=" * 70)

# %% cell 8
TRAIN_PATH    = "/kaggle/input/competitions/playground-series-s6e3/train.csv"
TEST_PATH     = "/kaggle/input/competitions/playground-series-s6e3/test.csv"
ORIGINAL_PATH = "/kaggle/input/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm/WA_Fn-UseC_-Telco-Customer-Churn.csv"

print("[1/3] Loading competition train set...")
train = pd.read_csv(TRAIN_PATH)
print(f"      Shape  : {train.shape}  |  Memory : {train.memory_usage(deep=True).sum()/1024**2:.2f} MB")

print("[2/3] Loading competition test set...")
test = pd.read_csv(TEST_PATH)
print(f"      Shape  : {test.shape}")

print("[3/3] Loading original IBM Telco dataset...")
orig = pd.read_csv(ORIGINAL_PATH)
print(f"      Shape  : {orig.shape}")

# ── Target: Yes/No → 1/0 ─────────────────────────────────────────────────────
train[CONFIG['TARGET']] = train[CONFIG['TARGET']].map({'No': 0, 'Yes': 1}).astype(int)
orig[CONFIG['TARGET']]  = orig[CONFIG['TARGET']].map({'No': 0, 'Yes': 1}).astype(int)

# ── Fix TotalCharges whitespace-NaN in original ───────────────────────────────
orig['TotalCharges'] = pd.to_numeric(orig['TotalCharges'], errors='coerce')
orig['TotalCharges'].fillna(orig['TotalCharges'].median(), inplace=True)

if 'customerID' in orig.columns:
    orig.drop(columns=['customerID'], inplace=True)

train_ids = train['id'].copy()
test_ids  = test['id'].copy()

print(f"\n  Train : {len(train):,} rows  |  Churn rate : {train[CONFIG['TARGET']].mean()*100:.2f}%")
print(f"  Orig  : {len(orig):,}   rows  |  Churn rate : {orig[CONFIG['TARGET']].mean()*100:.2f}%")
print(f"  Test  : {len(test):,} rows")
print("=" * 70)
print("  DATA LOADING COMPLETED ✅")
print("=" * 70)

# %% cell 12
CATS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']

NEW_NUMS    = []
NEW_CATS    = []
NUM_AS_CAT  = []
NON_TE_CATS = []
TO_REMOVE   = []

print(f"  Categorical : {len(CATS)} cols")
print(f"  Numerical   : {len(NUMS)} cols")

# %% cell 14
# ── 1. Frequency encoding of numerical values ────────────────────────────────
for col in NUMS:
    freq = pd.concat([train[col], orig[col], test[col]]).value_counts(normalize=True)
    for df in [train, test, orig]:
        df[f'FREQ_{col}'] = df[col].map(freq).fillna(0).astype('float32')
    NEW_NUMS.append(f'FREQ_{col}')

# ── 2. Arithmetic interaction features ───────────────────────────────────────
for df in [train, test, orig]:
    df['charges_deviation']      = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
    df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
    df['avg_monthly_charges']    = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')

NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']

# ── 3. Service count + binary flags ──────────────────────────────────────────
SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for df in [train, test, orig]:
    df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
    df['has_internet']  = (df['InternetService'] != 'No').astype('float32')
    df['has_phone']     = (df['PhoneService'] == 'Yes').astype('float32')

NEW_NUMS += ['service_count', 'has_internet', 'has_phone']

print(f"  Engineered numerical features : {len(NEW_NUMS)}")
print(f"  {NEW_NUMS}")

# %% cell 16
for col in CATS + NUMS:
    tmp   = orig.groupby(col)[CONFIG['TARGET']].mean()
    _name = f"ORIG_proba_{col}"
    train = train.merge(tmp.rename(_name), on=col, how="left")
    test  = test.merge(tmp.rename(_name), on=col, how="left")
    orig  = orig.merge(tmp.rename(_name), on=col, how="left")
    for df in [train, test, orig]:
        df[_name] = df[_name].fillna(0.5).astype('float32')
    NEW_NUMS.append(_name)

print(f"  ORIG_proba features : {len(CATS + NUMS)}")
print(f"  Total NEW_NUMS      : {len(NEW_NUMS)}")

# %% cell 18
for col in NUMS:
    _new = f'CAT_{col}'
    NUM_AS_CAT.append(_new)
    for df in [train, test, orig]:
        df[_new] = df[col].astype(str).astype('category')

FEATURES = NUMS + CATS + NEW_NUMS + NEW_CATS + NUM_AS_CAT + NON_TE_CATS
print(f"  NUM_AS_CAT     : {NUM_AS_CAT}")
print(f"  Total features : {len(FEATURES)}")
print("=" * 70)
print("  FEATURE ENGINEERING COMPLETED ✅")
print("=" * 70)

# %% cell 21
TE_COLUMNS = NUM_AS_CAT + CATS + NEW_CATS
TO_REMOVE  = NUM_AS_CAT + CATS + NEW_CATS
STATS      = ['std', 'min', 'max']

np.random.seed(CONFIG['RANDOM_SEED'])
skf_outer = StratifiedKFold(n_splits=CONFIG['N_FOLDS'],     shuffle=True, random_state=CONFIG['RANDOM_SEED'])
skf_inner = StratifiedKFold(n_splits=CONFIG['INNER_FOLDS'], shuffle=True, random_state=CONFIG['RANDOM_SEED'])

xgb_oof         = np.zeros(len(train))
xgb_pred        = np.zeros(len(test))
xgb_fold_scores = []
xgb_importances = []
xgb_pred_folds  = []
pl_results      = []   # track pseudo label outcomes per fold

t0 = time.time()
print(f"\n{'='*80}")
print("TRAINING XGBOOST")
print("="*80)

for i, (train_idx, val_idx) in enumerate(skf_outer.split(train, train[CONFIG['TARGET']])):
    print(f"\nFold {i+1}/{CONFIG['N_FOLDS']}")

    X_tr  = train.loc[train_idx, FEATURES + [CONFIG['TARGET']]].reset_index(drop=True).copy()
    y_tr  = train.loc[train_idx,  CONFIG['TARGET']].values
    X_val = train.loc[val_idx,   FEATURES].reset_index(drop=True).copy()
    y_val = train.loc[val_idx,   CONFIG['TARGET']].values
    X_te  = test[FEATURES].reset_index(drop=True).copy()

    # ── Inner KFold: leak-free TE aggregation ─────────────────────────────────
    for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
        print(f"   Inner Fold {j+1} (outer {i+1})")
        X_tr2 = X_tr.loc[in_tr, FEATURES + [CONFIG['TARGET']]].copy()
        X_va2 = X_tr.loc[in_va, FEATURES].copy()
        for col in TE_COLUMNS:
            tmp = X_tr2.groupby(col, observed=False)[CONFIG['TARGET']].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            X_va2 = X_va2.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                X_tr.loc[in_va, c] = X_va2[c].values.astype("float32")

    # ── Full-fold TE stats for val / test ─────────────────────────────────────
    for col in TE_COLUMNS:
        tmp = X_tr.groupby(col, observed=False)[CONFIG['TARGET']].agg(STATS)
        tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
        tmp = tmp.astype("float32")
        X_val = X_val.merge(tmp, on=col, how="left")
        X_te  = X_te.merge(tmp, on=col, how="left")
        for c in tmp.columns:
            for df in [X_tr, X_val, X_te]:
                df[c] = df[c].fillna(0)

    # ── sklearn TargetEncoder (mean, leak-safe via internal cv) ───────────────
    TE_MEAN_COLS = [f'TE_{col}' for col in TE_COLUMNS]
    te = TargetEncoder(cv=CONFIG['INNER_FOLDS'], shuffle=True,
                       smooth='auto', target_type='binary',
                       random_state=CONFIG['RANDOM_SEED'])
    X_tr[TE_MEAN_COLS]  = te.fit_transform(X_tr[TE_COLUMNS], y_tr)
    X_val[TE_MEAN_COLS] = te.transform(X_val[TE_COLUMNS])
    X_te[TE_MEAN_COLS]  = te.transform(X_te[TE_COLUMNS])

    # ── Cast categoricals, drop raw TE cols, drop target ─────────────────────
    for df in [X_tr, X_val, X_te]:
        df[CATS + NUM_AS_CAT] = df[CATS + NUM_AS_CAT].astype(str).astype("category")
        df.drop(columns=TO_REMOVE, inplace=True)
    X_tr.drop(columns=[CONFIG['TARGET']], inplace=True)
    COLS_XGB = X_tr.columns

    # ── Base XGBoost ──────────────────────────────────────────────────────────
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=1000)

    # ── Pseudo Labels ─────────────────────────────────────────────────────────
    if CONFIG['PSEUDO_LABELS']:
        oof_p    = model.predict_proba(X_val)[:, 1]
        test_p   = model.predict_proba(X_te)[:, 1]
        mask     = (test_p > CONFIG['TRES']) | (test_p < 1 - CONFIG['TRES'])
        base_auc = roc_auc_score(y_val, oof_p)

        X_tr_pl  = pd.concat([X_tr, X_te[mask]], axis=0)
        y_tr_pl  = np.concatenate([y_tr, (test_p[mask] > 0.5).astype(int)])

        print(f"   PL candidates : {mask.sum():,}  |  Base AUC : {base_auc:.5f}")
        model2 = xgb.XGBClassifier(**XGB_PARAMS)
        model2.fit(X_tr_pl, y_tr_pl, eval_set=[(X_val, y_val)], verbose=1000)
        oof_p2  = model2.predict_proba(X_val)[:, 1]
        pl_auc  = roc_auc_score(y_val, oof_p2)

        if pl_auc > base_auc:
            print(f"   ✅ PL improved : {base_auc:.5f} → {pl_auc:.5f}")
            model = model2
            pl_results.append(('improved', base_auc, pl_auc))
        else:
            print(f"   ❌ No PL gain  : {base_auc:.5f} vs {pl_auc:.5f}")
            pl_results.append(('no_gain', base_auc, pl_auc))
        del X_tr_pl, y_tr_pl, model2

    # ── Record results ────────────────────────────────────────────────────────
    xgb_oof[val_idx] = model.predict_proba(X_val)[:, 1]
    fold_auc = roc_auc_score(y_val, xgb_oof[val_idx])
    xgb_fold_scores.append(fold_auc)

    fold_test_p = model.predict_proba(X_te[COLS_XGB])[:, 1]
    xgb_pred   += fold_test_p / CONFIG['N_FOLDS']
    xgb_pred_folds.append(fold_test_p)
    xgb_importances.append(model.get_booster().get_score(importance_type='gain'))

    print(f"   Fold {i+1} AUC : {fold_auc:.5f}")

    del X_tr, X_val, X_te, y_tr, y_val, model
    gc.collect()

xgb_mean = np.mean(xgb_fold_scores)
xgb_std  = np.std(xgb_fold_scores)
xgb_auc  = roc_auc_score(train[CONFIG['TARGET']], xgb_oof)

print(f"\n{'='*70}")
print(f"  XGBoost Fold AUC : {xgb_mean:.5f} ± {xgb_std:.5f}")
print(f"  XGBoost OOF AUC  : {xgb_auc:.5f}")
print(f"  Wall time        : {(time.time()-t0)/60:.1f} min")
print(f"  Pseudo label     : {sum(1 for r in pl_results if r[0]=='improved')}/{CONFIG['N_FOLDS']} folds improved")
print("=" * 70)
print("  XGBOOST TRAINING COMPLETED ✅")
print("=" * 70)

# %% cell 24
y_true = train[CONFIG['TARGET']].values

summary_df = pd.DataFrame({
    'Fold'   : [f"Fold {i+1}" for i in range(CONFIG['N_FOLDS'])] + ['OOF (Overall)'],
    'AUC'    : xgb_fold_scores + [xgb_auc],
    'PL Used': [r[0]=='improved' for r in pl_results] + [None],
})
display(summary_df.style
    .background_gradient(cmap='Purples', subset=['AUC'])
    .format({'AUC': '{:.5f}'})
    .hide(axis='index'))

# %% cell 26
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── Left: OOF ROC curve ───────────────────────────────────────────────────────
ax = axes[0]
fpr, tpr, _ = roc_curve(y_true, xgb_oof)
ax.plot(fpr, tpr, color=COLORS[0], linewidth=2.5,
        label=f"XGBoost OOF  (AUC = {xgb_auc:.5f})")
ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS[0])
ax.plot([0,1],[0,1], color='gray', linestyle='--', linewidth=1.2, label='Random')
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("OOF ROC Curve"); ax.legend(loc='lower right')

# ── Right: per-fold bar chart ─────────────────────────────────────────────────
ax = axes[1]
fold_ids = [f"F{i+1}" for i in range(CONFIG['N_FOLDS'])]
bar_colors = [COLORS[0] if r[0]=='improved' else COLORS[2] for r in pl_results]
bars = ax.bar(fold_ids, xgb_fold_scores, color=bar_colors, edgecolor='white', linewidth=1.5)
ax.axhline(xgb_mean, color=COLORS[4], linestyle='--', linewidth=2,
           label=f'Mean = {xgb_mean:.5f}')
ax.set_ylabel("ROC AUC"); ax.set_title("Per-Fold AUC (Purple = PL improved)")
ax.set_ylim(min(xgb_fold_scores) - 0.002, max(xgb_fold_scores) + 0.002)
ax.legend()
for bar, score in zip(bars, xgb_fold_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
            f'{score:.5f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout(); plt.show()

# %% cell 28
fig, ax = plt.subplots(figsize=(9, 6))
prec, rec, _ = precision_recall_curve(y_true, xgb_oof)
ap = average_precision_score(y_true, xgb_oof)
ax.plot(rec, prec, color=COLORS[0], linewidth=2.5, label=f"XGBoost  (AP = {ap:.5f})")
ax.fill_between(rec, prec, alpha=0.1, color=COLORS[0])
ax.axhline(y=y_true.mean(), color='gray', linestyle='--', linewidth=1.2,
           label=f'No-skill (AP = {y_true.mean():.3f})')
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve (OOF)"); ax.legend()
plt.tight_layout(); plt.show()

# %% cell 30
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, thresh, label in zip(axes, [0.5, 0.3], ['Threshold = 0.5', 'Threshold = 0.3 (recall-focused)']):
    preds = (xgb_oof >= thresh).astype(int)
    cm = confusion_matrix(y_true, preds)
    cm_pct = cm / cm.sum() * 100
    ax.imshow(cm, cmap='Purples', aspect='auto')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred: No Churn', 'Pred: Churn'], fontsize=10)
    ax.set_yticklabels(['True: No Churn', 'True: Churn'], fontsize=10)
    ax.set_title(f"Confusion Matrix ({label})")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{cm[r,c]:,}\n({cm_pct[r,c]:.1f}%)",
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white' if cm[r,c] > cm.max()*0.5 else '#333333')

plt.tight_layout(); plt.show()

# %% cell 32
all_feats = set().union(*[d.keys() for d in xgb_importances])
imp_means = {f: np.mean([d.get(f, 0) for d in xgb_importances]) for f in all_feats}
top30 = pd.Series(imp_means).sort_values(ascending=False).head(30)

fig, ax = plt.subplots(figsize=(12, 10))
colors_bar = [COLORS[0] if 'ORIG' in f or 'TE' in f else COLORS[2] for f in top30.index[::-1]]
ax.barh(top30.index[::-1], top30.values[::-1], color=colors_bar, edgecolor='white', linewidth=0.8)
ax.set_title("XGBoost — Top 30 Features (Mean Gain)  |  Purple=TE/ORIG  Lavender=Other")
ax.set_xlabel("Mean Gain")
plt.tight_layout(); plt.show()

print("=" * 70)
print("  EVALUATION COMPLETED ✅")
print("=" * 70)

# %% cell 34
# ── Save OOF and per-fold test predictions ────────────────────────────────────
pd.DataFrame({'xgb_oof': xgb_oof}).to_csv('oof_predictions.csv', index=False)

fold_test_df = pd.DataFrame({f'fold_{i}': p for i, p in enumerate(xgb_pred_folds)})
fold_test_df['mean'] = xgb_pred
fold_test_df.to_csv('test_predictions_per_fold.csv', index=False)

# ── Submission ────────────────────────────────────────────────────────────────
sub = pd.DataFrame({'id': test_ids, CONFIG['TARGET']: xgb_pred})
sub.to_csv('submission.csv', index=False)

display(sub.head(10))

print(f"  OOF AUC              : {xgb_auc:.5f}")
print(f"  Fold AUC             : {xgb_mean:.5f} ± {xgb_std:.5f}")
print(f"  Prediction range     : [{xgb_pred.min():.5f},  {xgb_pred.max():.5f}]")
print(f"  Mean churn prob      : {xgb_pred.mean():.5f}")
print(f"  Features used        : {len(COLS_XGB)}")
print()
print(f"  Files saved:")
print(f"    submission.csv                ← submit this")
print(f"    oof_predictions.csv           ← OOF for stacking")
print(f"    test_predictions_per_fold.csv ← per-fold + mean preds")
print("=" * 70)
print("  SUBMISSION SAVED ✅")
print("=" * 70)

# %% cell 36
print("=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"  Model              : XGBoost  (GPU, device=cuda)")
print(f"  Features           : {len(FEATURES)} base → {len(COLS_XGB)} after TE expansion")
print(f"  CV Strategy        : StratifiedKFold ({CONFIG['N_FOLDS']}-fold outer, {CONFIG['INNER_FOLDS']}-fold inner TE)")
print(f"  Pseudo Labels      : {'ON' if CONFIG['PSEUDO_LABELS'] else 'OFF'}  (threshold={CONFIG['TRES']}, {sum(1 for r in pl_results if r[0]=="improved")}/{CONFIG['N_FOLDS']} folds improved)")
print()
print(f"  {'Fold':<12} | {'AUC':>9} | {'PL Used':>8}")
print(f"  {'-'*35}")
for idx, (score, res) in enumerate(zip(xgb_fold_scores, pl_results)):
    pl_tag = '✅' if res[0] == 'improved' else '❌'
    print(f"  Fold {idx+1:<7} | {score:>9.5f} | {pl_tag:>8}")
print(f"  {'-'*35}")
print(f"  {'OOF (Overall)':<12} | {xgb_auc:>9.5f} |")
print("=" * 70)
print()
print("  NOTE: Public LB uses ~20% of test data.")
print("  OOF CV AUC is the primary optimization signal.")
print("  Private LB (80%) determines final standings.")

