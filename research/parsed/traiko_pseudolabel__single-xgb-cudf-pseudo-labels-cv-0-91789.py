# %% cell 2
NAME = '010'

# %% cell 3
from sklearn.metrics import roc_auc_score, roc_curve, auc, RocCurveDisplay
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import seaborn as sns

# %% cell 4
%load_ext cudf.pandas

import numpy as np, pandas as pd, gc
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 200)

# %% cell 5
train_file = '/kaggle/input/competitions/playground-series-s6e3/train.csv'
test_file = '/kaggle/input/competitions/playground-series-s6e3/test.csv'
original_file = '/kaggle/input/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm/WA_Fn-UseC_-Telco-Customer-Churn.csv'

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
orig = pd.read_csv(original_file)

train_ids = train['id'].copy()
test_ids = test['id'].copy() # for submission

# %% cell 6
train['Churn'] = train['Churn'].map({'No': 0, 'Yes': 1})
orig['Churn'] = orig['Churn'].map({'No': 0, 'Yes': 1})

# %% cell 7
test.columns.shape

# %% cell 8
train.columns

# %% cell 9
CATS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]
print(f"There are {len(CATS)} categorical columns:")
print(CATS)

NUMS = ['tenure', 'MonthlyCharges', 'TotalCharges']
print(f"There are {len(NUMS)} numerical columns:")
print(NUMS)

# %% cell 11
NEW_NUMS = []
NEW_CATS = []
NUM_AS_CAT = []
TO_REMOVE = []
NON_TE_CATS = []

# %% cell 12
# Compute frequencies from ALL data (train + orig + test)
for cat in NUMS:
    freq = pd.concat([train[cat], orig[cat], test[cat]]).value_counts(normalize=True)
    for df in [train, test, orig]:
        df[f'FREQ_{cat}'] = df[cat].map(freq).fillna(0).astype('float32')
    NEW_NUMS.append(f'FREQ_{cat}')

# %% cell 13
_to_combo = CATS

for i, c1 in enumerate(_to_combo[:-1]):
    for j, c2 in enumerate(_to_combo[i + 1:]):
        _new_col = f'COMBO_{c1}_{c2}'

        for df in [train, test, orig]:
            df[_new_col] = df[c1].astype('str') + '_' + df[c2].astype('str')
        NEW_CATS.append(_new_col)

        # 3-combos
        # for k, c3 in enumerate(_to_combo[i+j+2:]):
        #     _new_col = f'COMBO_{c1}_{c2}_{c3}'
        #     for df in [train, test, orig]:
        #         df[_new_col] = df[c1].astype('str') + '_' + df[c2].astype('str') + '_' + df[c3].astype('str')
    
        #     NEW_CATS.append(_new_col)

            # print(c1, c2, c3)
            # 4-combos
            # for l, c4 in enumerate(_to_combo[i+j+k+3:]):
            #     _new_col = f'COMBO_{c1}_{c2}_{c3}_{c4}'
            #     for df in [train, test, orig]:
            #         df[_new_col] = df[c1].astype('str') + '_' + df[c2].astype('str') + '_' + df[c3].astype('str') + '_' + df[c4].astype('str')
        
            #     NEW_CATS.append(_new_col)

# %% cell 14
# FROM : https://www.kaggle.com/code/datasciencegrad/s6e3-detail-eda-baseline-xgb-auc-0-91808?scriptVersionId=300750597
# Arithmetic interaction features
for df in [train, test, orig]:
    # removed the difference here as XGB cares about ordering only
    df['charges_deviation']      = (df['tenure'] * df['MonthlyCharges']).astype('float32')
    df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
    df['avg_monthly_charges']    = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')

NEW_NUMS += ['charges_deviation', 'monthly_to_total_ratio', 'avg_monthly_charges']

# Service count + binary flags
SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for df in [train, test, orig]:
    df['service_count'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1).astype('float32')
    # removed binary flags - tree can map those already

NEW_NUMS += ['service_count']

# %% cell 15
# numerical as categorical
for col in NUMS:    
    _new_col = f'CAT_{col}'
    NUM_AS_CAT.append(_new_col)

    for df in [train, test, orig]:
        df[_new_col] = df[col].astype(str).astype('category')

# %% cell 16
train.isna().any().any(), test.isna().any().any()

# %% cell 17
for col in CATS + NUMS:
    stats = orig.groupby(col)['Churn'].agg(['mean', 'std']).reset_index()
    stats.columns = [col] + [f"ORIG_{col}_{s}" for s in ['mean', 'std']]

    train = train.merge(stats, on=col, how='left')
    test = test.merge(stats, on=col, how='left')

    fill_values = {
        f"ORIG_{col}_mean": 0.5,
        f"ORIG_{col}_std": 0,
    }
    train = train.fillna(value=fill_values)
    test = test.fillna(value=fill_values)

    NEW_NUMS.extend([f"ORIG_{col}_{s}" for s in ['mean', 'std']])

# %% cell 18
FEATURES = NUMS + CATS + NEW_NUMS + NEW_CATS + NUM_AS_CAT + NON_TE_CATS
print(f'We now have {len(FEATURES)} columns:')
print(FEATURES)

# %% cell 19
# TARGET ENCODING STATISTICS TO AGGEGATE FOR OUR FEATURE GROUPS
STATS = ['std'] # mean is probability, handled by TargetEncoder

TE_COLUMNS = NUM_AS_CAT + CATS + NEW_CATS
TO_REMOVE += NUM_AS_CAT + CATS + NEW_CATS
QUANTILE_COLUMNS = []

# %% cell 21
np.random.seed(11)

# %% cell 22
PSEUDO_LABELS = True
TRES = 0.999

FOLDS = 5
INNER_FOLDS = 5

# %% cell 23
xgb_params = {
    'n_estimators': 50000, 
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8, 
    'colsample_bytree': 0.8,
    'random_state': 11,
    'early_stopping_rounds': 1000,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'enable_categorical': True,
    'device': 'cuda'
}

# %% cell 25
%%time

print(f"\n{'='*80}")
print("TRAINING XGBOOST")
print("="*80)

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=11)

oof = np.zeros((len(train)))
pred = np.zeros((len(test)))
importances = []
roc_auc_folds = []
pred_per_fold = []


for i, (train_index, val_index) in enumerate(kf.split(train, train['Churn'])):
    print(f"\nFold {i+1}/{FOLDS}")

    # ===================================
    # TRAIN/VAL SPLIT
    # ===================================
    X_train = train.loc[train_index,FEATURES+['Churn']].reset_index(drop=True).copy()
    y_train = train.loc[train_index,'Churn']
    
    X_val = train.loc[val_index,FEATURES].reset_index(drop=True).copy()
    y_val = train.loc[val_index,'Churn']

    X_train = pd.concat([X_train],axis=0).reset_index(drop=True).copy()
    y_train = np.concatenate([y_train],axis=0).copy()
    

    X_test = test[FEATURES].reset_index(drop=True).copy()

    # ===================================
    # INNER K FOLD (TO PREVENT LEAKAGE WHEN USING TARGET)
    # ===================================
    kf2 = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=11)
    
    for j, (train_index2, val_index2) in enumerate(kf2.split(X_train, X_train['Churn'])):
        print(f" ## INNER Fold {j+1} (outer fold {i+1}) ##")

        X_train2 = X_train.loc[train_index2, FEATURES + ['Churn']].copy()
        X_val2   = X_train.loc[val_index2, FEATURES].copy()
        y_train2 = y_train[train_index2]
        y_val2   = y_train[val_index2]
        

        # ===================================
        # TARGET ENCODING/AGGREGATION
        # ===================================
    
        ### FEATURE SET 1 (uses exam_score) ###
        for col in TE_COLUMNS:
        # col = 'study_hours'
            tmp = X_train2.groupby(col, observed=False)['Churn'].agg(STATS)
            tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
            X_val2 = X_val2.merge(tmp, on=col, how="left") # these are the oof predictions
        
            for c in tmp.columns:
                X_train.loc[val_index2, c] = X_val2[c].values.astype("float32")

        # ===================================
        # END TARGET ENCODING/AGGREGATION
        # ===================================

    
    ### FEATURE SET 1 (uses Churn) ###
    for col in TE_COLUMNS:
        tmp = X_train.groupby(col, observed=False)['Churn'].agg(STATS)
        tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
        tmp = tmp.astype("float32")
        X_val = X_val.merge(tmp, on=col, how="left")
        X_test = X_test.merge(tmp, on=col, how="left")

    
        X_train[[f"TE1_{col}_{s}" for s in STATS]] = X_train[[f"TE1_{col}_{s}" for s in STATS]].fillna(0)
        X_val[[f"TE1_{col}_{s}" for s in STATS]] = X_val[[f"TE1_{col}_{s}" for s in STATS]].fillna(0)
        X_test[[f"TE1_{col}_{s}" for s in STATS]] = X_test[[f"TE1_{col}_{s}" for s in STATS]].fillna(0)
        
    
    NEW_COL_NAMES = [f'TE_{col}' for col in TE_COLUMNS]
    target_encoder = TargetEncoder(cv=INNER_FOLDS, shuffle=True, smooth='auto', target_type='binary', random_state=11)
    
    X_train[NEW_COL_NAMES] = target_encoder.fit_transform(X_train[TE_COLUMNS], y_train)
    X_val[NEW_COL_NAMES] = target_encoder.transform(X_val[TE_COLUMNS])
    X_test[NEW_COL_NAMES] = target_encoder.transform(X_test[TE_COLUMNS])

    # ===================================
    # end inner KFold for Target encoding
    # ===================================

    # ===================================    
    # CONVERT TO CATS SO XGBOOST RECOGNIZES THEM
    X_train[CATS + NEW_CATS + NUM_AS_CAT + NON_TE_CATS] = X_train[CATS + NEW_CATS + NUM_AS_CAT + NON_TE_CATS].astype(str).astype("category")
    X_val[CATS + NEW_CATS + NUM_AS_CAT + NON_TE_CATS] = X_val[CATS + NEW_CATS + NUM_AS_CAT + NON_TE_CATS].astype(str).astype("category")
    X_test[CATS + NEW_CATS + NUM_AS_CAT + NON_TE_CATS] = X_test[CATS + NEW_CATS + NUM_AS_CAT + NON_TE_CATS].astype(str).astype("category")

    X_train.drop(TO_REMOVE, axis=1, inplace=True)
    X_val.drop(TO_REMOVE, axis=1, inplace=True)
    X_test.drop(TO_REMOVE, axis=1, inplace=True)

    # DROP EXAM SCORE THAT WAS USED FOR TARGET ENCODING
    X_train = X_train.drop(['Churn'],axis=1)
    COLS = X_train.columns
    
    # ===================================
    # train XGB
    # ===================================

    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=1000)

    if PSEUDO_LABELS:

        oof_preds = model.predict_proba(X_val)[:,1]
        test_preds = model.predict_proba(X_test)[:,1]
    
        # PSEUDO LABEL WITH XGB FOR XGB
    
        
        X_train_xgb = pd.concat([
            X_train,
            X_test[(test_preds > TRES) | (test_preds < 1 - TRES)]
        ], axis=0)
        
        y_train_xgb = np.concatenate([
            y_train,
            (test_preds[(test_preds > TRES) | (test_preds < 1 - TRES)] > 0.5).astype(int)
        ], axis=0)
    
        roc_auc_xgb = roc_auc_score(y_val, oof_preds)
    
        print('Training XGB 2')
        # TRAIN XGB WITH PSEUDO LABELS
        model2 = xgb.XGBClassifier(**xgb_params)
        model2.fit(X_train_xgb, y_train_xgb, eval_set=[(X_val, y_val)], verbose=1000)
        oof_preds2 = model2.predict_proba(X_val)[:,1]
        roc_auc_xgb2 = roc_auc_score(y_val, oof_preds2)
    
        if roc_auc_xgb2 > roc_auc_xgb:
            print(f'Pseudo labels improvement: {roc_auc_xgb} to {roc_auc_xgb2}')
            model = model2
            oof_preds = oof_preds2
            roc_auc_xgb = roc_auc_xgb2
        else:
            print(f'NO pseudo labels improvement: {roc_auc_xgb} to {roc_auc_xgb2}')



    # ===================================
    # predict OOF and TEST
    # ===================================
    
    oof[val_index] = model.predict_proba(X_val)[:,1]
    roc_auc_fold = roc_auc_score(y_val, oof[val_index])
    roc_auc_folds.append(roc_auc_fold)
    print(f"Validation ROC AUC: {roc_auc_fold:.5f}")

    pred += (_test_preds := model.predict_proba(X_test[COLS])[:,1])
    pred_per_fold.append(_test_preds)

    importances.append(model.get_booster().get_score(importance_type='gain'))
    
    # CLEAR MEMORY
    del X_train, X_val, X_test
    del y_train, y_val
    if i != FOLDS-1: del model
    gc.collect()

pred /= FOLDS

# %% cell 26
print(f'Fold ROC AUC {np.mean(roc_auc_folds):.5f} +- {np.std(roc_auc_folds):.5f}')

true = train['Churn'].values
print(f'Overall ROC AUC: {roc_auc_score(true, oof)}')

# %% cell 27
model.get_booster().best_iteration

# %% cell 28
fig, ax = plt.subplots()

fpr, tpr, thresholds = roc_curve(train['Churn'], oof)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display.plot(ax=ax)
display.plot(ax=ax, name='XGB')

plt.plot([0., 1.], [0., 1.])
plt.show()

# %% cell 29
print(f'\nIn total, we used {len(COLS)} features, Wow!\n')
print(list(COLS))

# %% cell 31
feature_names = importances[0].keys()
importances_mean = [
    np.mean([imp[feat] if feat in imp else 0 for imp in importances])
    for feat in feature_names
]

# %% cell 32
indices = np.argsort(importances_mean)
sorted_features = np.array(list(feature_names))[indices]
sorted_importance = np.array(importances_mean)[indices]

# Plot horizontally (features on y-axis for better readability)
plt.figure(figsize=(10, 15))
plt.barh(range(len(sorted_importance)), sorted_importance)
plt.yticks(range(len(sorted_features)), sorted_features)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('XGBoost Feature Importance (Sorted)')
plt.tight_layout()
plt.show()

# %% cell 33
dict(zip(feature_names, [float(f) for f in importances_mean]))

# %% cell 35
# SAVE OOF TO DISK FOR ENSEMBLES
df_oof = pd.DataFrame({'xgb': oof})
df_oof.to_csv(f'{NAME}_oof.csv', index=False)

df_test = pd.DataFrame({'xgb': pred})
df_test.to_csv(f'{NAME}_test.csv',index=False)

df_test = pd.DataFrame({f'fold_{i}': pred for (i, pred) in enumerate(pred_per_fold)})
df_test.to_csv(f'{NAME}_test_per_fold.csv', index=False)

print("Saved oof to file")

# %% cell 36
test.shape

# %% cell 37
sub = pd.DataFrame({
    'id': test['id'],
    'Churn': pred
})

sub.to_csv(f'{NAME}_submission.csv', index=False)

# %% cell 38
import pprint

# %% cell 39
# pprint.pprint(xgb_params)

