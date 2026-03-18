# %% cell 3
## Import packages

# General purpose modules
import time
from copy import deepcopy
import warnings
import math
from itertools import combinations

# Data handling and visualization modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import cupy as cp

# Skikit-learn preprocessing and evaluation modules
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

# Skikit-learn ML modules
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

# Further ML modules
import xgboost as xgboost
import lightgbm as lightgbm
from catboost import CatBoostClassifier
import torch
from pytabkit import RealMLP_TD_Classifier

# %% cell 5
## Read csv files, discritize labels and calculate statistical features

ADD_EXTERN_DATA = True # Extend competition dataset with external dataset

# Read csv files
trainval = pd.read_csv('/kaggle/input/competitions/playground-series-s6e3/train.csv')
test = pd.read_csv('/kaggle/input/competitions/playground-series-s6e3/test.csv')
extern_data = pd.read_csv('/kaggle/input/datasets/cdeotte/s6e3-original-dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Replace missing values in 'TotalCharges' of extern data with None and convert to same datatype as of the competition dataset
extern_data.loc[extern_data['TotalCharges'] == ' ', 'TotalCharges'] = None
extern_data['TotalCharges'] = extern_data['TotalCharges'].astype('float64')
if ADD_EXTERN_DATA:
    trainval = pd.concat([trainval[trainval.columns[1:]], extern_data[trainval.columns[1:]]]
                         ).reset_index(drop=True).reset_index().rename(columns={'index':'id'})

# Discretization of labels
target = 'Churn'
trainval[target] = LabelEncoder().fit_transform(trainval[target]).astype(np.uint8)

# %% cell 6
## Basic feature engineering

phone_servie_columns = ['PhoneService', 'MultipleLines']
internet_service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies']
service_columns = phone_servie_columns + internet_service_columns

# Function to create n-grams from categorical columns
def create_ngrams(row, n=2):
    return list(combinations(row, n))

for df in [trainval, test]:
    
    df['charges_deviation']      = (df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']).astype('float32')
    df['monthly_to_total_ratio'] = (df['MonthlyCharges'] / (df['TotalCharges'] + 1)).astype('float32')
    df['avg_monthly_charges']    = (df['TotalCharges'] / (df['tenure'] + 1)).astype('float32')
    df['service_count'] = (df[service_columns] == 'Yes').sum(axis=1).astype('int8')
    # Remove reduntant categories from service features
    df[internet_service_columns] = df[internet_service_columns].replace(to_replace='No internet service', value='No')
    df['MultipleLines'] = df['MultipleLines'].replace(to_replace='No phone service', value='No')
    # Round mothly charges to nearest multiple of 5
    df['MonthlyCharges_R'] = ((df[['MonthlyCharges']] / 5).round() * 5).astype('int8')
    # Create Trigrams from service columns
    #for gram in combinations(service_columns,3):
    #    df[gram[0]+'_'+gram[1]] = df[list(gram)].astype(str).agg('_'.join, axis=1)

# %% cell 7
## Calculate statistical features of target values on feature groups

STAT_N_BINS = 500

# Bucketize numerical features
num_columns = make_column_selector(dtype_include='float')(trainval) + ['tenure'] + ['service_count']
bin_num_columns = ['bin_'+col for col in num_columns]
cat_columns = make_column_selector(dtype_include='object')(trainval) + ['SeniorCitizen'] + ['MonthlyCharges_R']
stat_bucket_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
            ('stat_bins', KBinsDiscretizer(n_bins=STAT_N_BINS, strategy='uniform', encode='ordinal', random_state=42))])
trainval[bin_num_columns] = stat_bucket_pipeline.fit_transform(trainval[num_columns]).astype('int16')
test[bin_num_columns] = stat_bucket_pipeline.transform(test[num_columns]).astype('int16')

# Calculate global statistic values as replacement value for unknown groups
df_stat = trainval
global_stats = {'mean': df_stat[target].mean(), 'std': df_stat[target].std(), 'skew': df_stat[target].skew(),
                'median': df_stat[target].median(), 'min': df_stat[target].min(), 'max': df_stat[target].max(), 'count': 0}

# Save statistical features of target values being grouped by feature values
for stat in ['mean', 'std', 'skew', 'median', 'min', 'max', 'count']:
    globals()['stats_' + stat] = {}
for column in cat_columns + bin_num_columns:
    for stat in ['mean', 'std', 'skew', 'median', 'min', 'max', 'count']:
        globals()['stats_' + stat][column] = (df_stat.groupby(column)[target].agg([stat])/(len(df_stat) if stat=='count' else 1)
                                             ).to_dict()[stat]

# %% cell 8
## Configure data spliting strategy for k-fold training

STRAT = True # Use stratification for data spliting
EXTENDED_STRAT = True # Stratification is based on multiple features
FOLDS = 25 # Number of folds for k-fold training

# Determine stratification bins
strat_encoder = LabelEncoder()
strat_encoder_eval = LabelEncoder()
strat_cols = ['Contract', 'InternetService', 'PaymentMethod', target]
strat_cols_eval = ['Contract', 'InternetService', 'PaymentMethod', 'OnlineSecurity', 'TechSupport', 'SeniorCitizen']
trainval['multicat'] = strat_encoder.fit_transform(trainval[strat_cols].astype(str).agg('_'.join, axis=1)).astype('int16')
trainval['multicat_eval'] = strat_encoder_eval.fit_transform(trainval[strat_cols_eval].astype(str).agg('_'.join, axis=1)).astype('int16')

# Merge rare multicats into one category
rare_multicats = trainval['multicat'].value_counts()[trainval['multicat'].value_counts() < 9].index.tolist()
if len(rare_multicats)>0:
    trainval['multicat'] = trainval['multicat'].replace(rare_multicats, rare_multicats[0])

# Experimental data spliting to check spliting quality
skf = (StratifiedKFold if STRAT else KFold)(n_splits=FOLDS, shuffle=True, random_state=333)
train_idx, val_idx = next(skf.split(trainval, trainval['multicat'] if EXTENDED_STRAT else trainval[target]))
train = trainval.iloc[train_idx].reset_index()
val = trainval.iloc[val_idx].reset_index()
trainval_labels = trainval.pop(target)
train_labels = train.pop(target)
val_labels = val.pop(target)

# Verify dataset sizes of first fold
print(f"Total rows:   {len(trainval)}")
print(f"Dev train:    {len(train)} ({len(train)/len(trainval):.2%})")
print(f"Dev valid:    {len(val)} ({len(val)/len(trainval):.2%})")
print(f"Number of unique elements in multicat column: {len(trainval['multicat'].unique())}")
print('-'*80, end='\n\n')

# Size of stratification bins
print(trainval['multicat'].value_counts().tail())
print('-'*80, end='\n\n')
print(trainval['multicat_eval'].value_counts().tail())

# %% cell 9
## Explore train dataset

print('List of dataset columns including data types and number of non-zero elements: ', end='\n\n')
train.info()
print('-'*80, end='\n\n')

# Explore categorical features
print('Number of unique elements of categorical features: ', end='\n\n')
for cat in cat_columns:
    print(train[cat].value_counts(), end='\n\n')

# Explore numerical features
train[num_columns].hist(bins=50, figsize=(14,12), layout=(3,3))
plt.suptitle('Probability distribution of numerical features: ')
print('-'*80, end='\n\n')

# %% cell 10
## Compare probabilty distribution of features and labels between train, validation and test sets (based on a single fold)

# Create a dataframe with assigned sources (train/validation/test)
df_plot = pd.concat([train[num_columns+cat_columns].assign(Set='Train'), val[num_columns+cat_columns].assign(Set='Validation'),
                     test[num_columns+cat_columns].assign(Set='Test')])
df_plot.insert(3, value=pd.concat([train_labels, val_labels, pd.Series([None] * len(test), name=target)]), column=target)
df_plot[cat_columns + [target]] = df_plot[cat_columns + [target]].astype('str')
df_plot.reset_index(drop=True, inplace=True)
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn") # Suppress the specific FutureWarning

# Plot probabilty distribution of features
n_cols = 4
n_rows = math.ceil(len(num_columns+cat_columns+[target]) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
fig.suptitle('Feature distributions in train and validation sets')
axes = axes.flatten()
for i, col in enumerate(num_columns):
    sns.kdeplot(data=df_plot, x=col, ax=axes[i], hue='Set', common_norm=False, fill=True)
for j, col in enumerate(cat_columns + [target]):
    sns.histplot(data=df_plot, x=col, ax=axes[i+1+j], hue='Set', bins=len(df_plot[col].unique()),
                 stat='density', discrete=True, multiple="dodge", common_norm=False, shrink=.8)
plt.tight_layout()
plt.show()
del df_plot

# %% cell 12
## Helping function for adding statistical features

def target_stats(X, features, st_type, global_stats=global_stats):
    stats = globals()['stats_' + st_type]
    X_stat = pd.DataFrame()
    for c in features:
        X_stat[c] = X[c].map(stats[c]).fillna(global_stats[st_type])
    return X_stat

# %% cell 13
## Define pipelines

# Pipelines for numerical features
robust_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                            ('robust_scaling', RobustScaler())])
log_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                         ('log_trans', FunctionTransformer(func=lambda x: np.log(x+0.001), feature_names_out='one-to-one')),
                         ('robust_scaling', RobustScaler())])
square_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                            ('square_trans', FunctionTransformer(func=np.square, feature_names_out='one-to-one')),
                            ('robust_scaling', RobustScaler())])
cube_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                          ('cube_trans', FunctionTransformer(func=lambda x: np.power(x, 3), feature_names_out='one-to-one')),
                          ('robust_scaling', RobustScaler())])
sqrt_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                          ('sqrt_trans', FunctionTransformer(func=np.sqrt, feature_names_out='one-to-one')),
                          ('robust_scaling', RobustScaler())])
cbrt_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                          ('cbrt_trans', FunctionTransformer(func=np.cbrt, feature_names_out='one-to-one')),
                          ('robust_scaling', RobustScaler())])
kbins_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                           ('kbins', KBinsDiscretizer(n_bins=500, strategy='uniform', encode='ordinal', random_state=42)),
                           ('kbins_cast', FunctionTransformer(lambda X: X.astype(np.uint8), feature_names_out='one-to-one'))])

# Pipelines for categorical features
ordinal_pipeline = Pipeline([('imputer', SimpleImputer(strategy="most_frequent")),
                             ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int8))])
onehot_pipeline = Pipeline([('imputer', SimpleImputer(strategy="most_frequent")),
                            ('onehot', OneHotEncoder(sparse_output=False))])

# Pipelines for statistical features
mean_pipeline = Pipeline([('mean', FunctionTransformer(func=lambda x: target_stats(x, x.columns, 'mean'), feature_names_out='one-to-one'))])
std_pipeline = Pipeline([('std', FunctionTransformer(func=lambda x: target_stats(x, x.columns, 'std'), feature_names_out='one-to-one'))])
skew_pipeline = Pipeline([('skew', FunctionTransformer(func=lambda x: target_stats(x, x.columns, 'skew'), feature_names_out='one-to-one'))])
median_pipeline = Pipeline([('median', FunctionTransformer(func=lambda x: target_stats(x, x.columns, 'median'), feature_names_out='one-to-one'))])
min_pipeline = Pipeline([('min', FunctionTransformer(func=lambda x: target_stats(x, x.columns, 'min'), feature_names_out='one-to-one'))])
max_pipeline = Pipeline([('max', FunctionTransformer(func=lambda x: target_stats(x, x.columns, 'max'), feature_names_out='one-to-one'))])
frq_pipeline = Pipeline([('frq', FunctionTransformer(func=lambda x: target_stats(x, x.columns, 'count'), feature_names_out='one-to-one'))])

# Pipeline for target encoding
te_pipeline = Pipeline([('te', TargetEncoder(cv=7, shuffle=True, smooth='auto',
                                             target_type='binary', random_state=42))])

# Pipeline for PCA on non-linear transformations of numerical features
nonlinear_transformer = ColumnTransformer([("scaled", robust_pipeline, num_columns),
                                           ("log", log_pipeline, num_columns),
                                           ("square", square_pipeline, num_columns),
                                           ("cube", cube_pipeline, num_columns),
                                           ("sqrt", log_pipeline, num_columns),
                                           ("cbrt", square_pipeline, num_columns)
                                          ])
pca_pipeline = Pipeline([('nonlinear', nonlinear_transformer), ('pca', PCA(random_state=42))])

# %% cell 14
## Deinfe preprocessing pipeline and fit preprocessing

# Preprocessing pipeline
preprocessing = ColumnTransformer([## Numerical transformations
                                   ("scaled", robust_pipeline, num_columns),
                                   #("nonlin", nonlinear_transformer, num_columns),
                                   #("pca", pca_pipeline, num_columns),
                                   #("cluster", kbins_pipeline, num_columns),

                                   ## Categorical encoding
                                   ("ordinal", ordinal_pipeline, cat_columns),
                                   #("onehot", onehot_pipeline, cat_columns),
                                   
                                   ## Statistical transformers
                                   ("mean", mean_pipeline, bin_num_columns+cat_columns),
                                   ("std", std_pipeline, bin_num_columns+cat_columns),
                                   ("skew", skew_pipeline, bin_num_columns+cat_columns),
                                   ("median", median_pipeline, bin_num_columns),
                                   ("min", min_pipeline, bin_num_columns),
                                   ("max", max_pipeline, bin_num_columns),
                                   ("frq", frq_pipeline, bin_num_columns+cat_columns),
                                   #("te", te_pipeline, bin_num_columns+cat_columns)
                                  ]).set_output(transform='pandas')

# Preprocess data
train_prepared = preprocessing.fit_transform(trainval, trainval_labels)
test_prepared = preprocessing.transform(test)
train_labels = trainval_labels
train = trainval
print(f'Number of unfiltered features: {train_prepared.shape[1]}')

# %% cell 15
## Calculate model based meta-features with linear models

META_FEAT = False # Add model based meta features
META_PCA = False # Use PCA for calculating model based meta features

if META_FEAT:
    preprocessing_meta = ColumnTransformer([("pca", pca_pipeline, num_columns) if META_PCA else (
                                                'nonlinear', nonlinear_transformer, num_columns),
                                            ("onehot", onehot_pipeline, cat_columns)]).set_output(transform='pandas')
    train_meta = preprocessing_meta.fit_transform(train)
    val_meta = preprocessing_meta.transform(val)
    test_meta = preprocessing_meta.transform(test)
    print(f'Number of unfiltered features for meta feature learning: {train_meta.shape[1]}')
    
    lr_meta = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42).fit(train_meta, train_labels)
    sgdc_meta = SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42).fit(train_meta, train_labels)
    
    train_prepared['lr_meta'] = lr_meta.predict_proba(train_meta)[:,1]
    train_prepared['sgdc_meta'] = sgdc_meta.predict_proba(train_meta)[:,1]
    test_prepared['lr_meta'] = lr_meta.predict_proba(test_meta)[:,1]
    test_prepared['sgdc_meta'] = sgdc_meta.predict_proba(test_meta)[:,1]
    print(f'Number of unfiltered features including meta features: {train_prepared.shape[1]}')

# %% cell 16
## Final feature selection based on XGBoost feature importances

GPU_ACC = True
MAX_FEAT = 100 # Max number of features after feature selection

if MAX_FEAT:
    # Define base model for feature selection
    xgbr_fs = xgboost.XGBClassifier(device ='gpu' if GPU_ACC else 'cpu', random_state=42
                                   ).fit(train_prepared, train_labels)
    model_fs = SelectFromModel(xgbr_fs, max_features=MAX_FEAT, threshold=1e-6, prefit=True
                              ).set_output(transform="pandas").fit(train_prepared, train_labels)
    # Perform feature selection with choosen model
    train_prepared = model_fs.transform(train_prepared)
    test_prepared = model_fs.transform(test_prepared)
print(f'Number of selected features: {train_prepared.shape[1]}')

# %% cell 18
## Helping function to create parameter grids

def make_param(param_dict, model='est'):
    for elem in param_dict.copy():
        if elem == 'n_components':
            param_dict['pca'+'__'+elem] = param_dict.pop(elem)
        else:
            param_dict[model+'__'+elem] = param_dict.pop(elem)
    return param_dict

# %% cell 19
## Machine learning models and their hyperparameter search space

# Models
svc = SVC(kernel='linear', class_weight='balanced')
rfc = RandomForestClassifier(random_state=42)
kneigh = KNeighborsClassifier()
gbc = GradientBoostingClassifier(random_state=42)
xgb = xgboost.XGBClassifier(objective='binary:logistic', enable_categorical=True, device='cuda' if GPU_ACC else 'cpu',
                            random_state=42, eval_metric="auc")
ada = AdaBoostClassifier(random_state=42)
hgbc = HistGradientBoostingClassifier(scoring='roc_auc', class_weight='balanced', random_state=42)
lgbm = lightgbm.LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, random_state=42,
                               device ='gpu' if GPU_ACC else 'cpu', verbosity=-1)
catc = CatBoostClassifier(eval_metric='AUC', auto_class_weights='Balanced', random_state=42,
                          task_type='GPU' if GPU_ACC else 'CPU', verbose=False)
sgdc = SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42)
rmlp = RealMLP_TD_Classifier(device='cuda' if GPU_ACC else 'cpu', random_state=42, verbosity=2)

# Model space
EstimatorStr = {1: 'svc', 2: 'rfc', 3: 'kneigh', 4: 'gbc', 5: 'xgb', 6: 'ada',
                7: 'hgbc', 8: 'lgbm', 9: 'catc', 10: 'sgdc', 11: 'rmlp'}
EstimatorMdl = {1: svc, 2: rfc, 3: kneigh, 4: gbc, 5: xgb, 6: ada, 7: hgbc, 8: lgbm, 9: catc, 10: sgdc, 11: rmlp}

# %% cell 20
## Tuned hyperparameter sets

# svc parameter
param_single_svc = make_param({}) #
# rfc parameter
param_single_rfc = make_param({}) # 
# kneight parameter
param_single_kneigh = make_param({}) # 
# gbc parameter
param_single_gbc = make_param({}) # 
# xgb parameter
param_single_xgb = make_param({'n_estimators': 50000, 'learning_rate': 0.05, 'early_stopping_rounds': 100,
                               'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8,
                               'reg_alpha': 0.1, 'reg_lambda': 1.0, 'gamma': 0.05, 'min_child_weight': 5
                              }) # 
# ada parameter
param_single_ada = make_param({}) # 
# hgbc parameter
param_single_hgbc = make_param({'max_iter': 5000, 'learning_rate': 0.05, #'max_bins': 160,
                                }) # 
# lgbm parameter
param_single_lgbm = make_param({'n_estimators': 50000, 'learning_rate': 0.05, 'early_stopping_rounds': 100,
                                #'num_leaves': 127, 'subsample': 0.8, 'colsample_bytree': 0.8,
                                }) # 
# catc parameter
param_single_catc = make_param({'n_estimators': 50000, 'learning_rate': 0.025, 'early_stopping_rounds': 350,
                                'max_depth': 5, 'l2_leaf_reg': 14.026538565247332, 'random_strength': 2.922287095621834,
                                'border_count': 153, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8244572613654564,
                                }) # 
# sgdc parameter
param_single_sgdc = make_param({}) # 

# rmlp parameter
param_single_rmlp = make_param({'n_epochs': 3, 'batch_size': 256, 'n_ens': 8, 'val_metric_name': '1-auc_ovr',
                                'use_early_stopping': True, 'early_stopping_additive_patience': 20,
                                'early_stopping_multiplicative_patience': 1, 'act': "mish", 'embedding_size': 6,
                                'first_layer_lr_factor': 0.25, 'hidden_sizes': "rectangular", 'hidden_width': 352,
                                'lr': 0.075, 'ls_eps': 0.01, 'ls_eps_sched': "coslog4", 'max_one_hot_cat_size': 18,
                                'n_hidden_layers': 4, 'p_drop': 0.05, 'p_drop_sched': "flat_cos", 'plr_hidden_1': 16,
                                'plr_hidden_2': 8, 'plr_lr_factor': 0.1151, 'plr_sigma': 2.33, 'scale_lr_factor': 2.24,
                                'sq_mom': 0.988, 'wd': 0.0236,})

# %% cell 21
## Hyperparameter sets for parameter tuning

# xgb parameter
param_grid_xgb = make_param({'n_estimators': [50000],
                             'early_stopping_rounds': [100],
                             'max_depth': [3, 4, 5],
                             'learning_rate': [0.05],
                             'subsample': [0.8, 1.0],
                             'colsample_bytree': [0.8, 1.0],
                             'reg_lambda': [0.1, 1, 10],
                             #'reg_alpha': [0.1, 1, 10],
                             })
# hgbc parameter
param_grid_hgbc = make_param({#'max_depth': [3, 4, 5, 6, 9, 12],
                              #'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                             'max_bins': [10,20,40,80,160,255]
                              })
# lgbm parameter
param_grid_lgbm = make_param({'num_leaves': [80,110,120,125,130,135,140,150,180],
                              'max_depth': [24,29,31,32,33,35,40],
                              'learning_rate': [0.05,0.1,0.2],
                              #'n_estimators': [100,200,300],
                              })
# catc parameter
param_grid_catc = make_param({'n_estimators': [8000], # 4000, 6000, 8000, 10000
                              'learning_rate': [0.03], # 0.005, 0.01, 0.03, 0.05, 0.1
                              'max_depth': [4,5,6,7,8], #
                              'early_stopping_rounds': [100],
                                #'learning_rate': 0.06, 'max_depth': 2, 'l2_leaf_reg': 0.3
                                })

# %% cell 23
## Single fitting with tuned parameters or grid search for machine learning methods

TUNING = False # Choose between single fitting or parameter tuning
EST_IDS = [5,8,9,11] # Choose model(s) {1: 'svc', 2: 'rfc', 3: 'kneigh', 4: 'gbc', 5: 'xgb', 6: 'ada', 7: 'hgbc', 8: 'lgbm', 9: 'catc', 10: 'sgdc'}
EST_IDS_W_EARLYSTOPPING = [5,8,9,11]
EST_IDS_W_CAT_FEAT = [5,7,8,9,11]
MULTI_SEED = True
val_preds = pd.DataFrame(train_labels)
test_pred = pd.DataFrame()

for est_id in EST_IDS:
    start_time = time.time()
    roc_auc_train_ls = []
    roc_auc_val_ls = []
    
    # Define pipeline
    pipeline = Pipeline([('est', EstimatorMdl[est_id])])

    # Cross-validation configuration w/wo extended stratification
    cv_gen = skf.split(train_prepared, train['multicat'] if EXTENDED_STRAT else train_labels)

    # Cast categorical features to 'category' for choosen estimators being able to handle it
    cat_feat = make_column_selector(pattern='ordinal|cluster|onehot')(train_prepared)
    #cat_feat = make_column_selector(pattern=r'^(?!.*(?:scaled|___bin)).*$')(train_prepared)
    train_prepared[cat_feat] = train_prepared[cat_feat].astype(str).astype('category' if est_id in EST_IDS_W_CAT_FEAT else 'uint8')
    test_prepared[cat_feat] = test_prepared[cat_feat].astype(str).astype('category' if est_id in EST_IDS_W_CAT_FEAT else 'uint8')
    catc.set_params(cat_features=cat_feat if est_id in EST_IDS_W_CAT_FEAT else None)
    
    # Fitting or tuning on train dataset with k-fold cross-validation
    param = globals()[f'param_grid_{EstimatorStr[est_id]}' if TUNING else f'param_single_{EstimatorStr[est_id]}']
    if TUNING:
        eval_set = {}
        if est_id in EST_IDS_W_EARLYSTOPPING:
            # Split data into train and validation set and set configuration parameters if early stopping configured
            sss = (StratifiedShuffleSplit if STRAT else ShuffleSplit)(n_splits=1, test_size=0.1, random_state=42)
            train_index, eval_index = next(sss.split(train, train['multicat'] if EXTENDED_STRAT else train[target]))
            X_train, X_eval = train_prepared.iloc[train_index], train_prepared.iloc[eval_index]
            train_multi = train['multicat'].iloc[train_index],
            y_train, y_eval = train_labels.iloc[train_index], train_labels.iloc[eval_index]
            eval_set['est__eval_set'] = [(X_eval, np.array(y_eval))]
            if est_id==5: eval_set['est__verbose'] = 0
            cv_gen = skf.split(X_train, y_train if EXTENDED_STRAT else y_train) #train_multi
        else:
            X_train, y_train = train_prepared, train_labels

        # Tune model
        grid = GridSearchCV(pipeline, param, scoring='roc_auc', verbose=3, cv=cv_gen)
        grid.fit(X_train, np.array(y_train), **eval_set)
        print(grid.best_params_)
        print(grid.cv_results_)
        pipeline_tune = grid.best_estimator_

        # Store predictions and model
        val_preds.loc[eval_index, f'pred_{EstimatorStr[est_id]}'] = pipeline_tune.predict_proba(X_eval)[:,1]
        val_preds = val_preds.dropna()
        test_pred[f'pred_{EstimatorStr[est_id]}'] = pipeline_tune.predict_proba(test_prepared)[:,1]
        globals()[f'model1_{EstimatorStr[est_id]}'] = pipeline_tune
    else:
        # Training with k-fold cross-validation
        for i, (train_index, eval_index) in enumerate(cv_gen):
            # Split data into train and validation set
            X_train, X_eval = train_prepared.iloc[train_index], train_prepared.iloc[eval_index]
            y_train, y_eval = train_labels.iloc[train_index], train_labels.iloc[eval_index]

            # Clon selected pipeline, set configuration parameters and train model
            pipeline_train = deepcopy(pipeline)
            pipeline_train.set_params(**param)
            if MULTI_SEED:
                pipeline_train.set_params(est__random_state=(42+1*i))  
            eval_set = {}
            if est_id in EST_IDS_W_EARLYSTOPPING:
                if est_id==11:
                    eval_set['est__X_val'] = X_eval
                    eval_set['est__y_val'] = np.array(y_eval)
                else:
                    eval_set['est__eval_set'] = [(X_eval, np.array(y_eval))]
                if est_id==5:
                    eval_set['est__verbose'] = 0
            pipeline_train.fit(X_train, np.array(y_train), **eval_set)

            # Calculate and show ROC AUC scores after training of each fold
            train_score = roc_auc_score(np.array(y_train), pipeline_train.predict_proba(X_train)[:,1])
            eval_preds = pipeline_train.predict_proba(X_eval)[:,1]
            val_score = roc_auc_score(np.array(y_eval), eval_preds)
            
            # Store oof predictions and the scores for each fold
            val_preds.loc[eval_index, f'pred_{EstimatorStr[est_id]}'] = eval_preds
            roc_auc_train_ls.append(train_score)
            roc_auc_val_ls.append(val_score)

            # Store predictions for test set
            test_pred[f'pred{i+1}_{EstimatorStr[est_id]}'] = pipeline_train.predict_proba(test_prepared)[:,1]

            # Save trained pipeline
            globals()[f'model{i+1}_{EstimatorStr[est_id]}'] = pipeline_train
            
            print(f'Estimator: {EstimatorStr[est_id]} of fold {i+1} is fitted')
            print(f'Train ROC_AUC score: {train_score}')
            print(f'Val ROC_AUC score: {val_score}')
            print(f'Elapsed time: {int(time.time() - start_time)} [s]')
            print('-'*40)

        print(f'Average train ROC_AUC score of {EstimatorStr[est_id]} estimator over all folds: {sum(roc_auc_train_ls)/len(roc_auc_train_ls)}')
        print(f'Average val ROC_AUC score of {EstimatorStr[est_id]} estimator over all folds: {sum(roc_auc_val_ls)/len(roc_auc_val_ls)}')
        print('-'*80)
        print('-'*80)

# %% cell 25
## Calculate ROC_AUC scores for each estimators and mean ROC_AUC score over all estimators based on accumulated Out-of-Fold (oof) samples

# ROC_AUC scores for each estimator based on accumulated oof samples
for est_id in EST_IDS:
    est_oof_score = roc_auc_score(val_preds[target], val_preds[f'pred_{EstimatorStr[est_id]}'])
    print(f'ROC_AUC score over all oof samples for {EstimatorStr[est_id]} estimator: {est_oof_score}')
    print('-'*80)

# ROC_AUC score on mean value of ensemble predictions based on accumulated oof samples
val_preds['pred_score'] = val_preds.filter(like='pred').mean(axis=1)
overall_oof_score = roc_auc_score(val_preds[target], val_preds['pred_score'])
print(f'Overall ROC_AUC score over all oof samples and all estimators: {overall_oof_score}')
print('-'*80)
print('Show predictions and labels of validation dataset: ')
print(val_preds)

# %% cell 26
## Show ROC_AUC scores of subcategory subsets

subcats = []
subcat_ROC_AUC_scores  = []
for cat in cat_columns:
    for subcat in np.sort(train[cat].unique()):
        subcats.append(cat+'_'+str(subcat))
        val_filtered = val_preds[target][train[cat] == subcat]
        val_labels_filtered = val_preds['pred_score'][train[cat] == subcat]
        subcat_ROC_AUC_scores.append(roc_auc_score(val_filtered, val_labels_filtered))

# Create bar chart
fig1, ax1 = plt.subplots(figsize=(10, 8))
cmap = plt.get_cmap('viridis')
rescale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
normalized_values = rescale(subcat_ROC_AUC_scores)
ax1.barh(subcats, np.array(subcat_ROC_AUC_scores)-overall_oof_score, color=cmap(normalized_values), left=overall_oof_score)
plt.title(f'ROC_AUC scores of subcategory subsets compared to the average ROC_AUC score')
plt.xlabel('ROC_AUC score')
plt.ylabel('Subcategories')
ax1.axvline(x=overall_oof_score, color='green', linestyle='-.')
plt.tight_layout()
ax1.tick_params(left=False, bottom=True)

# %% cell 27
## Show feature importances

if est_id in [5, 8, 9]:
    # Sort feature names and their importances
    categories = train_prepared.columns
    values = globals()[f'model1_{EstimatorStr[est_id]}'][-1].feature_importances_+1e-4
    sorted_values, sorted_categories = zip(*sorted(zip(values,categories), reverse=False))
    
    # Plot feature importances
    fig3, ax3 = plt.subplots(figsize=(10, 20))
    normalized_values = rescale(np.log(sorted_values))
    ax3.barh(sorted_categories, sorted_values, color=cmap(normalized_values), log=True)
    plt.title(f'Feature Importances (Magnitude)')
    plt.xlabel('Logarithmic importance score')
    plt.ylabel('Features')
    plt.tight_layout()
    ax3.tick_params(left=False, bottom=True)

# %% cell 28
## Show worse ROC_AUC scores of multicategory subsets

if not TUNING:
    multicats = []
    multicat_ROC_AUC_scores = []
    for multicat in train['multicat_eval'].unique():
        if len(train_labels[train['multicat_eval'] == multicat].unique()) > 1:
            multicats.append(multicat)
            val_filtered = val_preds[target][train['multicat_eval'] == multicat]
            val_labels_filtered = val_preds['pred_score'][train['multicat_eval'] == multicat]
            multicat_ROC_AUC_scores.append(roc_auc_score(val_filtered, val_labels_filtered))
    multicats = strat_encoder_eval.inverse_transform(multicats)
    sorted_multicat_ROC_AUC_scores, sorted_multicats = zip(*sorted(zip(multicat_ROC_AUC_scores,multicats), reverse=False))
    
    # Create bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    n_top = 10
    cmap = plt.get_cmap('viridis')
    rescale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    normalized_values = rescale(sorted_multicat_ROC_AUC_scores[:n_top])
    ax2.barh(sorted_multicats[:n_top], np.array(sorted_multicat_ROC_AUC_scores[:n_top])-overall_oof_score,
             color=cmap(normalized_values), left=overall_oof_score)
    plt.title(f'Worse {n_top} ROC_AUC scores of multicategory subsets')
    plt.xlabel('ROC_AUC score')
    plt.ylabel('Subcategories (Thallium_Chest pain type_Number of vessels fluro)')
    plt.tight_layout()
    ax2.tick_params(left=False, bottom=True)

# %% cell 30
## Test prediction & submission 

submission_df = test[['id']].copy()

# Take the mean value of prediction over all estimators and folds
submission_df[target] = test_pred.mean(axis=1)

# Write dataframe to .csv file
submission_df.to_csv("submission.csv", index=False)
print("✅ submission.csv saved!")
submission_df

