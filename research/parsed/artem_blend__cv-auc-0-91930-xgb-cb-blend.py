# %% cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

pred1 = pd.read_csv('/kaggle/input/notebooks/artemevstafyev/xgb-cb-best-cv-auc-0-91930-blend/submission.csv', float_precision='round_trip').iloc[:, 1].rename('pred1')
pred2 = pd.read_csv('/kaggle/input/notebooks/anthonytherrien/predict-customer-churn-blend/submission.csv', float_precision='round_trip').iloc[:, 1].rename('pred2')
adjustment = pd.read_csv('/kaggle/input/datasets/artemevstafyev/predict-customer-churn-adjustment/adjustment.csv', float_precision='round_trip').iloc[:, 1]

# %% cell 2
rank_new = rankdata(pred2) * 0.99 + adjustment * 0.01

# %% cell 3
def calibrate_rank_by_pred(rank, pred):
    df = pd.DataFrame()
    df['rank'] = rank
    df['pred'] = pred
    df_me = df.groupby('rank')['pred'].mean()
    df_me.loc[:] = np.sort(df_me.values)
    for i in range(1, df_me.shape[0]):
        if df_me[df_me.index[i]] <= df_me[df_me.index[i-1]]:
            df_me[df_me.index[i]] = min(df_me[df_me.index[i-1]] + 1e-6 / df.shape[0], 1.0)
    df = df.join(df_me, on = 'rank', rsuffix = '_new')
    return df.pred_new.values

pred_new = calibrate_rank_by_pred(rank_new, pred1)

# %% cell 4
h = plt.hist(pred_new, 100)

# %% cell 5
submission = pd.read_csv('/kaggle/input/competitions/playground-series-s6e3/sample_submission.csv')
submission['Churn'] = pred_new
submission.to_csv('submission.csv', index=False)

