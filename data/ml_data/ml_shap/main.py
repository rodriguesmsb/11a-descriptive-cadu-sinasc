# all these imports mean you can run this data on a linux machine with no monitor and it won't crash
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

# the rest of the imports
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import pickle
from os.path import join

import shap
from gbdt_utils import lightgbmExperiment, get_cv

# some settings
sns.set(context='poster')
warnings.filterwarnings("ignore")

# whether to run the full hyperopt or just to load a model
hyperopt_model = True
compute_shap = True

# lightgbm, hyperopt, shap_params params
hyperopt_evals = 25 # for hyperparam evaluation. each one takes a while
n_splits = 1 # just keep at 1 for the huge dataset
n_estimators = 2500 # number of trees for lightgbm
n_subset_for_shap = 25000 # shap takes a long time to return importances, this is how many samples to do it for

## import data
# all of the input variables
Xcols = ['m_age_yrs', 'marital_status', 'm_educ', 'n_live_child',
         'n_dead_child', 'preg_type', 'deliv_type', 'n_prenat_visit_cat',
         'sex', 'apgar1', 'apgar5', 'race', 'cong_anom', 'birth_year',
         'birth_state_code', 'm_state_code', 'm_educ_grade', 'm_race',
         'n_prev_preg', 'n_vag_deliv', 'n_ces_deliv', 'f_age_yrs',
         'gest_weeks', 'gest_method', 'n_prenat_visit', 'gest_month_precare',
         'presentation', 'labor_induced', 'ces_pre_labor', 'm_educ_2010',
         'birth_assist', 'm_educ_2010agg', 'hosp_pct_ces', 'hosp_deliv_type',
         'm_muni_mean_inc', 'm_muni_prop_2mw', 'm_muni_pop', 'birth_qtr']

# output variable
ycol = ['brthwt_z']

# categorical variables
categorical = {'marital_status', 'm_educ', 'preg_type', 'deliv_type', 'n_prenat_visit_cat',
               'sex', 'race', 'cong_anom', 'birth_state_code', 'm_state_code',
               'm_race', 'gest_method', 'presentation', 'labor_induced',
               'ces_pre_labor', 'm_educ_2010', 'birth_assist', 'm_educ_2010agg',
               'hosp_deliv_type', 'birth_qtr'}

# import data into pandas
df = pd.read_csv('data/snsc_ml.csv',
                 usecols=Xcols + ycol,
                 dtype={i: 'category' for i in categorical})

# convert categorical data integers 0, ..., k-1
# -1 means missing, as lightgbm expects
categorical_map_col = {}
for i, col in enumerate(categorical):
    categorical_map_col[col] = dict(enumerate(df[col].cat.categories))
    df[col] = df[col].cat.codes

# extract data into numpy
X = df[Xcols].values
y = df[ycol].values[:, 0]

# keep only rows that have non-nan non-inf outputs
flag = np.isfinite(y)
X = X[flag, :]
y = y[flag]

# split into test set and the rest
splitter = get_cv('regression', X, y, None, n_splits=1, random_state=0)
train_inds, test_inds = next(splitter)
X_train, y_train = X[train_inds], y[train_inds]
X_test, y_test = X[test_inds], y[test_inds]

# cross-validate a model OR load a model
model_name = 'lightgbm_' + str(n_estimators) + '_model.pickle'
if hyperopt_model:
    cab = lightgbmExperiment('regression',
                             'rmse',
                             n_estimators=n_estimators,
                             hyperopt_evals=hyperopt_evals,
                             n_splits=1)

    categorical_binary = np.array([i in categorical for i in Xcols])
    cab.run(X_train, y_train, X_test, y_test, None, categorical_binary)

    bst = cab.bst

    with open(join('..', 'data', 'ml_data', model_name), 'wb') as f:
        pickle.dump(bst, f)
else:
    # shap on a subset of values
    with open(join('..', 'data', 'ml_data', model_name), 'rb') as f:
        bst = pickle.load(f)

# get a subset of the test set and get shap values
# or just load it from the save file
if compute_shap:
    rand_inds = np.random.choice(np.arange(len(X_test)), n_subset_for_shap)
    X_test_sub = X_test[rand_inds, :]
    y_test_sub = y_test[rand_inds]
    shap_values = shap.TreeExplainer(bst).shap_values(X_test_sub)
    shap.summary_plot(shap_values, X_test_sub, feature_names=Xcols)

    with open(join('..', 'data', 'ml_data', 'lightgbm_' + str(n_estimators) + '_shap.pickle'), 'wb') as f:
        pickle.dump((X_test_sub, y_test_sub, shap_values), f)

else:
    with open(join('..', 'data', 'ml_data', 'lightgbm_' + str(n_estimators) + '_shap.pickle'), 'rb') as f:
        X_test_sub, y_test_sub, shap_values = pickle.load(f)

    shap.summary_plot(shap_values, X_test_sub, feature_names=Xcols)
