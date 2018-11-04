# -*- coding: utf-8 -*-
"""
Created on Tue May 29 12:35:14 2018

@author: sergey feldman
"""

import time
from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit, ShuffleSplit

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL

import lightgbm as lgb

'''
some shared functions
'''


def get_cv(learning_task, X, y, groups=None, n_splits=10, random_state=0):
    if n_splits > 1:
        if groups is not None:
            cv = GroupKFold(n_splits)
            return cv.split(X, y, groups)
        elif learning_task == 'classification':
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
            return cv.split(X, y)
        else:
            cv = KFold(n_splits, shuffle=True, random_state=random_state)
            return cv.split(X, y)
    elif n_splits == 1:
        if groups is not None:
            cv = GroupShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=random_state)
            return cv.split(X, y, groups)
        elif learning_task == 'classification':
            cv = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=random_state)
            return cv.split(X, y)
        else:
            cv = ShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=random_state)
            return cv.split(X, y)
    else:
        return None


'''
Experiment abstract class
'''


class Experiment(object, metaclass=ABCMeta):

    # This method must be overridden but can still be called via super by subclasses have shared construction logic
    @abstractmethod
    def __init__(self):
        pass

    def get_dataset_pair(self, X_train, y_train, X_test, y_test, cat_cols):
        raise NotImplementedError('Method get_dataset_pair is not implemented.')

    def fit(self, params, dtrain, dtest, n_estimators, seed=0):
        raise NotImplementedError('Method fit is not implemented.')

    def split_and_preprocess(self, X_train, y_train, X_test, y_test, groups_train=None, cat_cols=None, n_splits=10):
        cv = get_cv(self.learning_task, X_train, y_train, groups=groups_train, n_splits=n_splits,
                    random_state=self.random_seed)
        cv_pairs = []
        for train_index, test_index in cv:
            cv_pair = self.get_dataset_pair(X_train[train_index], y_train[train_index],
                                            X_train[test_index], y_train[test_index],
                                            cat_cols)
            cv_pairs.append(cv_pair)

        train_test_pair = self.get_dataset_pair(X_train, y_train, X_test, y_test, cat_cols)
        return cv_pairs, train_test_pair

    def run_cv(self, cv_pairs, params, n_estimators=None, verbose=False):
        n_estimators = n_estimators or self.n_estimators
        evals_results, start_time = [], time.time()
        for dtrain, dtest in cv_pairs:
            _, evals_result = self.fit(params, dtrain, dtest, n_estimators, self.random_seed)
            evals_results.append(evals_result)
        mean_evals_results = np.mean(evals_results, axis=0)
        # TODO: get best_n_estimators without overfitting by using an additional holdout
        # best_n_estimators = np.argmin(self.eval_multiplier * mean_evals_results) + 1
        best_n_estimators = n_estimators
        eval_time = time.time() - start_time

        cv_result = {'loss': self.eval_multiplier * mean_evals_results[best_n_estimators - 1],
                     'best_n_estimators': best_n_estimators,
                     'eval_time': eval_time,
                     'status': STATUS_FAIL if np.isnan(mean_evals_results[best_n_estimators - 1]) else STATUS_OK,
                     'params': params.copy()}
        self.best_loss = min(self.best_loss, cv_result['loss'])
        self.hyperopt_eval_num += 1
        cv_result.update({'hyperopt_eval_num': self.hyperopt_eval_num, 'best_loss': self.best_loss})

        if verbose:
            print('[{0}/{1}]\teval_time={2:.2f} sec\tcurrent_{3}={4:.6f}\tmin_{3}={5:.6f}'.format(
                self.hyperopt_eval_num, self.hyperopt_evals, eval_time,
                self.eval_metric, cv_result['loss'], self.best_loss))
        return cv_result

    def run_test(self, dtrain, dtest, params=None, n_estimators=None, custom_metric=None, seed=0):
        params = params or self.best_params
        n_estimators = n_estimators or self.best_n_estimators or self.n_estimators
        start_time = time.time()
        self.bst, evals_result = self.fit(params, dtrain, dtest, n_estimators, seed=seed)
        eval_time = time.time() - start_time
        result = {'loss': evals_result[-1], 'n_estimators': n_estimators,
                  'eval_time': eval_time, 'status': STATUS_OK, 'params': params.copy()}
        return result

    def run(self, X_train, y_train, X_test, y_test, groups_train=None, cat_cols=None):
        print('Loading and preprocessing dataset...')
        cv_pairs, (dtrain, dtest) = self.split_and_preprocess(X_train, y_train, X_test, y_test, groups_train, cat_cols,
                                                              self.n_splits)

        print('Optimizing params...')
        cv_result = self.optimize_params(cv_pairs)
        self.print_result(cv_result, '\nBest result on cv')

        print('\nTraining algorithm with the tuned parameters for different seed...')
        test_losses = []
        for seed in range(5):
            test_result = self.run_test(dtrain, dtest, seed=seed)
            test_losses.append(test_result['loss'])
            print('For seed=%d Test\'s %s : %.5f' % (seed, self.eval_metric, test_losses[-1]))

        print('\nTest\'s %s mean: %.5f, Test\'s %s std: %.5f' % (self.eval_metric,
                                                                 np.mean(test_losses),
                                                                 self.eval_metric,
                                                                 np.std(test_losses)))

        return test_losses

    def optimize_params(self, cv_pairs, max_evals=None, verbose=True):
        max_evals = max_evals or self.hyperopt_evals
        self.trials = Trials()
        self.hyperopt_eval_num, self.best_loss = 0, np.inf

        fmin(fn=lambda params: self.run_cv(cv_pairs, params, verbose=verbose),
             space=self.space, algo=tpe.suggest, max_evals=max_evals, trials=self.trials)

        self.best_params = self.trials.best_trial['result']['params']
        if 'best_n_estimators' in self.trials.best_trial['result'].keys():
            self.best_n_estimators = self.trials.best_trial['result']['best_n_estimators']
        return self.trials.best_trial['result']

    def print_result(self, result, name='', extra_keys=None):
        print('%s:\n' % name)
        print('%s = %s' % (self.eval_metric, result['loss']))
        if 'best_n_estimators' in result.keys():
            print('best_n_estimators = %s' % result['best_n_estimators'])
        elif 'n_estimators' in result.keys():
            print('n_estimators = %s' % result['n_estimators'])
        print('params = %s' % result['params'])
        if extra_keys is not None:
            for k in extra_keys:
                if k in result:
                    print("%s = %f" % (k, result[k]))


'''
lightgbm subclass
'''


class lightgbmExperiment(Experiment):

    def __init__(self, learning_task, metric, eval_metric=None, n_estimators=5000, hyperopt_evals=50, n_splits=10,
                 tree_learner='feature', random_seed=0, threads=6):
        # TODO: set param like num_class

        super().__init__()

        if learning_task == 'classification':
            assert metric in {'binary', 'multiclass', 'multiclassova'}
        elif learning_task == 'regression':
            assert metric in {'rmse', 'mae', 'huber', 'fair', 'poisson', 'quantile',
                              'mape', 'gamma', 'tweedie'}

        self.learning_task = learning_task
        self.metric = metric
        self.n_estimators = n_estimators
        self.hyperopt_evals = hyperopt_evals
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.threads = threads

        self.best_loss = np.inf
        self.best_n_estimators = None
        self.hyperopt_eval_num = 0

        # eval metric
        if eval_metric in {'auc', 'binary_logloss', 'binary_error', 'multi_logloss',
                           'multi_error', 'l1', 'l2', 'rmse', 'quantile', 'mape', 'huber',
                           'fair', 'poisson', 'gamma', 'gamma_deviance', 'tweedie'}:
            self.eval_metric = eval_metric
        else:
            self.eval_metric = metric  # same as objective

        self.space = {
            'learning_rate': hp.loguniform('learning_rate', -7, 0),
            'num_leaves': hp.qloguniform('num_leaves', 0, 7, 1),
            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
            'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 5, 1),
            'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
            'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
            'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
        }

        self.constant_params = {
            'num_threads ': self.threads,
            'tree_learner': tree_learner,
            'objective': self.metric,
            'metric': self.eval_metric,
            'bagging_freq': 1,
            'verbose': -1
        }

        # hyperopt minimizes metrics, so we have negate ones where larger is better
        if self.eval_metric == 'auc':
            self.eval_multiplier = -1
        else:
            self.eval_multiplier = 1

    def fit(self, params, dtrain, dtest, n_estimators, seed=0):
        params['num_leaves'] = max(int(params['num_leaves']), 2)
        params['min_data_in_leaf'] = int(params['min_data_in_leaf'])
        params.update(self.constant_params)
        params.update({
            'data_random_seed': 1 + seed,
            'feature_fraction_seed': 2 + seed,
            'bagging_seed': 3 + seed,
            'drop_seed': 4 + seed,
        })

        evals_result = {}
        bst = lgb.train(params, dtrain, valid_sets=dtest, valid_names=['test'], evals_result=evals_result,
                        num_boost_round=n_estimators, verbose_eval=False)

        results = evals_result['test'][self.eval_metric]

        return bst, results

    def get_dataset_pair(self, X_train, y_train, X_test, y_test, cat_cols=None):
        if cat_cols is None:
            categorical_feature = []
        else:
            categorical_feature = np.nonzero(cat_cols)[0].tolist()
        dtrain = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature, free_raw_data=False)
        dtest = lgb.Dataset(X_test, y_test, reference=dtrain, categorical_feature=categorical_feature,
                            free_raw_data=False)
        return (dtrain, dtest)
