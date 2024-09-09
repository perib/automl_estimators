
from sklearn.linear_model import SGDRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import available_if
import optuna
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score, auc, recall_score, precision_recall_curve, \
                             roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             f1_score)
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer


import traceback

import numpy as np
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
import sklearn.feature_selection
from functools import partial
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.svm import LinearSVC

from functools import partial
#import GaussianNB

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

import numpy as np
from functools import partial
import numpy as np

from sklearn.preprocessing import Binarizer
from sklearn.decomposition import FastICA
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


# from warnings import simplefilter
# simplefilter("ignore", category=RuntimeWarning)

# optuna.logging.disable_default_handler()
# optuna.logging.set_verbosity(optuna.logging.WARNING)

def params_LogisticRegression(trial, random_state, name=None, n_jobs=1):
    params = {}
    params['solver'] ="saga"
    params['dual'] = False
    params['penalty'] = trial.suggest_categorical(name=f'penalty_{name}', choices=['l1', 'l2',"elasticnet"])
    params['C'] = trial.suggest_float(f'C_{name}', 0.01, 1e5, log=True)
    params['l1_ratio'] = None
    if params['penalty'] == 'elasticnet':
        params['l1_ratio'] = trial.suggest_float(f'l1_ratio_{name}', 0.0, 1.0)

    params['class_weight'] = trial.suggest_categorical(name=f'class_weight_{name}', choices=['balanced', None])
    params['n_jobs'] = n_jobs
    params['random_state'] = random_state
    return params


def params_RandomForestClassifier(trial, random_state, name=None, n_jobs=1,):
    params = {
        'n_estimators': 128,
        'max_features': trial.suggest_float(f'max_features_{name}', 0.01, 1.0),
        'criterion': trial.suggest_categorical(name=f'criterion_{name}', choices=['gini', 'entropy']),
        'min_samples_split': trial.suggest_int(f'min_samples_split_{name}', 2, 20),
        'min_samples_leaf': trial.suggest_int(f'min_samples_leaf_{name}', 1, 20),
        'bootstrap': trial.suggest_categorical(name=f'bootstrap_{name}', choices=[True, False]),
        'class_weight': trial.suggest_categorical(name=f'class_weight_{name}', choices=['balanced', None]),

        'n_jobs': n_jobs,
        'random_state': random_state,
    }
    return params

def params_XGBClassifier(trial, random_state, name=None, n_jobs=1,):
    return {
        'learning_rate': trial.suggest_float(f'learning_rate_{name}', 1e-3, 1, log=True),
        'subsample': trial.suggest_float(f'subsample_{name}', 0.1, 1.0),
        'min_child_weight': trial.suggest_int(f'min_child_weight_{name}', 1, 21),
        #'booster': trial.suggest_categorical(name='booster_{name}', choices=['gbtree', 'dart']),
        'n_estimators': 100,
        'max_depth': trial.suggest_int(f'max_depth_{name}', 1, 11),
        'n_jobs': n_jobs,
        #'use_label_encoder' : True,
        'random_state': random_state,
    }

def params_KNeighborsClassifier(trial, random_state, name=None, n_jobs=1, n_samples=10):
    return {
        'n_neighbors': trial.suggest_int(f'n_neighbors_{name}', 1, n_samples, log=True ),
        'weights': trial.suggest_categorical(f'weights_{name}', ['uniform', 'distance']),
        'p': trial.suggest_int('p', 1, 3),
        'metric': str(trial.suggest_categorical(f'metric_{name}', ['euclidean', 'minkowski'])),
        'n_jobs': n_jobs,
    }

def params_sklearn_feature_selection_SelectPercentile(trial, random_state, name=None, n_jobs=1):
    return {
        'percentile': trial.suggest_float(f'percentile_{name}', 1, 100.0),
        'score_func' : sklearn.feature_selection.f_classif,
    }

def params_sklearn_feature_selection_VarianceThreshold(trial,random_state,  name=None, n_jobs=1):
    return {
        'threshold': trial.suggest_float(f'threshold_{name}', 1e-4, .2, log=True)
    }

def params_sklearn_decomposition_PCA(trial, random_state, name=None, n_features=100, n_jobs=1):
    # keep the number of components required to explain 'variance_explained' of the variance
    variance_explained = 1.0 - trial.suggest_float(f'n_components_{name}', 0.001, 0.5, log=True) #values closer to 1 are more likely

    return {
        'n_components': variance_explained,
        'random_state': random_state,
    }

all_params = {
                "RandomForestClassifier": params_RandomForestClassifier,
                "XGBClassifier": params_XGBClassifier,
              "LogisticRegression": params_LogisticRegression, 
              "KNeighborsClassifier": params_KNeighborsClassifier,
                "PCA": params_sklearn_decomposition_PCA,
                "VarianceThreshold": params_sklearn_feature_selection_VarianceThreshold,
                "SelectPercentile": params_sklearn_feature_selection_SelectPercentile,
            
              }

name_to_class = {
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
    "LogisticRegression": LogisticRegression,
    "KNeighborsClassifier": KNeighborsClassifier,
    "PCA": PCA,
    "VarianceThreshold": VarianceThreshold,
    "SelectPercentile": SelectPercentile,
                 }


def get_params(trial, sequence, random_state=None, n_jobs=1):
    params_list = []

    for i, step_options in enumerate(sequence):
        if len(step_options) == 1:
            step = step_options[0]
            params = all_params[step](trial, name=f'{step}_{i}', random_state=random_state, n_jobs=n_jobs)
            params_list.append((step, params))
        else:
            selected_step_i = trial.suggest_categorical(f'step_{i}', np.arange(len(step_options)).astype(np.float64))
            step = step_options[int(selected_step_i)]
            params = all_params[step](trial, name=f'{step}_{i}', random_state=random_state, n_jobs=n_jobs)
            params_list.append((step, params))
    
    return params_list

    # for step in sequence:
    #     params = all_params[step](trial)
    #     params_list.append((step, params))

    # return params_list

def params_to_pipeline(params):
    steps = []
    for step, param in params:
        steps.append((step, name_to_class[step](**param)))

    return Pipeline(steps)

def get_pipeline(trial, sequence):
    steps = []
    for i, step in enumerate(sequence):
        params = all_params[step](trial)
        steps.append((step, name_to_class[step](**params)))
        
    return Pipeline(steps)

def objective(trial, X_train, y_train, sequence, scoring, cv, random_state, n_jobs=1):
    try:
        params = get_params(trial, sequence, random_state=random_state, n_jobs=n_jobs)
        pipeline = params_to_pipeline(params)
        trial.set_user_attr('params', params)
        #cross val score
        return sklearn.model_selection.cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=cv).mean()
    except Exception as e:
        print(f"failed error {e}")
        print(traceback.format_exc())
        return np.nan
    

class OptunaEstimator(sklearn.base.BaseEstimator):
    def __init__(self, sequence, scorer, n_trials, n_jobs, est_n_jobs=1, cv=StratifiedKFold(n_splits=5, random_state=42, shuffle=True), timeout=None, random_state=None):
        self.sequence = sequence
        self.scorer = scorer
        self.n_trials = n_trials
        self.est_n_jobs = est_n_jobs
        self.n_jobs = n_jobs
        self.cv = cv
        self.timeout = timeout
        self.random_state = random_state

        self.fitted_pipeline_ = None

        
    def fit(self, X, y):
        print("start fitting")
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study = optuna.create_study(direction='maximize', sampler=sampler)
        objective_fn = lambda trial: objective(trial, X, y, self.sequence, self.scorer, cv=self.cv, random_state=self.random_state, n_jobs=self.est_n_jobs)
        self.study.optimize(objective_fn, n_trials=self.n_trials, n_jobs=self.n_jobs, timeout=self.timeout)
        best_trial = self.study.best_trial
        best_params = best_trial.user_attrs['params']
        self.fitted_pipeline_ = params_to_pipeline(best_params)

        self.fitted_pipeline_.fit(X, y)

    def _estimator_has(attr):
        '''Check if we can delegate a method to the underlying estimator.
        First, we check the first fitted final estimator if available, otherwise we
        check the unfitted final estimator.
        '''
        return  lambda self: (self.fitted_pipeline_ is not None and
            hasattr(self.fitted_pipeline_, attr)
        )
        
    @available_if(_estimator_has('predict'))
    def predict(self, X, **predict_params):
        check_is_fitted(self)
        #X = check_array(X)

        preds = self.fitted_pipeline_.predict(X,**predict_params)
        return preds

    @available_if(_estimator_has('predict_proba'))
    def predict_proba(self, X, **predict_params):
        check_is_fitted(self)
        #X = check_array(X)
        return self.fitted_pipeline_.predict_proba(X,**predict_params)

    @available_if(_estimator_has('decision_function'))
    def decision_function(self, X, **predict_params):
        check_is_fitted(self)
        #X = check_array(X)
        return self.fitted_pipeline_.decision_function(X,**predict_params)

    @available_if(_estimator_has('transform'))
    def transform(self, X, **predict_params):
        check_is_fitted(self)
        #X = check_array(X)
        return self.fitted_pipeline_.transform(X,**predict_params)
    
    @property
    def classes_(self):
        """The classes labels. Only exist if the last step is a classifier."""
        return self.fitted_pipeline_.classes_

    @property
    def _estimator_type(self):
        return self.fitted_pipeline_._estimator_type