from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
import time
import sklearn.metrics
from collections.abc import Iterable
import pandas as pd
import sklearn
import numpy as np

"""HPO
# Flags: doc-Runnable

!!! note "Dependencies"

    Requires the following integrations and dependencies:

    * `#!bash pip install openml amltk[smac, sklearn]`

This example shows the basic of setting up a simple HPO loop around a
`RandomForestClassifier`. We will use the [OpenML](https://openml.org) to
get a dataset and also use some static preprocessing as part of our pipeline
definition.

You can fine the [pipeline guide here](../guides/pipelines.md)
and the [optimization guide here](../guides/optimization.md) to learn more.

You can skip the imports sections and go straight to the
[pipeline definition](#pipeline-definition).

## Dataset

Below is just a small function to help us get the dataset from OpenML and encode the
labels.
"""

from typing import Any

import openml
from sklearn.preprocessing import LabelEncoder

from amltk.sklearn import split_data

import numpy as np
import openml
from sklearn.compose import make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC

from amltk.data.conversions import probabilities_to_classes
from amltk.ensembling.weighted_ensemble_caruana import weighted_ensemble_caruana
from amltk.optimization import History, Metric, Trial
from amltk.optimization.optimizers.smac import SMACOptimizer
from amltk.pipeline import Choice, Component, Sequential, Split
from amltk.scheduling import Scheduler
from amltk.sklearn.data import split_data
from amltk.store import PathBucket
from amltk.optimization import Metric
from amltk.scheduling import Scheduler
from amltk.optimization.optimizers.smac import SMACOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from amltk.pipeline import Component, Node, Sequential, Split
from functools import partial
import pandas as pd

def get_dataset(
    dataset_id: str | int,
    *,
    seed: int,
    splits: dict[str, float],
) -> dict[str, Any]:
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_features_meta_data=False,
        download_qualities=False,
    )

    target_name = dataset.default_target_attribute
    X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=target_name)
    _y = LabelEncoder().fit_transform(y)

    return split_data(X, _y, splits=splits, seed=seed)  # type: ignore

class AmltkEstimator(BaseEstimator):

    def __init__(self, pipeline, cv, scorers, other_objective_functions=None, max_time_seconds=5, n_jobs=1, seed=None, max_evals=None):
        self.pipeline = pipeline
        self.max_time_seconds = max_time_seconds
        self.n_jobs = n_jobs
        self.scorers = scorers
        self.other_objective_functions = other_objective_functions
        self.cv = cv
        self.seed = seed
        self.max_evals = max_evals
   

        self._scorers = [sklearn.metrics.get_scorer(scoring) for scoring in self.scorers]
        self._scorer_names = [scoring for scoring in self.scorers]



        self.objective_names = [f._score_func.__name__ if hasattr(f,"_score_func") else f.__name__ for f in self._scorers] + [f.__name__ for f in other_objective_functions]



    def fit(self, X, y):
        # bucket = PathBucket("AMLTKEstimator_TMP", clean=True, create=True)
        # data_bucket = bucket / "data"
        # data_bucket.store(
        #     {
        #         "X.csv": pd.DataFrame(X),
        #         "y.csv": pd.DataFrame(y),
        #     },
        # )

        scheduler = Scheduler.with_processes(1)

        metrics = [Metric(scorer, minimize=False) for scorer in self.objective_names] #bounds?
        optimizer = SMACOptimizer.create(
            space=self.pipeline,  # <!> (1)!
            metrics=metrics,
            # bucket=bucket,
            seed=self.seed,
        )

        #import partial
        # target_function_p = partial(target_function,X=X, y=y, scorers=self.scorers, cv=self.cv, other_objective_functions=self.other_objective_functions, objective_names=self.objective_names)
        target_function_p = partial(target_function,X=X, y=y, scorers=self.scorers, cv=self.cv, other_objective_functions=self.other_objective_functions, objective_names=self.objective_names)
        task = scheduler.task(target_function_p)

        @scheduler.on_start
        def launch_initial_tasks() -> None:
            """When we start, launch `n_workers` tasks."""
            trial = optimizer.ask()
            task.submit(trial, _pipeline=self.pipeline)

        @task.on_result
        def tell_optimizer(_, report: Trial.Report) -> None:
            """When we get a report, tell the optimizer."""
            optimizer.tell(report)

        trial_history = History()


        @task.on_result
        def add_to_history(_, report: Trial.Report) -> None:
            """When we get a report, print it."""
            trial_history.add(report)


        @task.on_result
        def launch_another_task(*_: Any) -> None:
            """When we get a report, evaluate another trial."""
            if scheduler.running():
                if len(trial_history) < self.max_evals:
                    trial = optimizer.ask()
                    # task.submit(trial, _pipeline=self.pipeline, data_bucket=data_bucket)
                    task.submit(trial, _pipeline=self.pipeline)


        @task.on_exception
        def stop_scheduler_on_exception(*_: Any) -> None:
            scheduler.stop()


        @task.on_cancelled
        def stop_scheduler_on_cancelled(_: Any) -> None:
            scheduler.stop()

        scheduler.run(timeout=self.max_time_seconds, wait=False)

        print("Trial history:")
        history_df = trial_history.df()
        print(history_df)

        self.history_ = trial_history

        self.fitted_pipeline_ = self.pipeline.configure(trial_history.best().config).build("sklearn")
        self.fitted_pipeline_.fit(X,y)

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
        if self.label_encoder_:
            return self.label_encoder_.classes_
        else:
            return self.fitted_pipeline_.classes_


    @property
    def _estimator_type(self):
        return self.fitted_pipeline_._estimator_type
    
#TODO: optionally use data bucket to store and load data.
def target_function(
    trial: Trial,
    _pipeline: Node,
    # data_bucket: PathBucket,
    scorers: list[str],
    other_objective_functions: list[callable],
    objective_names: list[str],
    X,
    y,
    cv,
) -> Trial.Report:
    trial.store({"config.json": trial.config})
    # Load in data
    # with trial.profile("data-loading"):
    #     X, y = (
    #         data_bucket["X.csv"].load(),
    #         data_bucket["y.csv"].load(),
    #     )

    # Configure the pipeline with the trial config before building it.
    sklearn_pipeline = _pipeline.configure(trial.config).build("sklearn")

    # Fit the pipeline, indicating when you want to start the trial timing
    try:
        with trial.profile("scoring"):
            #scores = cross_val_score_objective(sklearn_pipeline, X, y, scorers, cv)
            scores = objective_function_generator(sklearn_pipeline, X, y, scorers, cv, other_objective_functions)
    except Exception as e:
        return trial.fail(e)

    for key, value in zip(objective_names, scores):
        trial.summary[key] = value

    # Save all of this to the file system
    trial.store(
        {
            "model.pkl": sklearn_pipeline,
        },
    )

    # Finally report the success
    return trial.success(**{key: value for key, value in zip(objective_names, scores)})




def cross_val_score_objective(pipeline, X, y, scorers, cv, fold=None):
    #check if scores is not iterable
    if not isinstance(scorers, Iterable): 
        scorers = [scorers]
    scores = []
    if fold is None:
        for train_index, test_index in cv.split(X, y):
            this_fold_pipeline = sklearn.base.clone(pipeline)
            if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            else:
                X_train, X_test = X[train_index], X[test_index]

            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            else:
                y_train, y_test = y[train_index], y[test_index]


            start = time.time()
            this_fold_pipeline.fit(X_train,y_train)
            duration = time.time() - start

            this_fold_scores = [sklearn.metrics.get_scorer(scorer)(this_fold_pipeline, X_test, y_test) for scorer in scorers] 
            scores.append(this_fold_scores)
            del this_fold_pipeline
            del X_train
            del X_test
            del y_train
            del y_test
            

        return np.mean(scores,0)
    else:
        this_fold_pipeline = sklearn.base.clone(pipeline)
        train_index, test_index = list(cv.split(X, y))[fold]
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        else:
            X_train, X_test = X[train_index], X[test_index]

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        else:
            y_train, y_test = y[train_index], y[test_index]

        start = time.time()
        this_fold_pipeline.fit(X_train,y_train)
        duration = time.time() - start
        this_fold_scores = [sklearn.metrics.get_scorer(scorer)(this_fold_pipeline, X_test, y_test) for scorer in scorers] 
        return this_fold_scores

def objective_function_generator(pipeline, x,y, scorers, cv, other_objective_functions):
    #pipeline = pipeline.export_pipeline(**pipeline_kwargs)

    if len(scorers) > 0:
        cv_obj_scores = cross_val_score_objective(sklearn.base.clone(pipeline),x,y,scorers=scorers, cv=cv)
    else:
        cv_obj_scores = []

    if other_objective_functions is not None and len(other_objective_functions) >0:
        other_scores = [obj(sklearn.base.clone(pipeline)) for obj in other_objective_functions]
        #flatten
        other_scores = np.array(other_scores).flatten().tolist()
    else:
        other_scores = []

    return np.concatenate([cv_obj_scores,other_scores])
