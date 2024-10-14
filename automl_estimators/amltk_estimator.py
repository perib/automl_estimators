from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
import time
import sklearn.metrics
from collections.abc import Iterable
import pandas as pd
import sklearn
import numpy as np
from amltk.optimization.optimizers.optuna import OptunaOptimizer


from typing import Any

import numpy as np
from amltk.optimization import History, Metric, Trial
from amltk.optimization.optimizers.smac import SMACOptimizer
from amltk.scheduling import Scheduler
from amltk.optimization import Metric
from amltk.scheduling import Scheduler
from amltk.optimization.optimizers.smac import SMACOptimizer

from amltk.pipeline import Component, Node, Sequential, Split
from functools import partial
import pandas as pd



def recursive_set_n_jobs(est, n_jobs):
    if isinstance(est, sklearn.pipeline.Pipeline):
        [recursive_set_n_jobs(estimator, n_jobs=n_jobs) for _,estimator in est.steps]
        return est
    if isinstance(est, sklearn.pipeline.FeatureUnion):
        [recursive_set_n_jobs(estimator, n_jobs=n_jobs) for _,estimator in est.transformer_list]
        return est

    cur_params = est.get_params()
    if 'n_jobs' in cur_params:
        est.set_params(n_jobs=n_jobs)
    
    return est

class AMLTKEstimator(BaseEstimator):

    def __init__(self, 
                 pipeline, 
                 cv, 
                 scorers, 
                 other_objective_functions=None, 
                 max_time_mins=10, 
                 n_jobs=1, 
                 estimator_n_jobs_override=None,
                 seed=None, 
                 max_evals=None, 
                 backend="SMAC"):
        """
        Wraps the AMLTK optimization framework to create a scikit-learn compatible estimator.

        Parameters
        ----------
        pipeline: amltk.pipeline.Node
            The pipeline search space to optimize.
        cv: int or cross-validation generator
            The cross-validation strategy to use.
        scorers: list[str]
            The scoring functions to use. Can be any of the scoring functions in sklearn.metrics or a custom function with signature (estimator, X, y) -> float. 
            These take in a fitted estimator, X, and y and return a float. The scorers are evaluated with cross-validation as set by the CV parameter.
        other_objective_functions: list[callable]
            Other objective functions to use. Each callable should take in a unfitted estimator and return a float. These are not evaluated with cross-validation.
        max_time_mins: int
            The maximum time in seconds to run the optimization for.
        n_jobs: int
            The number of jobs to run in parallel. Used to evaluate pipelines in parallel with the AMLTK framework.
        estimator_n_jobs_override: int
            Sets the n_jobs parameter for all estimators in the pipeline. If None, the n_jobs parameter is not modified.
            Whereas n_jobs evaluates multiple pipelines in parallel, estimator_n_jobs_override parallelizes the fit of an individual pipeline.
        seed: int
            The seed to use for reproducibility.
        max_evals: int
            The maximum number of pipeline evaluations to run. If None, the optimization runs until the max_time_mins is reached.
        backend: str
            The optimization backend to use. Can be "SMAC" or "Optuna".
            
        """
        self.pipeline = pipeline
        self.max_time_mins = max_time_mins
        self.n_jobs = n_jobs
        self.estimator_n_jobs_override = estimator_n_jobs_override
        self.scorers = scorers
        self.other_objective_functions = other_objective_functions
        self.cv = cv
        self.seed = seed
        self.max_evals = max_evals
        self.backend = backend

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
        
        if self.backend.lower() == "smac":
            optimizer_class = SMACOptimizer
        elif self.backend.lower() == "optuna":
            optimizer_class = OptunaOptimizer
        else:
            raise ValueError(f"Backend {self.backend} not supported. Must be 'SMAC' or 'Optuna'")

        
        optimizer = optimizer_class.create(
            space=self.pipeline,  # <!> (1)!
            metrics=metrics,
            # bucket=bucket,
            seed=self.seed,
        )

        #import partial
        # target_function_p = partial(target_function,X=X, y=y, scorers=self.scorers, cv=self.cv, other_objective_functions=self.other_objective_functions, objective_names=self.objective_names)
        target_function_p = partial(target_function,X=X, y=y, scorers=self.scorers, cv=self.cv, other_objective_functions=self.other_objective_functions, objective_names=self.objective_names, estimator_n_jobs_override=self.estimator_n_jobs_override)
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

        scheduler.run(timeout=self.max_time_mins*60, wait=False)

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
    estimator_n_jobs_override=None,
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

    if estimator_n_jobs_override is not None:
        sklearn_pipeline = recursive_set_n_jobs(sklearn_pipeline, estimator_n_jobs_override)

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
