# ====== Base class imports ======

import importlib
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import pandas as pd

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import modeleval

_METRIC_ALIASES = {
    "accuracy": "accuracy",
    "f1": "f1",
    "f1_score": "f1",
    "roc_auc": "roc_auc",
    "auc": "roc_auc",
    "log_loss": "log_loss",
}

_AUTOGLUON_METRIC_ALIASES = {
    "accuracy": "accuracy",
    "f1": "f1_weighted",
    "f1_score": "f1_weighted",
    "roc_auc": "roc_auc",
    "auc": "roc_auc",
    "log_loss": "log_loss",
}


def _load_flaml_automl():
    try:
        flaml = importlib.import_module("flaml")
    except ImportError as exc:
        raise ImportError(
            "AutoML support requires the optional FLAML dependency. "
            "Install it with: pip install 'qbiocode[automl-flaml]' "
            "or install all AutoML backends with: "
            "pip install 'qbiocode[automl]'"
        ) from exc
    return flaml.AutoML


def _load_autogluon_predictor():
    try:
        tabular = importlib.import_module("autogluon.tabular")
    except ImportError as exc:
        raise ImportError(
            "AutoML support with AutoGluon requires the optional "
            "AutoGluon dependency. "
            "Install it with: pip install 'qbiocode[automl-autogluon]'"
        ) from exc
    return tabular.TabularPredictor


def _prediction_scores(automl, X_test):
    if not hasattr(automl, "predict_proba"):
        return None

    try:
        y_proba = automl.predict_proba(X_test)
    except Exception:
        return None

    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        return y_proba[:, 1]
    return y_proba


def _compute_flaml(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose,
    model,
    task,
    metric,
    time_budget,
    estimator_list,
    eval_method,
    n_splits,
    n_jobs,
    seed,
    log_file_name,
    automl_settings,
):
    AutoML = _load_flaml_automl()
    flaml_metric = _METRIC_ALIASES.get(metric, metric)

    settings = {
        "time_budget": time_budget,
        "metric": flaml_metric,
        "task": task,
        "eval_method": eval_method,
        "n_splits": n_splits,
        "n_jobs": n_jobs,
        "log_file_name": log_file_name,
        **automl_settings,
    }
    if estimator_list is not None:
        settings["estimator_list"] = estimator_list
    if seed is not None:
        settings["seed"] = seed

    beg_time = time.time()
    automl = AutoML()
    automl.fit(X_train=X_train, y_train=y_train, **settings)

    y_predicted = automl.predict(X_test)
    y_score = _prediction_scores(automl, X_test)

    model_params = {
        "backend": "flaml",
        "best_estimator": getattr(automl, "best_estimator", None),
        "best_config": getattr(automl, "best_config", None),
        "best_loss": getattr(automl, "best_loss", None),
        "selected_estimator": getattr(
            getattr(automl, "model", None), "estimator", None
        ),  # noqa: E501
        "time_budget": time_budget,
        "metric": metric,
        "flaml_metric": flaml_metric,
        "task": task,
        "estimator_list": estimator_list,
    }

    return modeleval(
        y_test,
        y_predicted,
        beg_time,
        model_params,
        args,
        model=model,
        verbose=verbose,
        y_score=y_score,
    )


def _compute_autogluon(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose,
    model,
    task,
    metric,
    time_budget,
    seed,
    automl_settings,
):
    TabularPredictor = _load_autogluon_predictor()
    autogluon_metric = _AUTOGLUON_METRIC_ALIASES.get(metric, metric)
    label = automl_settings.pop("label", "__qbiocode_label__")
    presets = automl_settings.pop("presets", "medium_quality")
    save_artifacts = automl_settings.pop("save_artifacts", False)
    artifact_dir = automl_settings.pop("artifact_dir", None)
    verbosity = automl_settings.pop("verbosity", 0 if not verbose else 2)
    num_cpus = automl_settings.pop("num_cpus", "auto")
    num_gpus = automl_settings.pop("num_gpus", 0)
    fit_weighted_ensemble = automl_settings.pop("fit_weighted_ensemble", True)
    problem_type = automl_settings.pop("problem_type", None)
    if problem_type is None and task in {"classification", "binary", "multiclass"}:  # noqa: E501
        problem_type = None
    elif problem_type is None:
        problem_type = task

    cleanup_dir = None
    predictor_path = artifact_dir
    if predictor_path is None:
        cleanup_dir = tempfile.mkdtemp(prefix="qbiocode-autogluon-")
        predictor_path = cleanup_dir

    train_data = pd.DataFrame(X_train).copy()
    train_data[label] = list(y_train)

    predictor_kwargs = {
        "label": label,
        "eval_metric": autogluon_metric,
        "path": predictor_path,
        "verbosity": verbosity,
    }
    if problem_type is not None:
        predictor_kwargs["problem_type"] = problem_type

    fit_kwargs = {
        "train_data": train_data,
        "time_limit": time_budget,
        "presets": presets,
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
        "fit_weighted_ensemble": fit_weighted_ensemble,
        **automl_settings,
    }
    if seed is not None:
        fit_kwargs["ag_args_fit"] = {"random_state": seed}

    beg_time = time.time()
    try:
        predictor = TabularPredictor(**predictor_kwargs).fit(**fit_kwargs)
        y_predicted = predictor.predict(X_test)
        y_score = _prediction_scores(predictor, X_test)

        leaderboard = None
        best_model = getattr(predictor, "model_best", None)
        if hasattr(predictor, "leaderboard"):
            try:
                leaderboard = predictor.leaderboard(silent=True)
                if best_model is None and "model" in leaderboard:
                    best_model = leaderboard.iloc[0]["model"]
            except Exception:
                leaderboard = None

        model_params = {
            "backend": "autogluon",
            "best_estimator": best_model,
            "best_config": None,
            "best_loss": None,
            "selected_estimator": best_model,
            "time_budget": time_budget,
            "metric": metric,
            "autogluon_metric": autogluon_metric,
            "task": task,
            "presets": presets,
            "path": predictor_path if save_artifacts else None,
            "leaderboard": (
                leaderboard.to_dict(orient="records")
                if leaderboard is not None
                else None  # noqa: E501
            ),
        }

        return modeleval(
            y_test,
            y_predicted,
            beg_time,
            model_params,
            args,
            model=model,
            verbose=verbose,
            y_score=y_score,
        )
    finally:
        if cleanup_dir is not None and not save_artifacts:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def compute_automl(
    X_train,
    X_test,
    y_train,
    y_test,
    args,
    verbose=False,
    model="automl",
    data_key="",
    backend="flaml",
    task="classification",
    metric="f1_score",
    time_budget=300,
    estimator_list=None,
    eval_method="cv",
    n_splits=5,
    n_jobs=-1,
    seed=None,
    log_file_name="",
    **automl_settings: Any,
):
    """
    Train an AutoML classifier and return QProfiler-compatible output.

    Supported backends are FLAML and AutoGluon. Both are optional dependencies
    so standard QBioCode installs do not need to install an AutoML stack.
    """
    if backend == "flaml":
        return _compute_flaml(
            X_train,
            X_test,
            y_train,
            y_test,
            args,
            verbose,
            model,
            task,
            metric,
            time_budget,
            estimator_list,
            eval_method,
            n_splits,
            n_jobs,
            seed,
            log_file_name,
            automl_settings,
        )
    if backend == "autogluon":
        return _compute_autogluon(
            X_train,
            X_test,
            y_train,
            y_test,
            args,
            verbose,
            model,
            task,
            metric,
            time_budget,
            seed,
            automl_settings,
        )
    raise ValueError("AutoML backend must be one of: 'flaml', 'autogluon'.")
