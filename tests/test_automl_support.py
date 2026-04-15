import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest
from conftest import ensure_package, load_module


def load_automl_module():
    ensure_package("qbiocode", "qbiocode")
    ensure_package("qbiocode.utils", "qbiocode/utils")
    ensure_package("qbiocode.evaluation", "qbiocode/evaluation")
    ensure_package("qbiocode.learning", "qbiocode/learning")

    load_module("qbiocode.utils.helper_fn", "qbiocode/utils/helper_fn.py")
    load_module(
        "qbiocode.evaluation.model_evaluation",
        "qbiocode/evaluation/model_evaluation.py",
    )
    return load_module(
        "qbiocode.learning.compute_automl",
        "qbiocode/learning/compute_automl.py",
    )


def load_model_run_module():
    ensure_package("qbiocode", "qbiocode")
    ensure_package("qbiocode.evaluation", "qbiocode/evaluation")
    return load_module(
        "qbiocode.evaluation.model_run",
        "qbiocode/evaluation/model_run.py",
    )


def test_compute_automl_requires_optional_flaml_dependency(monkeypatch):
    compute_automl = load_automl_module().compute_automl

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "flaml":
            raise ImportError("No module named flaml")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(
        ImportError,
        match="pip install 'qbiocode\\[automl-flaml\\]'",
    ):
        compute_automl(
            [[0], [1]],
            [[0], [1]],
            [0, 1],
            [0, 1],
            {"grid_search": False},
            time_budget=1,
        )


def test_compute_automl_requires_optional_autogluon_dependency(monkeypatch):
    compute_automl = load_automl_module().compute_automl

    real_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "autogluon.tabular":
            raise ImportError("No module named autogluon")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(
        ImportError,
        match="pip install 'qbiocode\\[automl-autogluon\\]'",
    ):
        compute_automl(
            [[0], [1]],
            [[0], [1]],
            [0, 1],
            [0, 1],
            {"grid_search": False},
            backend="autogluon",
            time_budget=1,
        )


def test_compute_automl_flaml_backend_returns_qprofiler_result(monkeypatch):
    compute_automl = load_automl_module().compute_automl

    captured = {}

    class FakeAutoML:
        def __init__(self):
            self.best_estimator = "rf"
            self.best_config = {"n_estimators": 4}
            self.best_loss = 0.25
            self.model = types.SimpleNamespace(estimator="fitted-rf")

        def fit(self, X_train, y_train, **settings):
            captured["X_train"] = X_train
            captured["y_train"] = y_train
            captured["settings"] = settings

        def predict(self, X_test):
            captured["X_test"] = X_test
            return np.array([0, 1, 1, 0])

        def predict_proba(self, X_test):
            return np.array(
                [
                    [0.9, 0.1],
                    [0.2, 0.8],
                    [0.3, 0.7],
                    [0.8, 0.2],
                ]
            )

    fake_flaml = types.ModuleType("flaml")
    fake_flaml.AutoML = FakeAutoML
    monkeypatch.setitem(sys.modules, "flaml", fake_flaml)

    result = compute_automl(
        pd.DataFrame({"a": [0, 1, 1, 0], "b": [1, 1, 0, 0]}),
        pd.DataFrame({"a": [0, 1, 1, 0], "b": [1, 1, 0, 0]}),
        pd.Series([0, 1, 1, 0]),
        pd.Series([0, 1, 1, 0]),
        {"grid_search": False},
        model="automl",
        time_budget=7,
        metric="f1_score",
        estimator_list=["rf"],
        eval_method="cv",
        n_splits=3,
        n_jobs=2,
        seed=11,
        verbose=False,
    )

    payload = result["results_automl"].iloc[0]

    assert payload["model"] == "automl"
    assert payload["accuracy"] == 1.0
    assert payload["f1_score"] == 1.0
    assert payload["auc"] == 1.0
    assert payload["Model_Parameters"] == {
        "backend": "flaml",
        "best_estimator": "rf",
        "best_config": {"n_estimators": 4},
        "best_loss": 0.25,
        "selected_estimator": "fitted-rf",
        "time_budget": 7,
        "metric": "f1_score",
        "flaml_metric": "f1",
        "task": "classification",
        "estimator_list": ["rf"],
    }
    assert captured["settings"]["metric"] == "f1"
    assert captured["settings"]["task"] == "classification"
    assert captured["settings"]["time_budget"] == 7


def test_compute_automl_autogluon_backend_result(monkeypatch):
    compute_automl = load_automl_module().compute_automl
    captured = {}

    class FakeTabularPredictor:
        def __init__(self, **kwargs):
            captured["predictor_kwargs"] = kwargs
            self.model_best = "WeightedEnsemble_L2"

        def fit(self, **kwargs):
            captured["fit_kwargs"] = kwargs
            return self

        def predict(self, X_test):
            captured["X_test"] = X_test
            return np.array([0, 1, 1, 0])

        def predict_proba(self, X_test):
            return pd.DataFrame({0: [0.9, 0.2, 0.3, 0.8], 1: [0.1, 0.8, 0.7, 0.2]})  # noqa: E501

        def leaderboard(self, silent=True):
            captured["leaderboard_silent"] = silent
            return pd.DataFrame(
                [
                    {
                        "model": "WeightedEnsemble_L2",
                        "score_val": 0.95,
                        "fit_time": 0.1,
                    }
                ]
            )

    fake_autogluon = types.ModuleType("autogluon")
    fake_tabular = types.ModuleType("autogluon.tabular")
    fake_tabular.TabularPredictor = FakeTabularPredictor
    monkeypatch.setitem(sys.modules, "autogluon", fake_autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", fake_tabular)

    result = compute_automl(
        pd.DataFrame({"a": [0, 1, 1, 0], "b": [1, 1, 0, 0]}),
        pd.DataFrame({"a": [0, 1, 1, 0], "b": [1, 1, 0, 0]}),
        pd.Series([0, 1, 1, 0]),
        pd.Series([0, 1, 1, 0]),
        {"grid_search": False},
        model="automl",
        backend="autogluon",
        time_budget=9,
        metric="f1_score",
        presets="medium_quality",
        num_cpus=1,
        num_gpus=0,
        fit_weighted_ensemble=False,
        verbosity=0,
        verbose=False,
    )

    payload = result["results_automl"].iloc[0]

    assert payload["model"] == "automl"
    assert payload["accuracy"] == 1.0
    assert payload["f1_score"] == 1.0
    assert payload["auc"] == 1.0
    assert payload["Model_Parameters"]["backend"] == "autogluon"
    model_params = payload["Model_Parameters"]
    assert model_params["best_estimator"] == "WeightedEnsemble_L2"
    assert model_params["selected_estimator"] == "WeightedEnsemble_L2"
    assert payload["Model_Parameters"]["autogluon_metric"] == "f1_weighted"
    assert payload["Model_Parameters"]["path"] is None
    assert payload["Model_Parameters"]["leaderboard"] == [
        {"model": "WeightedEnsemble_L2", "score_val": 0.95, "fit_time": 0.1}
    ]
    assert captured["predictor_kwargs"]["label"] == "__qbiocode_label__"
    assert captured["predictor_kwargs"]["eval_metric"] == "f1_weighted"
    assert captured["fit_kwargs"]["time_limit"] == 9
    assert captured["fit_kwargs"]["presets"] == "medium_quality"
    assert captured["fit_kwargs"]["fit_weighted_ensemble"] is False
    assert "__qbiocode_label__" in captured["fit_kwargs"]["train_data"].columns


def test_model_run_dispatches_automl(monkeypatch):
    model_run_module = load_model_run_module()

    def fake_compute_automl(*args, **kwargs):
        return pd.DataFrame(
            {
                "results_automl": [
                    {
                        "model": kwargs["model"],
                        "accuracy": 1.0,
                        "f1_score": 1.0,
                        "time": 0.1,
                        "auc": 1.0,
                        "Model_Parameters": {"backend": "flaml"},
                    }
                ]
            }
        )

    monkeypatch.setattr(
        model_run_module,
        "_load_compute_functions",
        lambda: {"automl": fake_compute_automl},
    )

    result = model_run_module.model_run(
        [[0], [1]],
        [[0], [1]],
        [0, 1],
        [0, 1],
        "dataset",
        {
            "model": ["automl"],
            "grid_search": False,
            "automl_args": {"time_budget": 1},
        },
    )

    assert result["results_automl"][0]["model"] == "automl"
    assert result["results_automl"][0]["Model_Parameters"] == {"backend": "flaml"}  # noqa: E501


def test_model_run_uses_automl_from_qprofiler_args(monkeypatch):
    compute_automl = load_automl_module().compute_automl
    model_run_module = load_model_run_module()
    captured = {}

    class FakeAutoML:
        def __init__(self):
            self.best_estimator = "xgboost"
            self.best_config = {"max_depth": 2}
            self.best_loss = 0.1
            self.model = types.SimpleNamespace(estimator="fitted-xgboost")

        def fit(self, X_train, y_train, **settings):
            captured["settings"] = settings
            captured["X_train_shape"] = X_train.shape
            captured["y_train"] = list(y_train)

        def predict(self, X_test):
            captured["X_test_shape"] = X_test.shape
            return np.array([0, 1, 1, 0])

        def predict_proba(self, X_test):
            return np.array(
                [
                    [0.95, 0.05],
                    [0.05, 0.95],
                    [0.10, 0.90],
                    [0.90, 0.10],
                ]
            )

    fake_flaml = types.ModuleType("flaml")
    fake_flaml.AutoML = FakeAutoML
    monkeypatch.setitem(sys.modules, "flaml", fake_flaml)
    monkeypatch.setattr(
        model_run_module,
        "_load_compute_functions",
        lambda: {"automl": compute_automl},
    )

    args = {
        "model": ["automl"],
        "grid_search": False,
        "automl_args": {
            "backend": "flaml",
            "task": "classification",
            "metric": "f1_score",
            "time_budget": 12,
            "estimator_list": ["xgboost"],
            "eval_method": "cv",
            "n_splits": 2,
            "n_jobs": 1,
            "seed": 123,
            "log_file_name": "",
        },
    }

    result = model_run_module.model_run(
        pd.DataFrame({"feature_a": [0, 1, 1, 0], "feature_b": [1, 1, 0, 0]}),
        pd.DataFrame({"feature_a": [0, 1, 1, 0], "feature_b": [1, 1, 0, 0]}),
        pd.Series([0, 1, 1, 0]),
        pd.Series([0, 1, 1, 0]),
        "dataset",
        args,
    )

    payload = result["results_automl"][0]

    assert payload["model"] == "automl"
    assert payload["accuracy"] == 1.0
    assert payload["f1_score"] == 1.0
    assert payload["auc"] == 1.0
    assert payload["Model_Parameters"]["best_estimator"] == "xgboost"
    assert payload["Model_Parameters"]["selected_estimator"] == ("fitted-xgboost")  # noqa: E501
    assert captured["settings"]["metric"] == "f1"
    assert captured["settings"]["time_budget"] == 12
    assert captured["settings"]["estimator_list"] == ["xgboost"]
    assert captured["X_train_shape"] == (4, 2)
    assert captured["X_test_shape"] == (4, 2)


@pytest.mark.automl
def test_compute_automl_with_real_flaml_smoke():
    pytest.importorskip("flaml")
    compute_automl = load_automl_module().compute_automl

    X_train = pd.DataFrame(
        {
            "feature_a": [0.0, 0.1, 0.9, 1.0, 0.2, 0.8],
            "feature_b": [0.0, 0.2, 0.8, 1.0, 0.1, 0.9],
        }
    )
    y_train = pd.Series([0, 0, 1, 1, 0, 1])
    X_test = pd.DataFrame(
        {
            "feature_a": [0.05, 0.95],
            "feature_b": [0.05, 0.95],
        }
    )
    y_test = pd.Series([0, 1])

    result = compute_automl(
        X_train,
        X_test,
        y_train,
        y_test,
        {"grid_search": False},
        model="automl",
        time_budget=2,
        estimator_list=["rf"],
        eval_method="holdout",
        split_ratio=0.8,
        n_jobs=1,
        seed=42,
        verbose=False,
    )

    payload = result["results_automl"].iloc[0]

    assert payload["model"] == "automl"
    assert 0.0 <= payload["accuracy"] <= 1.0
    assert 0.0 <= payload["f1_score"] <= 1.0
    assert payload["Model_Parameters"]["backend"] == "flaml"
    assert payload["Model_Parameters"]["best_estimator"] is not None


@pytest.mark.automl
def test_compute_automl_with_real_autogluon_smoke():
    pytest.importorskip("autogluon.tabular")
    compute_automl = load_automl_module().compute_automl

    X_train = pd.DataFrame(
        {
            "feature_a": [0.0, 0.1, 0.9, 1.0, 0.2, 0.8],
            "feature_b": [0.0, 0.2, 0.8, 1.0, 0.1, 0.9],
        }
    )
    y_train = pd.Series([0, 0, 1, 1, 0, 1])
    X_test = pd.DataFrame(
        {
            "feature_a": [0.05, 0.95],
            "feature_b": [0.05, 0.95],
        }
    )
    y_test = pd.Series([0, 1])

    result = compute_automl(
        X_train,
        X_test,
        y_train,
        y_test,
        {"grid_search": False},
        model="automl",
        backend="autogluon",
        time_budget=5,
        presets="medium_quality",
        hyperparameters={"RF": {}},
        num_cpus=1,
        num_gpus=0,
        fit_weighted_ensemble=False,
        save_artifacts=False,
        verbosity=0,
        verbose=False,
    )

    payload = result["results_automl"].iloc[0]

    assert payload["model"] == "automl"
    assert 0.0 <= payload["accuracy"] <= 1.0
    assert 0.0 <= payload["f1_score"] <= 1.0
    assert payload["Model_Parameters"]["backend"] == "autogluon"
    assert payload["Model_Parameters"]["best_estimator"] is not None
