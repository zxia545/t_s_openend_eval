import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


def _load_glm_winrate_module_with_stubs():
    repo_root = Path(__file__).resolve().parents[3]
    target = (
        repo_root
        / "external_evals"
        / "alpaca_eval"
        / "src"
        / "alpaca_eval"
        / "metrics"
        / "glm_winrate.py"
    )

    fake_constants = types.SimpleNamespace(
        DATASETS_TOKEN=None,
        DATASETS_FORCE_DOWNLOAD=False,
        DEFAULT_CACHE_DIR=None,
    )
    fake_types = types.SimpleNamespace(AnyLoadableDF=object)
    fake_utils = types.SimpleNamespace(convert_to_dataframe=lambda x: x)

    fake_alpaca_pkg = types.ModuleType("alpaca_eval")
    fake_alpaca_pkg.constants = fake_constants
    fake_alpaca_pkg.types = fake_types
    fake_alpaca_pkg.utils = fake_utils

    fake_metrics_pkg = types.ModuleType("alpaca_eval.metrics")
    fake_winrate_mod = types.ModuleType("alpaca_eval.metrics.winrate")
    fake_winrate_mod.get_winrate = lambda annotations: {"win_rate": 50.0}

    sys.modules["alpaca_eval"] = fake_alpaca_pkg
    sys.modules["alpaca_eval.metrics"] = fake_metrics_pkg
    sys.modules["alpaca_eval.metrics.winrate"] = fake_winrate_mod

    if "patsy" not in sys.modules:
        sys.modules["patsy"] = types.SimpleNamespace(
            build_design_matrices=lambda *args, **kwargs: None,
            dmatrix=lambda *args, **kwargs: None,
        )

    spec = importlib.util.spec_from_file_location(
        "alpaca_eval.metrics.glm_winrate", target
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fit_logistic_regressioncv_works_when_fit_rejects_groups(monkeypatch):
    glm_winrate = _load_glm_winrate_module_with_stubs()

    class _Scorer:
        def set_score_request(self, **kwargs):
            return self

    class _FakeLogisticRegressionCV:
        def __init__(self, cv=None, scoring=None, **kwargs):
            self.cv = cv

        def set_fit_request(self, **kwargs):
            return self

        def fit(
            self, X, y, sample_weight=None, true_sample_weight=None, true_prob=None
        ):
            self.fitted = True
            return self

    monkeypatch.setattr(glm_winrate, "make_scorer", lambda *args, **kwargs: _Scorer())
    monkeypatch.setattr(glm_winrate, "LogisticRegressionCV", _FakeLogisticRegressionCV)

    data = pd.DataFrame(
        {
            "feature": [0.1, 0.2, 0.3, 0.4],
            "preference": [0.2, 0.8, 0.3, 0.7],
        }
    )

    model = glm_winrate.fit_LogisticRegressionCV(
        data=data,
        col_y_true="preference",
        is_ytrue_proba=True,
        n_splits=2,
    )

    assert getattr(model, "fitted", False)
