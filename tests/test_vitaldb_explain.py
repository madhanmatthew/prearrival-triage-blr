import numpy as np
import pytest

from backend.ml import vitaldb_explain as ve
from backend.ml import vitaldb_model as vm


def toy(n, seed=0):
    """Synthetic windows: falling SBP -> positive (shape test data only, not a result)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int64)
    X = np.empty((n, 5, 5), np.float32)
    for i in range(n):
        X[i, :, 0] = 80 + rng.normal(0, 3, 5)
        X[i, :, 1] = 14 + rng.normal(0, 1, 5)
        X[i, :, 2] = 98 + rng.normal(0, 0.5, 5)
        X[i, :, 3] = 115 + (-6 if y[i] else 0) * np.arange(5) + rng.normal(0, 2, 5)
        X[i, :, 4] = 36.5
    return X, np.ones((n, 5), bool), y, np.arange(n) // 2


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    X, mask, y, groups = toy(600)
    path = tmp_path_factory.mktemp("d") / "w.npz"
    np.savez_compressed(path, X=X, mask=mask, y=y, groups=groups)
    return vm.train_hgb(path, quick=True), X, mask


def test_describe():
    assert ve.describe("change_sbp", -12.4) == "SBP fell 12 mmHg over 5 min"
    assert ve.describe("change_hr", 0.2) == "HR stable over 5 min"
    assert ve.describe("last_spo2", 91.0) == "SpO2 latest 91 %"
    assert ve.describe("min_temp", 35.25) == "Temp 5-min lowest 35.2 °C"
    assert ve.describe("max_rr", float("nan")) == "RR not measured"


def test_shap_shape_and_global_summary(bundle):
    b, X, mask = bundle
    sv = ve.shap_values(b, vm.tabular_features(X[:50], mask[:50]))
    assert sv.shape == (50, 25) and np.isfinite(sv).all()
    feat, by_vital, _, _ = ve.global_summary(b, X, mask, n_sample=200)
    assert len(feat) == 25 and set(by_vital.vital) == set(ve.LABEL)
    assert feat.iloc[0].vital == "sbp"                                   # toy signal is falling SBP


def test_reasons_point_at_sbp_and_skip_temp(bundle):
    b, X, mask = bundle
    falling = np.array([[[80, 14, 98, 115 - 6 * t, 36.5] for t in range(5)]], np.float32)
    r = ve.reasons(b, falling, np.ones((1, 5), bool))
    assert 1 <= len(r) <= 3 and any(x.startswith("SBP") for x in r)
    assert not any(x.startswith("Temp") for x in r)
    assert all(x.endswith(("(raises risk)", "(lowers risk)")) for x in r)
