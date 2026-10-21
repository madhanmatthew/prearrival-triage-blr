import numpy as np
import pytest

from backend.ml import vitaldb_model as vm


def toy(n=240, seed=0):
    """Synthetic windows: falling SBP -> positive. Shape test data only, not a result."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int64)
    X = np.empty((n, 5, 5), np.float32)
    for i in range(n):
        slope = -6 if y[i] else 0
        X[i, :, 0] = 80 + rng.normal(0, 3, 5)
        X[i, :, 1] = 14 + rng.normal(0, 1, 5)
        X[i, :, 2] = 98 + rng.normal(0, 0.5, 5)
        X[i, :, 3] = 115 + slope * np.arange(5) + rng.normal(0, 2, 5)
        X[i, :, 4] = 36.5
    mask = np.ones((n, 5), bool)
    X[0, :2] = np.nan
    mask[0, :2] = False
    groups = np.arange(n) // 2
    return X, mask, y, groups


def test_tabular_features():
    X = np.array([[[np.nan] * 5, [80, 14, 98, 120, 36], [90, 16, 96, 110, 37]]], np.float32)
    M = np.array([[False, True, True]])
    T = vm.tabular_features(X, M)
    assert T.shape == (1, 25) and len(vm.TAB_NAMES) == 25
    assert T[0, :5].tolist() == [90, 16, 96, 110, 37]                       # last
    assert T[0, 5] == 85 and T[0, 10] == 80 and T[0, 15] == 90             # hr mean/min/max
    assert T[0, 20:25] == pytest.approx([10, 2, -2, -10, 1])               # change


def test_news2_partial_matches_band_scores():
    last = np.array([[135, 26, 90, 88, 34.5], [70, 14, 98, 120, 37.0], [np.nan] * 5])
    s = vm.news2_partial(last)
    assert s.tolist() == [3 + 3 + 3 + 3 + 3, 0, 0]


def test_scaler_fills_missing_and_padding_with_zero():
    F = np.array([[[1.0, np.nan], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], np.float32)
    M = np.array([[False, True], [True, True]])
    Z = vm.Scaler().fit(F, M).transform(F, M)
    assert Z[0, 0].tolist() == [0, 0] and np.isfinite(Z).all()


def test_quick_evaluate_and_bigru_learns(tmp_path):
    X, mask, y, groups = toy()
    path = tmp_path / "w.npz"
    np.savez_compressed(path, X=X, mask=mask, y=y, groups=groups)
    folds = vm.evaluate(path, quick=True)
    s = vm.summarize(folds).set_index("model")
    assert set(s.index) == {"majority", "news2", "logreg", "hgb", "bigru"}
    assert (s.n_folds == vm.N_OUTER).all()
    assert s.loc["logreg", "auroc_mean"] > 0.9                              # trivially separable toy
    assert s.loc["majority", "auroc_mean"] == pytest.approx(0.5)


def test_outer_splits_keep_subjects_together():
    _, _, y, groups = toy()
    for tr, te in vm.outer_splits(y, groups):
        assert not set(groups[tr]) & set(groups[te])


def test_train_save_load_predict(tmp_path, monkeypatch):
    X, mask, y, groups = toy()
    path = tmp_path / "w.npz"
    np.savez_compressed(path, X=X, mask=mask, y=y, groups=groups)
    monkeypatch.setattr(vm, "MODEL_DIR", tmp_path)
    b = vm.train(path, epochs=3)
    p1 = vm.predict(b, X, mask)
    b2 = vm.load_model(vm.save(b))
    assert p1.shape == (len(y),) and np.allclose(p1, vm.predict(b2, X, mask), atol=1e-6)
