"""Tests for backend/ml/seq_model.py (synthetic windows, no MIMIC data)."""
import numpy as np
import pytest
import torch

from backend.ml.seq_model import (
    BiGRUTwoHead, Preprocessor, evaluate, fit_model, load, predict, save,
)
from tests.test_baselines import synthetic_hourly


def windows(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(80, 10, (n, 6, 7)).astype(np.float32)
    mask = np.ones((n, 6), dtype=bool)
    mask[: n // 2, :3] = False
    X[~mask] = np.nan
    X[0, 5, 1] = np.nan  # a missing value on a real step
    y_sev = np.arange(n) % 4
    y_trend = np.where(np.arange(n) % 5 == 0, 1.0, 0.0)
    y_trend[1::7] = np.nan
    return X, mask, y_sev, y_trend


def test_preprocessor_fills_and_standardises():
    X, mask, *_ = windows()
    pp = Preprocessor().fit(X, mask)
    Z = pp.transform(X, mask)
    assert Z.shape == (40, 6, 12) and not np.isnan(Z).any()
    assert (Z[~mask] == 0).all()
    assert np.allclose(Z[mask][:, :5].mean(axis=0), 0, atol=1e-4)


def test_preprocessor_uses_training_statistics_only():
    X, mask, *_ = windows()
    pp = Preprocessor().fit(X[:20], mask[:20])
    shifted = pp.transform(X[20:] + 1000, mask[20:])
    assert shifted[mask[20:]][:, 0].mean() > 10  # not re-centred on the test data


def test_forward_shapes_and_last_valid_step():
    model = BiGRUTwoHead().eval()
    x = torch.randn(3, 6, 12)
    mask = torch.ones(3, 6, dtype=torch.bool)
    mask[0, :4] = False
    logits, trend = model(x, mask)
    assert logits.shape == (3, 4) and trend.shape == (3,)
    assert torch.isfinite(logits).all() and torch.isfinite(trend).all()


def test_training_is_deterministic():
    X, mask, y_sev, y_trend = windows()
    pp = Preprocessor().fit(X, mask)
    Z = pp.transform(X, mask)
    m1, *_ = fit_model(Z, mask, y_sev, y_trend, 0.5, epochs=2)
    m2, *_ = fit_model(Z, mask, y_sev, y_trend, 0.5, epochs=2)
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


def test_early_stopping_returns_best_epoch():
    X, mask, y_sev, y_trend = windows()
    pp = Preprocessor().fit(X, mask)
    Z = pp.transform(X, mask)
    _, ep, f1 = fit_model(Z, mask, y_sev, y_trend, 0.5, epochs=15, val=(Z, mask, y_sev))
    assert 1 <= ep <= 15 and 0 <= f1 <= 1


def test_save_load_predict_roundtrip(tmp_path, monkeypatch):
    import backend.ml.seq_model as sm
    monkeypatch.setattr(sm, "MODEL_DIR", tmp_path)
    X, mask, y_sev, y_trend = windows()
    pp = Preprocessor().fit(X, mask)
    model, *_ = fit_model(pp.transform(X, mask), mask, y_sev, y_trend, 0.5, epochs=1)
    bundle = {"model": model, "preprocessor": pp, "lambda": 0.5, "epochs": 1}
    p1 = predict(bundle, X, mask)
    p2 = predict(load(save(bundle)), X, mask)
    assert np.allclose(p1[0], p2[0]) and np.allclose(p1[1], p2[1])
    assert np.allclose(p1[0].sum(axis=1), 1, atol=1e-5)


def test_evaluate_quick_on_synthetic(tmp_path):
    path = tmp_path / "hourly.csv.gz"
    synthetic_hourly().to_csv(path, index=False)
    summary, conf = evaluate(path, quick=True)
    assert summary["severity"][2] == 5
    assert conf.to_numpy().sum() == 30 * 12
    assert 0 <= summary["severity"][0] <= 1
