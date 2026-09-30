"""Tests for backend/ml/fusion_model.py (synthetic windows, no MIMIC data)."""
import numpy as np
import pytest

from backend.ml.fusion_model import (
    FusionNet, N_TEXT, evaluate, fit_model, load, load_generator, make_text_features, predict,
    risk_score, save,
)
from backend.ml.seq_model import Preprocessor
from tests.test_baselines import synthetic_hourly
from tests.test_seq_model import windows

GEN = load_generator()


def test_text_features_shape_and_ranges():
    y = np.arange(400) % 4
    T = make_text_features(y, GEN)
    assert T.shape == (400, N_TEXT)
    assert np.allclose(T[:, 15:19].sum(1), 1, atol=1e-5)
    assert (T[:, 10:15].sum(1) == 1).all()
    assert set(np.unique(T[:, [1, 3, 5, 7]])) <= {0.0, 1.0}
    unknown = T[:, 1] == 0
    assert (T[unknown, 0] == 0).all()          # unknown answer -> value 0


def test_text_is_deterministic():
    y = np.arange(200) % 4
    assert np.array_equal(make_text_features(y, GEN, seed=1), make_text_features(y, GEN, seed=1))


def test_unconscious_implies_cannot_speak():
    T = make_text_features(np.full(2000, 3), GEN)
    unresp = (T[:, 1] == 1) & (T[:, 0] == 0)
    assert unresp.any()
    assert (T[unresp, 6] == 0).all() and (T[unresp, 7] == 1).all()


def test_vitals_only_equals_bigru():
    """Same net + seed: the ablation's vitals_only row must reproduce seq_model's BiGRU."""
    from backend.ml import seq_model
    from backend.ml.fusion_model import _predict_tensors
    X, mask, y_sev, y_trend = windows(80)
    Z = Preprocessor().fit(X, mask).transform(X, mask)
    T = np.zeros((80, N_TEXT), np.float32)
    a, _, _ = seq_model.fit_model(Z, mask, y_sev, y_trend, 0.5, epochs=3)
    b, _, _ = fit_model(Z, mask, T, y_sev, y_trend, 0.5, "vitals_only", epochs=3)
    assert np.allclose(seq_model._predict_tensors(a, Z, mask)[0], _predict_tensors(b, Z, mask, T)[0])


def test_gain_table_is_paired_per_fold():
    from backend.ml.fusion_model import gain_table
    mk = lambda v, a, f: {"variant": v, "agreement": a, "fold_f1": f, "severity": (np.mean(f), 0, len(f))}
    g = gain_table([mk("vitals_only", None, [0.3, 0.5]), mk("text_only", 0.65, [0.2, 0.2]),
                    mk("fused", 0.65, [0.4, 0.5])])
    assert np.isclose(g.fusion_gain_mean[0], 0.05) and np.isclose(g.fusion_gain_std[0], 0.05)
    assert g.folds_fused_better[0] == 1


def test_agreement_controls_informativeness():
    y = np.arange(2000) % 4
    acc = lambda a: (make_text_features(y, GEN, a)[:, 15:19].argmax(1) == y).mean()
    assert acc(0.9) > acc(0.5) + 0.15


def test_risk_score_bounds():
    p = np.array([[1, 0, 0, 0], [0, 0, 0, 1.0], [0.25] * 4])
    assert np.allclose(risk_score(p), [0, 1, 0.5])


def test_forward_shapes_and_modality_masks():
    X, mask, y_sev, y_trend = windows()
    Z = Preprocessor().fit(X, mask).transform(X, mask)
    import torch
    T = torch.rand(40, N_TEXT)
    net = FusionNet("fused").eval()
    lg, tl = net(torch.from_numpy(Z), torch.from_numpy(mask), T)
    assert lg.shape == (40, 4) and tl.shape == (40,)
    zero = torch.zeros(40)
    a, _ = net(torch.from_numpy(Z), torch.from_numpy(mask), T, None, zero)
    b, _ = net(torch.from_numpy(Z), torch.from_numpy(mask), T * 0 + 5, None, zero)
    assert torch.allclose(a, b)                # masked text branch ignores its input


def test_text_only_variant_learns_from_text():
    X, mask, y_sev, y_trend = windows(200)
    Z = Preprocessor().fit(X, mask).transform(X, mask)
    T = make_text_features(y_sev, GEN, 0.9)
    model, _, _ = fit_model(Z, mask, T, y_sev, y_trend, 0.5, "text_only", epochs=30)
    from backend.ml.fusion_model import _predict_tensors
    p, _ = _predict_tensors(model, Z, mask, T)
    assert (p.argmax(1) == y_sev).mean() > 0.5


def test_evaluate_quick_and_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "hourly.csv.gz"
    synthetic_hourly().to_csv(path, index=False)
    results, confs, gain = evaluate(path, quick=True, agreements=[0.65])
    assert {r["variant"] for r in results} == {"vitals_only", "text_only", "fused"}
    assert list(gain.columns)[:1] == ["agreement"] and len(gain) == 1
    assert confs["fused_agree0.65"].to_numpy().sum() == 30 * 12

    from backend.ml.fusion_model import train
    import backend.ml.fusion_model as fm
    bundle = train(path, quick=True)
    monkeypatch.setattr(fm, "MODEL_DIR", tmp_path)
    p = save(bundle)
    b2 = load(p)
    X, mask, *_ = windows()
    T = make_text_features(np.arange(40) % 4, GEN)
    p1, t1 = predict(bundle, X, mask, T)
    p2, t2 = predict(b2, X, mask, T)
    assert np.allclose(p1, p2, atol=1e-5) and p1.shape == (40, 4)
    pn, _ = predict(b2, X, mask, None)          # vitals-only inference path
    assert pn.shape == (40, 4)
