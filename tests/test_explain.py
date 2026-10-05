"""Tests for XAI (docs/15 D13): SHAP on tree pipelines, Integrated Gradients on the BiGRU."""
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from backend.ml import explain as ex
from backend.ml.baselines import FEATURE_NAMES, _pipe
from backend.ml.seq_model import BiGRUTwoHead

rng = np.random.default_rng(0)


def tree_data(n=300, n_classes=4):
    F = rng.normal(size=(n, 12))
    F[rng.random(F.shape) < 0.05] = np.nan               # imputer must handle NaN
    y = (np.nan_to_num(F[:, 0]) > 0).astype(int) + (np.nan_to_num(F[:, 3]) > 0.5) * (n_classes > 2) * 2
    return F, y


@pytest.mark.parametrize("clf,n_classes", [
    (RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0), 4),
    (GradientBoostingClassifier(n_estimators=20, random_state=0), 4),
    (GradientBoostingClassifier(n_estimators=20, random_state=0), 2),
])
def test_shap_tree_shape_and_additivity(clf, n_classes):
    F, y = tree_data(n_classes=n_classes)
    pipe = _pipe(clf).fit(F, y)
    sv, ev = ex.shap_tree(pipe, F[:50])
    k = 1 if n_classes == 2 and isinstance(clf, GradientBoostingClassifier) else n_classes
    assert sv.shape == (50, 12, k) and ev.shape == (k,)
    if isinstance(clf, RandomForestClassifier):          # RF SHAP is in probability space
        np.testing.assert_allclose(sv.sum(axis=1) + ev, pipe.predict_proba(F[:50]), atol=1e-6)
    else:                                                # GB SHAP is in raw log-odds space
        raw = pipe.decision_function(F[:50]).reshape(50, -1)
        np.testing.assert_allclose(sv.sum(axis=1) + ev, raw, atol=1e-6)
    summ = ex.summarize_shap(sv)
    assert list(summ.columns[:1]) == ["mean_abs_all"] and set(summ.index) == set(FEATURE_NAMES)
    assert (summ.values >= 0).all() and summ["mean_abs_all"].is_monotonic_decreasing
    assert summ.index[0] in ("hr", "sbp")                # the informative features rank first


def ig_inputs(n=16):
    torch.manual_seed(0)
    model = BiGRUTwoHead().eval()
    Z = rng.normal(size=(n, 6, 12)).astype(np.float32)
    mask = np.ones((n, 6), bool)
    mask[:4, :3] = False                                 # left-padded windows
    Z[~mask] = 0.0
    return model, Z, mask


@pytest.mark.parametrize("head", ["severity", "trend"])
def test_ig_completeness_and_padding(head):
    model, Z, mask = ig_inputs()
    attr, delta, tgt = ex.ig_bigru(model, Z, mask, head, n_steps=128, batch=5)
    assert attr.shape == Z.shape and delta.shape == (len(Z),) and tgt.shape == (len(Z),)
    assert np.abs(delta).max() < 5e-2                    # sum(attr) ~= f(x) - f(baseline)
    assert np.all(attr[~mask] == 0)                      # padded steps get no attribution
    f = ex._Head(model, head)
    with torch.no_grad():
        m = torch.from_numpy(mask)
        diff = (f(torch.from_numpy(Z), m) - f(torch.zeros_like(torch.from_numpy(Z)), m)).numpy()
    np.testing.assert_allclose(attr.sum(axis=(1, 2)), diff[np.arange(len(Z)), tgt], atol=5e-2)
    if head == "severity":
        with torch.no_grad():
            assert (tgt == f(torch.from_numpy(Z), m).argmax(1).numpy()).all()


def test_summarize_ig_shapes():
    attr = rng.normal(size=(10, 6, 12))
    per_feat, heat = ex.summarize_ig(attr)
    assert set(per_feat.index) == set(FEATURE_NAMES) and per_feat["mean_abs"].is_monotonic_decreasing
    assert heat.shape == (6, 12) and list(heat.index) == ex.STEP_NAMES and heat.index[-1] == "t0"
    np.testing.assert_allclose(per_feat.loc["hr", "mean_abs"], np.abs(attr[..., 0]).sum(1).mean())


def test_sample_is_seeded_and_bounded():
    a, b = ex._sample(100, 30), ex._sample(100, 30)
    assert (a == b).all() and len(a) == 30 and len(set(a)) == 30
    assert len(ex._sample(10, 30)) == 10


def test_figures_written(tmp_path):
    df = pd.DataFrame({"mean_abs": [0.3, 0.1]}, index=["hr", "rr"])
    ex._bar(df, "mean_abs", "t", tmp_path / "b.png")
    ex._heatmap(pd.DataFrame(np.ones((6, 12)), index=ex.STEP_NAMES, columns=FEATURE_NAMES),
                "t", tmp_path / "h.png")
    assert (tmp_path / "b.png").stat().st_size > 0 and (tmp_path / "h.png").stat().st_size > 0
