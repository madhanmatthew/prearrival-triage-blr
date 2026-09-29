"""Tests for backend/ml/baselines.py on synthetic windows (no MIMIC data)."""
import numpy as np
import pandas as pd
import pytest

from backend.data.mimic_windows import FEATURES
from backend.ml.baselines import (
    FEATURE_NAMES, News2Classifier, evaluate, load, outer_fold_of_patient, predict, save,
    tabular_features,
)


def test_tabular_features_last_step_and_delta():
    X = np.full((1, 6, len(FEATURES)), np.nan, dtype=np.float32)
    X[0, 4, :] = [80, 16, 97, 120, 37.0, 0, 0]
    X[0, 5, :] = [95, 22, 94, 105, 38.5, 1, 1]
    mask = np.array([[False] * 4 + [True, True]])
    f = tabular_features(X, mask)
    assert f.shape == (1, 12)
    assert f[0, :7].tolist() == pytest.approx([95, 22, 94, 105, 38.5, 1, 1])
    assert f[0, 7:].tolist() == pytest.approx([15, 6, -3, -15, 1.5])


def test_news2_classifier_matches_hand_scores():
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    X = np.zeros((2, 12))
    # healthy: 0; docs/07 example (hr112 rr24 spo2 93 sbp98 temp36.4 A air): 8
    for i, (hr, rr, sp, sbp, t) in enumerate([(72, 16, 98, 120, 37.0), (112, 24, 93, 98, 36.4)]):
        X[i, [idx["hr"], idx["rr"], idx["spo2"], idx["sbp"], idx["temp"]]] = [hr, rr, sp, sbp, t]
    clf = News2Classifier("severity").fit(X, np.array([0, 3]))
    assert News2Classifier.totals(X).tolist() == [0, 8]
    assert clf.predict(X).tolist() == [0, 3]
    assert News2Classifier("trend").predict_proba(X)[:, 1].tolist() == pytest.approx([0, 0.4])


def test_outer_folds_disjoint_by_patient():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 10)
    y = rng.integers(0, 4, len(groups))
    fold = outer_fold_of_patient(y, groups)
    assert set(fold) == set(range(40)) and set(fold.values()) == set(range(5))


def synthetic_hourly(n_patients=30, hours=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        sev = p % 4
        for h in range(hours):
            rows.append({
                "stay_id": p, "hour": h, "patient_id": 1000 + p, "has_reading": True,
                "hr": 70 + 15 * sev + rng.normal(0, 5), "rr": 14 + 3 * sev + rng.normal(0, 1),
                "spo2": 98 - sev + rng.normal(0, 0.5), "sbp": 125 - 10 * sev + rng.normal(0, 5),
                "temp": 37 + 0.3 * sev, "avpu_bin": float(sev == 3), "supplemental_o2": float(sev >= 2),
                "severity_label": sev, "news2": np.nan,
                "trend_label": float(sev == 3 and h % 3 == 0) if sev != 2 else np.nan,
                "t": pd.Timestamp("2150-01-01") + pd.Timedelta(hours=h + 1),
            })
    return pd.DataFrame(rows)


def test_evaluate_quick_runs_and_beats_majority(tmp_path):
    path = tmp_path / "hourly.csv.gz"
    synthetic_hourly().to_csv(path, index=False)
    results, conf = evaluate(path, quick=True, models=["majority", "news2", "logreg"])
    sev = results[results.task == "severity"].set_index("model")
    assert sev.loc["logreg", "f1_macro_mean"] > sev.loc["majority", "f1_macro_mean"]
    assert set(results.task) == {"severity", "trend"}
    assert conf.groupby("model").sum(numeric_only=True).to_numpy().sum() == 3 * 30 * 12


def test_save_load_roundtrip(tmp_path, monkeypatch):
    import backend.ml.baselines as b
    monkeypatch.setattr(b, "MODEL_DIR", tmp_path)
    clf = News2Classifier("severity").fit(np.zeros((1, 12)), np.array([0]))
    path = save(clf, "news2", "severity")
    assert path.name.startswith("news2_severity_v1_")
    assert predict(load(path), np.zeros((1, 12))).tolist() == [3]  # all-zero vitals score high
