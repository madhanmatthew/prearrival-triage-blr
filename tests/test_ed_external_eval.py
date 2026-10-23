"""Tests for backend/data/mimic_ed_windows.py + backend/ml/ed_external_eval.py (synthetic ED tables)."""
import numpy as np
import pandas as pd
import pytest
import torch

from backend.data.mimic_ed_windows import (
    ed_hourly, ed_outcomes, first_hours_windows, load_ed_tables, temp_to_celsius,
)
from backend.ml import ed_external_eval as ev
from backend.ml.seq_model import BiGRUTwoHead, Preprocessor

T0 = pd.Timestamp("2150-01-01 10:00")
DISPOSITIONS = ["HOME", "ADMITTED", "EXPIRED", "TRANSFER", "HOME", "ADMITTED"]


def ed_tables(n=12, seed=0):
    rng = np.random.default_rng(seed)
    stays = pd.DataFrame({
        "subject_id": np.arange(n) // 2 + 100,           # 2 stays per patient
        "hadm_id": np.nan, "stay_id": np.arange(n) + 1,
        "intime": [T0 + pd.Timedelta(days=i) for i in range(n)],
        "outtime": [T0 + pd.Timedelta(days=i, hours=8) for i in range(n)],
        "disposition": [DISPOSITIONS[i % 6] for i in range(n)],
    })
    triage = pd.DataFrame({
        "subject_id": stays.subject_id, "stay_id": stays.stay_id,
        "temperature": rng.normal(98.6, 1, n), "heartrate": rng.normal(90, 15, n),
        "resprate": rng.normal(18, 3, n), "o2sat": rng.normal(96, 2, n),
        "sbp": rng.normal(125, 20, n), "dbp": 80, "pain": "3",
        "acuity": [1, 2, 3, 4, 3, 2] * (n // 6), "chiefcomplaint": "x",
    })
    rows = []
    for s, t in zip(stays.stay_id, stays.intime):
        for h in (0.5, 2.2, 3.1, 7.5):
            rows.append({"subject_id": 0, "stay_id": s, "charttime": t + pd.Timedelta(hours=h),
                         "temperature": 98.0, "heartrate": 80 + h, "resprate": 16, "o2sat": 97,
                         "sbp": 120, "dbp": 70, "rhythm": None, "pain": "2"})
    return stays, triage, pd.DataFrame(rows)


def test_temp_to_celsius_handles_both_units_and_junk():
    out = temp_to_celsius(pd.Series([98.6, 37.0, 200.0, "abc", np.nan]))
    assert out[0] == pytest.approx(37.0) and out[1] == 37.0
    assert out[2:].isna().all()


def test_ed_hourly_bins_triage_at_bin0_and_ffills_two_bins():
    stays, triage, vit = ed_tables()
    h = ed_hourly(stays, triage, vit)
    s1 = h[h.stay_id == 1].set_index("hour")
    assert len(s1) == 8                                       # ceil(8 h)
    assert s1.loc[0, "hr"] == pytest.approx(np.median([triage.heartrate[0], 80.5]))
    assert s1.loc[1, "hr"] == s1.loc[0, "hr"]                 # forward-filled
    assert s1.loc[5, "hr"] == s1.loc[3, "hr"] and np.isnan(s1.loc[6, "hr"])  # limit 2
    assert s1.has_reading.loc[[0, 2, 3, 7]].all() and not s1.has_reading.loc[6]
    assert (h.avpu_bin == 0).all() and (h.supplemental_o2 == 0).all()
    assert (h.severity_label == -1).all() and h.trend_label.isna().all()
    assert h.news2.notna().any()


def test_ed_hourly_drops_readings_before_intime_and_implausible_values():
    stays, triage, vit = ed_tables()
    vit.loc[0, "charttime"] = stays.intime[0] - pd.Timedelta(hours=1)  # 0.5 h reading -> before intime
    vit.loc[1, "heartrate"] = 999                                     # 2.2 h reading -> implausible
    h = ed_hourly(stays, triage, vit).set_index(["stay_id", "hour"])
    assert h.loc[(1, 0), "hr"] == pytest.approx(triage.heartrate[0])  # triage only
    assert h.loc[(1, 2), "hr"] == pytest.approx(triage.heartrate[0])  # 999 dropped, ffilled
    assert h.loc[(1, 2), "rr"] == 16                                  # other vitals of that row kept


def test_ed_outcomes_labels():
    stays, triage, _ = ed_tables()
    icu = pd.DataFrame({"subject_id": [100, 101],
                        "intime": [stays.outtime[0] + pd.Timedelta(hours=5),     # stay 1: within 24 h
                                   stays.outtime[2] + pd.Timedelta(hours=30)]})  # stay 3: too late,
    o = ed_outcomes(stays, triage, icu).set_index("stay_id")                    # but stay 4 (next day): in
    assert o.admitted.iloc[[0, 1, 2, 4, 5]].tolist() == [0, 1, 1, 0, 1] and np.isnan(o.admitted.iloc[3])
    assert o.high_acuity.iloc[:6].tolist() == [1, 1, 0, 0, 0, 1]
    assert o.icu_24h[o.icu_24h == 1].index.tolist() == [1, 4]


def test_ed_outcomes_icu_unavailable_when_demos_share_no_patients():
    stays, triage, _ = ed_tables()
    icu = pd.DataFrame({"subject_id": [999], "intime": [T0]})
    assert ed_outcomes(stays, triage, icu).icu_24h.isna().all()
    assert ed_outcomes(stays, triage, None).icu_24h.isna().all()


def test_first_hours_windows_ends_at_last_reading_before_6h():
    stays, triage, vit = ed_tables()
    X, mask, ids, n2 = first_hours_windows(ed_hourly(stays, triage, vit))
    assert X.shape == (12, 6, 7) and list(ids) == list(range(1, 13))
    # bins 0-5 all have a value (readings at 0, 2, 3 + ffill), window ends at bin 5
    assert mask[0].all()
    assert X[0, -1, 0] == pytest.approx(80 + 3.1)            # bin 5 = ffilled 3.1 h reading
    assert not np.isnan(n2).all()


def test_grouped_bootstrap_auroc_perfect_and_ci():
    y = np.array([0, 0, 1, 1] * 5)
    groups = np.repeat(np.arange(10), 2)
    auc, lo, hi = ev.grouped_bootstrap_auroc(y, y.astype(float), groups, n_boot=50)
    assert auc == 1.0 and lo == 1.0 and hi == 1.0


def test_expected_severity():
    p = np.array([[1, 0, 0, 0], [0, 0, 0.5, 0.5]])
    assert ev.expected_severity(p).tolist() == [0, 2.5]


def _write_tables(tmp_path):
    stays, triage, vit = ed_tables()
    d = tmp_path / "mimic-iv-ed-demo-2.2" / "ed"
    d.mkdir(parents=True)
    stays.to_csv(d / "edstays.csv.gz", index=False)
    triage.to_csv(d / "triage.csv.gz", index=False)
    vit.to_csv(d / "vitalsign.csv.gz", index=False)
    return tmp_path


def test_load_ed_tables_finds_nested_files_and_checks_columns(tmp_path):
    raw = _write_tables(tmp_path)
    tb = load_ed_tables(raw)
    assert set(tb) == {"edstays", "triage", "vitalsign"}
    assert pd.api.types.is_datetime64_any_dtype(tb["vitalsign"].charttime)
    pd.read_csv(raw / "mimic-iv-ed-demo-2.2/ed/triage.csv.gz").drop(columns="acuity").to_csv(
        raw / "mimic-iv-ed-demo-2.2/ed/triage.csv.gz", index=False)
    with pytest.raises(ValueError, match="acuity"):
        load_ed_tables(raw)


def test_evaluate_end_to_end_with_untrained_model(tmp_path):
    raw = _write_tables(tmp_path)
    torch.manual_seed(0)
    X = np.random.default_rng(0).normal(80, 10, (20, 6, 7)).astype(np.float32)
    mask = np.ones((20, 6), dtype=bool)
    pp = Preprocessor().fit(X, mask)
    path = tmp_path / "bigru_test.pt"
    torch.save({"state_dict": BiGRUTwoHead().state_dict(), "preprocessor": pp.to_dict(),
                "lambda": 0.5, "epochs": 1, "data_version": "synthetic"}, path)
    scores, table, used = ev.evaluate(raw, path, icu_dir=None)
    assert used == path and len(scores) == 12
    assert scores.model_risk.between(0, 3).all()
    assert set(table.outcome) == set(ev.OUTCOMES)
    assert table[table.outcome == "icu_24h"].subset.tolist() == ["unavailable"]
    ok = table.dropna(subset=["auroc"])
    assert ok.auroc.between(0, 1).all() and {"model_risk", "news2"} <= set(ok.scorer)
