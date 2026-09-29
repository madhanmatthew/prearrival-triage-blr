"""Tests for backend/data/mimic_windows.py on tiny hand-made tables (no MIMIC data)."""
import numpy as np
import pandas as pd
import pytest

from backend.data.mimic_windows import (
    FEATURES, ITEMS, add_deltas, add_labels, hourly_features, make_windows, severity_class,
)

T0 = pd.Timestamp("2150-01-01 00:00")
H = pd.Timedelta(hours=1)


def stays(n_hours=10, stay_id=1):
    return pd.DataFrame({"subject_id": [100], "hadm_id": [10], "stay_id": [stay_id],
                         "intime": [T0], "outtime": [T0 + n_hours * H]})


def ev(hour, name, valuenum=None, value=None, item_idx=0, minute=0):
    return {"stay_id": 1, "charttime": T0 + hour * H + pd.Timedelta(minutes=minute),
            "itemid": ITEMS[name][item_idx], "value": value, "valuenum": valuenum}


def chart(*rows):
    return pd.DataFrame(list(rows), columns=["stay_id", "charttime", "itemid", "value", "valuenum"])


def row(df, hour):
    return df[df.hour == hour].iloc[0]


# ---------------------------------------------------------------------------
# Severity rule (docs/08 §2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("death, interv, los, expected", [
    (True, False, 100, 3), (False, True, 5, 3), (False, False, 72.5, 2), (False, False, 72, 1),
    (False, False, 24, 1), (False, False, 23.9, 0),
])
def test_severity_class(death, interv, los, expected):
    assert severity_class(death, interv, los) == expected


# ---------------------------------------------------------------------------
# Hourly features
# ---------------------------------------------------------------------------

def test_median_per_hour_and_grid():
    df = hourly_features(chart(ev(0, "hr", 80), ev(0, "hr", 100, minute=30), ev(0, "hr", 90, minute=50)),
                         stays(4))
    assert len(df) == 4 and list(df.hour) == [0, 1, 2, 3]
    assert row(df, 0).hr == 90


def test_forward_fill_at_most_two_bins():
    df = hourly_features(chart(ev(0, "hr", 80)), stays(5))
    assert list(df.hr[:3]) == [80, 80, 80]
    assert df.hr[3:].isna().all()
    assert list(df.has_reading) == [True, True, True, False, False]


def test_temperature_fahrenheit_converted_and_celsius_preferred():
    df = hourly_features(chart(ev(0, "temp_f", 98.6), ev(3, "temp_f", 100.4), ev(3, "temp_c", 37.5)),
                         stays(4))
    assert row(df, 0).temp == pytest.approx(37.0)
    assert row(df, 3).temp == pytest.approx(37.5)


def test_sbp_prefers_nibp_over_arterial():
    df = hourly_features(chart(ev(0, "sbp_art", 130), ev(0, "sbp_nibp", 120), ev(3, "sbp_art", 110)),
                         stays(4))
    assert row(df, 0).sbp == 120 and row(df, 3).sbp == 110


def test_rr_prefers_primary_item():
    df = hourly_features(chart(ev(0, "rr", 18), ev(0, "rr", 30, item_idx=1), ev(3, "rr", 22, item_idx=1)),
                         stays(4))
    assert row(df, 0).rr == 18 and row(df, 3).rr == 22


def test_implausible_values_dropped():
    df = hourly_features(chart(ev(0, "hr", 0), ev(0, "spo2", 150), ev(0, "temp_c", 3.7)), stays(1))
    assert np.isnan(row(df, 0).hr) and np.isnan(row(df, 0).spo2) and np.isnan(row(df, 0).temp)


@pytest.mark.parametrize("eye, verbal, motor, avpu_bin", [(4, 5, 6, 0), (4, 4, 6, 1), (1, 1, 1, 1)])
def test_gcs_to_avpu_bin(eye, verbal, motor, avpu_bin):
    df = hourly_features(chart(ev(0, "gcs_eye", eye), ev(0, "gcs_verbal", verbal),
                               ev(0, "gcs_motor", motor)), stays(1))
    assert row(df, 0).avpu_bin == avpu_bin


def test_gcs_needs_all_three_parts():
    df = hourly_features(chart(ev(0, "gcs_eye", 4), ev(0, "gcs_motor", 6)), stays(1))
    assert np.isnan(row(df, 0).avpu_bin)


def test_supplemental_o2_from_device_or_flow():
    df = hourly_features(chart(ev(0, "o2_device", value="Nasal cannula"), ev(5, "o2_flow", 2.0),
                               ev(5, "hr", 80)), stays(7))
    assert list(df.supplemental_o2) == [1, 1, 1, 0, 0, 1, 1]  # ffill <= 2 h, else room air


def test_news2_computed_only_when_complete():
    full = [ev(0, "hr", 72), ev(0, "rr", 16), ev(0, "spo2", 98), ev(0, "sbp_nibp", 120),
            ev(0, "temp_c", 37.0), ev(0, "gcs_eye", 4), ev(0, "gcs_verbal", 5), ev(0, "gcs_motor", 6)]
    df = hourly_features(chart(*full), stays(4))
    assert row(df, 0).news2 == 0
    assert pd.isna(row(df, 3).news2)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def labelled(n_hours=100, deathtime=None, expire=0, interventions=()):
    st = stays(n_hours)
    adm = pd.DataFrame({"hadm_id": [10], "dischtime": [T0 + (n_hours + 5) * H],
                        "deathtime": [deathtime], "hospital_expire_flag": [expire]})
    iv = pd.DataFrame([{"stay_id": 1, "starttime": T0 + a * H, "endtime": T0 + b * H}
                       for a, b in interventions], columns=["stay_id", "starttime", "endtime"])
    hourly = hourly_features(chart(ev(0, "hr", 80)), st)
    return add_labels(hourly, st, adm, iv)


def test_los_bands():
    df = labelled(100)  # t = hour + 1, remaining = 100 - (hour + 1)
    assert row(df, 0).severity_label == 2      # 99 h left
    assert row(df, 27).severity_label == 1     # 72 h left
    assert row(df, 76).severity_label == 0     # 23 h left
    assert row(df, 0).patient_id == 100


def test_death_within_24h_is_critical():
    df = labelled(100, deathtime=T0 + 50 * H, expire=1)
    assert row(df, 25).death_24h and row(df, 25).severity_label == 3  # t=26, death in 24 h
    assert not row(df, 24).death_24h                                   # t=25, death in 25 h
    assert not row(df, 49).death_24h                                   # t=50: death at t, not after


def test_expire_flag_without_deathtime_uses_discharge():
    df = labelled(100, deathtime=pd.NaT, expire=1)  # discharge at 105 h
    assert row(df, 85).death_24h and not row(df, 79).death_24h


def test_intervention_overlap_and_trend():
    df = labelled(100, interventions=[(10, 12)])
    assert row(df, 4).intervention_6h and row(df, 4).trend_label == 1   # t=5: (5,11] overlaps [10,12)
    assert not row(df, 2).intervention_6h and row(df, 2).trend_label == 0  # t=3: (3,9] no overlap
    assert np.isnan(row(df, 10).trend_label)  # t=11, active in (10,11]: excluded from trend
    assert row(df, 10).severity_label == 3
    assert not row(df, 12).intervention_6h  # t=13: ended at 12


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def test_make_windows_shapes_and_padding():
    df = labelled(8)
    df.loc[df.hour >= 3, "has_reading"] = False
    df.loc[df.hour == 6, "has_reading"] = True
    X, mask, ys, yt, groups, n2 = make_windows(df)
    assert X.shape == (4, 6, len(FEATURES)) and mask.shape == (4, 6)
    assert mask[0].tolist() == [False] * 5 + [True]          # first bin: left-padded
    assert mask[2].tolist() == [False, False, False, True, True, True]
    assert mask[3].tolist() == [True, True, False, False, False, True]  # hour 6: bins 1..6
    assert (groups == 100).all() and len(ys) == 4


def test_add_deltas():
    X = np.zeros((1, 6, len(FEATURES)), dtype=np.float32)
    X[0, :, 0] = [0, 0, 80, 90, 85, 85]  # hr
    mask = np.array([[False, False, True, True, True, True]])
    Xd = add_deltas(X, mask)
    assert Xd.shape == (1, 6, 12)
    assert Xd[0, :, 7].tolist() == [0, 0, 0, 10, -5, 0]  # hr_delta, 0 at first real step
