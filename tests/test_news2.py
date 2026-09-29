"""Tests for backend/ml/news2.py.

Band edges come from docs/02 §3 (checked against the RCP NEWS2 chart, SpO2 Scale 1).
Each boundary is tested on both sides. Worked examples were totalled by hand; the
arithmetic is written next to each one so it can be re-checked.
"""
import math

import pytest

from backend.ml.news2 import (
    gcs_to_avpu,
    news2_class,
    news2_score,
    score_avpu,
    score_hr,
    score_reading,
    score_rr,
    score_sbp,
    score_spo2,
    score_supplemental_o2,
    score_temp,
)
from backend.schemas import VitalsReading

NORMAL = dict(rr=16, spo2=98, supplemental_o2=False, sbp=120, hr=72, avpu="A", temp=37.0)


# ---------------------------------------------------------------------------
# Component bands: both sides of every boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rr, expected", [
    (0, 3), (8, 3), (9, 1), (11, 1), (12, 0), (20, 0), (21, 2), (24, 2), (25, 3), (40, 3),
])
def test_rr_bands(rr, expected):
    assert score_rr(rr) == expected


@pytest.mark.parametrize("spo2, expected", [
    (80, 3), (91, 3), (92, 2), (93, 2), (94, 1), (95, 1), (96, 0), (100, 0),
])
def test_spo2_bands(spo2, expected):
    assert score_spo2(spo2) == expected


@pytest.mark.parametrize("sbp, expected", [
    (60, 3), (90, 3), (91, 2), (100, 2), (101, 1), (110, 1), (111, 0), (219, 0), (220, 3),
])
def test_sbp_bands(sbp, expected):
    assert score_sbp(sbp) == expected


@pytest.mark.parametrize("hr, expected", [
    (30, 3), (40, 3), (41, 1), (50, 1), (51, 0), (90, 0), (91, 1), (110, 1),
    (111, 2), (130, 2), (131, 3), (180, 3),
])
def test_hr_bands(hr, expected):
    assert score_hr(hr) == expected


@pytest.mark.parametrize("temp, expected", [
    (34.0, 3), (35.0, 3), (35.1, 1), (36.0, 1), (36.1, 0), (38.0, 0),
    (38.1, 1), (39.0, 1), (39.1, 2), (41.0, 2),
])
def test_temp_bands(temp, expected):
    assert score_temp(temp) == expected


@pytest.mark.parametrize("avpu, expected", [("A", 0), ("V", 3), ("P", 3), ("U", 3)])
def test_avpu(avpu, expected):
    assert score_avpu(avpu) == expected


@pytest.mark.parametrize("on_o2, expected", [(True, 2), (False, 0), (1, 2), (0, 0)])
def test_supplemental_o2(on_o2, expected):
    assert score_supplemental_o2(on_o2) == expected


# ---------------------------------------------------------------------------
# Non-integer values between integer bands: upper-edge rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn, value, expected", [
    (score_rr, 8.5, 1), (score_rr, 11.5, 0), (score_rr, 20.5, 2), (score_rr, 24.5, 3),
    (score_spo2, 91.5, 2), (score_spo2, 95.5, 0),
    (score_sbp, 90.5, 2), (score_sbp, 110.5, 0), (score_sbp, 219.5, 3),
    (score_hr, 40.5, 1), (score_hr, 90.5, 1), (score_hr, 130.5, 3),
    (score_temp, 35.05, 1), (score_temp, 38.05, 1), (score_temp, 39.05, 2),
])
def test_values_between_bands(fn, value, expected):
    assert fn(value) == expected


# ---------------------------------------------------------------------------
# GCS -> AVPU (docs/08 §2): every valid value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gcs, expected", [
    (15, "A"), (14, "V"), (13, "V"), (12, "P"), (11, "P"), (10, "P"), (9, "P"),
    (8, "U"), (7, "U"), (6, "U"), (5, "U"), (4, "U"), (3, "U"), (15.0, "A"),
])
def test_gcs_to_avpu(gcs, expected):
    assert gcs_to_avpu(gcs) == expected


@pytest.mark.parametrize("gcs", [2, 16, 0, -1, 14.5, True, "15"])
def test_gcs_to_avpu_rejects_invalid(gcs):
    with pytest.raises(ValueError):
        gcs_to_avpu(gcs)


def test_gcs_missing():
    assert gcs_to_avpu(None) is None
    assert gcs_to_avpu(float("nan")) is None


# ---------------------------------------------------------------------------
# Total -> 4-class project convention (docs/08 §3)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("total, expected", [
    (0, 0), (1, 1), (4, 1), (5, 2), (6, 2), (7, 3), (20, 3),
])
def test_news2_class_boundaries(total, expected):
    assert news2_class(total) == expected


@pytest.mark.parametrize("total", [-1, 21, 3.0, True])
def test_news2_class_rejects_invalid(total):
    with pytest.raises(ValueError):
        news2_class(total)


def test_news2_class_missing():
    assert news2_class(None) is None


# ---------------------------------------------------------------------------
# Worked examples (hand-checked)
# ---------------------------------------------------------------------------

def test_healthy_adult_scores_zero():
    # every parameter in its 0 band
    r = news2_score(**NORMAL)
    assert r.total == 0 and r.severity_class == 0 and r.missing == ()


def test_mild_fever_tachycardia_is_moderate():
    # HR 95 -> 1, temp 38.5 -> 1, rest 0: total 2 -> class 1
    r = news2_score(**{**NORMAL, "hr": 95, "temp": 38.5})
    assert r.total == 2 and r.severity_class == 1


def test_mixed_derangement_is_severe():
    # RR 22 -> 2, SpO2 95 -> 1, SBP 105 -> 1, HR 95 -> 1, temp 38.5 -> 1, air 0, A 0: total 6
    r = news2_score(rr=22, spo2=95, supplemental_o2=False, sbp=105, hr=95, avpu="A", temp=38.5)
    assert r.total == 6 and r.severity_class == 2


def test_docs07_example_reading():
    # docs/07 §2.2: HR 112 -> 2, RR 24 -> 2, SpO2 93 -> 2, SBP 98 -> 2, temp 36.4 -> 0,
    # A -> 0, air -> 0: total 8 -> class 3
    reading = VitalsReading(
        incident_id="INC-20260928-0001", patient_id="P-1", timestamp="2026-09-28T09:24:00Z",
        hr=112, rr=24, spo2=93, sbp=98, temp=36.4, avpu="A", supplemental_o2=False,
    )
    r = score_reading(reading)
    assert r.components == {
        "rr": 2, "spo2": 2, "supplemental_o2": 0, "sbp": 2, "hr": 2, "avpu": 0, "temp": 0,
    }
    assert r.total == 8 and r.severity_class == 3


def test_maximum_score():
    # RR 3 + SpO2 3 + O2 2 + SBP 3 + HR 3 + AVPU 3 + temp 3 = 20
    r = news2_score(rr=30, spo2=85, supplemental_o2=True, sbp=80, hr=140, avpu="U", temp=34.5)
    assert r.total == 20 and r.severity_class == 3


def test_single_red_parameter_uses_total_only():
    # GCS 12 -> P -> 3, rest 0: total 3 -> class 1. The official single-parameter red
    # escalation is deliberately not applied (docs/08 §3 convention).
    r = news2_score(**{**NORMAL, "avpu": gcs_to_avpu(12)})
    assert r.components["avpu"] == 3
    assert r.total == 3 and r.severity_class == 1


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [
    score_rr, score_spo2, score_sbp, score_hr, score_temp, score_avpu, score_supplemental_o2,
])
def test_component_missing_is_none(fn):
    assert fn(None) is None
    assert fn(float("nan")) is None


def test_any_missing_parameter_gives_no_total():
    r = news2_score(**{**NORMAL, "spo2": None, "temp": math.nan})
    assert r.total is None and r.severity_class is None
    assert r.missing == ("spo2", "temp")
    assert r.components["hr"] == 0  # present parameters are still scored


def test_all_missing():
    r = news2_score()
    assert r.total is None and len(r.missing) == 7


def test_missing_o2_strict_by_default():
    r = news2_score(**{**NORMAL, "supplemental_o2": None})
    assert r.total is None and r.missing == ("supplemental_o2",)
    assert r.o2_assumed_air is False


def test_missing_o2_as_air_opt_in():
    r = news2_score(**{**NORMAL, "supplemental_o2": None}, missing_o2_as_air=True)
    assert r.total == 0 and r.components["supplemental_o2"] == 0
    assert r.o2_assumed_air is True


def test_missing_o2_as_air_does_not_override_recorded_o2():
    r = news2_score(**{**NORMAL, "supplemental_o2": True}, missing_o2_as_air=True)
    assert r.components["supplemental_o2"] == 2 and r.o2_assumed_air is False


# ---------------------------------------------------------------------------
# Invalid inputs are rejected, not scored
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn, value", [
    (score_rr, -1), (score_hr, -5), (score_sbp, -10), (score_temp, -1.0),
    (score_spo2, -1), (score_spo2, 101),
    (score_hr, True), (score_rr, "16"),
    (score_avpu, "a"), (score_avpu, "X"), (score_supplemental_o2, "yes"),
    (score_supplemental_o2, 2),
])
def test_invalid_inputs_raise(fn, value):
    with pytest.raises(ValueError):
        fn(value)
