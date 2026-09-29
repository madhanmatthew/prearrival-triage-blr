"""Tests for the docs/07 contract in backend/schemas.py."""
import pytest
from pydantic import ValidationError

from backend.schemas import (
    TEXT_FEATURE_DIM,
    FusedRisk,
    HospitalRecommendation,
    IncidentReport,
    VitalsReading,
    risk_score_from_probs,
)

TS = "2026-09-28T09:15:00Z"


def _incident(**overrides):
    base = dict(
        incident_id="INC-20260928-0001",
        timestamp=TS,
        gps={"lat": 12.9166, "lon": 77.6231},
        language="kn",
        transcript="bike accident near silk board",
        text_features=[0.0] * TEXT_FEATURE_DIM,
        text_severity_probs=[0.1, 0.3, 0.4, 0.2],
        injury_type="road_accident",
        num_injured=2,
    )
    base.update(overrides)
    return base


def test_risk_score_endpoints_and_example():
    assert risk_score_from_probs([1, 0, 0, 0]) == 0.0
    assert risk_score_from_probs([0, 0, 0, 1]) == 1.0
    # docs/07 §2.3 example: 0.18/3 + 2*0.55/3 + 3*0.25/3 = 0.6767
    assert risk_score_from_probs([0.02, 0.18, 0.55, 0.25]) == pytest.approx(0.6767, abs=1e-3)


def test_incident_report_valid():
    r = IncidentReport(**_incident())
    assert r.language == "kn"


@pytest.mark.parametrize("dim", [18, 20])
def test_incident_report_rejects_wrong_text_dim(dim):
    with pytest.raises(ValidationError):
        IncidentReport(**_incident(text_features=[0.0] * dim))


def test_incident_report_rejects_bad_language_and_probs():
    with pytest.raises(ValidationError):
        IncidentReport(**_incident(language="ta"))
    with pytest.raises(ValidationError):
        IncidentReport(**_incident(text_severity_probs=[0.5, 0.5, 0.5, 0.0]))


def test_vitals_reading_allows_nulls_and_flags_all_null():
    v = VitalsReading(incident_id="I", patient_id="P-1", timestamp=TS)
    assert v.all_null
    v2 = VitalsReading(incident_id="I", patient_id="P-1", timestamp=TS, hr=112, avpu="A")
    assert not v2.all_null
    with pytest.raises(ValidationError):
        VitalsReading(incident_id="I", patient_id="P-1", timestamp=TS, spo2=130)


def test_fused_risk_from_probs_derives_class_and_score():
    fr = FusedRisk.from_probs(
        [0.02, 0.18, 0.55, 0.25],
        incident_id="I", patient_id="P-1", timestamp=TS,
        trend_prob=0.41, modalities_used=["text", "vitals"], model_version="test",
    )
    assert fr.severity_class == 2
    assert fr.risk_score == pytest.approx(0.6767, abs=1e-3)


def test_fused_risk_rejects_inconsistent_score():
    with pytest.raises(ValidationError):
        FusedRisk(
            incident_id="I", patient_id="P-1", timestamp=TS,
            severity_probs=[0.25, 0.25, 0.25, 0.25], severity_class=0, risk_score=0.9,
            modalities_used=["text"], model_version="test",
        )


def test_fused_risk_text_only_mode_has_no_trend():
    fr = FusedRisk.from_probs(
        [0.1, 0.2, 0.3, 0.4], incident_id="I", patient_id="P-1", timestamp=TS,
        modalities_used=["text"], model_version="test",
    )
    assert fr.trend_prob is None


def _ranked(score, hid):
    return dict(
        hospital_id=hid, name=hid, total_score=score, eta_min=10,
        breakdown=dict(proximity=1, trauma_match=1, beds=1, specialty=1, blood_match=1),
        blood_flag="matched",
    )


def test_hospital_recommendation_must_be_sorted():
    HospitalRecommendation(incident_id="I", ranked=[_ranked(0.9, "H1"), _ranked(0.5, "H2")])
    with pytest.raises(ValidationError):
        HospitalRecommendation(incident_id="I", ranked=[_ranked(0.5, "H1"), _ranked(0.9, "H2")])
