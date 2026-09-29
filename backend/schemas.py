"""Shared message schemas: the contract between components (docs/07_interfaces_and_schemas.md).

Change a schema only by editing docs/07 first, then this file, then the code that uses it.
Every component (reporting, vitals/fusion, traffic, hospital, coordinator) imports from here,
so teammates can build against mocks of these objects before upstream modules exist.

Research prototype, not a medical device.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants (docs/07 §1, §3, docs/15 D1)
# ---------------------------------------------------------------------------

N_SEVERITY_CLASSES = 4  # 0 stable, 1 moderate, 2 severe, 3 critical (D2 / 15 D1)
SEVERITY_NAMES = ("stable", "moderate", "severe", "critical")
TEXT_FEATURE_DIM = 19
TREND_HORIZON_READINGS = 6

Language = Literal["en", "hi", "kn"]
AVPU = Literal["A", "V", "P", "U"]
AnswerValue = Literal["yes", "no", "unknown"]
QuestionId = Literal[
    "Q_CONSCIOUS", "Q_BREATHING", "Q_BLEEDING", "Q_SPEAK", "Q_NUM_INJURED", "Q_MECHANISM"
]
InjuryType = Literal["road_accident", "fall", "burn", "cardiac_medical", "other"]
INJURY_TYPES: tuple[str, ...] = ("road_accident", "fall", "burn", "cardiac_medical", "other")


def _check_prob_vector(v: list[float], name: str) -> list[float]:
    if len(v) != N_SEVERITY_CLASSES:
        raise ValueError(f"{name} must have {N_SEVERITY_CLASSES} entries, got {len(v)}")
    if any(p < 0 or p > 1 for p in v):
        raise ValueError(f"{name} entries must be in [0, 1]")
    if not math.isclose(sum(v), 1.0, abs_tol=1e-3):
        raise ValueError(f"{name} must sum to 1 (got {sum(v):.4f})")
    return v


def risk_score_from_probs(severity_probs: list[float]) -> float:
    """Expected severity normalised to [0, 1]: sum_i p_i * (i / 3)  (docs/07 §2.3)."""
    _check_prob_vector(severity_probs, "severity_probs")
    return sum(p * i for i, p in enumerate(severity_probs)) / (N_SEVERITY_CLASSES - 1)


class GPS(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


# ---------------------------------------------------------------------------
# 2.1 IncidentReport (Component 1 -> Coordinator)
# ---------------------------------------------------------------------------

class DialogueTurn(BaseModel):
    q_id: QuestionId
    question: str
    answer_raw: str
    answer_value: AnswerValue


class IncidentReport(BaseModel):
    incident_id: str
    timestamp: str  # ISO-8601 UTC (D4)
    gps: GPS  # device GPS only, never parsed from speech (hard rule 5)
    language: Language
    transcript: str
    dialogue: list[DialogueTurn] = Field(default_factory=list, max_length=6)
    text_features: list[float]
    text_severity_probs: list[float]
    injury_type: InjuryType
    num_injured: int = Field(ge=0)

    @field_validator("text_features")
    @classmethod
    def _tf_dim(cls, v: list[float]) -> list[float]:
        if len(v) != TEXT_FEATURE_DIM:
            raise ValueError(f"text_features must be {TEXT_FEATURE_DIM}-dim, got {len(v)}")
        return v

    @field_validator("text_severity_probs")
    @classmethod
    def _tsp(cls, v: list[float]) -> list[float]:
        return _check_prob_vector(v, "text_severity_probs")


# ---------------------------------------------------------------------------
# 2.2 VitalsReading (Ambulance -> Component 2). Any field may be null (sensor dropout).
# ---------------------------------------------------------------------------

class VitalsReading(BaseModel):
    incident_id: str
    patient_id: str
    timestamp: str
    hr: Optional[float] = None
    rr: Optional[float] = None
    spo2: Optional[float] = Field(default=None, ge=0, le=100)
    sbp: Optional[float] = None
    temp: Optional[float] = None  # degrees Celsius
    avpu: Optional[AVPU] = None
    supplemental_o2: Optional[bool] = None

    @property
    def all_null(self) -> bool:
        """True when every measurement is missing (whole timestep masked, docs/07 §4.3)."""
        return all(
            getattr(self, f) is None
            for f in ("hr", "rr", "spo2", "sbp", "temp", "avpu", "supplemental_o2")
        )


# ---------------------------------------------------------------------------
# 2.3 FusedRisk (Component 2/5 -> Hospital agent, Coordinator)
# ---------------------------------------------------------------------------

class FusedRisk(BaseModel):
    incident_id: str
    patient_id: str
    timestamp: str
    severity_probs: list[float]
    severity_class: int = Field(ge=0, le=N_SEVERITY_CLASSES - 1)
    risk_score: float = Field(ge=0, le=1)
    trend_prob: Optional[float] = Field(default=None, ge=0, le=1)  # None before vitals exist
    trend_horizon_readings: int = TREND_HORIZON_READINGS
    modalities_used: list[Literal["text", "vitals"]]
    news2_score: Optional[int] = Field(default=None, ge=0, le=20)
    model_version: str
    stale: bool = False  # docs/07 §6: True after 3 missed vitals intervals

    @field_validator("severity_probs")
    @classmethod
    def _sp(cls, v: list[float]) -> list[float]:
        return _check_prob_vector(v, "severity_probs")

    @model_validator(mode="after")
    def _consistency(self) -> "FusedRisk":
        if not self.modalities_used:
            raise ValueError("modalities_used cannot be empty")
        expected = risk_score_from_probs(self.severity_probs)
        if not math.isclose(self.risk_score, expected, abs_tol=1e-3):
            raise ValueError(f"risk_score {self.risk_score} != expected severity {expected:.4f}")
        return self

    @classmethod
    def from_probs(cls, severity_probs: list[float], **kwargs) -> "FusedRisk":
        """Build a FusedRisk, deriving severity_class (argmax) and risk_score from probs."""
        cls_idx = max(range(len(severity_probs)), key=lambda i: severity_probs[i])
        return cls(
            severity_probs=severity_probs,
            severity_class=cls_idx,
            risk_score=risk_score_from_probs(severity_probs),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# 2.4 HospitalRecommendation (Hospital agent -> Coordinator)
# ---------------------------------------------------------------------------

class ScoreBreakdown(BaseModel):
    proximity: float
    trauma_match: float
    beds: float
    specialty: float
    blood_match: float
    rf_qvalue: Optional[float] = None


class RankedHospital(BaseModel):
    hospital_id: str
    name: str
    total_score: float
    eta_min: float = Field(ge=0)
    breakdown: ScoreBreakdown
    blood_flag: Literal["matched", "unknown", "short"]


class HospitalRecommendation(BaseModel):
    incident_id: str
    ranked: list[RankedHospital] = Field(min_length=1)

    @model_validator(mode="after")
    def _sorted(self) -> "HospitalRecommendation":
        scores = [h.total_score for h in self.ranked]
        if scores != sorted(scores, reverse=True):
            raise ValueError("ranked must be sorted by total_score, highest first")
        return self


# ---------------------------------------------------------------------------
# 2.5 SignalState / CorridorStatus (Traffic agent -> Coordinator)
# ---------------------------------------------------------------------------

class JunctionState(BaseModel):
    id: Literal["silkboard", "bellandur", "marathahalli", "krpuram"]
    phase: Literal["main_green", "cross_green", "yellow"]
    ambulance_eta_s: Optional[float] = Field(default=None, ge=0)
    preempted: bool = False


class Conflict(BaseModel):
    type: str
    detail: str = ""


class SignalState(BaseModel):
    incident_id: str
    route_id: str
    junctions: list[JunctionState]
    conflicts: list[Conflict] = Field(default_factory=list)
    estimated_transit_s: Optional[float] = Field(default=None, ge=0)


# ---------------------------------------------------------------------------
# 2.6 PoliceAlert (Coordinator -> Police dashboard)
# ---------------------------------------------------------------------------

class PoliceAlert(BaseModel):
    incident_id: str
    gps: GPS
    severity_class: int = Field(ge=0, le=N_SEVERITY_CLASSES - 1)
    route_polyline: list[tuple[float, float]] = Field(default_factory=list)
    units_required: int = Field(ge=0)
    perimeter_m: float = Field(ge=0)


# ---------------------------------------------------------------------------
# 2.7 DispatchDecision (Coordinator -> all clients)
# ---------------------------------------------------------------------------

class ConflictResolution(BaseModel):
    type: str
    resolution: str


class DispatchDecision(BaseModel):
    incident_id: str
    ambulance_id: str
    hospital_id: str
    route_id: str
    police_units: int = Field(ge=0)
    conflicts_resolved: list[ConflictResolution] = Field(default_factory=list)
    explanation: str  # human-readable top reasons (decision support, not automation)
    timestamp: str
    route_estimated: bool = False  # docs/07 §6: True when ORS was down


# ---------------------------------------------------------------------------
# WebSocket envelope (docs/07 §5)
# ---------------------------------------------------------------------------

class WSMessage(BaseModel):
    type: Literal["PoliceAlert", "FusedRisk", "SignalState", "DispatchDecision"]
    data: dict
