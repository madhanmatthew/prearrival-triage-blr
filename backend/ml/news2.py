"""NEWS2 (National Early Warning Score 2) scoring: the clinical baseline for Component 2.

Implements the scoring bands in docs/02 §3 exactly (these match the Royal College of
Physicians NEWS2 chart, SpO2 Scale 1), the GCS -> AVPU mapping in docs/08 §2, and the
total -> 4-class banding in docs/08 §3.

NEWS2 is a fixed rule, not a trained model, so this module has no train/evaluate/save/load.
It is wrapped as a baseline predictor alongside the tabular baselines in
`backend/ml/baselines.py` (nested grouped CV, docs/15 D5).

Rule 1 (AGENTS.md): NEWS2 is a *baseline to compare against*. Never use it to build
`severity_label`; labels come from real outcomes (docs/08 §2).

Design choices (documented so they can be defended in the viva):

* **Missing values.** `None` and NaN both mean "not measured". A missing parameter gets a
  component score of `None`, and the total is `None` if any of the 7 parameters is missing:
  missing is never silently scored as 0 (normal). Filling gaps (forward-fill, train-median)
  happens upstream per docs/07 §4.3, before scoring.
  Opt-in exception: `missing_o2_as_air=True` scores a missing supplemental-O2 flag as
  "on room air" (0). [ASSUMPTION] MIMIC often does not chart O2 delivery explicitly;
  any result that uses this flag must say so.
* **Non-integer values.** NEWS2 bands are written for integers (e.g. RR 21-24), but hourly
  MIMIC medians can fall between them (RR 24.5). Each band is applied by its upper edge,
  so every real value falls in exactly one band: RR 24.5 is above 24 and scores 3;
  temperature 35.05 is above 35.0 and scores 1.
* **Invalid values.** Negative vitals, SpO2 above 100, booleans passed as numbers, unknown
  AVPU letters and GCS outside 3-15 raise `ValueError` rather than being scored.

Not implemented (state as limitations):

* SpO2 Scale 2 (target 88-92% for hypercapnic respiratory failure).
* "New confusion" scores 3 in NEWS2 but is not recorded in MIMIC; only GCS-derived AVPU is used.
* Official NEWS2 escalates when any single parameter scores 3 ("red score"). The 4-class
  banding in `news2_class` uses the total only, as fixed in docs/08 §3.

Research prototype, not a medical device. Not for clinical use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

from backend.schemas import VitalsReading

Number = Union[int, float]

PARAMETERS = ("rr", "spo2", "supplemental_o2", "sbp", "hr", "avpu", "temp")
MAX_TOTAL = 20  # 3 * 6 parameters + 2 for supplemental O2
AVPU_VALUES = ("A", "V", "P", "U")


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------

def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _check_number(value: object, name: str) -> Optional[float]:
    """Return a float, or None when missing. Rejects booleans and negative values."""
    if _is_missing(value):
        return None
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        v = float(value)  # type: ignore[arg-type]  # also accepts numpy scalars
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if math.isnan(v):  # numpy NaN
        return None
    if v < 0:
        raise ValueError(f"{name} cannot be negative, got {v}")
    return v


def _band(value: float, upper_edges: tuple[tuple[float, int], ...], above: int) -> int:
    """Score of the first band whose upper edge is >= value; `above` if past the last edge."""
    for edge, score in upper_edges:
        if value <= edge:
            return score
    return above


# ---------------------------------------------------------------------------
# Component scores (docs/02 §3)
# ---------------------------------------------------------------------------

def score_rr(rr: Optional[Number]) -> Optional[int]:
    """Respiratory rate (breaths/min): <=8 -> 3; 9-11 -> 1; 12-20 -> 0; 21-24 -> 2; >=25 -> 3."""
    v = _check_number(rr, "rr")
    return None if v is None else _band(v, ((8, 3), (11, 1), (20, 0), (24, 2)), 3)


def score_spo2(spo2: Optional[Number]) -> Optional[int]:
    """SpO2 (%), Scale 1: <=91 -> 3; 92-93 -> 2; 94-95 -> 1; >=96 -> 0."""
    v = _check_number(spo2, "spo2")
    if v is None:
        return None
    if v > 100:
        raise ValueError(f"spo2 cannot exceed 100, got {v}")
    return _band(v, ((91, 3), (93, 2), (95, 1)), 0)


def score_supplemental_o2(on_o2: Optional[bool]) -> Optional[int]:
    """Supplemental oxygen: yes -> 2; no (room air) -> 0."""
    if _is_missing(on_o2):
        return None
    if on_o2 not in (True, False):  # also accepts 0/1 and numpy bools
        raise ValueError(f"supplemental_o2 must be True/False, got {on_o2!r}")
    return 2 if on_o2 else 0


def score_sbp(sbp: Optional[Number]) -> Optional[int]:
    """Systolic BP (mmHg): <=90 -> 3; 91-100 -> 2; 101-110 -> 1; 111-219 -> 0; >=220 -> 3."""
    v = _check_number(sbp, "sbp")
    return None if v is None else _band(v, ((90, 3), (100, 2), (110, 1), (219, 0)), 3)


def score_hr(hr: Optional[Number]) -> Optional[int]:
    """Heart rate (bpm): <=40 -> 3; 41-50 -> 1; 51-90 -> 0; 91-110 -> 1; 111-130 -> 2; >=131 -> 3."""
    v = _check_number(hr, "hr")
    return None if v is None else _band(
        v, ((40, 3), (50, 1), (90, 0), (110, 1), (130, 2)), 3
    )


def score_avpu(avpu: Optional[str]) -> Optional[int]:
    """Consciousness: A (alert) -> 0; V, P or U -> 3."""
    if _is_missing(avpu):
        return None
    if avpu not in AVPU_VALUES:
        raise ValueError(f"avpu must be one of {AVPU_VALUES}, got {avpu!r}")
    return 0 if avpu == "A" else 3


def score_temp(temp: Optional[Number]) -> Optional[int]:
    """Temperature (deg C): <=35.0 -> 3; 35.1-36.0 -> 1; 36.1-38.0 -> 0; 38.1-39.0 -> 1; >=39.1 -> 2."""
    v = _check_number(temp, "temp")
    return None if v is None else _band(
        v, ((35.0, 3), (36.0, 1), (38.0, 0), (39.0, 1)), 2
    )


# ---------------------------------------------------------------------------
# GCS -> AVPU (docs/08 §2)
# ---------------------------------------------------------------------------

def gcs_to_avpu(gcs_total: Optional[int]) -> Optional[str]:
    """Map total GCS to AVPU: 15 -> A; 13-14 -> V; 9-12 -> P; <=8 -> U.

    Project convention from docs/08 §2 (MIMIC records GCS, not AVPU); not an official
    equivalence. Accepts whole numbers 3-15 only (15.0 is accepted, 14.5 is not).
    """
    g = _check_number(gcs_total, "gcs_total")
    if g is None:
        return None
    if not g.is_integer() or not 3 <= g <= 15:
        raise ValueError(f"gcs_total must be a whole number from 3 to 15, got {gcs_total!r}")
    if g == 15:
        return "A"
    if g >= 13:
        return "V"
    if g >= 9:
        return "P"
    return "U"


# ---------------------------------------------------------------------------
# Total and class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class News2Result:
    components: dict[str, Optional[int]]  # keyed by PARAMETERS
    total: Optional[int]  # None if any parameter is missing
    missing: tuple[str, ...]  # parameters that were None/NaN
    o2_assumed_air: bool = False  # True if missing_o2_as_air filled the O2 flag

    @property
    def severity_class(self) -> Optional[int]:
        return news2_class(self.total)


def news2_score(
    *,
    rr: Optional[Number] = None,
    spo2: Optional[Number] = None,
    supplemental_o2: Optional[bool] = None,
    sbp: Optional[Number] = None,
    hr: Optional[Number] = None,
    avpu: Optional[str] = None,
    temp: Optional[Number] = None,
    missing_o2_as_air: bool = False,
) -> News2Result:
    """Score all 7 NEWS2 parameters. See the module docstring for missing-value rules."""
    o2_assumed_air = False
    if missing_o2_as_air and _is_missing(supplemental_o2):
        supplemental_o2 = False
        o2_assumed_air = True

    components = {
        "rr": score_rr(rr),
        "spo2": score_spo2(spo2),
        "supplemental_o2": score_supplemental_o2(supplemental_o2),
        "sbp": score_sbp(sbp),
        "hr": score_hr(hr),
        "avpu": score_avpu(avpu),
        "temp": score_temp(temp),
    }
    missing = tuple(p for p in PARAMETERS if components[p] is None)
    total = None if missing else sum(components.values())  # type: ignore[arg-type]
    return News2Result(components, total, missing, o2_assumed_air)


def news2_class(total: Optional[int]) -> Optional[int]:
    """Total NEWS2 -> 4 severity classes: 0 -> 0 stable; 1-4 -> 1 moderate; 5-6 -> 2 severe;
    >=7 -> 3 critical.

    PROJECT CONVENTION (docs/08 §3), not official NEWS2 banding: the official clinical
    response tiers (low / low-medium / medium / high) differ, and the official "any single
    parameter scores 3" escalation is not applied here.
    """
    if total is None:
        return None
    if isinstance(total, bool) or not isinstance(total, int) or not 0 <= total <= MAX_TOTAL:
        raise ValueError(f"total must be an int from 0 to {MAX_TOTAL}, got {total!r}")
    if total == 0:
        return 0
    if total <= 4:
        return 1
    if total <= 6:
        return 2
    return 3


def score_reading(reading: VitalsReading, *, missing_o2_as_air: bool = False) -> News2Result:
    """Score a `VitalsReading` (docs/07 §2.2)."""
    return news2_score(
        rr=reading.rr,
        spo2=reading.spo2,
        supplemental_o2=reading.supplemental_o2,
        sbp=reading.sbp,
        hr=reading.hr,
        avpu=reading.avpu,
        temp=reading.temp,
        missing_o2_as_air=missing_o2_as_air,
    )
