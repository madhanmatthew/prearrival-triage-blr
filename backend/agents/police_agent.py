"""Police alert sizing (docs/03 §8, docs/07 §2.6). Rule-based, NOT machine learning.

    size_alert(incident_id, gps, severity_class, injury_type, num_injured, route_polyline)
        -> PoliceAlert(units_required, perimeter_m, ...)

Units and perimeter scale with severity, injured count and injury type. The numbers below are
hand-set and not calibrated against real police deployment data.
[ASSUMPTION] all constants; [TODO-VERIFY] with Bengaluru traffic-police practice before any
claim beyond "rule-based demo logic". Research prototype, not a medical device.
"""
from __future__ import annotations

from typing import Sequence

from backend.schemas import GPS, INJURY_TYPES, N_SEVERITY_CLASSES, PoliceAlert

# Units by severity class (0 stable .. 3 critical): scene control + route clearing.
BASE_UNITS = {0: 0, 1: 1, 2: 2, 3: 3}
# Extra unit per 2 injured beyond the first 2 (crowd/scene), capped.
EXTRA_PER_INJURED = 2
MAX_UNITS = 6
# Road accidents need traffic diversion on top of scene control.
ROAD_EXTRA_UNITS = 1
# Medical-only incidents (cardiac, other) need a route-clearing escort but no cordon.
NO_CORDON_TYPES = ("cardiac_medical", "other")
PERIMETER_M = {0: 0.0, 1: 50.0, 2: 100.0, 3: 200.0}
PERIMETER_EXTRA_M = {"road_accident": 50.0, "burn": 50.0, "fall": 0.0}
PERIMETER_PER_EXTRA_INJURED_M = 20.0
MAX_PERIMETER_M = 400.0


def size_alert(incident_id: str, gps: GPS, severity_class: int, injury_type: str,
               num_injured: int, route_polyline: Sequence[tuple[float, float]] = ()) -> PoliceAlert:
    if not 0 <= severity_class < N_SEVERITY_CLASSES:
        raise ValueError(f"severity_class must be 0..{N_SEVERITY_CLASSES - 1}, got {severity_class}")
    if injury_type not in INJURY_TYPES:
        raise ValueError(f"unknown injury_type {injury_type!r}")
    n = max(int(num_injured), 0)
    units = BASE_UNITS[severity_class]
    if units > 0:
        units += max(n - 2, 0) // EXTRA_PER_INJURED
        if injury_type == "road_accident":
            units += ROAD_EXTRA_UNITS
        units = min(units, MAX_UNITS)
    perimeter = 0.0
    if units > 0 and injury_type not in NO_CORDON_TYPES:
        perimeter = PERIMETER_M[severity_class] + PERIMETER_EXTRA_M.get(injury_type, 0.0)
        perimeter += PERIMETER_PER_EXTRA_INJURED_M * max(n - 1, 0)
        perimeter = min(perimeter, MAX_PERIMETER_M)
    return PoliceAlert(incident_id=incident_id, gps=gps, severity_class=severity_class,
                       route_polyline=[(float(a), float(b)) for a, b in route_polyline],
                       units_required=units, perimeter_m=perimeter)
