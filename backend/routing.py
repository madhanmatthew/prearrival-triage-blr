"""Routing and ETAs: OpenRouteService directions (AGENTS rule 11, kept as-is) with a straight-line
fallback when ORS is down or no key is set (docs/07 §6: `route_estimated=true`).

Both routers implement the coordinator's `Router` protocol plus `eta_min(a, b)`.
Locations are (lat, lon). ORS takes [lon, lat].

[ASSUMPTION]s: fallback speed 25 km/h and road/straight-line factor 1.4 for Bengaluru traffic
(fixed placeholders; ORS durations are the primary ETA, docs/15 D16).
[TODO-VERIFY] ORS request/response fields (`/v2/directions/driving-car/geojson`,
`options.avoid_polygons`, `features[0].properties.summary.duration`) are from memory of the ORS
docs: smoke-test with a real (rotated) key before relying on it.
"""
from __future__ import annotations

import json
import math
import os
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from backend.agents.coordinator import LatLon, Route

FALLBACK_SPEED_KMH = 25.0
ROAD_FACTOR = 1.4
ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"

Locator = Callable[[str], Optional[LatLon]]   # id -> (lat, lon)


def haversine_km(a: LatLon, b: LatLon) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))


@dataclass
class StraightLineRouter:
    """Fallback: straight line, ETA from distance x ROAD_FACTOR at FALLBACK_SPEED_KMH.
    Cannot avoid polygons (returns None when asked to, so the coordinator flags it)."""
    locate_incident: Locator
    locate_hospital: Locator

    def eta_min(self, a: LatLon, b: LatLon) -> tuple[float, bool]:
        return haversine_km(a, b) * ROAD_FACTOR / FALLBACK_SPEED_KMH * 60.0, True

    def route(self, incident_id: str, hospital_id: str,
              avoid: Sequence[list[LatLon]] = ()) -> Optional[Route]:
        a, b = self.locate_incident(incident_id), self.locate_hospital(hospital_id)
        if a is None or b is None or avoid:
            return None
        eta, _ = self.eta_min(a, b)
        return Route(f"SL-{incident_id}-{hospital_id}", [a, b], round(eta, 2), estimated=True)


@dataclass
class OrsRouter:
    api_key: str
    fallback: StraightLineRouter
    timeout: float = 10.0

    @classmethod
    def from_env(cls, fallback: StraightLineRouter):
        key = os.getenv("ORS_API_KEY", "").strip()
        return cls(key, fallback) if key else fallback

    def _directions(self, a: LatLon, b: LatLon, avoid: Sequence[list[LatLon]] = ()) -> dict:
        body: dict = {"coordinates": [[a[1], a[0]], [b[1], b[0]]]}
        if avoid:
            rings = [[[lon, lat] for lat, lon in (*poly, poly[0])] for poly in avoid]
            body["options"] = {"avoid_polygons": {"type": "MultiPolygon",
                                                  "coordinates": [[r] for r in rings]}}
        req = urllib.request.Request(ORS_URL, data=json.dumps(body).encode(), headers={
            "Authorization": self.api_key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["features"][0]

    def eta_min(self, a: LatLon, b: LatLon) -> tuple[float, bool]:
        try:
            return self._directions(a, b)["properties"]["summary"]["duration"] / 60.0, False
        except Exception:
            return self.fallback.eta_min(a, b)

    def route(self, incident_id: str, hospital_id: str,
              avoid: Sequence[list[LatLon]] = ()) -> Optional[Route]:
        a = self.fallback.locate_incident(incident_id)
        b = self.fallback.locate_hospital(hospital_id)
        if a is None or b is None:
            return None
        try:
            f = self._directions(a, b, avoid)
        except Exception:
            return self.fallback.route(incident_id, hospital_id, avoid)
        line = [(lat, lon) for lon, lat in f["geometry"]["coordinates"]]
        eta = f["properties"]["summary"]["duration"] / 60.0
        tag = "-avoid" if avoid else ""
        return Route(f"ORS-{incident_id}-{hospital_id}{tag}", line, round(eta, 2))
