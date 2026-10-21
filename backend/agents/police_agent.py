"""Police alerting (docs/03 §8, docs/07 §2.6). Rule-based, NOT machine learning.

    size_alert(...)          -> PoliceAlert(units_required, perimeter_m, ...)
    plan_junction_posts(...) -> JunctionPost per ORR corridor junction on the route, in travel
                                order, with the ambulance ETA and the "officer in place by" time
    AlertBoard               -> alert lifecycle: acknowledge, ETA updates from ambulance GPS,
                                release each post once the ambulance has passed, close

Units and perimeter scale with severity, injured count and injury type. ETAs at junctions are
the route duration spread over route distance (constant speed), not a traffic prediction.
Junction posts exist only for the 4 corridor junctions in scripts/traffic_points.json
(corridor-only demo); other routes get an alert without posts.
[ASSUMPTION] all constants; [TODO-VERIFY] with Bengaluru traffic-police practice before any
claim beyond "rule-based demo logic". Research prototype, not a medical device.
"""
from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from backend.schemas import (GPS, INJURY_TYPES, N_SEVERITY_CLASSES, JunctionPost, PoliceAlert,
                             SignalState)

LatLon = tuple[float, float]

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


# ---------------------------------------------------------------------------
# Junction posts (corridor junctions on the route, with timing)
# ---------------------------------------------------------------------------

JUNCTIONS_FILE = Path(__file__).resolve().parents[2] / "scripts" / "traffic_points.json"
JUNCTION_RADIUS_M = 150.0   # junction counts as "on the route" within this distance
OFFICER_LEAD_S = 120.0      # officer in place this long before the ambulance arrives
ON_SCENE_MIN = 10.0         # time at the scene before leaving for hospital
PASS_MARGIN_M = 50.0        # post released once the ambulance is this far past the junction
OFF_ROUTE_M = 300.0         # GPS further than this from the route: ETAs are not updated


def load_corridor_junctions(path: Path = JUNCTIONS_FILE) -> dict[str, LatLon]:
    """Junction id -> (lat, lon) from scripts/traffic_points.json."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: (float(v["lat"]), float(v["lon"])) for k, v in data["junctions"].items()}


def _xy_km(p: LatLon, origin: LatLon) -> tuple[float, float]:
    """Local equirectangular projection (fine over a city)."""
    return ((p[1] - origin[1]) * 111.320 * math.cos(math.radians(origin[0])),
            (p[0] - origin[0]) * 110.574)


def route_length_km(polyline: Sequence[LatLon]) -> float:
    o = polyline[0]
    pts = [_xy_km(p, o) for p in polyline]
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def project_onto_route(point: LatLon, polyline: Sequence[LatLon]) -> tuple[float, float]:
    """(km from the route, km along the route) of the nearest point on the route."""
    if len(polyline) < 2:
        raise ValueError("route polyline needs at least 2 points")
    o = polyline[0]
    pts = [_xy_km(p, o) for p in polyline]
    px, py = _xy_km(point, o)
    best, along, run = math.inf, 0.0, 0.0
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        seg = math.hypot(dx, dy)
        t = 0.0 if seg == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg ** 2))
        d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
        if d < best:
            best, along = d, run + t * seg
        run += seg
    return best, along


def _parse(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _fmt(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def junctions_on_route(polyline: Sequence[LatLon], junctions: dict[str, LatLon],
                       radius_m: float = JUNCTION_RADIUS_M) -> list[tuple[str, float]]:
    """(junction_id, km along the route) for junctions within radius_m, in travel order."""
    if len(polyline) < 2:
        return []
    hits = []
    for jid, pos in junctions.items():
        off, along = project_onto_route(pos, polyline)
        if off * 1000.0 <= radius_m:
            hits.append((jid, along))
    return sorted(hits, key=lambda h: (h[1], h[0]))


def depart_time(now: str, ambulance_to_scene_min: float, on_scene_min: float = ON_SCENE_MIN) -> str:
    """When the ambulance leaves the scene for hospital."""
    return _fmt(_parse(now) + dt.timedelta(minutes=ambulance_to_scene_min + on_scene_min))


def plan_junction_posts(polyline: Sequence[LatLon], depart_at: str, route_eta_min: float,
                        junctions: dict[str, LatLon], signals: Optional[SignalState] = None,
                        radius_m: float = JUNCTION_RADIUS_M,
                        lead_s: float = OFFICER_LEAD_S) -> list[JunctionPost]:
    """Posts for corridor junctions on the scene->hospital route. The ambulance leaves the scene
    at depart_at and covers the route in route_eta_min at constant speed."""
    hits = junctions_on_route(polyline, junctions, radius_m)
    if not hits:
        return []
    total = route_length_km(polyline)
    t0 = _parse(depart_at)
    lost = {j.id for j in (signals.junctions if signals else [])
            if j.ambulance_eta_s is not None and not j.preempted}
    posts = []
    for jid, along in hits:
        frac = along / total if total > 0 else 0.0
        eta = t0 + dt.timedelta(minutes=route_eta_min * frac)
        lat, lon = junctions[jid]
        posts.append(JunctionPost(junction_id=jid, gps=GPS(lat=lat, lon=lon),
                                  ambulance_eta=_fmt(eta),
                                  officer_by=_fmt(eta - dt.timedelta(seconds=lead_s)),
                                  manual_clearing=jid in lost))
    return posts


# ---------------------------------------------------------------------------
# Alert lifecycle
# ---------------------------------------------------------------------------

@dataclass
class _Tracked:
    alert: PoliceAlert
    along_km: dict[str, float]      # junction id -> km along the route
    speed_kmh: float


class AlertBoard:
    """In-memory alert state per incident (one police dashboard, demo scale)."""

    def __init__(self, lead_s: float = OFFICER_LEAD_S, pass_margin_m: float = PASS_MARGIN_M,
                 off_route_m: float = OFF_ROUTE_M):
        self.lead_s, self.pass_margin_m, self.off_route_m = lead_s, pass_margin_m, off_route_m
        self._alerts: dict[str, _Tracked] = {}

    def __contains__(self, incident_id: str) -> bool:
        return incident_id in self._alerts

    def open(self, alert: PoliceAlert, route_eta_min: float) -> PoliceAlert:
        poly = alert.route_polyline
        along, speed = {}, 0.0
        if len(poly) >= 2:
            along = {p.junction_id: project_onto_route((p.gps.lat, p.gps.lon), poly)[1]
                     for p in alert.junction_posts}
            if route_eta_min > 0:
                speed = route_length_km(poly) / (route_eta_min / 60.0)
        self._alerts[alert.incident_id] = _Tracked(alert.model_copy(deep=True), along, speed)
        return self.get(alert.incident_id)

    def get(self, incident_id: str) -> PoliceAlert:
        return self._tracked(incident_id).alert.model_copy(deep=True)

    def acknowledge(self, incident_id: str, junction_id: Optional[str] = None) -> PoliceAlert:
        a = self._tracked(incident_id).alert
        if a.alert_status == "closed":
            raise ValueError("alert is closed")
        if junction_id is None:
            a.alert_status = "acknowledged"
        else:
            post = next((p for p in a.junction_posts if p.junction_id == junction_id), None)
            if post is None:
                raise KeyError(junction_id)
            if post.status == "pending":
                post.status = "acknowledged"
        return self.get(incident_id)

    def update_position(self, incident_id: str, lat: float, lon: float, now: str) -> PoliceAlert:
        """Release passed posts and re-time the rest from the ambulance's position."""
        tr = self._tracked(incident_id)
        a = tr.alert
        if a.alert_status == "closed" or len(a.route_polyline) < 2:
            return self.get(incident_id)
        off, pos = project_onto_route((lat, lon), a.route_polyline)
        if off * 1000.0 > self.off_route_m:
            return self.get(incident_id)
        t = _parse(now)
        for p in a.junction_posts:
            if p.status == "released":
                continue
            ahead = tr.along_km[p.junction_id] - pos
            if ahead * 1000.0 < -self.pass_margin_m:
                p.status = "released"
            elif tr.speed_kmh > 0:
                eta = t + dt.timedelta(hours=max(ahead, 0.0) / tr.speed_kmh)
                p.ambulance_eta = _fmt(eta)
                p.officer_by = _fmt(eta - dt.timedelta(seconds=self.lead_s))
        if a.junction_posts and all(p.status == "released" for p in a.junction_posts):
            a.alert_status = "closed"
        return self.get(incident_id)

    def close(self, incident_id: str) -> PoliceAlert:
        a = self._tracked(incident_id).alert
        for p in a.junction_posts:
            p.status = "released"
        a.alert_status = "closed"
        return self.get(incident_id)

    def _tracked(self, incident_id: str) -> _Tracked:
        if incident_id not in self._alerts:
            raise KeyError(incident_id)
        return self._alerts[incident_id]
