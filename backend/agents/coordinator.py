"""Rule-based coordinator: resolves the 6 conflicts of docs/12 §05 and issues DispatchDecisions.

Intentionally rule-based (AGENTS rule 8, docs/05 §5): an RL coordinator was scoped out as too
ambitious to build and validate in the timeline. Every decision is a fixed, inspectable rule,
and every rule that fired is listed in `DispatchDecision.conflicts_resolved` and `explanation`
(decision support for a human dispatcher, not automation).

Conflicts (docs/12 §05 table):
  1. cordon_on_route          route polyline crosses a police cordon polygon -> re-request a route
                              avoiding it; flag the dispatcher if ETA grows > CORDON_MAX_EXTRA_MIN.
  2. hospital_bed_shortfall   chosen hospital has 0 beds -> next-ranked hospital with beds; notify
                              the original hospital.
  3. ambulance_contention     two incidents want the same ambulance -> priority bid
                              (risk, ETA, incident age); higher bid wins, loser takes its next one.
  4. police_units_short       units requested > available -> allocate by severity priority,
                              escalate the remainder.
  5. junction_preemption      two ambulances reach one junction within PREEMPT_WINDOW_S ->
                              preempt for the higher risk_score, delay the other.
  6. blood_shortage           top hospital blood_flag == "short" and transfusion risk -> best
                              non-short hospital within BLOOD_MAX_EXTRA_MIN, else keep and flag.

Inputs come from the other agents (FusedRisk, HospitalRecommendation, SignalState, PoliceAlert
sizes) plus live state passed in by the caller (beds, ambulance ETAs, cordons). Routing (ORS) is
injected as a `Router`, so this module makes no network calls (docs/07 D5: in-process agents).

[ASSUMPTION]s (no values exist in the docs; fixed before any benchmark run):
  * CORDON_MAX_EXTRA_MIN = 5, BLOOD_MAX_EXTRA_MIN = 10, PREEMPT_WINDOW_S = 30.
  * Bid = 0.6 * risk_score + 0.25 * (1 - min(eta / 30 min, 1)) + 0.15 * min(age / 10 min, 1).
    Ties: earlier incident, then incident_id (deterministic).
  * Police priority: severity_class, then risk_score, then age; partial allocation allowed.
  * Junction conflict = overlapping ETAs, whatever the approach direction (SignalState has no
    approach field, docs/07 §2.5); conservative, may delay an ambulance that could share green.
  * Hospitals missing from `beds_now` count as having beds (no live data = no evidence of 0).
  * Coordinates are (lat, lon) treated as planar; fine for city-scale polygons.
Research prototype, not a medical device.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

from backend.agents.hospital_agent import blood_needed
from backend.schemas import (ConflictResolution, DispatchDecision, FusedRisk,
                             HospitalRecommendation, RankedHospital, SignalState)

CORDON_MAX_EXTRA_MIN = 5.0
BLOOD_MAX_EXTRA_MIN = 10.0
PREEMPT_WINDOW_S = 30.0
BID_W_RISK, BID_W_ETA, BID_W_AGE = 0.6, 0.25, 0.15
BID_ETA_CAP_MIN, BID_AGE_CAP_MIN = 30.0, 10.0

LatLon = tuple[float, float]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class Route:
    route_id: str
    polyline: list[LatLon]
    eta_min: float
    estimated: bool = False          # True when ORS was down (straight-line fallback, docs/07 §6)


@dataclass
class Cordon:
    cordon_id: str
    polygon: list[LatLon]


class Router(Protocol):
    def route(self, incident_id: str, hospital_id: str,
              avoid: Sequence[list[LatLon]] = ()) -> Optional[Route]: ...


@dataclass
class IncidentCase:
    incident_id: str
    risk: FusedRisk
    injury_type: str
    reported_at: str                                   # ISO-8601 UTC
    recommendation: HospitalRecommendation
    ambulance_etas_min: dict[str, float]               # ambulance_id -> ETA to scene (ORS)
    police_units_required: int = 0
    bleeding_heavy: bool = False


@dataclass
class Resolution:
    """Coordinator output for one incident before it becomes a DispatchDecision."""
    incident_id: str
    ambulance_id: Optional[str] = None
    hospital: Optional[RankedHospital] = None
    route: Optional[Route] = None
    police_units: int = 0
    conflicts: list[ConflictResolution] = field(default_factory=list)
    notify_hospitals: list[str] = field(default_factory=list)   # hospitals to un-alert
    needs_dispatcher: list[str] = field(default_factory=list)   # reasons a human must look


# ---------------------------------------------------------------------------
# Geometry (conflict 1)
# ---------------------------------------------------------------------------

def _point_in_polygon(p: LatLon, poly: Sequence[LatLon]) -> bool:
    x, y = p
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _segments_cross(a: LatLon, b: LatLon, c: LatLon, d: LatLon) -> bool:
    def orient(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if abs(v) < 1e-15 else (1 if v > 0 else -1)

    def on_seg(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
                and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and on_seg(a, c, b)) or (o2 == 0 and on_seg(a, d, b))
            or (o3 == 0 and on_seg(c, a, d)) or (o4 == 0 and on_seg(c, b, d)))


def route_crosses(polyline: Sequence[LatLon], polygon: Sequence[LatLon]) -> bool:
    """True if any route vertex lies inside the polygon or any route segment crosses an edge."""
    if len(polygon) < 3 or not polyline:
        return False
    if any(_point_in_polygon(p, polygon) for p in polyline):
        return True
    edges = [(polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon))]
    return any(_segments_cross(polyline[i], polyline[i + 1], c, d)
               for i in range(len(polyline) - 1) for c, d in edges)


# ---------------------------------------------------------------------------
# Per-incident rules: hospital (conflicts 2, 6) and route (conflict 1)
# ---------------------------------------------------------------------------

def choose_hospital(case: IncidentCase, beds_now: dict[str, int], res: Resolution) -> None:
    ranked = case.recommendation.ranked
    top = ranked[0]
    with_beds = [h for h in ranked if beds_now.get(h.hospital_id, 1) > 0]
    if not with_beds:
        res.hospital = top
        res.needs_dispatcher.append("no ranked hospital has a free bed")
        res.conflicts.append(ConflictResolution(
            type="hospital_bed_shortfall",
            resolution=f"no ranked hospital has beds; kept {top.hospital_id}, dispatcher alerted"))
        return
    chosen = with_beds[0]
    if chosen.hospital_id != top.hospital_id:
        rank = ranked.index(chosen) + 1
        res.notify_hospitals.append(top.hospital_id)
        res.conflicts.append(ConflictResolution(
            type="hospital_bed_shortfall",
            resolution=f"{top.hospital_id} has 0 beds; switched to rank-{rank} hospital "
                       f"{chosen.hospital_id}; {top.hospital_id} notified"))

    need_blood = blood_needed(case.risk.severity_class, case.injury_type, case.bleeding_heavy)
    if need_blood and chosen.blood_flag == "short":
        alt = next((h for h in with_beds if h.blood_flag != "short"
                    and h.eta_min <= chosen.eta_min + BLOOD_MAX_EXTRA_MIN), None)
        if alt is not None:
            res.conflicts.append(ConflictResolution(
                type="blood_shortage",
                resolution=f"{chosen.hospital_id} short of blood; chose {alt.hospital_id} "
                           f"(+{alt.eta_min - chosen.eta_min:.1f} min, blood {alt.blood_flag})"))
            if chosen.hospital_id not in res.notify_hospitals:
                res.notify_hospitals.append(chosen.hospital_id)
            chosen = alt
        else:
            res.needs_dispatcher.append("arrange blood transfer")
            res.conflicts.append(ConflictResolution(
                type="blood_shortage",
                resolution=f"{chosen.hospital_id} short of blood and no alternative within "
                           f"{BLOOD_MAX_EXTRA_MIN:.0f} min; kept, blood transfer requested"))
    res.hospital = chosen


def resolve_route(case: IncidentCase, res: Resolution, router: Router,
                  cordons: Sequence[Cordon]) -> None:
    route = router.route(case.incident_id, res.hospital.hospital_id)
    if route is None:
        res.needs_dispatcher.append("no route available")
        return
    hit = [c for c in cordons if route_crosses(route.polyline, c.polygon)]
    if hit:
        ids = ", ".join(c.cordon_id for c in hit)
        alt = router.route(case.incident_id, res.hospital.hospital_id,
                           avoid=[c.polygon for c in hit])
        if alt is None or any(route_crosses(alt.polyline, c.polygon) for c in hit):
            res.needs_dispatcher.append(f"route crosses cordon {ids}, no detour found")
            res.conflicts.append(ConflictResolution(
                type="cordon_on_route",
                resolution=f"route crosses {ids}; no avoiding route, kept original, dispatcher alerted"))
        else:
            extra = alt.eta_min - route.eta_min
            text = f"route crossed {ids}; re-routed via {alt.route_id} (+{extra:.1f} min)"
            if extra > CORDON_MAX_EXTRA_MIN:
                res.needs_dispatcher.append(f"cordon detour adds {extra:.1f} min")
                text += ", dispatcher alerted"
            res.conflicts.append(ConflictResolution(type="cordon_on_route", resolution=text))
            route = alt
    res.route = route


# ---------------------------------------------------------------------------
# Cross-incident rules: ambulances (3), police (4), junctions (5)
# ---------------------------------------------------------------------------

def _age_min(reported_at: str, now: str) -> float:
    parse = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    return max((parse(now) - parse(reported_at)).total_seconds() / 60.0, 0.0)


def bid(risk_score: float, eta_min: float, age_min: float) -> float:
    return (BID_W_RISK * risk_score
            + BID_W_ETA * (1.0 - min(eta_min / BID_ETA_CAP_MIN, 1.0))
            + BID_W_AGE * min(age_min / BID_AGE_CAP_MIN, 1.0))


def allocate_ambulances(cases: Sequence[IncidentCase], available: Sequence[str], now: str,
                        res: dict[str, Resolution]) -> None:
    """Greedy by bid: each incident, highest bid first, takes its fastest free ambulance."""
    free = set(available)
    order = []
    for c in cases:
        etas = {a: e for a, e in c.ambulance_etas_min.items() if a in free}
        best = min(etas, key=lambda a: (etas[a], a)) if etas else None
        b = bid(c.risk.risk_score, etas[best], _age_min(c.reported_at, now)) if best else -1.0
        order.append((-b, c.reported_at, c.incident_id, c, best, b))
    order.sort(key=lambda t: t[:3])
    taken_by: dict[str, tuple[str, float]] = {}
    for _, _, iid, c, wanted, b in order:
        etas = {a: e for a, e in c.ambulance_etas_min.items() if a in free}
        if not etas:
            res[iid].needs_dispatcher.append("no free ambulance")
            if wanted in taken_by:
                w_id, w_bid = taken_by[wanted]
                res[iid].conflicts.append(ConflictResolution(
                    type="ambulance_contention",
                    resolution=f"{wanted} went to {w_id} (bid {w_bid:.3f} > {b:.3f}); "
                               "no other ambulance free, queued"))
            continue
        got = min(etas, key=lambda a: (etas[a], a))
        if wanted is not None and got != wanted and wanted in taken_by:
            w_id, w_bid = taken_by[wanted]
            res[iid].conflicts.append(ConflictResolution(
                type="ambulance_contention",
                resolution=f"{wanted} went to {w_id} (bid {w_bid:.3f} > {b:.3f}); "
                           f"assigned {got} (+{etas[got] - c.ambulance_etas_min[wanted]:.1f} min)"))
        free.discard(got)
        taken_by[got] = (iid, b)
        res[iid].ambulance_id = got


def allocate_police(cases: Sequence[IncidentCase], units_available: int, now: str,
                    res: dict[str, Resolution]) -> int:
    """Allocate by severity priority; returns units escalated (still missing)."""
    order = sorted(cases, key=lambda c: (-c.risk.severity_class, -c.risk.risk_score,
                                         -_age_min(c.reported_at, now), c.incident_id))
    left, escalated = units_available, 0
    short = sum(c.police_units_required for c in cases) > units_available
    for c in order:
        give = min(c.police_units_required, left)
        left -= give
        res[c.incident_id].police_units = give
        missing = c.police_units_required - give
        if missing:
            escalated += missing
            res[c.incident_id].needs_dispatcher.append(f"{missing} police units escalated")
        if short and c.police_units_required:
            res[c.incident_id].conflicts.append(ConflictResolution(
                type="police_units_short",
                resolution=f"allocated {give}/{c.police_units_required} units by severity priority"
                           + (f"; {missing} escalated" if missing else "")))
    return escalated


def resolve_preemption(signals: Sequence[SignalState], risk_by_incident: dict[str, float],
                       res: dict[str, Resolution]) -> list[SignalState]:
    """Per junction, ambulances arriving within PREEMPT_WINDOW_S of a higher-risk one lose
    preemption there. Returns updated SignalStates (copies)."""
    out = [s.model_copy(deep=True) for s in signals]
    by_junction: dict[str, list[tuple[SignalState, int]]] = {}
    for s in out:
        for k, j in enumerate(s.junctions):
            if j.ambulance_eta_s is not None:
                by_junction.setdefault(j.id, []).append((s, k))
    for jid, entries in by_junction.items():
        entries.sort(key=lambda e: (-risk_by_incident.get(e[0].incident_id, 0.0),
                                    e[0].junctions[e[1]].ambulance_eta_s, e[0].incident_id))
        winners: list[tuple[str, float]] = []
        for s, k in entries:
            eta = s.junctions[k].ambulance_eta_s
            clash = next((w for w in winners if abs(w[1] - eta) < PREEMPT_WINDOW_S), None)
            if clash is None:
                s.junctions[k].preempted = True
                winners.append((s.incident_id, eta))
                continue
            s.junctions[k].preempted = False
            text = (f"{jid}: preemption given to {clash[0]} (higher risk); this ambulance "
                    f"delayed up to {PREEMPT_WINDOW_S:.0f} s")
            if s.incident_id in res:
                res[s.incident_id].conflicts.append(
                    ConflictResolution(type="junction_preemption", resolution=text))
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _explain(case: IncidentCase, r: Resolution) -> str:
    h = r.hospital
    parts = [f"Severity {case.risk.severity_class} (risk {case.risk.risk_score:.2f})."]
    if h is not None:
        parts.append(f"Hospital {h.name} ({h.hospital_id}): score {h.total_score:.2f}, "
                     f"ETA {h.eta_min:.0f} min, blood {h.blood_flag}.")
    if r.ambulance_id:
        parts.append(f"Ambulance {r.ambulance_id}, "
                     f"{case.ambulance_etas_min[r.ambulance_id]:.0f} min to scene.")
    parts.append(f"Police units {r.police_units}.")
    parts += [f"{c.type}: {c.resolution}." for c in r.conflicts]
    if r.needs_dispatcher:
        parts.append("Dispatcher review: " + "; ".join(r.needs_dispatcher) + ".")
    parts.append("Rule-based decision support; a human dispatcher confirms.")
    return " ".join(parts)


@dataclass
class CoordinatorResult:
    decisions: list[DispatchDecision]
    resolutions: dict[str, Resolution]
    signals: list[SignalState]
    police_escalated: int


def coordinate(cases: Sequence[IncidentCase], *, now: str, router: Router,
               available_ambulances: Sequence[str], police_units_available: int,
               beds_now: Optional[dict[str, int]] = None, cordons: Sequence[Cordon] = (),
               signals: Sequence[SignalState] = ()) -> CoordinatorResult:
    """Run all six rules for the active incidents and return one DispatchDecision each.

    Incidents with no free ambulance get no DispatchDecision (queued; see `resolutions`).
    """
    beds_now = beds_now or {}
    res = {c.incident_id: Resolution(c.incident_id) for c in cases}
    allocate_ambulances(cases, available_ambulances, now, res)
    escalated = allocate_police(cases, police_units_available, now, res)
    for c in cases:
        choose_hospital(c, beds_now, res[c.incident_id])
        resolve_route(c, res[c.incident_id], router, cordons)
    new_signals = resolve_preemption(signals, {c.incident_id: c.risk.risk_score for c in cases}, res)

    decisions = []
    for c in cases:
        r = res[c.incident_id]
        if r.ambulance_id is None or r.route is None:
            continue
        decisions.append(DispatchDecision(
            incident_id=c.incident_id, ambulance_id=r.ambulance_id,
            hospital_id=r.hospital.hospital_id, route_id=r.route.route_id,
            police_units=r.police_units, conflicts_resolved=r.conflicts,
            explanation=_explain(c, r), timestamp=now, route_estimated=r.route.estimated))
    return CoordinatorResult(decisions, res, new_signals, escalated)
