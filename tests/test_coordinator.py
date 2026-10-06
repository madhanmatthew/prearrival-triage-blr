"""Tests for the rule-based coordinator (docs/12 §05, six conflicts)."""
import pytest

from backend.agents import coordinator as co
from backend.schemas import (FusedRisk, HospitalRecommendation, JunctionState, RankedHospital,
                             ScoreBreakdown, SignalState)

NOW = "2026-10-06T10:10:00Z"
PROBS = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1]}


def risk(iid, sev):
    return FusedRisk.from_probs(PROBS[sev], incident_id=iid, patient_id="p", timestamp=NOW,
                                modalities_used=["text"], model_version="test")


def hosp(hid, score, eta, blood="matched"):
    bd = ScoreBreakdown(proximity=0, trauma_match=0, beds=0, specialty=0, blood_match=0)
    return RankedHospital(hospital_id=hid, name=f"Hosp {hid}", total_score=score, eta_min=eta,
                          breakdown=bd, blood_flag=blood)


def case(iid, sev=2, hospitals=None, etas=None, police=0, reported="2026-10-06T10:00:00Z",
         injury="road_accident", bleeding=False):
    hospitals = hospitals or [hosp("H1", 0.9, 10), hosp("H2", 0.8, 14), hosp("H3", 0.7, 25)]
    return co.IncidentCase(
        incident_id=iid, risk=risk(iid, sev), injury_type=injury, reported_at=reported,
        recommendation=HospitalRecommendation(incident_id=iid, ranked=hospitals),
        ambulance_etas_min=etas or {"A1": 5.0, "A2": 9.0}, police_units_required=police,
        bleeding_heavy=bleeding)


class FakeRouter:
    """Straight route through (0,0)-(0,10); detour via y=5 when asked to avoid anything."""

    def __init__(self, detour_extra=2.0, detour_ok=True):
        self.detour_extra, self.detour_ok, self.calls = detour_extra, detour_ok, []

    def route(self, incident_id, hospital_id, avoid=()):
        self.calls.append((incident_id, hospital_id, len(avoid)))
        if not avoid:
            return co.Route(f"R-{hospital_id}", [(0, 0), (0, 10)], 10.0)
        if not self.detour_ok:
            return None
        return co.Route(f"R-{hospital_id}-alt", [(0, 0), (5, 0), (5, 10), (0, 10)],
                        10.0 + self.detour_extra)


CORDON = co.Cordon("C1", [(-1, 4), (1, 4), (1, 6), (-1, 6)])  # straddles x=0 between y 4..6


def run(cases, **kw):
    kw.setdefault("router", FakeRouter())
    kw.setdefault("available_ambulances", ["A1", "A2"])
    kw.setdefault("police_units_available", 10)
    return co.coordinate(cases, now=NOW, **kw)


def types(decision):
    return [c.type for c in decision.conflicts_resolved]


# ---- geometry --------------------------------------------------------------

def test_route_crosses_geometry():
    sq = CORDON.polygon
    assert co.route_crosses([(0, 0), (0, 10)], sq)               # passes through
    assert co.route_crosses([(0, 5)], sq)                         # vertex inside
    assert not co.route_crosses([(5, 0), (5, 10)], sq)           # beside it
    assert co.route_crosses([(1, 0), (1, 10)], sq)               # runs along an edge (touches)
    assert not co.route_crosses([(0, 0), (0, 10)], sq[:2])       # not a polygon


# ---- no conflicts ----------------------------------------------------------

def test_clean_dispatch_no_conflicts():
    out = run([case("I1", police=2)])
    (d,) = out.decisions
    assert (d.ambulance_id, d.hospital_id, d.route_id, d.police_units) == ("A1", "H1", "R-H1", 2)
    assert d.conflicts_resolved == [] and out.police_escalated == 0
    assert "Rule-based" in d.explanation and "H1" in d.explanation


# ---- 1 cordon ----------------------------------------------------------------

def test_cordon_reroute_small_detour():
    (d,) = run([case("I1")], cordons=[CORDON]).decisions
    assert d.route_id == "R-H1-alt" and types(d) == ["cordon_on_route"]
    assert "dispatcher" not in d.conflicts_resolved[0].resolution


def test_cordon_big_detour_flags_dispatcher():
    out = run([case("I1")], cordons=[CORDON], router=FakeRouter(detour_extra=8.0))
    assert out.decisions[0].route_id == "R-H1-alt"
    assert any("detour adds 8.0" in r for r in out.resolutions["I1"].needs_dispatcher)


def test_cordon_no_detour_keeps_route_and_flags():
    out = run([case("I1")], cordons=[CORDON], router=FakeRouter(detour_ok=False))
    assert out.decisions[0].route_id == "R-H1"
    assert "no avoiding route" in out.decisions[0].conflicts_resolved[0].resolution


# ---- 2 beds ------------------------------------------------------------------

def test_bed_shortfall_switches_and_notifies():
    out = run([case("I1")], beds_now={"H1": 0, "H2": 3})
    d = out.decisions[0]
    assert d.hospital_id == "H2" and types(d) == ["hospital_bed_shortfall"]
    assert out.resolutions["I1"].notify_hospitals == ["H1"]


def test_no_beds_anywhere_keeps_top_and_flags():
    out = run([case("I1")], beds_now={"H1": 0, "H2": 0, "H3": 0})
    assert out.decisions[0].hospital_id == "H1"
    assert "no ranked hospital has a free bed" in out.resolutions["I1"].needs_dispatcher


def test_unknown_beds_count_as_available():
    assert run([case("I1")], beds_now={"H2": 0}).decisions[0].hospital_id == "H1"


# ---- 6 blood -----------------------------------------------------------------

def test_blood_short_switches_within_tolerance():
    hs = [hosp("H1", 0.9, 10, "short"), hosp("H2", 0.8, 14, "matched")]
    out = run([case("I1", sev=3, hospitals=hs)])
    d = out.decisions[0]
    assert d.hospital_id == "H2" and types(d) == ["blood_shortage"]
    assert out.resolutions["I1"].notify_hospitals == ["H1"]


def test_blood_short_no_alternative_in_time_keeps_and_flags():
    hs = [hosp("H1", 0.9, 10, "short"), hosp("H2", 0.8, 10 + co.BLOOD_MAX_EXTRA_MIN + 1)]
    out = run([case("I1", sev=3, hospitals=hs)])
    assert out.decisions[0].hospital_id == "H1"
    assert "arrange blood transfer" in out.resolutions["I1"].needs_dispatcher


def test_blood_short_ignored_without_transfusion_risk():
    hs = [hosp("H1", 0.9, 10, "short"), hosp("H2", 0.8, 14)]
    d = run([case("I1", sev=1, hospitals=hs, injury="cardiac_medical")]).decisions[0]
    assert d.hospital_id == "H1" and d.conflicts_resolved == []


def test_beds_then_blood_combined():
    hs = [hosp("H1", 0.9, 10), hosp("H2", 0.8, 12, "short"), hosp("H3", 0.7, 15)]
    out = run([case("I1", sev=3, hospitals=hs)], beds_now={"H1": 0})
    d = out.decisions[0]
    assert d.hospital_id == "H3" and types(d) == ["hospital_bed_shortfall", "blood_shortage"]
    assert out.resolutions["I1"].notify_hospitals == ["H1", "H2"]


# ---- 3 ambulances ------------------------------------------------------------

def test_ambulance_contention_higher_bid_wins():
    hi = case("I_hi", sev=3, etas={"A1": 5.0, "A2": 12.0})
    lo = case("I_lo", sev=1, etas={"A1": 4.0, "A2": 9.0})
    out = run([lo, hi])
    by = {d.incident_id: d for d in out.decisions}
    assert by["I_hi"].ambulance_id == "A1" and by["I_lo"].ambulance_id == "A2"
    assert types(by["I_lo"]) == ["ambulance_contention"] and types(by["I_hi"]) == []
    assert "+5.0 min" in by["I_lo"].conflicts_resolved[0].resolution


def test_ambulance_contention_loser_queued_when_none_left():
    out = run([case("I_hi", sev=3, etas={"A1": 5.0}), case("I_lo", sev=0, etas={"A1": 3.0})],
              available_ambulances=["A1"])
    assert [d.incident_id for d in out.decisions] == ["I_hi"]
    assert "queued" in out.resolutions["I_lo"].conflicts[0].resolution
    assert "no free ambulance" in out.resolutions["I_lo"].needs_dispatcher


def test_bid_formula_and_age_tiebreak():
    assert co.bid(1.0, 0.0, 10.0) == pytest.approx(1.0)
    assert co.bid(0.0, 30.0, 0.0) == pytest.approx(0.0)
    assert co.bid(0.5, 15.0, 5.0) == pytest.approx(0.3 + 0.125 + 0.075)
    old = case("I_old", sev=2, etas={"A1": 5.0}, reported="2026-10-06T09:55:00Z")
    new = case("I_new", sev=2, etas={"A1": 5.0}, reported="2026-10-06T10:09:00Z")
    out = run([new, old], available_ambulances=["A1"])
    assert out.decisions[0].incident_id == "I_old"


# ---- 4 police ----------------------------------------------------------------

def test_police_allocated_by_severity_and_escalated():
    cs = [case("I1", sev=1, police=3, etas={"A1": 5}), case("I3", sev=3, police=4, etas={"A2": 5})]
    out = run(cs, police_units_available=5)
    r = out.resolutions
    assert (r["I3"].police_units, r["I1"].police_units) == (4, 1)
    assert out.police_escalated == 2
    assert "2 escalated" in r["I1"].conflicts[0].resolution
    assert types(next(d for d in out.decisions if d.incident_id == "I3")) == ["police_units_short"]


def test_police_enough_units_no_conflict():
    out = run([case("I1", police=3)], police_units_available=3)
    assert out.decisions[0].police_units == 3 and out.decisions[0].conflicts_resolved == []


# ---- 5 junction preemption ---------------------------------------------------

def sig(iid, eta):
    return SignalState(incident_id=iid, route_id="ORR", junctions=[
        JunctionState(id="silkboard", phase="main_green", ambulance_eta_s=eta, preempted=True)])


def test_preemption_to_higher_risk_other_delayed():
    cs = [case("I_hi", sev=3, etas={"A1": 5}), case("I_lo", sev=1, etas={"A2": 5})]
    out = run(cs, signals=[sig("I_lo", 100.0), sig("I_hi", 110.0)])
    s = {x.incident_id: x.junctions[0].preempted for x in out.signals}
    assert s == {"I_hi": True, "I_lo": False}
    lo = next(d for d in out.decisions if d.incident_id == "I_lo")
    assert types(lo) == ["junction_preemption"] and "I_hi" in lo.conflicts_resolved[0].resolution


def test_preemption_no_conflict_when_far_apart():
    cs = [case("I_hi", sev=3, etas={"A1": 5}), case("I_lo", sev=1, etas={"A2": 5})]
    out = run(cs, signals=[sig("I_lo", 100.0), sig("I_hi", 100.0 + co.PREEMPT_WINDOW_S)])
    assert all(x.junctions[0].preempted for x in out.signals)
    assert all(d.conflicts_resolved == [] for d in out.decisions)


def test_signals_input_not_mutated():
    signals = [sig("I_lo", 100.0), sig("I_hi", 110.0)]
    run([case("I_hi", sev=3, etas={"A1": 5}), case("I_lo", sev=1, etas={"A2": 5})], signals=signals)
    assert all(s.junctions[0].preempted for s in signals)


# ---- determinism ---------------------------------------------------------------

def test_deterministic_regardless_of_input_order():
    cs = [case("I1", sev=2, police=3), case("I2", sev=2, police=3)]
    a = run(cs, police_units_available=4)
    b = run(list(reversed(cs)), police_units_available=4)
    key = lambda o: sorted((d.incident_id, d.ambulance_id, d.police_units) for d in o.decisions)
    assert key(a) == key(b)
