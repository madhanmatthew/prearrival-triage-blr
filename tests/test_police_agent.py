import pytest

from backend.agents.police_agent import MAX_PERIMETER_M, MAX_UNITS, size_alert
from backend.schemas import GPS

G = GPS(lat=12.9, lon=77.6)


def alert(sev, typ="road_accident", n=1, poly=()):
    return size_alert("I", G, sev, typ, n, poly)


def test_stable_needs_no_police():
    a = alert(0)
    assert (a.units_required, a.perimeter_m) == (0, 0.0)


def test_units_and_perimeter_grow_with_severity():
    u = [alert(s).units_required for s in range(4)]
    p = [alert(s).perimeter_m for s in range(4)]
    assert u == sorted(u) and p == sorted(p) and u[3] > u[1] and p[3] > p[1]


def test_road_accident_gets_extra_unit_vs_fall():
    assert alert(2, "road_accident").units_required == alert(2, "fall").units_required + 1


def test_more_injured_more_units_and_perimeter():
    assert alert(2, n=6).units_required > alert(2, n=2).units_required
    assert alert(2, n=6).perimeter_m > alert(2, n=2).perimeter_m


def test_medical_incident_has_escort_but_no_cordon():
    a = alert(3, "cardiac_medical")
    assert a.units_required >= 1 and a.perimeter_m == 0.0


def test_caps():
    a = alert(3, "road_accident", n=50)
    assert a.units_required == MAX_UNITS and a.perimeter_m == MAX_PERIMETER_M


def test_polyline_is_passed_through():
    assert alert(2, poly=[(12.9, 77.6), (12.95, 77.65)]).route_polyline == [(12.9, 77.6), (12.95, 77.65)]


def test_negative_injured_is_treated_as_zero():
    assert alert(2, n=-3).units_required == alert(2, n=0).units_required


@pytest.mark.parametrize("sev,typ", [(4, "fall"), (-1, "fall"), (1, "alien")])
def test_bad_input_rejected(sev, typ):
    with pytest.raises(ValueError):
        alert(sev, typ)


# ---------------------------------------------------------------------------
# Junction posts + lifecycle
# ---------------------------------------------------------------------------

from backend.agents import police_agent as pa  # noqa: E402
from backend.schemas import JunctionState, SignalState  # noqa: E402

J = {"silkboard": (12.9177, 77.6238), "bellandur": (12.9258, 77.6761),
     "marathahalli": (12.9567, 77.7010), "krpuram": (13.0002, 77.6770)}
# scene west of Silk Board -> along the ORR -> hospital past Marathahalli
ROUTE = [(12.9100, 77.6100), J["silkboard"], J["bellandur"], J["marathahalli"], (12.9600, 77.7000)]


def test_real_corridor_file_loads():
    j = pa.load_corridor_junctions()
    assert set(j) == {"silkboard", "bellandur", "marathahalli", "krpuram"}


def test_projection_on_and_off_route():
    off, along = pa.project_onto_route(J["silkboard"], ROUTE)
    assert off < 1e-6 and along == pytest.approx(pa.route_length_km(ROUTE[:2]))
    off2, _ = pa.project_onto_route((12.80, 77.60), ROUTE)
    assert off2 > 5


def test_junctions_on_route_in_travel_order_and_skips_far_ones():
    hits = [j for j, _ in pa.junctions_on_route(ROUTE, J)]
    assert hits == ["silkboard", "bellandur", "marathahalli"]          # KR Puram not on route
    assert [j for j, _ in pa.junctions_on_route(list(reversed(ROUTE)), J)] == hits[::-1]
    assert pa.junctions_on_route([(12.80, 77.50), (12.81, 77.51)], J) == []


def test_posts_timing():
    posts = pa.plan_junction_posts(ROUTE, "2026-10-21T09:00:00Z", 30.0, J)
    etas = [p.ambulance_eta for p in posts]
    assert etas == sorted(etas) and all(p.status == "pending" for p in posts)
    p0 = posts[0]
    gap = pa._parse(p0.ambulance_eta) - pa._parse(p0.officer_by)
    assert gap.total_seconds() == pa.OFFICER_LEAD_S
    # last post ETA = depart + 30 min * share of route length up to Marathahalli
    share = pa.junctions_on_route(ROUTE, J)[-1][1] / pa.route_length_km(ROUTE)
    expect = pa._parse("2026-10-21T09:00:00Z") + pa.dt.timedelta(minutes=30 * share)
    assert abs((pa._parse(posts[-1].ambulance_eta) - expect).total_seconds()) <= 1


def test_manual_clearing_where_preemption_lost():
    sig = SignalState(incident_id="I", route_id="R", junctions=[
        JunctionState(id="silkboard", phase="main_green", ambulance_eta_s=60, preempted=True),
        JunctionState(id="bellandur", phase="cross_green", ambulance_eta_s=400, preempted=False),
        JunctionState(id="krpuram", phase="cross_green", ambulance_eta_s=None, preempted=False)])
    posts = {p.junction_id: p.manual_clearing
             for p in pa.plan_junction_posts(ROUTE, "2026-10-21T09:00:00Z", 30.0, J, signals=sig)}
    assert posts == {"silkboard": False, "bellandur": True, "marathahalli": False}


def test_depart_time_adds_travel_and_scene_time():
    assert pa.depart_time("2026-10-21T09:00:00Z", 8.0) == "2026-10-21T09:18:00Z"


def _board():
    alert = size_alert("I", G, 3, "road_accident", 2, ROUTE).model_copy(
        update={"junction_posts": pa.plan_junction_posts(ROUTE, "2026-10-21T09:00:00Z", 30.0, J)})
    b = pa.AlertBoard()
    b.open(alert, 30.0)
    return b


def test_acknowledge_alert_and_post():
    b = _board()
    assert b.acknowledge("I").alert_status == "acknowledged"
    a = b.acknowledge("I", "bellandur")
    assert {p.junction_id: p.status for p in a.junction_posts}["bellandur"] == "acknowledged"
    with pytest.raises(KeyError):
        b.acknowledge("I", "krpuram")
    with pytest.raises(KeyError):
        b.get("nope")


def test_position_releases_passed_posts_and_retimes_rest():
    b = _board()
    before = {p.junction_id: p.ambulance_eta for p in b.get("I").junction_posts}
    a = b.update_position("I", 12.9230, 77.6500, "2026-10-21T09:05:00Z")   # between SB and Bellandur
    st = {p.junction_id: p for p in a.junction_posts}
    assert st["silkboard"].status == "released"
    assert st["bellandur"].status == "pending" and st["bellandur"].ambulance_eta != before["bellandur"]
    assert st["bellandur"].ambulance_eta < st["marathahalli"].ambulance_eta
    assert a.alert_status == "active"
    a = b.update_position("I", 12.9600, 77.7000, "2026-10-21T09:20:00Z")   # at hospital end
    assert all(p.status == "released" for p in a.junction_posts) and a.alert_status == "closed"


def test_off_route_position_changes_nothing():
    b = _board()
    before = b.get("I")
    assert b.update_position("I", 12.80, 77.50, "2026-10-21T09:05:00Z") == before


def test_close_releases_everything_and_blocks_ack():
    b = _board()
    a = b.close("I")
    assert a.alert_status == "closed" and all(p.status == "released" for p in a.junction_posts)
    with pytest.raises(ValueError):
        b.acknowledge("I")


def test_alert_without_route_has_no_posts():
    b = pa.AlertBoard()
    a = b.open(size_alert("X", G, 2, "fall", 1), 0.0)
    assert a.junction_posts == [] and b.update_position("X", 12.9, 77.6, "2026-10-21T09:00:00Z") == a
