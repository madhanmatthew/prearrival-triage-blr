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
