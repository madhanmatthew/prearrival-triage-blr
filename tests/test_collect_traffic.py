"""Offline tests for the TomTom Flow / Routing collectors (no network, no keys)."""
import datetime as dt
import json

import pytest

from scripts import collect_tomtom_typical as g
from scripts import traffic_common as tc
from scripts import collect_tomtom as t
from scripts.traffic_common import POINTS_PATH, append_rows, load_points

CFG = json.loads(POINTS_PATH.read_text())


def test_unverified_config_is_refused():
    with pytest.raises(SystemExit):
        load_points()  # shipped config is unverified until the user checks it
    assert load_points(allow_unverified=True)["junctions"]


def test_tomtom_points_cover_each_junction_once():
    pts = t.collection_points(CFG)
    assert [p["point"] for p in pts] == ["silkboard", "bellandur", "marathahalli", "krpuram"]
    cfg = json.loads(json.dumps(CFG))
    cfg["segments"][0]["extra_points"] = [{"lat": 12.92, "lon": 77.65}]
    assert any(p["point"] == "silkboard_bellandur_x1" for p in t.collection_points(cfg))


def test_parse_flow():
    payload = {"flowSegmentData": {
        "frc": "FRC1", "currentSpeed": 18, "freeFlowSpeed": 52, "currentTravelTime": 300,
        "freeFlowTravelTime": 104, "confidence": 0.9, "roadClosure": False,
        "coordinates": {"coordinate": [{"latitude": 12.9, "longitude": 77.6}]}}}
    r = t.parse_flow(payload)
    assert r["current_speed_kmph"] == 18 and r["frc"] == "FRC1" and r["snapped_lat"] == 12.9
    assert t.parse_flow({"flowSegmentData": {}})["current_speed_kmph"] is None
    with pytest.raises(KeyError):
        t.parse_flow({})


def test_parse_route():
    r = g.parse_route({"routes": [{"summary": {
        "lengthInMeters": 5800, "travelTimeInSeconds": 1300,
        "noTrafficTravelTimeInSeconds": 900, "historicTrafficTravelTimeInSeconds": 1234}}]})
    assert r == {"historic_s": 1234, "no_traffic_s": 900, "travel_time_s": 1300, "length_m": 5800}
    with pytest.raises(KeyError):
        g.parse_route({"routes": [{"summary": {"lengthInMeters": 1}}]})


def test_next_departure_is_future_right_weekday_ist():
    now = dt.datetime(2026, 9, 30, 23, 30, tzinfo=tc.IST)  # a Wednesday
    for dow in range(7):
        for hour in (0, 12, 23):
            d = tc.next_departure(dow, hour, now)
            assert d > now + dt.timedelta(hours=1)
            assert d.weekday() == dow and d.hour == hour and d.minute == 0
            assert d - now <= dt.timedelta(days=8)


def test_jobs_count_and_reverse_swaps_endpoints():
    js = tc.segment_jobs(CFG)
    assert len(js) == 4 * 2 * 168
    f = next(j for j in js if j["segment"] == "silkboard_bellandur" and j["direction"] == "forward")
    r = next(j for j in js if j["segment"] == "silkboard_bellandur" and j["direction"] == "reverse")
    assert f["origin"] == r["dest"] and f["dest"] == r["origin"]


def test_build_url_shape():
    j = tc.segment_jobs(CFG)[0]
    url = g.build_url(j["origin"], j["dest"], [{"lat": 1.0, "lon": 2.0}],
                      dt.datetime(2026, 10, 5, 8, tzinfo=tc.IST), "KEY")
    assert "/calculateRoute/12.9177,77.6238:1.0,2.0:12.926,77.6762/json?" in url
    assert "departAt=2026-10-05T08%3A00%3A00%2B05%3A30" in url
    assert "traffic=true" in url and "computeTravelTimeFor=all" in url and "travelMode=car" in url


def test_pending_skips_cached_and_respects_limit():
    done = {("silkboard_bellandur", "forward", 0, 0)}
    todo = g.pending(CFG, done)
    assert len(todo) == 1343
    assert g.pending(CFG, done, 10)[0]["hour"] == 1
    assert len(g.pending(CFG, done, 10)) == 10


def test_append_and_resume_keys(tmp_path):
    p = tmp_path / "tomtom_typical.csv"
    row = {"collected_at": "x", "segment": "s", "direction": "forward", "dow": 2, "hour": 7,
           "historic_s": 1, "no_traffic_s": 1, "travel_time_s": 1, "length_m": 1}
    append_rows(p, g.COLUMNS, [row])
    append_rows(p, g.COLUMNS, [{**row, "hour": 8}])
    assert p.read_text().count("collected_at") == 1  # header once
    assert g.done_keys(p) == {("s", "forward", 2, 7), ("s", "forward", 2, 8)}
