import pandas as pd
import pytest

from backend.agents import hospital_agent as ha
from backend.benchmarks import benchmark as bm


def hosp(rows):
    out = []
    for hid, lat, lon, level, beds, o_pos in rows:
        out.append({"hospital_id": hid, "name": hid, "lat": lat, "lon": lon, "trauma_level": level,
                    "beds_available": beds, "specialties": "trauma",
                    **{c: 10 for c in ha.BLOOD_COLS}, "blood_O+": o_pos})
    return pd.DataFrame(out)


def scen(**kw):
    row = dict(scenario_id="T1", name="t", lat=12.90, lon=77.60, injury_type="road_accident",
               severity_class=3, num_injured=1, bleeding_heavy=False, blood_group=None,
               expected_best_hospital_rule="", hospital_overrides={})
    row.update(kw)
    return pd.DataFrame([row])


AMB = {"A1": (12.90, 77.60)}


def test_naive_takes_nearest_ai_takes_capable():
    h = hosp([("NEAR", 12.901, 77.600, 1, 10, 10), ("FAR", 12.93, 77.60, 3, 10, 10)])
    r = bm.run(scen(), h, AMB).set_index("method")
    assert r.loc["naive", "hospital_id"] == "NEAR" and not r.loc["naive", "appropriate"]
    assert r.loc["ai", "hospital_id"] == "FAR" and r.loc["ai", "appropriate"]
    assert r.loc["ai", "total_min"] > r.loc["naive", "total_min"]


def test_blood_override_changes_ai_pick_only():
    h = hosp([("H_A", 12.9005, 77.600, 2, 10, 10), ("H_B", 12.9015, 77.600, 2, 10, 10)])
    sc = scen(severity_class=2, bleeding_heavy=True, blood_group="O+",
              hospital_overrides={"H_A": {"blood_O+": 0}})
    r = bm.run(sc, h, AMB).set_index("method")
    assert r.loc["naive", "hospital_id"] == "H_A" and not r.loc["naive", "blood_ok"]
    assert r.loc["ai", "hospital_id"] == "H_B" and r.loc["ai", "blood_ok"]


def test_fairness_same_ambulance_and_times_model():
    h = hosp([("H1", 12.95, 77.60, 3, 10, 10)])
    r = bm.run(scen(), h, {"A1": (12.90, 77.60), "A2": (13.0, 77.7)}).set_index("method")
    assert r.loc["naive", "ambulance_id"] == r.loc["ai", "ambulance_id"] == "A1"
    assert r.loc["naive", "total_min"] == r.loc["ai", "total_min"]       # same pick -> same time
    assert r.loc["ai", "response_min"] >= bm.DISPATCH_OVERHEAD_MIN


def test_unknown_override_hospital_rejected():
    with pytest.raises(KeyError):
        bm.apply_overrides(hosp([("H1", 12.9, 77.6, 3, 10, 10)]), {"NOPE": {"blood_O+": 0}})


def test_real_scenarios_run_and_blood_cases_discriminate():
    sc = bm.load_scenarios()
    assert len(sc) == 10 and set(sc.scenario_id) >= {"B1", "B2"}
    r = bm.run(sc, ha.load_hospitals(bm.HOSPITALS_PATH), bm.load_ambulances())
    b = r[r.scenario_id.isin(["B1", "B2"])].set_index(["scenario_id", "method"])
    for s in ("B1", "B2"):
        assert not b.loc[(s, "naive"), "blood_ok"] and b.loc[(s, "ai"), "blood_ok"]
    s = bm.summarize(r).set_index("method")
    assert s.loc["ai", "appropriate"] >= s.loc["naive", "appropriate"]
