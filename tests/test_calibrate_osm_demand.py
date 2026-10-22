import xml.etree.ElementTree as ET

import pytest

from scripts import calibrate_osm_demand as cd


def test_mix_sums_to_one_and_vtypes_parse():
    assert sum(cd.MIX.values()) == pytest.approx(1.0)
    root = ET.fromstring("<additional>" + cd.vtypes_xml(0.8) + "</additional>")
    dist = root.find("vTypeDistribution")
    assert {v.get("id") for v in dist} == set(cd.MIX)
    assert all("normc(0.800" in v.get("speedFactor") for v in root.iter("vType"))


def test_build_demand_flows_and_probes(tmp_path):
    routes = {"eb": ["a", "b"], "wb": ["b2", "a2"], "silkboard_bellandur": ["a"],
              "bellandur_marathahalli": ["b"], "marathahalli_krpuram": ["c"], "silkboard_krpuram": ["a", "b", "c"]}
    root = ET.parse(cd.build_demand(tmp_path / "d.rou.xml", routes, 0.5, 0.8, 1)).getroot()
    flows = {f.get("id"): f for f in root.iter("flow")}
    assert float(flows["main_eb"].get("vehsPerHour")) == pytest.approx(cd.MAIN_VPH * 0.5 * cd.THROUGH_SHARE)
    assert flows["main_wb"].get("type") == "mix"
    local = {k for k in flows if k.startswith("local_")}
    assert local == {f"local_{leg}" for leg in cd.LEGS}                    # eb legs only (no *_wb routes given)
    assert float(flows["local_silkboard_bellandur"].get("vehsPerHour")) == pytest.approx(
        cd.MAIN_VPH * 0.5 * (1 - cd.THROUGH_SHARE))
    probes = [k for k in flows if k.startswith("probe_")]
    assert len(probes) == 4 and all(flows[k].get("begin") == str(cd.WARMUP_S) for k in probes)


def test_probe_speeds_from_tripinfo(tmp_path):
    p = tmp_path / "t.xml"
    p.write_text('<tripinfos><tripinfo id="probe_silkboard_bellandur.0" duration="600"/>'
                 '<tripinfo id="probe_silkboard_bellandur.1" duration="1200"/>'
                 '<tripinfo id="main_eb.3" duration="10"/></tripinfos>')
    s = cd.probe_speeds(p, {"silkboard_bellandur": 6000.0})
    assert s == {"silkboard_bellandur": pytest.approx(6000 / 900 * 3.6)}


def test_bisect_finds_monotone_target():
    log = []
    x, v = cd._bisect(lambda s: 60 / (1 + s), 0.0, 4.0, 20.0, increasing=False, iters=12, log=log)
    assert abs(v - 20) / 20 < 0.03 and x == pytest.approx(2.0, abs=0.15) and log
    x2, _ = cd._bisect(lambda f: 50 * f, 0.3, 1.2, 40.0, increasing=True, iters=12)
    assert x2 == pytest.approx(0.8, abs=0.03)


def test_flow_check_pcu(tmp_path):
    (tmp_path / "edgedata_x.xml").write_text('<meandata><interval><edge id="e1" left="900"/>'
                                             '<edge id="e2" left="1100"/><edge id="z" left="5"/></interval></meandata>')
    pcu = cd.flow_check(tmp_path, "x", ["e1", "e2"])
    per_veh = sum(cd.MIX[k] * cd.PCU[k] for k in cd.MIX)
    assert pcu == pytest.approx(1000 * 3600 / cd.MEASURE_S * per_veh)
    assert cd.flow_check(tmp_path, "missing", ["e1"]) is None


def test_leg_multiplier_scales_only_that_leg_both_directions(tmp_path):
    routes = {"eb": ["a"], "wb": ["b"], **{leg: ["a"] for leg in cd.LEGS},
              **{f"{leg}_wb": ["b"] for leg in cd.LEGS}}
    root = ET.parse(cd.build_demand(tmp_path / "d.rou.xml", routes, 1.0, 0.8, 1,
                                    leg_scale={"bellandur_marathahalli": 2.0})).getroot()
    vph = {f.get("id"): float(f.get("vehsPerHour")) for f in root.iter("flow") if f.get("vehsPerHour")}
    base = cd.MAIN_VPH * (1 - cd.THROUGH_SHARE)
    assert vph["local_bellandur_marathahalli"] == pytest.approx(2 * base)
    assert vph["local_bellandur_marathahalli_wb"] == pytest.approx(2 * base)
    assert vph["local_silkboard_bellandur"] == pytest.approx(base)
    assert vph["main_eb"] == pytest.approx(cd.MAIN_VPH * cd.THROUGH_SHARE)


def test_speed_fallback_vss_and_initial_limits():
    import pandas as pd
    from scripts import calibrate_osm_speed as cs
    xml = cs.vss_xml({"silkboard_bellandur": ["e1_0", "e1_1"]}, {"silkboard_bellandur": 10.0})
    root = ET.fromstring(xml)
    v = root.find("variableSpeedSign")
    assert v.get("lanes") == "e1_0 e1_1" and v.find("step").get("speed") == "10.000"
    t = pd.DataFrame([{"slot": "peak", "segment": leg, "typical_speed_kmh": 27.0} for leg in cd.LEGS])
    lim = cs.initial_limits(t, "peak")
    assert all(x == pytest.approx(27.0 / 3.6 / cs.SPEED_FACTOR) for x in lim.values())
