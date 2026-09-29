"""Tests for the SUMO corridor files and sumo/gen_routes.py (docs/03 §4-5, docs/08 §6)."""
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SUMO_DIR = Path(__file__).resolve().parents[1] / "sumo"
sys.path.insert(0, str(SUMO_DIR))
from gen_routes import DEMAND, JUNCTIONS, VEHICLE_MIX, build_routes  # noqa: E402


def test_demand_levels_within_docs08_ranges():
    for main_vph, cross_vph in DEMAND.values():
        assert 900 <= main_vph <= 1500
        assert 300 <= cross_vph <= 600


def test_vehicle_mix():
    assert VEHICLE_MIX == {"car": 0.6, "two_wheeler": 0.3, "bus_truck": 0.1}


@pytest.mark.parametrize("demand", list(DEMAND))
def test_routes_xml(demand):
    root = ET.fromstring(build_routes(demand, 300))
    main_vph, cross_vph = DEMAND[demand]
    flows = {f.get("id"): f for f in root.iter("flow")}
    assert len(flows) == 2 + 2 * len(JUNCTIONS)
    for fid, f in flows.items():
        rate = float(re.fullmatch(r"exp\((.+)\)", f.get("period")).group(1)) * 3600
        expected = main_vph if fid.startswith("f_main") else cross_vph
        assert rate == pytest.approx(expected, rel=1e-4)
    amb = root.find("vehicle[@id='ambulance']")
    assert amb.get("route") == "main_eb" and amb.get("depart") == "300"
    assert root.find("vType[@id='ambulance']").get("vClass") == "emergency"


@pytest.mark.parametrize("bad", [("rush", 300), ("medium", -1), ("medium", 1801)])
def test_invalid_args(bad):
    with pytest.raises(ValueError):
        build_routes(*bad)


def test_network_edges_and_lanes():
    root = ET.parse(SUMO_DIR / "corridor.net.xml").getroot()
    edges = {e.get("id"): e for e in root.iter("edge") if e.get("function") != "internal"}
    assert len(edges["silkboard_bellandur"].findall("lane")) == 3
    assert len(edges["bellandur_n_bellandur"].findall("lane")) == 2
    assert {t.get("id") for t in root.iter("tlLogic")} == set(JUNCTIONS)


@pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO not installed / not on PATH")
def test_ambulance_completes_corridor(tmp_path):
    routes = tmp_path / "r.rou.xml"
    routes.write_text(build_routes("low", 0))
    trip = tmp_path / "trip.xml"
    subprocess.run(
        ["sumo", "-n", str(SUMO_DIR / "corridor.net.xml"), "-r", str(routes), "--seed", "1",
         "--end", "2400", "--no-step-log", "--tripinfo-output", str(trip)],
        check=True, capture_output=True,
    )
    amb = ET.parse(trip).getroot().find("tripinfo[@id='ambulance']")
    assert amb is not None, "ambulance did not finish within 2400 s"
    assert float(amb.get("routeLength")) == pytest.approx(16594, abs=50)
