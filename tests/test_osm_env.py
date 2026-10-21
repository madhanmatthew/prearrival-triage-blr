"""OSM corridor (docs/15 D21): tls_map, route, 2-phase states and a short env episode."""
import json
import os
import shutil

import numpy as np
import pytest

from backend.rl.osm_env import OSM_DIR, OsmCorridorEnv, load_tls_map
from scripts import build_osm_corridor as bo

pytestmark = pytest.mark.skipif(shutil.which("sumo") is None and "SUMO_HOME" not in os.environ
                                and not (OSM_DIR / "corridor.net.xml").exists(), reason="no SUMO / OSM net")


def test_tls_map_four_junctions_in_route_order():
    tm = load_tls_map()
    assert list(tm) == bo.NAMES
    offs = [j["route_offset_m"] for j in tm.values()]
    assert offs == sorted(offs) and offs[-1] < 7000                     # all on the Silk Board -> Iblur stretch
    for j in tm.values():
        assert not set(j["main_links"]) & set(j["cross_links"]) and j["cross_links"]
        for st in j["states"].values():
            assert len(st) == j["n_links"]
        assert all(j["states"]["main_green"][k] == "g" for k in j["main_links"])
        assert all(j["states"]["main_green"][k] == "r" for k in j["cross_links"])
        assert all(j["states"]["cross_yellow"][k] == "y" for k in j["cross_links"])


def test_states_helper():
    s = bo.states({"n_links": 4, "main_links": [0, 1], "cross_links": [3]})
    assert s == {"main_green": "ggrr", "main_yellow": "yyrr", "cross_green": "rrrg", "cross_yellow": "rrry"}


def test_route_matches_tomtom_length():
    r = json.loads((OSM_DIR / "orr_eastbound_route.json").read_text())
    assert 15500 < r["length_m"] < 18500                                  # TomTom 16.8 km + 0.9 km lead-in


def tiny_demand(path):
    edges = " ".join(json.loads((OSM_DIR / "orr_eastbound_route.json").read_text())["edges"])
    veh = "\n".join(f'  <vehicle id="c{i}" depart="{10 * i}" route="r"/>' for i in range(5))
    path.write_text(f'<routes>\n  <route id="r" edges="{edges}"/>\n{veh}\n</routes>\n')
    return path


@pytest.mark.parametrize("control", ["rl", "fixed"])
def test_episode_runs_and_ambulance_arrives(tmp_path, control):
    env = OsmCorridorEnv("medium", control=control, demand_files={"medium": tiny_demand(tmp_path / "d.rou.xml")})
    try:
        obs, _ = env.reset(seed=3)
        assert obs.shape == (24,) and env.action_space.n == 16
        assert all(len(env._lanes[j][0]) and len(env._lanes[j][1]) for j in env.junctions)
        done, info = False, {}
        while not done:
            obs, r, term, trunc, info = env.step(0)
            assert np.isfinite(r) and ((obs >= 0) & (obs <= 1)).all()
            done = term or trunc
        assert info["amb_completed"] and info["amb_transit_s"] > 0
    finally:
        env.close()


def test_missing_demand_file_has_hint(tmp_path):
    env = OsmCorridorEnv("low", demand_files={"low": tmp_path / "nope.rou.xml"})
    try:
        with pytest.raises(FileNotFoundError, match="fit demand"):
            env.reset(seed=0)
    finally:
        env.close()
