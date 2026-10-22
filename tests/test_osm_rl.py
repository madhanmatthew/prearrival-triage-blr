"""Randomized OSM env + RL runner (short episodes with a tiny demand file)."""
import json

import numpy as np
import pandas as pd
import pytest

from backend.rl import demand as dm
from backend.rl import osm_rl
from backend.rl.osm_env import OSM_DIR


def tiny_files(tmp_path):
    edges = " ".join(json.loads((OSM_DIR / "orr_eastbound_route.json").read_text())["edges"])
    veh = "\n".join(f'  <vehicle id="c{i}" depart="{5 * i}" route="r"/>' for i in range(8))
    files = {}
    for slot in ("low", "medium", "peak"):
        p = tmp_path / f"{slot}.rou.xml"
        p.write_text(f'<routes>\n  <route id="r" edges="{edges}"/>\n{veh}\n</routes>\n')
        files[slot] = p
    return files


def test_randomized_env_jitters_scale_and_can_place_incident(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "INCIDENT_P", 1.0)
    env = dm.RandomizedOsmEnv(demand_files=tiny_files(tmp_path))
    try:
        scales = []
        for s in (1, 2):
            env.reset(seed=s)
            scales.append(env.scale)
            assert dm.DEMAND_JITTER[0] <= env.scale <= dm.DEMAND_JITTER[1]
            assert env.incident_lane and env._conn.lane.getMaxSpeed(env.incident_lane) == dm.INCIDENT_SPEED_MS
            assert env.demand in ("low", "medium", "peak")
        assert scales[0] != scales[1]
        env.reset(seed=1)
        assert env.scale == scales[0]                                     # reproducible
    finally:
        env.close()


def test_summarize_groups_policy_and_slot():
    df = pd.DataFrame([
        {"policy": "fixed_time", "slot": "peak", "seed": 1, "amb_completed": True, "amb_transit_s": 100.0,
         "amb_wait_s": 10, "amb_progress_m": 7700.0, "mean_general_wait_s": 2.0, "max_cross_queue_veh": 3.0,
         "throughput_veh": 50},
        {"policy": "fixed_time", "slot": "peak", "seed": 2, "amb_completed": False, "amb_transit_s": np.nan,
         "amb_wait_s": 30, "amb_progress_m": 5000.0, "mean_general_wait_s": 4.0, "max_cross_queue_veh": 5.0,
         "throughput_veh": 70}])
    s = osm_rl.summarize(df).iloc[0]
    assert s.episodes == 2 and s.completed == 1 and s.amb_progress_m == 6350.0 and s.amb_wait_s == 20.0
    assert s.amb_transit_s == 100.0                                       # mean over finished episodes
