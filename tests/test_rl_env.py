"""Tests for the corridor RL env and the traffic baselines (runs real SUMO; ~1 min)."""
import math

import numpy as np
import pytest

pytest.importorskip("traci")

from backend.rl.baselines import POLICIES, run_episode, summarise  # noqa: E402
from backend.rl.env import JUNCTIONS, MAIN_GREEN, CorridorEnv  # noqa: E402


def test_spaces_and_bad_args():
    env = CorridorEnv(control="fixed")
    assert env.action_space.n == 16
    assert env.observation_space.shape == (24,)
    env.close()
    with pytest.raises(ValueError):
        CorridorEnv(control="nope")
    with pytest.raises(ValueError):
        CorridorEnv(demand="huge")


def test_reset_obs_step_and_determinism():
    runs = []
    for _ in range(2):
        env = CorridorEnv(demand="low", control="rl")
        obs, _ = env.reset(seed=7)
        assert obs.shape == (24,) and obs.dtype == np.float32
        seq = [obs]
        for a in [0, 15, 3, 12, 0, 5, 10, 0]:
            obs, r, term, trunc, _ = env.step(a)
            assert env.observation_space.contains(obs)
            assert r <= 0 and not term and not trunc
            seq.append(obs)
        runs.append(np.stack(seq))
        env.close()
    assert np.array_equal(runs[0], runs[1])
    # no ambulance yet: flag 0, distance 1.0; phase one-hot is main at start
    first = runs[0][0].reshape(4, 6)
    assert (first[:, 4] == 0).all() and (first[:, 5] == 1).all()
    assert (first[:, 2] == 1).all() and (first[:, 3] == 0).all()


def test_rl_respects_min_green_and_yellow():
    env = CorridorEnv(demand="low", control="rl")  # min_green 10, yellow 3
    env.reset(seed=1)
    env.step(15)  # want cross everywhere; only 5 s elapsed < min_green
    assert all(env._phase[j] == MAIN_GREEN for j in JUNCTIONS)
    env.step(15)  # 10 s elapsed -> yellow starts
    assert all(env._phase[j] == 1 for j in JUNCTIONS)
    env.step(15)  # yellow (3 s) done -> cross green
    assert all(env._phase[j] == 2 for j in JUNCTIONS)
    env.close()


def test_preempt_episode_completes_with_metrics():
    env = CorridorEnv(demand="low", control=POLICIES["always_green"])
    info = run_episode(env, seed=3)
    env.close()
    assert info["amb_completed"] and math.isfinite(info["amb_transit_s"])
    assert 200 <= info["amb_depart_s"] <= 600
    assert info["throughput_veh"] > 0 and info["mean_general_wait_s"] >= 0


def test_summarise_columns():
    import pandas as pd
    df = pd.DataFrame([{"policy": "fixed_time", "demand": "low", "amb_completed": True,
                        "amb_transit_s": t, "amb_wait_s": 1, "mean_general_wait_s": 1.0,
                        "throughput_veh": 10, "max_cross_queue_veh": 2.0} for t in (100, 110)])
    s = summarise(df)
    assert s.loc[0, "amb_transit_s_mean"] == 105 and s.loc[0, "completed"] == 2
