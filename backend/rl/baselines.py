"""Traffic baselines for the ORR corridor (docs/15 D8, docs/09 §4).

  fixed_time    SUMO's default static program, untouched (baseline a).
  always_green  naive ambulance preemption (baseline b): ORR green for a junction while the
                ambulance is within 1 km upstream, default cycle otherwise.

Each (policy, demand) is run on the same episode seeds (seed + episode index), so the same
background traffic and ambulance depart time are seen by every policy: paired comparison.
Writes reports/rl_baselines_episodes.csv (per episode) and rl_baselines_summary.csv, and
appends one row per (policy, demand) to experiments/log.csv.

Run: python -m backend.rl.baselines --episodes 20 [--quick]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from backend.rl.env import DEMANDS, CorridorEnv

POLICIES = {"fixed_time": "fixed", "always_green": "preempt"}
SEED = 42
LOG_PATH = Path("experiments/log.csv")
REPORT_DIR = Path("reports")
DATA_VERSION = "sumo-corridor-v1"
METRIC_COLS = ["amb_transit_s", "amb_wait_s", "mean_general_wait_s", "throughput_veh",
               "max_cross_queue_veh"]


def run_episode(env: CorridorEnv, seed: int, policy=None) -> dict:
    """Run one episode. `policy(obs) -> action`; baselines ignore actions (control != rl)."""
    obs, _ = env.reset(seed=seed)
    info = {}
    done = False
    while not done:
        action = 0 if policy is None else policy(obs)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
    return info


def evaluate_policy(name: str, demand: str, episodes: int, seed: int = SEED) -> pd.DataFrame:
    env = CorridorEnv(demand=demand, control=POLICIES[name])
    try:
        rows = []
        for ep in range(episodes):
            rows.append({"policy": name, "seed": seed + ep, **run_episode(env, seed + ep)})
    finally:
        env.close()
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (policy, demand), g in df.groupby(["policy", "demand"], sort=False):
        row = {"policy": policy, "demand": demand, "episodes": len(g),
               "completed": int(g.amb_completed.sum())}
        for m in METRIC_COLS:
            row[f"{m}_mean"], row[f"{m}_std"] = g[m].mean(), g[m].std(ddof=1)
        out.append(row)
    return pd.DataFrame(out)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_results(summary: pd.DataFrame, seed: int) -> None:
    today, commit = dt.date.today().isoformat(), _git_commit()
    lines = []
    for r in summary.itertuples():
        note = (f"task=ambulance_transit_s demand={r.demand} episodes={r.episodes} "
                f"completed={r.completed} gen_wait_s={r.mean_general_wait_s_mean:.2f} "
                f"throughput={r.throughput_veh_mean:.0f} "
                f"max_cross_q={r.max_cross_queue_veh_mean:.1f} SUMO simulation")
        lines.append([today, commit, "traffic_rl", r.policy, json.dumps({"demand": r.demand}),
                      f"{r.amb_transit_s_mean:.1f}", f"{r.amb_transit_s_std:.1f}", "",
                      DATA_VERSION, seed, note])
    pd.DataFrame(lines).to_csv(LOG_PATH, mode="a", header=False, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Fixed-time and always-green baselines")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--demands", nargs="*", default=DEMANDS)
    p.add_argument("--policies", nargs="*", default=list(POLICIES))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--quick", action="store_true", help="2 episodes, medium only; not logged")
    a = p.parse_args()
    if a.quick:
        a.episodes, a.demands = 2, ["medium"]
    random.seed(a.seed)
    np.random.seed(a.seed)
    frames = []
    for demand in a.demands:
        for name in a.policies:
            print(f"{name} / {demand}: {a.episodes} episodes", flush=True)
            frames.append(evaluate_policy(name, demand, a.episodes, a.seed))
    eps = pd.concat(frames, ignore_index=True)
    summary = summarise(eps)
    print(summary.round(2).to_string(index=False))
    REPORT_DIR.mkdir(exist_ok=True)
    suffix = "_quick" if a.quick else ""
    eps.to_csv(REPORT_DIR / f"rl_baselines_episodes{suffix}.csv", index=False)
    summary.to_csv(REPORT_DIR / f"rl_baselines_summary{suffix}.csv", index=False)
    if not a.quick:
        log_results(summary, a.seed)
        print(f"appended {len(summary)} rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
