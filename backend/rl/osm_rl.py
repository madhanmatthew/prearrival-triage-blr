"""RL on the calibrated OSM ORR corridor (D21, docs/19 §6–7): baselines, DQN training, evaluation.

    python -m backend.rl.osm_rl baselines --episodes 6 --workers 6
    python -m backend.rl.osm_rl train --seed 1 --hours 8            # time-boxed (deadline), checkpoints
    python -m backend.rl.osm_rl eval --seed 1 --episodes 6 --workers 3
    python -m backend.rl.osm_rl summary                              # comparison table + log rows

Evaluation: OsmCorridorEnv on each calibrated slot (low/medium/peak), episode seeds EVAL_SEED + i,
identical for every policy (paired). Training: RandomizedOsmEnv (backend/rl/demand.py). Agent spec and
DQN hyper-parameters as the schematic corridor (backend/rl/train.py, docs/09 §4). Training length is set
by a wall-clock budget because of the project deadline (2026-10-24), not by convergence: state it.
Metrics per episode (OsmCorridorEnv): ambulance completed / transit s / wait s / progress m over the
7.7 km controlled stretch, mean general wait, throughput, max cross-street queue. Simulation only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SLOTS = ("low", "medium", "peak")
EVAL_SEED = 1000
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")
LOG_PATH = Path("experiments/log.csv")
DATA_VERSION = "osm-orr-speed-calibrated-2026-10-22"
BASELINES = {"fixed_time": "fixed", "always_green": "preempt"}


def _episode(args) -> dict:
    """One evaluation episode in its own process (own SUMO). args = (policy, control, slot, seed, model)."""
    policy, control, slot, seed, model_path = args
    from backend.rl.osm_env import OsmCorridorEnv
    model = None
    if model_path:
        from stable_baselines3 import DQN
        model = DQN.load(model_path)
    env = OsmCorridorEnv(slot, control=control)
    try:
        obs, _ = env.reset(seed=seed)
        info, done, t0 = {}, False, time.time()
        while not done:
            a = 0 if model is None else int(model.predict(obs, deterministic=True)[0])
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
    finally:
        env.close()
    return {"policy": policy, "slot": slot, "seed": seed, "wall_s": round(time.time() - t0, 1),
            **{k: v for k, v in info.items() if k != "demand"}}


def run_parallel(jobs: list, workers: int) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_episode, jobs))
    return pd.DataFrame(rows)


def baselines(episodes: int, workers: int) -> pd.DataFrame:
    jobs = [(name, ctrl, slot, EVAL_SEED + i, None)
            for name, ctrl in BASELINES.items() for slot in SLOTS for i in range(episodes)]
    df = run_parallel(jobs, workers)
    REPORT_DIR.mkdir(exist_ok=True)
    df.to_csv(REPORT_DIR / "rl_osm_baselines_episodes.csv", index=False)
    return df


def latest_model(seed: int) -> Path | None:
    files = sorted(MODEL_DIR.glob(f"dqn_osm_s{seed}_*.zip"))
    return files[-1] if files else None


def evaluate(seed: int, episodes: int, workers: int, model: Path | None = None) -> pd.DataFrame:
    model = model or latest_model(seed)
    if model is None:
        raise FileNotFoundError(f"no models/dqn_osm_s{seed}_*.zip: train first")
    jobs = [(f"dqn_s{seed}", "rl", slot, EVAL_SEED + i, str(model)) for slot in SLOTS for i in range(episodes)]
    df = run_parallel(jobs, workers)
    df["model"] = model.name
    df.to_csv(REPORT_DIR / f"rl_osm_dqn_s{seed}_episodes.csv", index=False)
    return df


def train(seed: int, hours: float, checkpoint_steps: int = 5000) -> Path:
    """DQN on RandomizedOsmEnv until the wall-clock budget runs out; checkpoints every N steps."""
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
    from backend.rl.demand import RandomizedOsmEnv
    from backend.rl.train import HPARAMS, make_model, set_seeds

    class TimeBudget(BaseCallback):
        def __init__(self, seconds: float):
            super().__init__()
            self.deadline = time.time() + seconds

        def _on_step(self) -> bool:
            return time.time() < self.deadline

    set_seeds(seed)
    env = RandomizedOsmEnv()
    env.reset(seed=seed)
    ckpt = MODEL_DIR / f"dqn_osm_s{seed}_ckpt"
    try:
        model = make_model(env, seed)
        cb = CallbackList([TimeBudget(hours * 3600),
                           CheckpointCallback(checkpoint_steps, str(ckpt), name_prefix="dqn")])
        model.learn(total_timesteps=10_000_000, callback=cb)
    finally:
        env.close()
    out = MODEL_DIR / f"dqn_osm_s{seed}_v1_{dt.date.today():%Y%m%d}.zip"
    model.save(str(out.with_suffix("")))
    (MODEL_DIR / f"dqn_osm_s{seed}_train.json").write_text(json.dumps(
        {"seed": seed, "hours": hours, "timesteps": int(model.num_timesteps), "hparams": HPARAMS}))
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["policy", "slot"], sort=False)
    out = g.agg(episodes=("seed", "count"), completed=("amb_completed", "sum"),
                amb_transit_s=("amb_transit_s", "mean"), amb_wait_s=("amb_wait_s", "mean"),
                amb_progress_m=("amb_progress_m", "mean"), general_wait_s=("mean_general_wait_s", "mean"),
                max_cross_queue=("max_cross_queue_veh", "mean"), throughput=("throughput_veh", "mean"))
    return out.round(1).reset_index()


def summary_table() -> pd.DataFrame:
    parts = [pd.read_csv(p) for p in sorted(REPORT_DIR.glob("rl_osm_*_episodes.csv"))]
    if not parts:
        raise FileNotFoundError("no reports/rl_osm_*_episodes.csv yet")
    s = summarize(pd.concat(parts, ignore_index=True))
    s.to_csv(REPORT_DIR / "rl_osm_comparison.csv", index=False)
    return s


def log_rows(s: pd.DataFrame) -> None:
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    rows = [[dt.date.today().isoformat(), commit, "traffic_rl", f"osm_{r.policy}",
             json.dumps({"slot": r.slot}), f"{r.amb_progress_m:.0f}", "", f"{r.amb_wait_s:.1f}", DATA_VERSION,
             EVAL_SEED, f"task=ambulance_on_7.7km_controlled_stretch slot={r.slot} episodes={r.episodes} "
             f"completed={r.completed} transit_s={r.amb_transit_s} gen_wait_s={r.general_wait_s} "
             f"max_cross_q={r.max_cross_queue} metric(cv_mean)=progress_m test_score=amb_wait_s; "
             "SUMO, OSM corridor, speed-only calibration"] for r in s.itertuples()]
    pd.DataFrame(rows).to_csv(LOG_PATH, mode="a", header=False, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines"); b.add_argument("--episodes", type=int, default=6); b.add_argument("--workers", type=int, default=6)
    t = sub.add_parser("train"); t.add_argument("--seed", type=int, required=True); t.add_argument("--hours", type=float, default=8.0)
    e = sub.add_parser("eval"); e.add_argument("--seed", type=int, required=True); e.add_argument("--episodes", type=int, default=6)
    e.add_argument("--workers", type=int, default=3); e.add_argument("--model", type=Path)
    s = sub.add_parser("summary"); s.add_argument("--log", action="store_true")
    a = ap.parse_args()
    if a.cmd == "baselines":
        print(summarize(baselines(a.episodes, a.workers)).to_string(index=False))
    elif a.cmd == "train":
        print(f"saved {train(a.seed, a.hours)}")
    elif a.cmd == "eval":
        print(summarize(evaluate(a.seed, a.episodes, a.workers, a.model)).to_string(index=False))
    else:
        tab = summary_table()
        print(tab.to_string(index=False))
        if a.log:
            log_rows(tab)


if __name__ == "__main__":
    main()
