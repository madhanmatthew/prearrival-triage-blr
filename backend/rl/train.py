"""DQN training + deterministic evaluation for the ORR corridor (docs/09 §4).

Train on randomized demand (episode samples low/medium/peak, random ambulance depart time);
evaluate deterministically on each demand level with the same episode seeds the baselines use
(seed + episode index), so results pair with `reports/rl_baselines_episodes.csv`.

    python -m backend.rl.train --seed 1 --timesteps 100000
    python -m backend.rl.train --seed 1 --eval-only --model models/dqn_corridor_s1.zip
    python -m backend.rl.train --quick        # ~2k steps, 2 eval episodes; not logged

Outputs: models/dqn_corridor_s<seed>_v1_<YYYYMMDD>.zip (best-by-eval checkpoint, gitignored),
reports/rl_dqn_{episodes,summary}_s<seed>.csv, one row per demand appended to
experiments/log.csv (model = dqn_s<seed>). Run >= 3 seeds (docs/09 §4).

Training demand is drawn from the schematic corridor's 3 fixed levels; the +-20 % demand
jitter, mix jitter and incidents of docs/19 §7 need the OSM network and are not built yet.
[ASSUMPTION] DQN hyper-parameters below are SB3-style starting values, not tuned.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from backend.rl.baselines import (DATA_VERSION, LOG_PATH, REPORT_DIR, SEED, _git_commit,
                                  run_episode, summarise)
from backend.rl.env import DEMANDS, CorridorEnv

MODEL_DIR = Path("models")
HPARAMS = dict(learning_rate=1e-3, buffer_size=50_000, learning_starts=1_000, batch_size=64,
               gamma=0.99, train_freq=4, target_update_interval=500,
               exploration_fraction=0.3, exploration_final_eps=0.05)


def set_seeds(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(env, seed: int, **overrides):
    from stable_baselines3 import DQN
    return DQN("MlpPolicy", env, seed=seed, verbose=0, policy_kwargs=dict(net_arch=[64, 64]),
               **{**HPARAMS, **overrides})


def _eval_callback(eval_env, path: Path, freq: int, episodes: int):
    from stable_baselines3.common.callbacks import EvalCallback
    return EvalCallback(eval_env, n_eval_episodes=episodes, eval_freq=freq, deterministic=True,
                        best_model_save_path=str(path), log_path=str(path), verbose=0)


def train(seed: int, timesteps: int, eval_freq: int = 1000, eval_episodes: int = 5,
          **overrides) -> Path:
    """Train one seed; returns the path of the best-by-eval model."""
    set_seeds(seed)
    env = CorridorEnv(demand=DEMANDS, control="rl")
    eval_env = CorridorEnv(demand="medium", control="rl")
    ckpt = MODEL_DIR / f"dqn_corridor_s{seed}_ckpt"
    try:
        model = make_model(env, seed, **overrides)
        model.learn(timesteps, callback=_eval_callback(eval_env, ckpt, eval_freq, eval_episodes))
    finally:
        env.close()
        eval_env.close()
    best = ckpt / "best_model.zip"
    final = MODEL_DIR / f"dqn_corridor_s{seed}_v1_{dt.date.today():%Y%m%d}.zip"
    if not best.exists():  # eval never ran (very short run): keep the last model
        model.save(str(final.with_suffix("")))
    else:
        final.write_bytes(best.read_bytes())
    return final


def evaluate(model_path: Path, seed: int, episodes: int, demands=DEMANDS) -> pd.DataFrame:
    from stable_baselines3 import DQN
    model = DQN.load(str(model_path))
    rows = []
    for demand in demands:
        env = CorridorEnv(demand=demand, control="rl")
        try:
            for ep in range(episodes):
                # eval seeds = baseline seeds (SEED + ep), independent of the training seed
                info = run_episode(env, SEED + ep,
                                   lambda o: int(model.predict(o, deterministic=True)[0]))
                rows.append({"policy": f"dqn_s{seed}", "seed": SEED + ep, **info})
        finally:
            env.close()
    return pd.DataFrame(rows)


def log_results(summary: pd.DataFrame, seed: int, timesteps: int) -> None:
    today, commit = dt.date.today().isoformat(), _git_commit()
    lines = []
    for r in summary.itertuples():
        note = (f"task=ambulance_transit_s demand={r.demand} episodes={r.episodes} "
                f"completed={r.completed} gen_wait_s={r.mean_general_wait_s_mean:.2f} "
                f"throughput={r.throughput_veh_mean:.0f} "
                f"max_cross_q={r.max_cross_queue_veh_mean:.1f} timesteps={timesteps} "
                "SUMO simulation, schematic corridor")
        lines.append([today, commit, "traffic_rl", r.policy,
                      json.dumps({"demand": r.demand, **HPARAMS}),
                      f"{r.amb_transit_s_mean:.1f}", f"{r.amb_transit_s_std:.1f}", "",
                      DATA_VERSION, seed, note])
    pd.DataFrame(lines).to_csv(LOG_PATH, mode="a", header=False, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Train/evaluate DQN on the ORR corridor")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--timesteps", type=int, default=100_000)
    p.add_argument("--eval-episodes", type=int, default=20, help="final eval episodes per demand")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--model", type=Path, help="model .zip for --eval-only")
    p.add_argument("--quick", action="store_true", help="tiny smoke run; not logged")
    a = p.parse_args()
    if a.quick:
        a.timesteps, a.eval_episodes = 2_000, 2
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    if a.eval_only:
        if a.model is None:
            p.error("--eval-only needs --model")
        path = a.model
    else:
        kw = dict(learning_starts=200, eval_freq=1000, eval_episodes=1) if a.quick else {}
        path = train(a.seed, a.timesteps, **kw)
        print(f"saved {path}")
    eps = evaluate(path, a.seed, a.eval_episodes)
    summary = summarise(eps)
    print(summary.round(2).to_string(index=False))
    suffix = f"_s{a.seed}" + ("_quick" if a.quick else "")
    eps.to_csv(REPORT_DIR / f"rl_dqn_episodes{suffix}.csv", index=False)
    summary.to_csv(REPORT_DIR / f"rl_dqn_summary{suffix}.csv", index=False)
    if not a.quick:
        log_results(summary, a.seed, a.timesteps)
        print(f"appended {len(summary)} rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
