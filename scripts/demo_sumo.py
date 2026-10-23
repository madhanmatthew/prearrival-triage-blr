"""Watch one ambulance episode on the real ORR corridor in sumo-gui (demo step 7, docs/09 §6).

    python -m scripts.demo_sumo --policy fixed_time --slot peak
    python -m scripts.demo_sumo --policy dqn --slot peak            # latest models/dqn_osm_s1_*.zip
    python -m scripts.demo_sumo --policy always_green --slot medium --delay 20

Same paired episode seed as the evaluation (EVAL_SEED), so the run matches a row of
reports/rl_osm_*_episodes.csv. The ambulance is red; sumo-gui opens zoomed out on the corridor
(use the mouse wheel to zoom to Silk Board). Prints the ambulance transit time at the end.
Fallback if the GUI misbehaves on stage: screen-record a run beforehand.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    from backend.rl.osm_env import OsmCorridorEnv
    from backend.rl.osm_rl import EVAL_SEED, latest_model

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", choices=["fixed_time", "always_green", "dqn"], default="fixed_time")
    ap.add_argument("--slot", choices=["low", "medium", "peak"], default="peak")
    ap.add_argument("--seed", type=int, default=1, help="DQN training seed whose model to load")
    ap.add_argument("--episode", type=int, default=0, help="evaluation episode index (seed EVAL_SEED + i)")
    ap.add_argument("--delay", type=int, default=0, help="ms per simulation step in the GUI")
    ap.add_argument("--model", type=Path)
    a = ap.parse_args()
    control = {"fixed_time": "fixed", "always_green": "preempt", "dqn": "rl"}[a.policy]
    model = None
    if a.policy == "dqn":
        from stable_baselines3 import DQN
        path = a.model or latest_model(a.seed)
        if path is None:
            raise SystemExit("no models/dqn_osm_s*_*.zip")
        model = DQN.load(str(path))
        print(f"model {path.name}")
    env = OsmCorridorEnv(a.slot, control=control, gui=True, gui_delay_ms=a.delay)
    try:
        obs, _ = env.reset(seed=EVAL_SEED + a.episode)
        done, info = False, {}
        while not done:
            act = 0 if model is None else int(model.predict(obs, deterministic=True)[0])
            obs, _, term, trunc, info = env.step(act)
            done = term or trunc
    finally:
        env.close()
    t = info.get("amb_transit_s")
    print(f"{a.policy} / {a.slot}: ambulance {'arrived' if info.get('amb_completed') else 'did NOT arrive'}"
          f" — transit {t:.0f} s, waited {info.get('amb_wait_s')} s over {info.get('amb_route_m', 0) / 1000:.1f} km")


if __name__ == "__main__":
    main()
