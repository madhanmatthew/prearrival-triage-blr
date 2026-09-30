"""Gymnasium env: joint signal control of the 4-junction ORR corridor in SUMO (docs/09 §4).

One agent controls all 4 junctions. Action = int in [0, 16): bit i of the action picks
junction i's green (0 = ORR main, 1 = cross street). Decisions every `delta_time` = 5 s.
Observation (24-d, docs/09 §4): per junction [main queue, cross queue, phase one-hot (main,
cross), ambulance_within_1km, ambulance distance / 1000 m (1.0 if none)].
Reward (docs/09 §4): -w_amb*amb_wait_step - w_gen*mean_wait_others - w_cross*max_cross_queue.

`control` selects who drives the lights:
  "rl"       action-driven (min_green 10 s, yellow 3 s).
  "fixed"    SUMO's own static program is never touched (baseline a, docs/15 D8).
  "preempt"  naive "always green for the ambulance" (baseline b): the default 41/4/41/4 s
             cycle, except a junction holds/switches to ORR green while the ambulance is
             within `preempt_m` upstream of it.

[ASSUMPTION]s (not in docs): queue normalisation cap QUEUE_CAP vehicles per approach group;
ambulance departs at a random second in AMB_DEPART_RANGE (traffic warm-up); episode cap
MAX_SIM_S; ambulance is inserted with traci (the route file's own copy is stripped);
queues are SUMO "halting" vehicles (speed < 0.1 m/s).
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import gymnasium as gym
import numpy as np

SUMO_DIR = Path(__file__).resolve().parents[2] / "sumo"
JUNCTIONS = ["silkboard", "bellandur", "marathahalli", "krpuram"]
DEMANDS = ["low", "medium", "peak"]
AMB_ID = "ambulance"
AMB_ROUTE = "main_eb"
AMB_DEPART_RANGE = (200, 600)
MAX_SIM_S = 2400
QUEUE_CAP = 40.0
MAIN_GREEN, CROSS_GREEN = 0, 2          # phase indices in the netconvert program
FIXED_GREEN_S = 41                      # default program durations (see corridor.sumocfg)
FIXED_YELLOW_S = 4
HOLD = 100_000                          # "stay in this phase" duration


def _locate_sumo() -> str:
    import sumo
    import sumolib
    os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)
    return sumolib.checkBinary("sumo")


def _is_cross(edge_id: str) -> bool:
    return bool(re.search(r"_[ns]_", edge_id))


class CorridorEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, demand: str | list[str] = "medium", control: str = "rl",
                 delta_time: int = 5, min_green: int = 10, yellow: int = 3,
                 preempt_m: float = 1000.0, w_amb: float = 1.0, w_gen: float = 0.1,
                 w_cross: float = 0.05):
        if control not in ("rl", "fixed", "preempt"):
            raise ValueError(f"unknown control {control!r}")
        self.demands = [demand] if isinstance(demand, str) else list(demand)
        for d in self.demands:
            if d not in DEMANDS:
                raise ValueError(f"unknown demand {d!r}")
        self.control, self.dt, self.min_green = control, delta_time, min_green
        self.yellow = yellow if control == "rl" else FIXED_YELLOW_S
        self.preempt_m = preempt_m
        self.w = (w_amb, w_gen, w_cross)
        self.action_space = gym.spaces.Discrete(2 ** len(JUNCTIONS))
        self.observation_space = gym.spaces.Box(0.0, 1.0, (6 * len(JUNCTIONS),), np.float32)
        self._sumo = _locate_sumo()
        self._tmp = Path(tempfile.mkdtemp(prefix="corridor_env_"))
        self._label = f"env_{uuid.uuid4().hex[:8]}"
        self._conn = None
        self._routes: dict[str, Path] = {}

    # ---- SUMO plumbing -------------------------------------------------------------
    def _route_file(self, demand: str) -> Path:
        """Route file without the built-in ambulance (it is added by traci at a random time)."""
        if demand not in self._routes:
            text = (SUMO_DIR / f"routes_{demand}.rou.xml").read_text()
            text = re.sub(rf'^.*<vehicle id="{AMB_ID}".*$\n?', "", text, flags=re.M)
            path = self._tmp / f"routes_{demand}.rou.xml"
            path.write_text(text)
            self._routes[demand] = path
        return self._routes[demand]

    def _close_sumo(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001 - SUMO may already be gone
                pass
            self._conn = None

    def close(self) -> None:
        self._close_sumo()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def reset(self, *, seed=None, options=None):
        import traci
        super().reset(seed=seed)
        self._close_sumo()
        self.demand = self.demands[int(self.np_random.integers(len(self.demands)))]
        sumo_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.amb_depart = int(self.np_random.integers(*AMB_DEPART_RANGE, endpoint=True))
        cmd = [self._sumo, "-n", str(SUMO_DIR / "corridor.net.xml"),
               "-r", str(self._route_file(self.demand)), "--seed", str(sumo_seed),
               "--time-to-teleport", "-1", "--no-step-log", "true", "--no-warnings", "true"]
        traci.start(cmd, label=self._label)
        self._conn = traci.getConnection(self._label)
        c = self._conn
        c.vehicle.add(AMB_ID, AMB_ROUTE, typeID=AMB_ID, depart=str(self.amb_depart),
                      departLane="best", departSpeed="max")

        self._lanes = {}
        for j in JUNCTIONS:
            lanes = sorted(set(c.trafficlight.getControlledLanes(j)))
            self._lanes[j] = ([l for l in lanes if not _is_cross(l.rsplit("_", 1)[0])],
                              [l for l in lanes if _is_cross(l.rsplit("_", 1)[0])])
        self._jx = {j: c.junction.getPosition(j)[0] for j in JUNCTIONS}

        self._phase = {j: MAIN_GREEN for j in JUNCTIONS}
        self._elapsed = {j: 0 for j in JUNCTIONS}
        self._target = {j: MAIN_GREEN for j in JUNCTIONS}
        if self.control != "fixed":
            for j in JUNCTIONS:
                c.trafficlight.setPhase(j, MAIN_GREEN)
                c.trafficlight.setPhaseDuration(j, HOLD)

        self._t = 0
        self._amb_in = self._amb_out = None
        self._amb_wait = 0
        self._arrived_others = 0
        self._wait_hist: list[float] = []
        self._max_cross_q = 0.0
        return self._obs(), {}

    # ---- signals -------------------------------------------------------------------
    def _amb_x(self):
        c = self._conn
        if self._amb_in is not None and self._amb_out is None:
            return c.vehicle.getPosition(AMB_ID)[0]
        return None

    def _tick_signals(self, amb_x) -> None:
        """Advance every junction's phase machine by one sim second (min-green/yellow safe)."""
        if self.control == "fixed":
            return
        tl = self._conn.trafficlight
        for j in JUNCTIONS:
            self._elapsed[j] += 1
            p, t = self._phase[j], self._elapsed[j]
            if p in (MAIN_GREEN, CROSS_GREEN):
                if self.control == "rl":
                    switch = self._target[j] != p and t >= self.min_green
                else:
                    near = amb_x is not None and 0 <= self._jx[j] - amb_x <= self.preempt_m
                    if near:
                        switch = p == CROSS_GREEN and t >= self.min_green
                    else:
                        switch = t >= FIXED_GREEN_S
                if switch:
                    self._phase[j], self._elapsed[j] = p + 1, 0
                    tl.setPhase(j, p + 1)
                    tl.setPhaseDuration(j, HOLD)
            elif t >= self.yellow:
                nxt = (p + 1) % 4
                self._phase[j], self._elapsed[j] = nxt, 0
                tl.setPhase(j, nxt)
                tl.setPhaseDuration(j, HOLD)

    # ---- observation / reward ------------------------------------------------------
    def _queue(self, lanes) -> float:
        return float(sum(self._conn.lane.getLastStepHaltingNumber(l) for l in lanes))

    def _obs(self) -> np.ndarray:
        amb_x = self._amb_x()
        out = []
        for j in JUNCTIONS:
            main_l, cross_l = self._lanes[j]
            main_q = min(self._queue(main_l) / QUEUE_CAP, 1.0)
            cross_q = min(self._queue(cross_l) / QUEUE_CAP, 1.0)
            p = self._phase[j]
            # yellow counts as the phase it is leaving toward: 1 -> cross, 3 -> main
            cross_on = p in (CROSS_GREEN, 1)
            d = None if amb_x is None else self._jx[j] - amb_x
            near = d is not None and 0 <= d <= 1000
            out += [main_q, cross_q, float(not cross_on), float(cross_on),
                    float(near), min(d / 1000.0, 1.0) if near else 1.0]
        return np.asarray(out, dtype=np.float32)

    def _mean_wait_others(self) -> float:
        v = self._conn.vehicle
        w = [v.getWaitingTime(i) for i in v.getIDList() if i != AMB_ID]
        return float(np.mean(w)) if w else 0.0

    # ---- gym API -------------------------------------------------------------------
    def step(self, action):
        c = self._conn
        act = int(action)
        for i, j in enumerate(JUNCTIONS):
            self._target[j] = CROSS_GREEN if (act >> i) & 1 else MAIN_GREEN
        amb_wait_step = 0
        for _ in range(self.dt):
            self._tick_signals(self._amb_x())
            c.simulationStep()
            self._t += 1
            for vid in c.simulation.getDepartedIDList():
                if vid == AMB_ID:
                    self._amb_in = self._t
            for vid in c.simulation.getArrivedIDList():
                if vid == AMB_ID:
                    self._amb_out = self._t
                else:
                    self._arrived_others += 1
            if self._amb_x() is not None and c.vehicle.getSpeed(AMB_ID) < 0.1:
                amb_wait_step += 1
        self._amb_wait += amb_wait_step
        cross_q = max(self._queue(self._lanes[j][1]) for j in JUNCTIONS)
        self._max_cross_q = max(self._max_cross_q, cross_q)
        wait = self._mean_wait_others()
        self._wait_hist.append(wait)
        r = -(self.w[0] * amb_wait_step + self.w[1] * wait
              + self.w[2] * min(cross_q / QUEUE_CAP, 1.0))
        terminated = self._amb_out is not None
        truncated = not terminated and self._t >= MAX_SIM_S
        info = {}
        if terminated or truncated:
            info = self.episode_metrics()
        return self._obs(), float(r), terminated, truncated, info

    def episode_metrics(self) -> dict:
        done = self._amb_out is not None
        return {
            "demand": self.demand,
            "amb_depart_s": self.amb_depart,
            "amb_completed": done,
            "amb_transit_s": float(self._amb_out - self._amb_in) if done else float("nan"),
            "amb_wait_s": self._amb_wait,
            "mean_general_wait_s": float(np.mean(self._wait_hist)) if self._wait_hist else 0.0,
            "throughput_veh": self._arrived_others,
            "max_cross_queue_veh": self._max_cross_q,
            "episode_s": self._t,
        }
