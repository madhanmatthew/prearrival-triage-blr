"""Randomized training traffic on the calibrated OSM corridor (docs/19 §7, D21).

    env = RandomizedOsmEnv()          # each reset: random slot + demand jitter + maybe an incident

Per episode (seeded from the env's np_random, so reproducible with reset(seed=...)):
  * slot ~ uniform{low, medium, peak} (the calibrated route + speed-limit files in sumo/demand/);
  * demand x U(0.8, 1.2) via SUMO's --scale (all flows and background trips);
  * with probability INCIDENT_P a lane-blocking incident: one random lane of one random ambulance-route
    edge after the first controlled signal gets max speed 1 m/s for the whole episode
    [ASSUMPTION: a stalled vehicle / works zone; no other lane is closed];
  * random ambulance depart time (already in the base env).
Vehicle-mix jitter (docs/19 §7, +-5 pp) is NOT applied: it needs regenerated route files; stated limit.
Evaluation never uses this class: it runs OsmCorridorEnv on a fixed slot (backend/rl/osm_rl.py).
"""
from __future__ import annotations

from backend.rl.osm_env import OsmCorridorEnv

DEMAND_JITTER = (0.8, 1.2)
INCIDENT_P = 0.1
INCIDENT_SPEED_MS = 1.0


class RandomizedOsmEnv(OsmCorridorEnv):
    """OsmCorridorEnv with docs/19 §7 domain randomization (training only)."""

    def __init__(self, slots=("low", "medium", "peak"), control: str = "rl", **kw):
        super().__init__(demand=list(slots), control=control, **kw)
        self.scale = 1.0
        self.incident_lane = None

    def _extra_sumo_args(self) -> list[str]:
        self.scale = float(self.np_random.uniform(*DEMAND_JITTER))
        return ["--scale", f"{self.scale:.4f}"]

    def _init_junctions(self) -> None:
        super()._init_junctions()
        self.incident_lane = None
        if self.np_random.random() < INCIDENT_P:
            first = min(j["route_offset_m"] for j in self._tls.values())
            c = self._conn
            run, cands = 0.0, []
            for e in self._route_edges:
                run += c.lane.getLength(f"{e}_0")
                if run > first:
                    cands.append(e)
            edge = cands[int(self.np_random.integers(len(cands)))]
            lane = f"{edge}_{int(self.np_random.integers(c.edge.getLaneNumber(edge)))}"
            c.lane.setMaxSpeed(lane, INCIDENT_SPEED_MS)
            self.incident_lane = lane

    def episode_metrics(self) -> dict:
        m = super().episode_metrics()
        m.update({"demand_scale": round(self.scale, 4), "incident_lane": self.incident_lane or ""})
        return m
