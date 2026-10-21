"""RL env on the real OSM ORR corridor (docs/15 D21, docs/19 §1).

Same agent spec as backend/rl/env.py (docs/09 §4: 16 actions = main/cross green per junction for 4
junctions, 24-d observation, same reward and baselines); only the plumbing differs:
  * network sumo/osm/corridor.net.xml, junctions from sumo/osm/tls_map.json (built by
    scripts/build_osm_corridor.py): the 4 signalised junctions on the Silk Board -> Iblur stretch;
  * each junction is driven as 2 phases with traci.setRedYellowGreenState, using the main / cross
    link indices in tls_map.json (phase 0 main green, 1 main yellow, 2 cross green, 3 cross yellow);
  * "fixed" control leaves the OSM/netconvert programs untouched (baseline a);
  * the ambulance follows the ORR eastbound route (orr_eastbound_route.json); its distance to a
    junction = junction route offset - ambulance odometer (same formula as the schematic env);
  * demand: one route file per slot in sumo/demand/<slot>.rou.xml (fitted by the calibration step,
    docs/19 §5). Missing files raise with a hint; tests pass their own small demand.
[ASSUMPTION]s: ambulance vType = emergency class, speedFactor 1.3 (mirrors the schematic env's
elevated speed factor); ambulance departs at the route start ~0.9 km before Silk Board.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.rl.env import AMB_ID, HOLD, CorridorEnv

OSM_DIR = Path(__file__).resolve().parents[2] / "sumo" / "osm"
DEMAND_DIR = Path(__file__).resolve().parents[2] / "sumo" / "demand"
AMB_ROUTE_ID = "amb_orr_eb"
PHASE_STATE = {0: "main_green", 1: "main_yellow", 2: "cross_green", 3: "cross_yellow"}


def load_tls_map(path: Path = OSM_DIR / "tls_map.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["junctions"]


class OsmCorridorEnv(CorridorEnv):
    """CorridorEnv on the real OSM corridor. `demand_files` maps slot -> route file."""

    DEMANDS = ["low", "medium", "peak"]

    def __init__(self, demand="medium", control: str = "rl", demand_files: dict | None = None,
                 tls_map: dict | None = None, **kw):
        self._tls = tls_map if tls_map is not None else load_tls_map()
        self.JUNCTIONS = list(self._tls)
        self._demand_files = {k: Path(v) for k, v in (demand_files or {}).items()}
        self._route_edges = json.loads((OSM_DIR / "orr_eastbound_route.json").read_text())["edges"]
        super().__init__(demand=demand, control=control, **kw)

    def _demand_file(self, slot: str) -> Path:
        p = self._demand_files.get(slot, DEMAND_DIR / f"{slot}.rou.xml")
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: fit demand first (docs/19 §5) or pass demand_files")
        return p

    def _start_sumo(self, sumo_seed: int):
        import traci
        cmd = [self._sumo, "-n", str(OSM_DIR / "corridor.net.xml"), "-r", str(self._demand_file(self.demand)),
               "--seed", str(sumo_seed), "--time-to-teleport", "300", "--no-step-log", "true",
               "--no-warnings", "true", "--ignore-route-errors", "true"]
        traci.start(cmd, label=self._label)
        self._conn = c = traci.getConnection(self._label)
        c.route.add(AMB_ROUTE_ID, self._route_edges)
        if AMB_ID not in c.vehicletype.getIDList():
            c.vehicletype.copy("DEFAULT_VEHTYPE", AMB_ID)
            c.vehicletype.setVehicleClass(AMB_ID, "emergency")
            c.vehicletype.setSpeedFactor(AMB_ID, 1.3)
            c.vehicletype.setColor(AMB_ID, (255, 0, 0, 255))
        c.vehicle.add(AMB_ID, AMB_ROUTE_ID, typeID=AMB_ID, depart=str(self.amb_depart),
                      departLane="best", departSpeed="max")

    def _init_junctions(self) -> None:
        c = self._conn
        self._lanes, self._jx = {}, {}
        for name, j in self._tls.items():
            links = c.trafficlight.getControlledLinks(j["tls_id"])     # per link index: [(in, out, via)]
            main = {links[k][0][0] for k in j["main_links"] if k < len(links) and links[k]}
            cross = {links[k][0][0] for k in j["cross_links"] if k < len(links) and links[k]}
            self._lanes[name] = (sorted(main), sorted(cross - main))
            self._jx[name] = float(j["route_offset_m"])

    def _set_phase(self, j: str, phase: int) -> None:
        tid = self._tls[j]["tls_id"]
        self._conn.trafficlight.setRedYellowGreenState(tid, self._tls[j]["states"][PHASE_STATE[phase]])

    def _amb_x(self):
        """Ambulance odometer (m along its route) while it is driving, else None."""
        if self._amb_in is not None and self._amb_out is None:
            return self._conn.vehicle.getDistance(AMB_ID)
        return None


__all__ = ["OsmCorridorEnv", "load_tls_map", "HOLD"]
