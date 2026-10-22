"""Build the real-geometry ORR corridor for SUMO (docs/19 §1, decision docs/15 D21).

    python -m scripts.build_osm_corridor

Input:  data/raw/osm/orr_corridor.osm (Overpass export, main roads, ODbL)
Output: sumo/osm/corridor.net.xml       netconvert of the OSM extract (signals kept from OSM tags)
        sumo/osm/orr_eastbound_route.json  ORR eastbound edges Silk Board -> KR Puram (ambulance route)
        sumo/osm/tls_map.json           the 4 RL-controlled junctions (D21): TLS id, route offset,
                                        main / cross link indices, 2-phase signal states
The route is found leg by leg through the 4 corridor points in scripts/traffic_points.json
(plain shortest path cuts through Koramangala), preferring ORR-named edges and TomTom leg lengths,
and starts ~1 km before Silk Board so the Silk Board signal is on it.
Junction choice (D21): the 4 junction-type signals the eastbound route crosses (a signal counts
as a junction when it has a cross-street approach); mid-block / pedestrian signals stay on their
default OSM program. Junction names come from OSM cross-street names [TODO-VERIFY on a map].
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

OSM = Path("data/raw/osm/orr_corridor.osm")
OUT_DIR = Path("sumo/osm")
NET = OUT_DIR / "corridor.net.xml"
ROUTE = OUT_DIR / "orr_eastbound_route.json"
TLS_MAP = OUT_DIR / "tls_map.json"
POINTS = Path("scripts/traffic_points.json")
LEG_LENGTH_M = {"silkboard_bellandur": 6274, "bellandur_marathahalli": 4618, "marathahalli_krpuram": 5918}
APPROACH_M = 900          # route starts this far before the Silk Board stop line (approx.)
NETCONVERT_ARGS = ["--geometry.remove", "--roundabouts.guess", "--ramps.guess", "--junctions.join",
                   "--tls.join", "--output.street-names", "--remove-edges.isolated",
                   "--keep-edges.by-vclass", "passenger", "--seed", "42"]


def _sumo_env():
    import sumo
    os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)


def run_netconvert(osm: Path = OSM, net: Path = NET) -> None:
    import sumolib
    _sumo_env()
    net.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sumolib.checkBinary("netconvert"), "--osm-files", str(osm), *NETCONVERT_ARGS,
                    "-o", str(net)], check=True, capture_output=True)


def _orr(e) -> bool:
    return "Outer Ring" in e.getName() or "Underpass" in e.getName()


NON_MAIN_PENALTY = 4.0    # cost x this on edges that are not the ORR main carriageway (side/service roads)


def _edge_cost(e) -> float:
    main = _orr(e) and e.getLaneNumber() >= 2
    return e.getLength() * (1.0 if main else NON_MAIN_PENALTY)


def main_carriageway_path(net, a, b, max_cost: float = 60000.0):
    """Dijkstra from edge a to edge b preferring the ORR main carriageway (>= 2 lanes, ORR-named).
    A plain length-shortest path took a 1-lane at-grade side road at the Sarjapur Road junction,
    where the real ORR is grade-separated (2026-10-22)."""
    import heapq
    best = {a.getID(): 0.0}
    prev = {}
    heap = [(0.0, a.getID(), a)]
    while heap:
        c, eid, e = heapq.heappop(heap)
        if e is b:
            path = [e]
            while path[-1].getID() in prev:
                path.append(prev[path[-1].getID()])
            return list(reversed(path))
        if c > best.get(eid, float("inf")) or c > max_cost:
            continue
        for nxt in e.getOutgoing():
            if not nxt.allows("passenger"):
                continue
            nc = c + _edge_cost(nxt)
            if nc < best.get(nxt.getID(), float("inf")):
                best[nxt.getID()] = nc
                prev[nxt.getID()] = e
                heapq.heappush(heap, (nc, nxt.getID(), nxt))
    return None


EASTBOUND = ("silkboard", "bellandur", "marathahalli", "krpuram")
WESTBOUND = tuple(reversed(EASTBOUND))


def _leg_target(a: str, b: str) -> float:
    return LEG_LENGTH_M.get(f"{a}_{b}") or LEG_LENGTH_M[f"{b}_{a}"]     # TomTom eastbound lengths


def find_legs(net, points: dict, order: tuple = EASTBOUND) -> list[list]:
    """One edge path per leg through the corridor points; best = closest to the TomTom leg
    length and mostly ORR-named (a plain shortest path cuts through Koramangala)."""
    def near(name, r=200):
        p = points[name]
        x, y = net.convertLonLat2XY(p["lon"], p["lat"])
        return [e for e, _ in net.getNeighboringEdges(x, y, r)
                if e.allows("passenger") and (_orr(e) or e.getName() == "")]
    legs, last = [], None
    for a, b in zip(order, order[1:]):
        target, best = _leg_target(a, b), None
        starts = [last] if last is not None else near(a)
        for ea in starts:
            for eb in near(b):
                path = main_carriageway_path(net, ea, eb)
                if not path:
                    continue
                length = sum(e.getLength() for e in path)
                orr_share = sum(e.getLength() for e in path if _orr(e)) / length
                score = abs(length - target) / target + (1 - orr_share)
                if best is None or score < best[0]:
                    best = (score, path)
        if best is None:
            raise RuntimeError(f"no ORR path {a} -> {b}")
        legs.append(best[1])
        last = best[1][-1]
    return legs


def find_route(net, points: dict, order: tuple = EASTBOUND, lead_in_m: float = APPROACH_M) -> list:
    """Legs joined into one continuous route, extended `lead_in_m` upstream along the ORR (so the
    Silk Board signal is on the eastbound route)."""
    route = []
    for path in find_legs(net, points, order):
        if route and route[-1] == path[0]:
            path = path[1:]
        route += path
    head, back = route[0], 0.0
    while back < lead_in_m:
        inc = [e for e in head.getIncoming() if e.allows("passenger") and _orr(e) and e not in route]
        if not inc:
            break
        head = max(inc, key=lambda e: (e.getLaneNumber(), e.getLength()))
        route.insert(0, head)
        back += head.getLength()
    for i in range(len(route) - 1):
        if route[i + 1] not in route[i].getOutgoing():
            raise RuntimeError(f"route not continuous at {route[i].getID()}")
    return route


def _bearing(e) -> float:
    """Heading (deg) of the last 30 m of an edge, i.e. the approach direction at its stop line."""
    import math
    shape = e.getShape()
    (x0, y0), (x1, y1) = shape[-2], shape[-1]
    return math.degrees(math.atan2(x1 - x0, y1 - y0)) % 360


def _parallel(a: float, b: float, tol: float = 35.0) -> bool:
    d = abs(a - b) % 180
    return min(d, 180 - d) <= tol


def junctions_on_route(net, route: list) -> list[dict]:
    """Signals on the route in travel order, `kind` = "junction" (has a cross-street approach) or
    "midblock" (pedestrian crossing: no cross approach; its cross phase = all-red for vehicles).
    An approach is MAIN when it runs parallel (either direction) to the route's approach edge."""
    tls = {t.getID(): t for t in net.getTrafficLights()}
    out, run, seen = [], 0.0, set()
    route_ids = {e.getID() for e in route}
    for i, e in enumerate(route[:-1]):
        run += e.getLength()
        for c in e.getConnections(route[i + 1]):
            tid = c.getTLSID()
            if not tid or tid in seen:
                continue
            seen.add(tid)
            conns = tls[tid].getConnections()        # (inLane, outLane, linkIndex)
            ref = _bearing(e)
            cross_in = {cc[0].getEdge().getID() for cc in conns
                        if cc[0].getEdge().getID() not in route_ids
                        and not _parallel(_bearing(cc[0].getEdge()), ref)}
            main_idx = sorted({cc[2] for cc in conns if cc[0].getEdge().getID() not in cross_in})
            # main links coming from the route's own direction (vs the opposite ORR direction)
            same_dir = sorted({cc[2] for cc in conns if cc[0].getEdge().getID() not in cross_in and (
                cc[0].getEdge().getID() in route_ids or abs(((_bearing(cc[0].getEdge()) - ref + 180) % 360) - 180) <= 35)})
            cross_idx = sorted({cc[2] for cc in conns if cc[0].getEdge().getID() in cross_in})
            n = max(cc[2] for cc in conns) + 1
            lon, lat = net.convertXY2LonLat(*e.getToNode().getCoord())
            cross_names = sorted({net.getEdge(x).getName() for x in cross_in if net.getEdge(x).getName()})
            out.append({"tls_id": tid, "kind": "junction" if cross_in else "midblock",
                        "route_offset_m": round(run, 1), "lat": round(lat, 6),
                        "lon": round(lon, 6), "in_edge": e.getID(), "n_links": n,
                        "main_links": main_idx, "main_same_dir_links": same_dir,
                        "cross_links": cross_idx, "cross_streets": cross_names,
                        "cross_in_edges": sorted(cross_in)})
    return out


N_CONTROLLED = 4      # agent spec (docs/09 §4): 4 junctions -> 16 actions, 24-d observation


def select_controlled(signals: list[dict], n: int = N_CONTROLLED) -> list[dict]:
    """All junction signals first, then mid-block signals in travel order until n; travel order."""
    juncs = [s for s in signals if s["kind"] == "junction"]
    mids = [s for s in signals if s["kind"] == "midblock"]
    if len(juncs) + len(mids) < n:
        raise RuntimeError(f"need {n} signals on the route, found {len(juncs) + len(mids)}")
    chosen = juncs[:n] + mids[:max(0, n - len(juncs))]
    return sorted(chosen, key=lambda s: s["route_offset_m"])


def name_of(j: dict, i: int) -> str:
    """Stable readable name: first signal = silkboard; junctions by cross street; mid-block by km."""
    import re
    if i == 0:
        return "silkboard"
    km = f"{j['route_offset_m'] / 1000:.1f}".replace(".", "_")
    if j["kind"] == "midblock":
        return f"ped_crossing_{km}km"
    street = re.sub(r"[^a-z0-9]+", "_", (j["cross_streets"] or ["junction"])[0].lower()).strip("_")
    return f"{street}_{km}km" if street != "junction" else f"junction_{km}km"


def states(j: dict) -> dict[str, str]:
    """2-phase control: main green, cross green and their yellows (links neither main nor cross: red)."""
    def mk(green, other_yellow=None):
        s = []
        for k in range(j["n_links"]):
            if k in green:
                s.append("g")
            elif other_yellow is not None and k in other_yellow:
                s.append("y")
            else:
                s.append("r")
        return "".join(s)
    main, cross = set(j["main_links"]), set(j["cross_links"])
    return {"main_green": mk(main), "main_yellow": mk(set(), main),
            "cross_green": mk(cross), "cross_yellow": mk(set(), cross)}


SILKBOARD_2024 = (("route_dir", 147), ("route_dir_yellow", 3), ("opposite", 87), ("opposite_yellow", 3),
                  ("cross", 42), ("cross_yellow", 3))   # 285 s cycle, Deccan Herald 18 Jul 2024 (D20)
SILKBOARD_PROGRAM = OUT_DIR / "silkboard_2024.tll.xml"


def silkboard_program(j: dict) -> str:
    """Fixed-time program for Silk Board from the reported 2024 peak timings (D20): BTM/ORR-west
    (the route's approach) ~150 s, HSR/ORR-east ~90 s, Hosur Road ~45 s, cycle 285 s; 3 s yellow
    taken out of each green. Which OSM approach is which is inferred from geometry [TODO-VERIFY]."""
    groups = {"route_dir": set(j["main_same_dir_links"]),
              "opposite": set(j["main_links"]) - set(j["main_same_dir_links"]),
              "cross": set(j["cross_links"])}
    phases = []
    for name, dur in SILKBOARD_2024:
        g = groups[name.replace("_yellow", "")]
        ch = "y" if name.endswith("_yellow") else "G"
        state = "".join(ch if k in g else "r" for k in range(j["n_links"]))
        phases.append(f'    <phase duration="{dur}" state="{state}"/>')
    head = f'<additional>\n  <tlLogic id="{j["tls_id"]}" type="static" programID="silkboard_2024" offset="0">\n'
    return head + "\n".join(phases) + "\n  </tlLogic>\n</additional>\n"


def build() -> dict:
    import sumolib
    _sumo_env()
    run_netconvert()
    net = sumolib.net.readNet(str(NET), withPrograms=True)
    points = json.loads(POINTS.read_text(encoding="utf-8"))["junctions"]
    route = find_route(net, points)
    found = select_controlled(junctions_on_route(net, route))
    tls_map = {"_note": "docs/15 D21 (rev. 2026-10-22): the 4 RL-controlled signals on the ORR main "
                        "carriageway = every junction signal + mid-block crossings in travel order to make 4; "
                        "names from OSM cross streets [TODO-VERIFY on a map]. Other signals keep their OSM program.",
               "junctions": {name_of(j, i): {**j, "states": states(j)} for i, j in enumerate(found)}}
    length = sum(e.getLength() for e in route)
    ROUTE.write_text(json.dumps({"edges": [e.getID() for e in route], "length_m": round(length, 1)}))
    TLS_MAP.write_text(json.dumps(tls_map, indent=1))
    SILKBOARD_PROGRAM.write_text(silkboard_program(tls_map["junctions"]["silkboard"]))
    return {"route_length_m": length, "junctions": tls_map["junctions"]}


def main() -> None:
    r = build()
    print(f"route {r['route_length_m']:.0f} m -> {ROUTE}")
    for name, j in r["junctions"].items():
        print(f"{name:24s} {j['route_offset_m']:7.0f} m  {j['tls_id'][:40]:40s} cross={j['cross_streets']} "
              f"main={len(j['main_links'])} cross={len(j['cross_links'])}")
    print(f"wrote {TLS_MAP}")


if __name__ == "__main__":
    main()
