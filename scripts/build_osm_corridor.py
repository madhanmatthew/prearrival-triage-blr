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
NAMES = ["silkboard", "hsr_19th_main", "sarjapur_rd_iblur", "orr_5_9km"]   # travel order [TODO-VERIFY names]
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


def find_route(net, points: dict) -> list:
    """Leg by leg through the corridor points; best = closest to TomTom length, mostly ORR."""
    def near(name, r=200):
        p = points[name]
        x, y = net.convertLonLat2XY(p["lon"], p["lat"])
        return [e for e, _ in net.getNeighboringEdges(x, y, r)
                if e.allows("passenger") and (_orr(e) or e.getName() == "")]
    route = []
    for a, b in (("silkboard", "bellandur"), ("bellandur", "marathahalli"), ("marathahalli", "krpuram")):
        target, best = LEG_LENGTH_M[f"{a}_{b}"], None
        for ea in near(a):
            for eb in near(b):
                path, _ = net.getShortestPath(ea, eb)
                if not path:
                    continue
                length = sum(e.getLength() for e in path)
                orr_share = sum(e.getLength() for e in path if _orr(e)) / length
                score = abs(length - target) / target + (1 - orr_share)
                if best is None or score < best[0]:
                    best = (score, path)
        if best is None:
            raise RuntimeError(f"no ORR path {a} -> {b}")
        path = best[1]
        if route and route[-1] == path[0]:
            path = path[1:]
        route += path
    # extend upstream along the ORR so the Silk Board signal is on the route
    head, back = route[0], 0.0
    while back < APPROACH_M:
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
    """Junction-type signals (with a cross-street approach) in travel order. An approach is MAIN
    when it runs parallel (either direction) to the route's approach edge, else CROSS."""
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
            if not cross_in:
                continue                             # mid-block / pedestrian signal
            main_idx = sorted({cc[2] for cc in conns if cc[0].getEdge().getID() not in cross_in})
            cross_idx = sorted({cc[2] for cc in conns if cc[0].getEdge().getID() in cross_in})
            n = max(cc[2] for cc in conns) + 1
            lon, lat = net.convertXY2LonLat(*e.getToNode().getCoord())
            cross_names = sorted({net.getEdge(x).getName() for x in cross_in if net.getEdge(x).getName()})
            out.append({"tls_id": tid, "route_offset_m": round(run, 1), "lat": round(lat, 6),
                        "lon": round(lon, 6), "in_edge": e.getID(), "n_links": n,
                        "main_links": main_idx, "cross_links": cross_idx, "cross_streets": cross_names,
                        "cross_in_edges": sorted(cross_in)})
    return out


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


def build() -> dict:
    import sumolib
    _sumo_env()
    run_netconvert()
    net = sumolib.net.readNet(str(NET), withPrograms=True)
    points = json.loads(POINTS.read_text(encoding="utf-8"))["junctions"]
    route = find_route(net, points)
    found = junctions_on_route(net, route)
    if len(found) < len(NAMES):
        raise RuntimeError(f"expected {len(NAMES)} signalised junctions on the route, found {len(found)}")
    found = found[:len(NAMES)]
    tls_map = {"_note": "docs/15 D21: the 4 RL-controlled junctions; names from OSM cross streets "
                        "[TODO-VERIFY on a map]. Other signals on the route keep their OSM program.",
               "junctions": {name: {**j, "states": states(j)} for name, j in zip(NAMES, found)}}
    length = sum(e.getLength() for e in route)
    ROUTE.write_text(json.dumps({"edges": [e.getID() for e in route], "length_m": round(length, 1)}))
    TLS_MAP.write_text(json.dumps(tls_map, indent=1))
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
