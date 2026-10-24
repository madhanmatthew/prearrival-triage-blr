"""Real road geometry for the web app's Replay (mock) mode, fetched once from ORS.

Replay runs fully in the browser (no backend), so without this its routes were drawn as straight
segments. For every replay scene (data/benchmark_scenarios.csv + the corridor demo at Silk Board)
this fetches ORS driving routes scene -> each hospital, nearest ambulance base -> scene (same
nearest-by-haversine rule as web/src/mocks/scenario.ts), and the corridor demo through the 4 ORR
junctions. Output: web/src/data/demo_routes.json  {key: [[lat, lon], ...]}, key = see `key()`.

    python scripts/build_demo_routes.py            # needs ORS_API_KEY in .env (free tier)

Free-tier ORS limits requests per minute, so calls are spaced (`--pause`). Geometry is
© OpenStreetMap contributors (ODbL), via openrouteservice.

`--provider osrm` fills only the routes still missing from the public OSRM demo server (no key,
same OSM roads; light use only). Used once on 2026-10-24 for the 20 routes left when the ORS
daily quota ran out. Replay geometry only: live dispatch routing stays ORS (AGENTS rule 11).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "src" / "data" / "demo_routes.json"
ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving/"
JUNCTION_ORDER = ("silkboard", "bellandur", "marathahalli", "krpuram")
CORRIDOR_DEST = "H20"   # the corridor demo marks every other hospital full (scenario.ts forceCorridor)


def key(points) -> str:
    """'lat,lon|lat,lon[|...]' with 5 decimals; web/src/mocks/scenario.ts builds the same key."""
    return "|".join(f"{float(a):.5f},{float(b):.5f}" for a, b in points)


def haversine_m(a, b) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(h))


def read_csv(rel: str) -> list[dict]:
    with open(ROOT / rel, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ors_line(points, api_key: str) -> list[list[float]]:
    body = {"coordinates": [[lon, lat] for lat, lon in points], "geometry_simplify": True}
    req = urllib.request.Request(ORS_URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": api_key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        coords = json.loads(r.read())["features"][0]["geometry"]["coordinates"]
    return [[round(lat, 5), round(lon, 5)] for lon, lat in coords]


def osrm_line(points) -> list[list[float]]:
    coords = ";".join(f"{lon},{lat}" for lat, lon in points)
    req = urllib.request.Request(f"{OSRM_URL}{coords}?overview=simplified&geometries=geojson",
                                 headers={"User-Agent": "prearrival-triage-blr demo route build"})
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.loads(r.read())
    if d.get("code") != "Ok":
        raise KeyError(d.get("code"))
    return [[round(lat, 5), round(lon, 5)] for lon, lat in d["routes"][0]["geometry"]["coordinates"]]


def jobs() -> list[list[tuple[float, float]]]:
    hospitals = {h["hospital_id"]: (float(h["lat"]), float(h["lon"])) for h in read_csv("data/hospitals.csv")}
    bases = [(float(b["lat"]), float(b["lon"])) for b in read_csv("data/ambulance_bases.csv")]
    junctions = json.loads((ROOT / "scripts" / "traffic_points.json").read_text(encoding="utf-8"))["junctions"]
    jpts = [(float(junctions[j]["lat"]), float(junctions[j]["lon"])) for j in JUNCTION_ORDER]
    scenes = [(float(s["lat"]), float(s["lon"])) for s in read_csv("data/benchmark_scenarios.csv")]
    scenes.append(jpts[0])                                     # corridor demo starts at Silk Board
    out = []
    for sc in scenes:
        out.extend([sc, h] for h in hospitals.values())
        out.append([min(bases, key=lambda b: haversine_m(b, sc)), sc])
    if CORRIDOR_DEST in hospitals:
        out.append([*jpts, hospitals[CORRIDOR_DEST]])
    uniq = {key(j): j for j in out}
    return list(uniq.values())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pause", type=float, default=1.6, help="seconds between calls (free tier rate limit)")
    p.add_argument("--provider", choices=("ors", "osrm"), default="ors",
                   help="osrm = public OSRM demo server, only to fill routes ORS could not provide")
    a = p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("ORS_API_KEY", "").strip()
    if a.provider == "ors" and not api_key:
        raise SystemExit("ORS_API_KEY missing in .env")
    routes = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    todo = [j for j in jobs() if key(j) not in routes]
    print(f"{len(routes)} cached, {len(todo)} to fetch")
    failed = 0
    for i, pts in enumerate(todo, 1):
        try:
            routes[key(pts)] = ors_line(pts, api_key) if a.provider == "ors" else osrm_line(pts)
        except (urllib.error.URLError, KeyError, IndexError) as e:   # no route / quota: replay keeps its fallback
            failed += 1
            print(f"  skip {key(pts)}: {getattr(e, 'code', '')} {e}")
        if i % 20 == 0:
            print(f"  {i}/{len(todo)}")
            OUT.write_text(json.dumps(routes, separators=(",", ":")), encoding="utf-8")
        time.sleep(a.pause)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(routes, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(routes)} routes, {failed} failed, "
          f"{OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
