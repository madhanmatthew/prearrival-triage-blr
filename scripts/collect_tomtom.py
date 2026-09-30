"""TomTom Traffic Flow polling (docs/19 §3). Run >= 7 days in the background.

Every `--interval-min` (default 30) queries Flow Segment Data for each corridor point and
appends to data/raw/traffic/tomtom_flow.csv. Errors on a point are logged and skipped; the
loop keeps going. Ctrl+C or killing the process is safe (rows are appended per round).
Key: TOMTOM_API_KEY in .env. ~4-7 points x 48 rounds/day is small next to TomTom's free
daily quota [TODO-VERIFY current free-tier limit].

TomTom snaps each point to the nearest road segment, so we also store `frc` and the snapped
segment's first coordinate to check afterwards that it was the ORR and not a side street.

Run: python -m scripts.collect_tomtom [--once] [--interval-min 30]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

from scripts.traffic_common import OUT_DIR, append_rows, get_json, load_points, require_key

OUT_PATH = OUT_DIR / "tomtom_flow.csv"
URL = ("https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/{zoom}/json"
       "?point={lat},{lon}&unit=KMPH&key={key}")
COLUMNS = ["collected_at", "segment", "point", "lat", "lon", "current_speed_kmph",
           "free_flow_speed_kmph", "current_travel_time_s", "free_flow_travel_time_s",
           "confidence", "road_closure", "frc", "snapped_lat", "snapped_lon"]


def collection_points(cfg: dict) -> list[dict]:
    """Junction points (assigned to the segment they start) + each segment's extra points."""
    pts, seen = [], set()
    for seg in cfg["segments"]:
        if seg.get("full_corridor"):
            continue  # the full corridor is covered by its sub-segments
        for jn in (seg["from"], seg["to"]):
            if jn not in seen:
                seen.add(jn)
                pts.append({"segment": seg["name"], "point": jn, **cfg["junctions"][jn]})
        for i, p in enumerate(seg.get("extra_points", [])):
            pts.append({"segment": seg["name"], "point": f"{seg['name']}_x{i + 1}", **p})
    return pts


def parse_flow(payload: dict) -> dict:
    d = payload["flowSegmentData"]
    first = (d.get("coordinates", {}).get("coordinate") or [{}])[0]
    return {"current_speed_kmph": d.get("currentSpeed"),
            "free_flow_speed_kmph": d.get("freeFlowSpeed"),
            "current_travel_time_s": d.get("currentTravelTime"),
            "free_flow_travel_time_s": d.get("freeFlowTravelTime"),
            "confidence": d.get("confidence"), "road_closure": d.get("roadClosure"),
            "frc": d.get("frc"), "snapped_lat": first.get("latitude"),
            "snapped_lon": first.get("longitude")}


def poll_once(points: list[dict], key: str, zoom: int = 10) -> int:
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows = []
    for p in points:
        try:
            flow = parse_flow(get_json(URL.format(zoom=zoom, lat=p["lat"], lon=p["lon"], key=key)))
        except (RuntimeError, KeyError) as e:
            print(f"{now} {p['point']}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            continue
        rows.append({"collected_at": now, "segment": p["segment"], "point": p["point"],
                     "lat": p["lat"], "lon": p["lon"], **flow})
    if rows:
        append_rows(OUT_PATH, COLUMNS, rows)
    print(f"{now} wrote {len(rows)}/{len(points)} rows", flush=True)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="TomTom Flow polling")
    ap.add_argument("--interval-min", type=float, default=30)
    ap.add_argument("--once", action="store_true", help="single round, then exit (test)")
    ap.add_argument("--allow-unverified", action="store_true")
    a = ap.parse_args()
    points = collection_points(load_points(allow_unverified=a.allow_unverified))
    key = require_key("TOMTOM_API_KEY")
    print(f"polling {len(points)} points every {a.interval_min} min -> {OUT_PATH}", flush=True)
    while True:
        started = time.monotonic()
        poll_once(points, key)
        if a.once:
            return
        time.sleep(max(0.0, a.interval_min * 60 - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
