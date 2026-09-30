"""TomTom Routing API typical travel times (docs/19 §2): 24 h x 7 days, resumable.

For every segment, both directions, every (day-of-week, hour): one `calculateRoute` call with
traffic=true, computeTravelTimeFor=all, travelMode=car and a FUTURE `departAt` (next occurrence
of that weekday/hour, IST offset). `historicTrafficTravelTimeInSeconds` is TomTom's typical
(historic-pattern) time, NOT an observation: label it "typical (TomTom historic)".

Existing rows in the CSV are skipped, so a run can be stopped and resumed and nothing is ever
queried twice. `--max-requests` (default 600) spreads the 1,344 calls over >= 3 runs/days so
that, together with the Flow poller, daily usage stays inside the free tier
[TODO-VERIFY current TomTom free daily limit before raising it]. Key: TOMTOM_API_KEY in .env.

Run: python -m scripts.collect_tomtom_typical [--dry-run] [--max-requests 600]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time
import urllib.parse
from pathlib import Path

from scripts.traffic_common import (OUT_DIR, append_rows, get_json, load_points, next_departure,
                                    require_key, segment_jobs)

OUT_PATH = OUT_DIR / "tomtom_typical.csv"
LOCK_PATH = OUT_DIR / ".typical.lock"
URL = "https://api.tomtom.com/routing/1/calculateRoute/{locations}/json?{query}"
COLUMNS = ["collected_at", "segment", "direction", "dow", "hour", "historic_s", "no_traffic_s",
           "travel_time_s", "length_m"]


def build_url(origin: dict, dest: dict, via: list[dict], depart: dt.datetime, key: str) -> str:
    """Waypoints are `lat,lon` joined by ':'. departAt carries the IST offset (+05:30), which
    must be percent-encoded."""
    locations = ":".join(f"{p['lat']},{p['lon']}" for p in [origin, *via, dest])
    query = urllib.parse.urlencode({
        "key": key, "traffic": "true", "computeTravelTimeFor": "all", "travelMode": "car",
        "routeType": "fastest", "departAt": depart.isoformat(timespec="seconds")})
    return URL.format(locations=locations, query=query)


def parse_route(payload: dict) -> dict:
    s = payload["routes"][0]["summary"]
    return {"historic_s": s["historicTrafficTravelTimeInSeconds"],
            "no_traffic_s": s["noTrafficTravelTimeInSeconds"],
            "travel_time_s": s["travelTimeInSeconds"], "length_m": s["lengthInMeters"]}


def length_ok(cfg: dict, segment: str, length_m: float) -> bool:
    """False if the route length is far from the nominal segment length (left the ORR)."""
    exp = cfg.get("expected_length_m", {})
    if segment not in exp:
        return True
    tol = exp.get("length_tolerance", 0.3)
    return abs(length_m - exp[segment]) <= tol * exp[segment]


def done_keys(path: Path = OUT_PATH) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {(r["segment"], r["direction"], int(r["dow"]), int(r["hour"]))
                for r in csv.DictReader(f)}


def pending(cfg: dict, done: set[tuple], limit: int | None = None) -> list[dict]:
    todo = [j for j in segment_jobs(cfg)
            if (j["segment"], j["direction"], j["dow"], j["hour"]) not in done]
    return todo[:limit] if limit else todo


def main() -> None:
    ap = argparse.ArgumentParser(description="TomTom Routing typical travel times")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-requests", type=int, default=600, help="cap for this run")
    ap.add_argument("--min-gap-s", type=float, default=0.3)
    ap.add_argument("--allow-unverified", action="store_true")
    a = ap.parse_args()
    cfg = load_points(allow_unverified=a.allow_unverified)
    done = done_keys()
    todo = pending(cfg, done, a.max_requests)
    print(f"{len(done)} cached, {len(todo)} requests this run -> {OUT_PATH}", flush=True)
    if a.dry_run or not todo:
        return
    key = require_key("TOMTOM_API_KEY")
    try:  # one run at a time: two writers corrupt/duplicate the CSV
        lock_fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"another run is active (or crashed): delete {LOCK_PATH} if not") from None
    try:
        run(cfg, todo, key, a.min_gap_s)
    finally:
        os.close(lock_fd)
        os.remove(LOCK_PATH)


def run(cfg: dict, todo: list[dict], key: str, min_gap_s: float) -> None:
    ok, rejected, streak = 0, 0, 0
    for j in todo:
        url = build_url(j["origin"], j["dest"], j["via"], next_departure(j["dow"], j["hour"]), key)
        try:
            route = parse_route(get_json(url))
        except (RuntimeError, KeyError, IndexError) as e:
            print(f"{j['segment']} {j['direction']} dow={j['dow']} h={j['hour']}: {e}",
                  file=sys.stderr, flush=True)
            continue
        if not length_ok(cfg, j["segment"], route["length_m"]):
            rejected += 1
            streak += 1
            print(f"REJECTED {j['segment']} {j['direction']} dow={j['dow']} h={j['hour']}: "
                  f"length {route['length_m']} m is off the nominal route; add via points",
                  file=sys.stderr, flush=True)
            if streak >= 5:
                raise SystemExit("5 wrong-length routes in a row: fix via points in "
                                 "scripts/traffic_points.json before spending more quota")
            continue
        streak = 0
        append_rows(OUT_PATH, COLUMNS, [{
            "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            **{k: j[k] for k in ("segment", "direction", "dow", "hour")}, **route}])
        ok += 1
        time.sleep(min_gap_s)
    print(f"done: {ok}/{len(todo)} rows written, {rejected} rejected; re-run later for the rest",
          flush=True)


if __name__ == "__main__":
    main()
