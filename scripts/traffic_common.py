"""Shared helpers for the docs/19 traffic collectors (config, .env, HTTP, CSV append)."""
from __future__ import annotations

import csv
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

POINTS_PATH = Path(__file__).with_name("traffic_points.json")
OUT_DIR = Path("data/raw/traffic")


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:  # python-dotenv is in requirements; fall back to the process env
        pass


def require_key(name: str) -> str:
    load_env()
    key = os.environ.get(name, "").strip()
    if not key:
        raise SystemExit(f"{name} is empty: put it in .env (see .env.example)")
    return key


def load_points(path: Path = POINTS_PATH, allow_unverified: bool = False) -> dict:
    cfg = json.loads(path.read_text())
    if not cfg.get("verified") and not allow_unverified:
        raise SystemExit(f"{path}: coordinates are not verified. Check them on a map, then set "
                         '"verified": true (or pass --allow-unverified for a dry test).')
    return cfg


def post_json(url: str, body: dict, headers: dict, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    return _send(req, timeout)


def get_json(url: str, timeout: int = 30) -> dict:
    return _send(urllib.request.Request(url), timeout)


def _send(req: urllib.request.Request, timeout: int) -> dict:
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:  # keep the API's message, never the URL (holds the key)
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"network error: {e.reason}") from None


def append_rows(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if new:
            w.writeheader()
        w.writerows(rows)


# ---- shared by the typical-time collector -------------------------------------------
import datetime as dt  # noqa: E402

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))  # no DST in India
DIRECTIONS = ("forward", "reverse")  # forward = west->east (segment from -> to)


def next_departure(dow: int, hour: int, now: dt.datetime | None = None) -> dt.datetime:
    """Next IST datetime with weekday `dow` (Mon=0) and `hour`:00, more than an hour after
    `now` (routing APIs reject past departure times)."""
    now = (now or dt.datetime.now(IST)).astimezone(IST)
    cand = (now + dt.timedelta(days=(dow - now.weekday()) % 7)).replace(
        hour=hour, minute=0, second=0, microsecond=0)
    while cand <= now + dt.timedelta(hours=1):
        cand += dt.timedelta(days=7)
    return cand


def segment_jobs(cfg: dict) -> list[dict]:
    """Every (segment, direction, dow, hour) slot: 4 x 2 x 7 x 24 = 1,344."""
    out = []
    for seg in cfg["segments"]:
        a, b = cfg["junctions"][seg["from"]], cfg["junctions"][seg["to"]]
        via = cfg.get("via_points", {}).get(seg["name"], [])
        via_rev = cfg.get("via_points_reverse", {}).get(seg["name"]) or via[::-1]
        for d in DIRECTIONS:
            o, t, v = (a, b, via) if d == "forward" else (b, a, via_rev)
            for dow in range(7):
                for hour in range(24):
                    out.append({"segment": seg["name"], "direction": d, "dow": dow,
                                "hour": hour, "origin": o, "dest": t, "via": v})
    return out
