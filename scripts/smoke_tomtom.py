"""Smoke-test TOMTOM_API_KEY: Flow Segment Data (19 §3) + Routing calculateRoute (19 §2).

Usage (repo root):  python scripts/smoke_tomtom.py
Needs: requests, python-dotenv. Never prints the key. Uses 2 API calls.
Coordinates are approximate test points only, not the final segment endpoints.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.environ.get("TOMTOM_API_KEY", "").strip()
if not KEY:
    sys.exit("TOMTOM_API_KEY missing or empty in .env")

SILK_BOARD = "12.9170,77.6230"  # approx, test only
KR_PURAM = "13.0075,77.6950"    # approx, test only
IST = timezone(timedelta(hours=5, minutes=30))
ok = True

# 1. Flow Segment Data (live)
r = requests.get(
    "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json",
    params={"point": SILK_BOARD, "key": KEY}, timeout=20)
if r.status_code == 200:
    f = r.json()["flowSegmentData"]
    print(f"[Flow]    200  currentSpeed={f['currentSpeed']} km/h  "
          f"freeFlowSpeed={f['freeFlowSpeed']} km/h  confidence={f['confidence']}")
else:
    ok = False
    print(f"[Flow]    {r.status_code}  {r.text[:200]}")

# 2. Routing with future departAt (typical / historic)
depart = (datetime.now(IST) + timedelta(days=3)).replace(
    hour=18, minute=0, second=0, microsecond=0)
r = requests.get(
    f"https://api.tomtom.com/routing/1/calculateRoute/{SILK_BOARD}:{KR_PURAM}/json",
    params={"key": KEY, "departAt": depart.isoformat(timespec="seconds"),
            "traffic": "true", "computeTravelTimeFor": "all", "travelMode": "car"},
    timeout=20)
if r.status_code == 200:
    s = r.json()["routes"][0]["summary"]
    print(f"[Routing] 200  departAt={depart:%a %H:%M} IST  length={s['lengthInMeters']} m")
    for k in ("travelTimeInSeconds", "historicTrafficTravelTimeInSeconds",
              "noTrafficTravelTimeInSeconds", "liveTrafficIncidentsTravelTimeInSeconds"):
        print(f"          {k} = {s.get(k, 'MISSING')}")
    if "historicTrafficTravelTimeInSeconds" not in s:
        ok = False
        print("          historicTraffic field missing -> 19 §2 plan needs rethinking")
else:
    ok = False
    print(f"[Routing] {r.status_code}  {r.text[:200]}")

print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)