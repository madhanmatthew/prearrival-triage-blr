"""Calibration targets per time slot for the SUMO ORR corridor (docs/19 §6–7, docs/15 D20).

    python -m scripts.build_calibration_targets

Inputs (all already collected):
  data/raw/traffic/tomtom_typical.csv   TomTom Routing historic travel times, eastbound, 24 h x 7
  data/raw/traffic/tomtom_flow.csv      TomTom Flow live point speeds, 2026-09-30..10-07 (UTC stamps)
  data/traffic/published_counts.csv     peak volumes (BTP 2025, Silk Board 2024), D20
Output: data/traffic/calibration_targets.csv, one row per (slot, segment):
  typical_s       mean weekday TomTom historic travel time over the slot hours (IST)
  live_speed_kmh  mean weekday live speed at the segment's start point over the slot hours
  peak_volume     published peak volume where one exists for that segment (peak slot only)
Slots (docs/19 §7), weekday only (Mon–Fri): low 02–04, medium 12–14, peak 17–19 IST, chosen from
the typical curve (night minimum 25.6 min, evening maximum 49.6 min Silk Board -> KR Puram).
TomTom `dow`: 0 = Monday [ASSUMPTION, as written by scripts/collect_tomtom_typical.py].
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TYPICAL = Path("data/raw/traffic/tomtom_typical.csv")
FLOW = Path("data/raw/traffic/tomtom_flow.csv")
COUNTS = Path("data/traffic/published_counts.csv")
OUT = Path("data/traffic/calibration_targets.csv")
SLOTS = {"low": (2, 4), "medium": (12, 14), "peak": (17, 19)}      # [start, end) IST hours
SEGMENT_START = {"silkboard_bellandur": "silkboard", "bellandur_marathahalli": "bellandur",
                 "marathahalli_krpuram": "marathahalli", "silkboard_krpuram": "silkboard"}
PEAK_VOLUME = {"silkboard_bellandur": ("BTP2025_DH", "peak_volume")}  # Iblur–Bellandur segment


def typical_targets(typical: pd.DataFrame) -> pd.DataFrame:
    t = typical[(typical.direction == "forward") & (typical.dow < 5)]
    rows = []
    for slot, (h0, h1) in SLOTS.items():
        s = t[(t.hour >= h0) & (t.hour < h1)].groupby("segment").agg(
            typical_s=("historic_s", "mean"), length_m=("length_m", "median"), n_typical=("hour", "size"))
        rows.append(s.reset_index().assign(slot=slot))
    return pd.concat(rows, ignore_index=True)


def live_targets(flow: pd.DataFrame) -> pd.DataFrame:
    f = flow.copy()
    ist = pd.to_datetime(f.collected_at, utc=True) + pd.Timedelta(hours=5, minutes=30)
    f["hour"], f["weekday"] = ist.dt.hour, ist.dt.dayofweek < 5
    f = f[f.weekday]
    rows = []
    for slot, (h0, h1) in SLOTS.items():
        s = f[(f.hour >= h0) & (f.hour < h1)].groupby("point").agg(
            live_speed_kmh=("current_speed_kmph", "mean"), free_flow_kmh=("free_flow_speed_kmph", "mean"),
            n_live=("hour", "size"))
        rows.append(s.reset_index().assign(slot=slot))
    return pd.concat(rows, ignore_index=True)


def build(typical: pd.DataFrame, flow: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    out = typical_targets(typical)
    out["point"] = out.segment.map(SEGMENT_START)
    out = out.merge(live_targets(flow), on=["slot", "point"], how="left")
    out["typical_speed_kmh"] = out.length_m / out.typical_s * 3.6
    out["peak_volume"] = None
    out["peak_volume_source"] = None
    for seg, (src, metric) in PEAK_VOLUME.items():
        v = counts[(counts.source_id == src) & (counts.metric == metric)]
        if len(v):
            m = (out.segment == seg) & (out.slot == "peak")
            out.loc[m, "peak_volume"] = float(v.value.iloc[0])
            out.loc[m, "peak_volume_source"] = f"{src} ({v.unit.iloc[0]})"
    cols = ["slot", "segment", "length_m", "typical_s", "typical_speed_kmh", "n_typical", "point",
            "live_speed_kmh", "free_flow_kmh", "n_live", "peak_volume", "peak_volume_source"]
    order = {s: i for i, s in enumerate(SLOTS)}
    return out[cols].sort_values(["slot", "segment"], key=lambda c: c.map(order) if c.name == "slot" else c) \
        .reset_index(drop=True)


def main() -> None:
    t = build(pd.read_csv(TYPICAL), pd.read_csv(FLOW), pd.read_csv(COUNTS))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.round(2).to_csv(OUT, index=False)
    print(t.drop(columns=["peak_volume_source"]).round(1).to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
