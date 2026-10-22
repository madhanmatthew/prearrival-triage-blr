"""Speed-only calibration of the OSM ORR corridor (fallback, docs/19 §6, D20, D21; 2026-10-22).

    python -m scripts.calibrate_osm_speed            # fit + validate + write sumo/demand/<slot>.*
    python -m scripts.calibrate_osm_speed --validate-only

Why a fallback: fitting demand alone (scripts/calibrate_osm_demand.py) could not reproduce TomTom's
peak speeds on this network: matching needed ~14x the local demand on Bellandur -> Marathahalli and
then gridlocked in validation (reports/rl_calibration_demand_v2_failed.md). Real ORR friction (bus
stops, U-turns, pedestrians, merge weaving) is not in the model.

Method (stated as a limitation, not hidden):
  * Demand is moderate and plausible, fixed per slot: low = LOW_SCALE, medium/peak = 1.0 (through +
    per-leg local + background, scripts/calibrate_osm_demand.py), vehicle mix and driver speed factor
    as fitted there (speed factor fitted on the night slot).
  * Each leg gets a per-slot lane speed limit (SUMO variableSpeedSign on every lane of the leg, both
    directions), fitted so the probe speed on that leg matches the TomTom typical leg speed:
    L <- L * clip(target / sim, 0.6, 1.6), a few iterations. Slot speeds are therefore IMPOSED, not
    emergent from demand; queues at the 4 RL signals still come from the simulated traffic.
  * Validation: N_SEEDS seeds per slot, per-leg + full-corridor error vs TomTom typical (target 15 %).
Outputs: sumo/demand/<slot>.rou.xml + <slot>.vss.add.xml (loaded by backend/rl/osm_env.py),
sumo/demand/fit_speed.json, reports/rl_calibration.md/.csv, experiments/log.csv rows.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import calibrate_osm_demand as cd

SPEED_FACTOR = 0.75        # fitted on the night slot by calibrate_osm_demand (2026-10-22 run)
SLOT_SCALE = {"low": cd.LOW_SCALE, "medium": 1.0, "peak": 1.0}
ITERS = 4
CLIP = (0.6, 1.6)
TOL = 0.08                 # stop early when every leg is within 8 %
FIT_PATH = cd.DEMAND_DIR / "fit_speed.json"
SILKBOARD_PROGRAM = Path("sumo/osm/silkboard_2024.tll.xml")   # reported 2024 timings (D20), built by build_osm_corridor


def leg_lanes(net, routes: dict) -> dict[str, list[str]]:
    """Lane ids per leg (eastbound leg edges + the matching westbound leg edges)."""
    out = {}
    for leg in cd.LEGS:
        edges = list(routes[leg]) + list(routes.get(f"{leg}_wb", []))
        out[leg] = [ln.getID() for e in dict.fromkeys(edges) for ln in net.getEdge(e).getLanes()]
    return out


def vss_xml(lanes: dict[str, list[str]], limits_ms: dict[str, float]) -> str:
    rows = ["<additional>"]
    for leg, ids in lanes.items():
        rows.append(f'  <variableSpeedSign id="vss_{leg}" lanes="{" ".join(ids)}">'
                    f'<step time="0" speed="{limits_ms[leg]:.3f}"/></variableSpeedSign>')
    rows.append("</additional>")
    return "\n".join(rows) + "\n"


def initial_limits(targets: pd.DataFrame, slot: str) -> dict[str, float]:
    t = targets.set_index(["slot", "segment"])
    return {leg: float(t.loc[(slot, leg), "typical_speed_kmh"]) / 3.6 / SPEED_FACTOR for leg in cd.LEGS}


LIMIT_CAP_MS = 60 / 3.6    # never above the OSM ORR limit; beyond it a leg is signal-limited (2026-10-22)


def run_slot(slot, limits, routes, lengths, lanes, seed, tag, leg_scale=None) -> dict[str, float]:
    work = cd._work_dir()
    add = work / f"vss_{tag}.add.xml"
    add.write_text(vss_xml(lanes, limits))
    extra = f"{add},{SILKBOARD_PROGRAM.resolve()}" if SILKBOARD_PROGRAM.exists() else add
    return cd.simulate(routes, lengths, SLOT_SCALE[slot], SPEED_FACTOR, seed, tag, extra_add=extra,
                       leg_scale=leg_scale)


def fit(targets: pd.DataFrame, routes, lengths, lanes) -> dict:
    t = targets.set_index(["slot", "segment"])
    out = {"speed_factor": SPEED_FACTOR, "scale": dict(SLOT_SCALE), "limits_ms": {}, "leg_scale": {}, "fit_log": {}}
    for slot in cd.SLOTS:
        goal = {leg: float(t.loc[(slot, leg), "typical_speed_kmh"]) for leg in cd.LEGS}
        lim, best, log = initial_limits(targets, slot), None, []
        lim = {k: min(v, LIMIT_CAP_MS) for k, v in lim.items()}
        dm = {leg: 1.0 for leg in cd.LEGS}       # local-demand multiplier, only lowered for signal-limited legs
        for it in range(ITERS):
            sp = run_slot(slot, lim, routes, lengths, lanes, cd.SEED, f"spd_{slot}_{it}", leg_scale=dm)
            err = {leg: (sp.get(leg, np.nan) - goal[leg]) / goal[leg] for leg in cd.LEGS}
            worst = max((abs(e) for e in err.values() if not np.isnan(e)), default=np.inf)
            log.append({"iter": it, "limits_kmh": {k: round(v * 3.6, 1) for k, v in lim.items()},
                        "leg_scale": dict(dm),
                        "speed": {k: round(float(v), 1) for k, v in sp.items()}, "max_abs_err": round(float(worst), 3)})
            print(f"  {slot} iter {it}: max |err| {worst:.1%} limits km/h "
                  f"{ {k: round(v * 3.6, 1) for k, v in lim.items()} } demand x {dm}", flush=True)
            if best is None or worst < best[0]:
                best = (worst, dict(lim), dict(dm))
            if worst <= TOL:
                break
            for leg in cd.LEGS:
                if np.isnan(err[leg]) or not sp.get(leg):
                    continue
                want = lim[leg] * float(np.clip(goal[leg] / sp[leg], *CLIP))
                if want > LIMIT_CAP_MS and sp[leg] < goal[leg]:
                    # signal-limited: the limit cannot help, so lower that leg's local demand instead
                    lim[leg] = LIMIT_CAP_MS
                    dm[leg] = round(dm[leg] * float(np.clip((sp[leg] / goal[leg]) ** 1.5, 0.5, 1.0)), 4)
                else:
                    lim[leg] = min(want, LIMIT_CAP_MS)
        out["limits_ms"][slot] = {k: round(v, 3) for k, v in best[1].items()}
        out["leg_scale"][slot] = best[2]
        out["fit_log"][slot] = log
    return out


def validate(targets: pd.DataFrame, routes, lengths, lanes, params: dict, seeds: int) -> pd.DataFrame:
    t = targets.set_index(["slot", "segment"])
    rows = []
    for slot in cd.SLOTS:
        speeds = [run_slot(slot, params["limits_ms"][slot], routes, lengths, lanes, cd.SEED + k, f"spdval_{slot}_{k}",
                           leg_scale=params.get("leg_scale", {}).get(slot))
                  for k in range(seeds)]
        for seg in (*cd.LEGS, "silkboard_krpuram"):
            sim = [s.get(seg, np.nan) for s in speeds]
            typ = float(t.loc[(slot, seg), "typical_speed_kmh"])
            live = t.loc[(slot, seg), "live_speed_kmh"]
            m = float(np.nanmean(sim)) if not all(np.isnan(sim)) else np.nan
            rows.append({"slot": slot, "segment": seg, "tomtom_typical_kmh": round(typ, 1),
                         "tomtom_live_kmh": round(float(live), 1) if pd.notna(live) else None,
                         "sumo_kmh_mean": round(m, 1), "sumo_kmh_std": round(float(np.nanstd(sim)), 1),
                         "probe_seeds_ok": int(sum(not np.isnan(x) for x in sim)),
                         "err_vs_typical_pct": round(100 * (m - typ) / typ, 1),
                         "err_vs_live_pct": round(100 * (m - live) / live, 1) if pd.notna(live) else None,
                         "within_15pct": bool(abs(m - typ) / typ <= cd.TOL), "seeds": seeds})
    return pd.DataFrame(rows)


def write_outputs(val: pd.DataFrame, params: dict, routes, lanes, lengths) -> None:
    import xml.etree.ElementTree as ET
    cd.DEMAND_DIR.mkdir(parents=True, exist_ok=True)
    work = cd._work_dir()
    (work / "mix_types.add.xml").write_text("<additional>\n" + cd.vtypes_xml(SPEED_FACTOR) + "</additional>\n")
    rt = {k: v for k, v in routes.items() if k in ("eb", "wb") or k.endswith("_wb") or k in cd.LEGS}
    for slot in cd.SLOTS:
        main = cd.build_demand(work / f"final_{slot}.rou.xml", rt, SLOT_SCALE[slot], SPEED_FACTOR, cd.SEED,
                               params.get("leg_scale", {}).get(slot))
        text = "\n".join(l for l in main.read_text().replace("</routes>\n", "").splitlines() if "probe_" not in l)
        bg = ET.parse(cd.background_trips(work, SLOT_SCALE[slot], cd.SEED)).getroot()
        veh = [ET.tostring(v, encoding="unicode").strip() for v in bg if v.tag == "vehicle"]
        (cd.DEMAND_DIR / f"{slot}.rou.xml").write_text(
            f"<!-- scripts/calibrate_osm_speed.py, slot {slot}: demand scale {SLOT_SCALE[slot]}, speed factor "
            f"{SPEED_FACTOR}; leg speeds imposed by {slot}.vss.add.xml (speed-only calibration) -->\n"
            + text + "\n  " + "\n  ".join(veh) + "\n</routes>\n")
        (cd.DEMAND_DIR / f"{slot}.vss.add.xml").write_text(vss_xml(lanes, params["limits_ms"][slot]))
    val.to_csv(cd.REPORT_CSV, index=False)
    lines = [
        "# RL corridor calibration (OSM ORR) — speed-only fallback", "",
        f"Generated {dt.date.today()} by `python -m scripts.calibrate_osm_speed`. Simulation, not field data.", "",
        "**Method.** Demand-only fitting failed (peak needed ~14x local demand on one leg, then gridlocked; see "
        "`rl_calibration_demand_v2_failed.md`). Here demand is fixed and moderate per slot and each leg's lane "
        "speed limit is fitted per slot so probe speeds match TomTom typical speeds. **Slot speeds are imposed, "
        "not emergent from demand.** Queues at the 4 RL signals still come from the simulated traffic.", "",
        f"* Driver speed factor {SPEED_FACTOR} (fitted on the night slot); demand scale per slot: "
        + ", ".join(f"{k} {v}" for k, v in SLOT_SCALE.items()),
        "* Lane limits capped at 60 km/h (OSM ORR limit); where a leg was still too slow at the cap (signal-limited), "
        "its local demand was lowered instead. Local-demand multipliers: " + "; ".join(
            f"{slot}: " + ", ".join(f"{k} {v}" for k, v in dm.items()) for slot, dm in params.get("leg_scale", {}).items()),
        "* Fitted lane speed limits (km/h): " + "; ".join(
            f"{slot}: " + ", ".join(f"{k} {v * 3.6:.1f}" for k, v in lim.items()) for slot, lim in params["limits_ms"].items()),
        "* Silk Board runs the reported 2024 peak program (285 s cycle: BTM/ORR-west 150 s, HSR/ORR-east 90 s, "
        "Hosur Rd 45 s; `sumo/osm/silkboard_2024.tll.xml`); other signals use OSM/netconvert programs.",
        "* Leg lengths in SUMO: " + ", ".join(f"{k} {lengths[k] / 1000:.2f} km" for k in (*cd.LEGS, "silkboard_krpuram")),
        "", "## Validation (5 seeds, target |error| ≤ 15 % vs TomTom typical)", "",
        "| slot | segment | TomTom typical km/h | TomTom live km/h | SUMO km/h (mean ± sd) | err vs typical | ≤15 % |",
        "|---|---|---|---|---|---|---|"]
    for r in val.itertuples():
        lines.append(f"| {r.slot} | {r.segment} | {r.tomtom_typical_kmh} | {r.tomtom_live_kmh} | "
                     f"{r.sumo_kmh_mean} ± {r.sumo_kmh_std} | {r.err_vs_typical_pct:+.1f} % | "
                     f"{'yes' if r.within_15pct else 'NO'} |")
    lines += ["", "## Limits", "",
              "* Speeds per leg and slot are imposed with variable speed signs; congestion is not reproduced by demand.",
              "* Demand level, through share and background trips are assumptions (scripts/calibrate_osm_demand.py).",
              "* Vehicle mix from a 2016 model table (D20); TomTom typical and live are one provider (docs/19 §9).",
              "* Westbound uses the eastbound leg limits (TomTom westbound typical times were not collected)."]
    cd.REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def log_rows(val: pd.DataFrame, params: dict) -> None:
    import subprocess as sp
    try:
        commit = sp.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    rows = []
    for slot in cd.SLOTS:
        v = val[val.slot == slot]
        full = v[v.segment == "silkboard_krpuram"].iloc[0]
        rows.append([dt.date.today().isoformat(), commit, "rl_calibration", f"osm_speed_{slot}",
                     json.dumps({"scale": SLOT_SCALE[slot], "speed_factor": SPEED_FACTOR,
                                 "leg_scale": params.get("leg_scale", {}).get(slot),
                                 "limits_kmh": {k: round(x * 3.6, 1) for k, x in params["limits_ms"][slot].items()}}),
                     f"{full.err_vs_typical_pct:.1f}", f"{full.sumo_kmh_std:.1f}", "",
                     "osm-orr-2026-10-22 + tomtom-typical", cd.SEED,
                     f"metric=err_vs_typical_pct (full corridor) SPEED-ONLY calibration (imposed leg speeds) "
                     f"seeds={full.seeds} legs_within_15pct={int(v.within_15pct.sum())}/{len(v)}"])
    pd.DataFrame(rows).to_csv(cd.LOG_PATH, mode="a", header=False, index=False)


def main() -> None:
    import sumolib
    ap = argparse.ArgumentParser(description="Speed-only calibration of the OSM ORR corridor")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=cd.N_SEEDS)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    net = sumolib.net.readNet(str(cd.NET))
    routes = cd.corridor_routes(net)
    lengths = cd.route_lengths(net, routes)
    lanes = leg_lanes(net, routes)
    targets = pd.read_csv(cd.TARGETS)
    if a.validate_only:
        params = json.loads(FIT_PATH.read_text())
    else:
        params = fit(targets, routes, lengths, lanes)
        cd.DEMAND_DIR.mkdir(parents=True, exist_ok=True)
        FIT_PATH.write_text(json.dumps(params, indent=1, default=float))
    val = validate(targets, routes, lengths, lanes, params, a.seeds)
    write_outputs(val, params, routes, lanes, lengths)
    print(val.to_string(index=False))
    if not a.no_log:
        log_rows(val, params)


if __name__ == "__main__":
    main()
