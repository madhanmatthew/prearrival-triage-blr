"""Fit SUMO demand on the OSM ORR corridor to the per-slot targets (docs/19 §5–6, D20, D21).

    python -m scripts.calibrate_osm_demand                    # fit + validate, write everything
    python -m scripts.calibrate_osm_demand --validate-only    # re-validate the saved fit

Outputs: sumo/demand/{low,medium,peak}.rou.xml (fitted demand, used by backend/rl/osm_env.py),
sumo/demand/fit.json (fitted parameters), reports/rl_calibration.md (+ .csv), rows in
experiments/log.csv (component=rl_calibration).

Model (every number here is a knob, documented):
  * Vehicle mix [ASSUMPTION, source KRDCL DFR 2019, BAU 2016 model PCUs, Old Madras Rd - KR Puram,
    converted to vehicles with IRC 106-1990 PCU factors 2W 0.5, auto 1.2, car 1.0, bus 3.7]:
    two-wheeler 0.49, car 0.38, auto 0.11, bus 0.02. Not a 2026 count (D20).
  * Demand = ORR through-traffic both directions (eastbound / westbound routes from
    scripts/build_osm_corridor.py) at MAIN_VPH x scale, plus random background trips on the extract
    at BG_VPH x scale (randomTrips, fringe factor 10, validated by duarouter).
  * Driver desired-speed factor f (all types, normc(f, 0.1)): OSM limits (60 km/h on the ORR) are
    faster than real night traffic, so f is fitted on the LOW slot to the TomTom night corridor
    speed with demand fixed at LOW_SCALE. Then `scale` is fitted per slot (medium, peak) to the
    TomTom typical Silk Board -> KR Puram speed, f fixed.
  * Measurement: probe cars (type `probe`, not counted as demand) every PROBE_PERIOD_S on each leg
    and on the full corridor, after WARMUP_S; speed = route length / tripinfo duration. Speeds are
    compared (SUMO leg lengths differ from TomTom's by up to 12%).
  * Validation: N_SEEDS seeds per slot; error vs TomTom typical and vs live (live = mean point speed
    at the leg start: indicative only, a point is not a segment). Target |error| <= 15 % (docs/19 §6).
  * Volume check: SUMO eastbound flow on the Iblur–Bellandur stretch vs the BTP 2025 peak figure
    (PCU, peak hours, direction not stated) -> reported, not fitted.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import build_osm_corridor as bo

NET = Path("sumo/osm/corridor.net.xml")
DEMAND_DIR = Path("sumo/demand")
TARGETS = Path("data/traffic/calibration_targets.csv")
REPORT_MD = Path("reports/rl_calibration.md")
REPORT_CSV = Path("reports/rl_calibration.csv")
LOG_PATH = Path("experiments/log.csv")
SLOTS = ("low", "medium", "peak")
LEGS = ("silkboard_bellandur", "bellandur_marathahalli", "marathahalli_krpuram")
MIX = {"motorcycle": 0.49, "car": 0.38, "autorickshaw": 0.11, "bus": 0.02}
PCU = {"motorcycle": 0.5, "car": 1.0, "autorickshaw": 1.2, "bus": 3.7}           # IRC 106-1990
MAIN_VPH = 3000.0          # ORR vehicles/h per direction on each leg at scale 1 [ASSUMPTION, fitted via scale]
THROUGH_SHARE = 0.4        # share of MAIN_VPH driving the whole corridor; the rest enters/leaves per leg
                           # (on-ramps, side roads, service roads) [ASSUMPTION]. Added 2026-10-22: v1 fed all
                           # demand in before Silk Board and its queue starved the downstream legs.
BG_VPH = 3000.0            # background vehicles/h on the extract at scale 1 [ASSUMPTION, fitted via scale]
LOW_SCALE = 0.15           # night demand while fitting the speed factor [ASSUMPTION]
WARMUP_S = 900
MEASURE_S = 2700
PROBE_PERIOD_S = 120
N_SEEDS = 5
TOL = 0.15
SEED = 42


def _sumo_bin(name: str) -> str:
    import sumo
    import sumolib
    os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)
    return sumolib.checkBinary(name)


def _work_dir() -> Path:
    # SUMO writes paths into XML comments; a '--' in the path breaks them, so avoid such folders
    base = Path(tempfile.gettempdir())
    if "--" in str(base):
        base = Path(os.environ.get("SystemDrive", "C:") + "/") / "Temp"
    d = base / "orr_calibration"
    d.mkdir(parents=True, exist_ok=True)
    return d


def corridor_routes(net=None) -> dict[str, list[str]]:
    """eb / wb through routes and the probe routes per leg (eastbound) + full corridor."""
    import sumolib
    net = net or sumolib.net.readNet(str(NET))
    points = json.loads(bo.POINTS.read_text(encoding="utf-8"))["junctions"]
    eb_legs = bo.find_legs(net, points, bo.EASTBOUND)
    eb = json.loads((bo.OUT_DIR / "orr_eastbound_route.json").read_text())["edges"]
    wb = [e.getID() for e in bo.find_route(net, points, bo.WESTBOUND, lead_in_m=0)]
    out = {"eb": eb, "wb": wb}
    for name, leg in zip(LEGS, eb_legs):
        out[name] = [e.getID() for e in leg]
    wb_legs = bo.find_legs(net, points, bo.WESTBOUND)          # krpuram->marathahalli->bellandur->silkboard
    for name, leg in zip(reversed(LEGS), wb_legs):
        out[f"{name}_wb"] = [e.getID() for e in leg]
    out["silkboard_krpuram"] = [e.getID() for e in bo.find_route(net, points, bo.EASTBOUND, lead_in_m=0)]
    return out


def route_lengths(net, routes: dict) -> dict[str, float]:
    return {k: sum(net.getEdge(e).getLength() for e in v) for k, v in routes.items()}


def vtypes_xml(speed_factor: float) -> str:
    sf = f"normc({speed_factor:.3f},0.1,0.2,2)"
    return f"""  <vTypeDistribution id="mix">
    <vType id="car" vClass="passenger" probability="{MIX['car']}" speedFactor="{sf}"/>
    <vType id="motorcycle" vClass="motorcycle" length="2.2" minGap="1.0" maxSpeed="22" probability="{MIX['motorcycle']}" speedFactor="{sf}" latAlignment="arbitrary"/>
    <vType id="autorickshaw" vClass="passenger" length="2.6" maxSpeed="13.9" accel="1.5" probability="{MIX['autorickshaw']}" speedFactor="{sf}"/>
    <vType id="bus" vClass="bus" probability="{MIX['bus']}" speedFactor="{sf}"/>
  </vTypeDistribution>
  <vType id="probe" vClass="passenger" speedFactor="{sf}" color="0,0,255"/>
"""


def build_demand(path: Path, routes: dict, scale: float, speed_factor: float, seed: int,
                 leg_scale: dict[str, float] | None = None) -> Path:
    """Route file: mix flows on eb/wb, background trips (pre-routed), probes on legs + full corridor."""
    end = WARMUP_S + MEASURE_S
    lines = ['<routes>', vtypes_xml(speed_factor)]
    for rid, edges in routes.items():
        lines.append(f'  <route id="{rid}" edges="{" ".join(edges)}"/>')
    for d in ("eb", "wb"):
        lines.append(f'  <flow id="main_{d}" type="mix" route="{d}" begin="0" end="{end}" '
                     f'vehsPerHour="{MAIN_VPH * scale * THROUGH_SHARE:.1f}" departLane="best" departSpeed="max"/>')
        for leg in LEGS:
            rid = leg if d == "eb" else f"{leg}_wb"
            if rid in routes:
                m = (leg_scale or {}).get(leg, 1.0)            # same multiplier both directions [ASSUMPTION]
                lines.append(f'  <flow id="local_{rid}" type="mix" route="{rid}" begin="0" end="{end}" '
                             f'vehsPerHour="{MAIN_VPH * scale * m * (1 - THROUGH_SHARE):.1f}" departLane="best" '
                             f'departSpeed="max"/>')
    for leg in (*LEGS, "silkboard_krpuram"):
        if leg not in routes:
            continue
        lines.append(f'  <flow id="probe_{leg}" type="probe" route="{leg}" begin="{WARMUP_S}" '
                     f'end="{end}" period="{PROBE_PERIOD_S}" departLane="best" departSpeed="max"/>')
    lines.append("</routes>")
    path.write_text("\n".join(lines) + "\n")
    return path


def background_trips(work: Path, scale: float, seed: int) -> Path:
    """randomTrips + duarouter on the extract; vehicles get type `mix`."""
    import sumo
    out = work / f"bg_{scale:.3f}_{seed}.rou.xml"
    if out.exists():
        return out
    period = 3600.0 / max(BG_VPH * scale, 1e-6)
    tool = Path(sumo.SUMO_HOME) / "tools" / "randomTrips.py"
    import sys
    subprocess.run([sys.executable, str(tool), "-n", str(NET), "-o", str(work / f"bg_{scale:.3f}_{seed}.trips.xml"),
                    "-r", str(out), "-b", "0", "-e", str(WARMUP_S + MEASURE_S), "-p", f"{period:.4f}",
                    "--fringe-factor", "10", "--seed", str(seed), "--validate", "--trip-attributes",
                    'type="mix" departLane="best"', "--additional-file", str(work / "mix_types.add.xml")],
                   check=True, capture_output=True)
    # duarouter copies the type definitions into the output; the demand file already has them
    tree = ET.parse(out)
    root = tree.getroot()
    for el in [e for e in root if e.tag in ("vType", "vTypeDistribution")]:
        root.remove(el)
    tree.write(out, encoding="unicode")
    return out


def run_sumo(work: Path, demand: Path, bg: Path, seed: int, tag: str, edge_data: bool = False,
             extra_add: Path | None = None) -> Path:
    trip = work / f"tripinfo_{tag}.xml"
    cmd = [_sumo_bin("sumo"), "-n", str(NET), "-r", f"{demand},{bg}", "--seed", str(seed),
           "--tripinfo-output", str(trip), "--time-to-teleport", "300", "--no-step-log", "true",
           "--no-warnings", "true", "--ignore-route-errors", "true", "--end", str(WARMUP_S + MEASURE_S + 1800),
           "--device.rerouting.probability", "0"]
    if edge_data:
        add = work / f"edgedata_{tag}.add.xml"
        add.write_text(f'<additional><edgeData id="ed" file="{work / f"edgedata_{tag}.xml"}" '
                       f'begin="{WARMUP_S}" end="{WARMUP_S + MEASURE_S}"/></additional>\n')
        cmd += ["-a", str(add) + (f",{extra_add}" if extra_add else "")]
    elif extra_add:
        cmd += ["-a", str(extra_add)]
    subprocess.run(cmd, check=True, capture_output=True)
    return trip


def probe_speeds(trip: Path, lengths: dict[str, float]) -> dict[str, float]:
    """Mean probe speed (km/h) per probe route from tripinfo (probes that finished)."""
    durs: dict[str, list[float]] = {}
    for t in ET.parse(trip).getroot().iter("tripinfo"):
        vid = t.get("id")
        if not vid.startswith("probe_"):
            continue
        leg = vid[len("probe_"):].rsplit(".", 1)[0]
        durs.setdefault(leg, []).append(float(t.get("duration")))
    return {leg: lengths[leg] / np.mean(d) * 3.6 for leg, d in durs.items() if d}


def simulate(routes, lengths, scale, speed_factor, seed, tag, edge_data=False,
             leg_scale: dict[str, float] | None = None, extra_add: Path | None = None) -> dict[str, float]:
    work = _work_dir()
    (work / "mix_types.add.xml").write_text("<additional>\n" + vtypes_xml(speed_factor) + "</additional>\n")
    bg = background_trips(work, scale, seed)
    demand = build_demand(work / f"demand_{tag}.rou.xml", routes, scale, speed_factor, seed, leg_scale)
    trip = run_sumo(work, demand, bg, seed, tag, edge_data, extra_add)
    return probe_speeds(trip, lengths)


def _bisect(fn, lo, hi, target, increasing: bool, iters: int = 7, log=None):
    """Find x in [lo, hi] with fn(x) ~ target for a monotone fn (speed vs factor/scale)."""
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        v = fn(mid)
        if log is not None:
            log.append((mid, v))
        if best is None or abs(v - target) < abs(best[1] - target):
            best = (mid, v)
        if abs(v - target) / target < 0.03:
            break
        if (v < target) == increasing:
            lo = mid
        else:
            hi = mid
    return best


BASE_SCALE = 1.0          # through + background demand for medium/peak; per-leg multipliers do the rest
LEG_ITERS = 6
LEG_EXPONENT = 1.5        # multiplicative update m *= (sim / target) ** k  (speed falls as demand rises)
LEG_CLIP = (0.5, 2.0)     # max change per iteration
LEG_TOL = 0.10            # stop when every leg is within 10 % (validation target is 15 %)


def fit_legs(slot: str, t: pd.DataFrame, routes, lengths, f: float) -> tuple[dict, list]:
    """Per-leg demand multipliers so each leg's SUMO speed matches its TomTom typical speed.
    Added 2026-10-22: one global scale over-congested the signalised first leg while the
    signal-free legs stayed near free flow."""
    goal = {leg: float(t.loc[(slot, leg), "typical_speed_kmh"]) for leg in LEGS}
    m = {leg: 1.0 for leg in LEGS}
    log, best = [], None
    for it in range(LEG_ITERS):
        sp = simulate(routes, lengths, BASE_SCALE, f, SEED, f"fit{slot}_{it}", leg_scale=m)
        err = {leg: (sp.get(leg, np.nan) - goal[leg]) / goal[leg] for leg in LEGS}
        worst = max(abs(e) for e in err.values() if not np.isnan(e))
        log.append({"iter": it, "leg_scale": dict(m), "speed": {k: round(float(v), 1) for k, v in sp.items()},
                    "max_abs_err": round(worst, 3)})
        print(f"  {slot} iter {it}: max |err| {worst:.1%} m={ {k: round(v, 2) for k, v in m.items()} }", flush=True)
        if best is None or worst < best[0]:
            best = (worst, dict(m))
        if worst <= LEG_TOL:
            break
        for leg in LEGS:
            if not np.isnan(err[leg]):
                m[leg] *= float(np.clip((sp[leg] / goal[leg]) ** LEG_EXPONENT, *LEG_CLIP))
    return {k: round(v, 4) for k, v in best[1].items()}, log


def fit(targets: pd.DataFrame, routes, lengths) -> dict:
    t = targets.set_index(["slot", "segment"])
    log: list = []
    goal = t.loc[("low", "silkboard_krpuram"), "typical_speed_kmh"]
    f, v = _bisect(lambda x: simulate(routes, lengths, LOW_SCALE, x, SEED, f"fitf_{x:.3f}")["silkboard_krpuram"],
                   0.3, 1.2, goal, increasing=True, log=log)
    out = {"speed_factor": round(f, 4), "scale": {"low": LOW_SCALE},
           "leg_scale": {"low": {leg: 1.0 for leg in LEGS}}, "fit_log": {"low": log}}
    print(f"speed factor {f:.3f} (low slot corridor {v:.1f} vs TomTom {goal:.1f} km/h)", flush=True)
    for slot in ("medium", "peak"):
        legs, log = fit_legs(slot, t, routes, lengths, f)
        out["scale"][slot] = BASE_SCALE
        out["leg_scale"][slot] = legs
        out["fit_log"][slot] = log
    return out


def flow_check(work: Path, tag: str, edge_ids: list[str]) -> float | None:
    """Mean eastbound PCU/h on the given edges over the measurement window (SUMO edgeData)."""
    f = work / f"edgedata_{tag}.xml"
    if not f.exists():
        return None
    root = ET.parse(f).getroot()
    vals = []
    for e in root.iter("edge"):
        if e.get("id") in edge_ids and e.get("left") is not None:
            vals.append(float(e.get("left")))
    if not vals:
        return None
    pcu_per_veh = sum(MIX[k] * PCU[k] for k in MIX)
    return float(np.mean(vals)) * 3600.0 / MEASURE_S * pcu_per_veh


def validate(targets: pd.DataFrame, routes, lengths, params: dict, seeds: int = N_SEEDS) -> pd.DataFrame:
    import sumolib
    net = sumolib.net.readNet(str(NET))
    work = _work_dir()
    t = targets.set_index(["slot", "segment"])
    # Iblur -> Bellandur: eastbound edges between the 4th junction and the Bellandur point
    tm = json.loads((bo.OUT_DIR / "tls_map.json").read_text())["junctions"]
    off0 = list(tm.values())[-1]["route_offset_m"]
    run, check_edges = 0.0, []
    for e in routes["eb"]:
        run += net.getEdge(e).getLength()
        if off0 < run < off0 + 1500:
            check_edges.append(e)
    rows = []
    for slot in SLOTS:
        speeds, flows = [], []
        for k in range(seeds):
            tag = f"val_{slot}_{k}"
            speeds.append(simulate(routes, lengths, params["scale"][slot], params["speed_factor"], SEED + k, tag,
                                   edge_data=True, leg_scale=params.get("leg_scale", {}).get(slot)))
            flows.append(flow_check(work, tag, check_edges))
        for seg in (*LEGS, "silkboard_krpuram"):
            sim = [s.get(seg, np.nan) for s in speeds]
            typ = float(t.loc[(slot, seg), "typical_speed_kmh"])
            live = t.loc[(slot, seg), "live_speed_kmh"]
            m = float(np.nanmean(sim))
            rows.append({"slot": slot, "segment": seg, "tomtom_typical_kmh": round(typ, 1),
                         "tomtom_live_kmh": round(float(live), 1) if pd.notna(live) else None,
                         "sumo_kmh_mean": round(m, 1), "sumo_kmh_std": round(float(np.nanstd(sim)), 1),
                         "err_vs_typical_pct": round(100 * (m - typ) / typ, 1),
                         "err_vs_live_pct": round(100 * (m - live) / live, 1) if pd.notna(live) else None,
                         "within_15pct": abs(m - typ) / typ <= TOL, "seeds": seeds,
                         "eb_pcu_per_h_after_5_9km": round(float(np.nanmean([x for x in flows if x is not None])), 0)
                         if any(x is not None for x in flows) else None})
    return pd.DataFrame(rows)


def write_report(val: pd.DataFrame, params: dict, lengths: dict) -> None:
    REPORT_DIR = REPORT_MD.parent
    REPORT_DIR.mkdir(exist_ok=True)
    val.to_csv(REPORT_CSV, index=False)
    full = val[val.segment == "silkboard_krpuram"]
    lines = [
        "# RL corridor calibration (OSM ORR, docs/19 §6, D20, D21)", "",
        f"Generated {dt.date.today()} by `python -m scripts.calibrate_osm_demand`. Simulation, not field data.", "",
        f"* Fitted desired-speed factor (LOW slot, night): **{params['speed_factor']}**",
        "* Fitted demand scale per slot: " + ", ".join(f"{k} **{v}**" for k, v in params["scale"].items())
        + f" (per direction: through {MAIN_VPH * THROUGH_SHARE:.0f} veh/h x scale; local per leg "
        f"{MAIN_VPH * (1 - THROUGH_SHARE):.0f} veh/h x scale x leg multiplier; background {BG_VPH:.0f} veh/h x scale)",
        "* Per-leg multipliers (fitted to each leg's TomTom typical speed, same both directions): "
        + "; ".join(f"{slot}: " + ", ".join(f"{k} {v}" for k, v in legs.items())
                    for slot, legs in params.get("leg_scale", {}).items()),
        "* Vehicle mix: " + ", ".join(f"{k} {v:.0%}" for k, v in MIX.items())
        + " (KRDCL DFR 2019, 2016 model PCUs converted with IRC 106 factors; not a 2026 count)",
        f"* Speeds compared (SUMO leg lengths: " + ", ".join(f"{k} {lengths[k] / 1000:.2f} km" for k in (*LEGS, "silkboard_krpuram"))
        + "); target |error| <= 15 % vs TomTom typical. Live = point speed at leg start (indicative only).",
        "", "## Validation", "",
        "| slot | segment | TomTom typical km/h | TomTom live km/h | SUMO km/h (mean ± sd) | err vs typical | err vs live | ≤15 % |",
        "|---|---|---|---|---|---|---|---|"]
    for r in val.itertuples():
        lines.append(f"| {r.slot} | {r.segment} | {r.tomtom_typical_kmh} | {r.tomtom_live_kmh} | "
                     f"{r.sumo_kmh_mean} ± {r.sumo_kmh_std} | {r.err_vs_typical_pct:+.1f} % | "
                     f"{'' if r.err_vs_live_pct is None or pd.isna(r.err_vs_live_pct) else f'{r.err_vs_live_pct:+.1f} %'} | "
                     f"{'yes' if r.within_15pct else 'NO'} |")
    pk = full[full.slot == "peak"].eb_pcu_per_h_after_5_9km.iloc[0] if len(full) else None
    lines += ["", "## Volume check (not fitted)", "",
              f"SUMO eastbound flow just past the 5.9 km junction (toward Bellandur), peak slot: **{pk} PCU/h**. "
              "BTP 2025 (Deccan Herald): Iblur–Bellandur ~10,657 PCU at peak vs 4,800 design capacity; "
              "direction and exact hour not stated, so this is an order-of-magnitude check only.", "",
              "## Limits", "",
              "* One speed factor for all vehicle types and slots; demand scale fitted on the full corridor only.",
              "* Background trips are random OD pairs on the extract, not observed OD flows.",
              "* Signal programs other than the 4 RL junctions are OSM/netconvert defaults.",
              "* TomTom typical and live are one provider (docs/19 §9); live is a point speed."]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_demand_files(routes, params: dict) -> None:
    """Final per-slot route files for the RL env (seed SEED): through flows + background, no probes."""
    import shutil
    DEMAND_DIR.mkdir(parents=True, exist_ok=True)
    work = _work_dir()
    (work / "mix_types.add.xml").write_text("<additional>\n" + vtypes_xml(params["speed_factor"]) + "</additional>\n")
    rt = {k: v for k, v in routes.items() if k in ("eb", "wb") or k.endswith("_wb") or k in LEGS}
    for slot in SLOTS:
        s = params["scale"][slot]
        main = build_demand(work / f"final_{slot}.rou.xml", rt, s, params["speed_factor"], SEED,
                            params.get("leg_scale", {}).get(slot))
        text = main.read_text().replace("</routes>\n", "")
        text = "\n".join(l for l in text.splitlines() if "probe_" not in l)
        bg = ET.parse(background_trips(work, s, SEED)).getroot()
        veh = [ET.tostring(v, encoding="unicode").strip() for v in bg if v.tag == "vehicle"]
        (DEMAND_DIR / f"{slot}.rou.xml").write_text(
            f"<!-- fitted by scripts/calibrate_osm_demand.py, slot {slot}, scale {s}, "
            f"speed factor {params['speed_factor']} -->\n" + text + "\n  " + "\n  ".join(veh) + "\n</routes>\n")
    shutil.rmtree(work / "unused", ignore_errors=True)


def log_rows(val: pd.DataFrame, params: dict) -> None:
    import subprocess as sp
    try:
        commit = sp.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    rows = []
    for slot in SLOTS:
        v = val[val.slot == slot]
        full = v[v.segment == "silkboard_krpuram"].iloc[0]
        rows.append([dt.date.today().isoformat(), commit, "rl_calibration", f"osm_demand_{slot}",
                     json.dumps({"scale": params["scale"][slot], "speed_factor": params["speed_factor"],
                                 "leg_scale": params.get("leg_scale", {}).get(slot)}),
                     f"{full.err_vs_typical_pct:.1f}", f"{full.sumo_kmh_std:.1f}", "",
                     "osm-orr-2026-10-21 + tomtom-typical", SEED,
                     f"metric=err_vs_typical_pct (full corridor) seeds={full.seeds} legs_within_15pct="
                     f"{int(v.within_15pct.sum())}/{len(v)} sumo_kmh={full.sumo_kmh_mean} tomtom_kmh={full.tomtom_typical_kmh}"])
    pd.DataFrame(rows).to_csv(LOG_PATH, mode="a", header=False, index=False)


def main() -> None:
    import sumolib
    ap = argparse.ArgumentParser(description="Fit + validate SUMO demand on the OSM ORR corridor")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    net = sumolib.net.readNet(str(NET))
    routes = corridor_routes(net)
    lengths = route_lengths(net, routes)
    targets = pd.read_csv(TARGETS)
    fit_path = DEMAND_DIR / "fit.json"
    if a.validate_only:
        params = json.loads(fit_path.read_text())
    else:
        params = fit(targets, routes, lengths)
        DEMAND_DIR.mkdir(parents=True, exist_ok=True)
        fit_path.write_text(json.dumps(params, indent=1, default=float))
        print("fit:", {k: v for k, v in params.items() if k != "fit_log"}, flush=True)
    val = validate(targets, routes, lengths, params, a.seeds)
    write_report(val, params, lengths)
    write_demand_files(routes, params)
    print(val.to_string(index=False))
    if not a.no_log:
        log_rows(val, params)
        print(f"appended {len(SLOTS)} rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
