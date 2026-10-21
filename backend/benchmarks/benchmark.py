"""System benchmark: AI dispatch vs a fair naive baseline on 8 + 2 scenarios (docs/15 D15, docs/08 §7.4).

    python -m backend.benchmarks.benchmark            # writes reports/benchmark_*.csv, logs summary

Scenarios: data/benchmark_scenarios.csv (8 Bengaluru incidents + 2 blood-match cases where two
hospitals are equidistant and only one has the needed blood group; that scenario's simulated stock
is set in `hospital_overrides`). Hospitals / ambulances: data/hospitals.csv, data/ambulance_bases.csv.

Both methods share EVERYTHING except the decision logic (fairness, D15):
  * same ETA model (straight-line x road factor at a fixed speed, backend/routing.py; ORS is not
    used so runs are reproducible and free),
  * same dispatch overhead and on-scene time, same ambulances and hospital state,
  * same severity input: the scenario's severity_class (isolates dispatch logic from the triage model).
naive:  nearest ambulance, nearest hospital (no capability, bed or blood check).
ai:     rule-based coordinator (backend/agents/coordinator.py) + hospital agent, rule-only scores
        (the RF surrogate is not trained; say so).
Outcome per scenario: times (minutes, estimated) and whether the hospital is APPROPRIATE:
trauma_level >= severity, beds > 0, and blood stock when blood is needed (hospital agent rule).
Hospital beds/blood are SIMULATED, so this measures decision logic on simulated hospital state,
not patient outcomes. Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from backend.agents import hospital_agent as ha
from backend.agents.coordinator import IncidentCase, coordinate
from backend.routing import StraightLineRouter
from backend.schemas import FusedRisk

SCENARIOS_PATH = Path("data/benchmark_scenarios.csv")
HOSPITALS_PATH = Path("data/hospitals.csv")
AMBULANCES_PATH = Path("data/ambulance_bases.csv")
REPORT_DIR = Path("reports")
LOG_PATH = Path("experiments/log.csv")
DISPATCH_OVERHEAD_MIN = 2.0   # [ASSUMPTION] call-taking + dispatch, identical for both methods
ON_SCENE_MIN = 10.0           # [ASSUMPTION] identical for both methods
NOW = "2026-10-21T09:00:00Z"


def load_scenarios(path: Path = SCENARIOS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"blood_group": str}, keep_default_na=False)
    df["bleeding_heavy"] = df["bleeding_heavy"].astype(str).str.lower().eq("true")
    df["blood_group"] = df["blood_group"].replace("", None)
    df["hospital_overrides"] = df["hospital_overrides"].apply(lambda s: json.loads(s) if s else {})
    return df


def load_ambulances(path: Path = AMBULANCES_PATH) -> dict[str, tuple[float, float]]:
    df = pd.read_csv(path, dtype={"ambulance_id": str})
    return {r.ambulance_id: (float(r.lat), float(r.lon)) for r in df.itertuples()}


def apply_overrides(hospitals: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    h = hospitals.copy()
    for hid, cols in overrides.items():
        if hid not in set(h.hospital_id):
            raise KeyError(f"override for unknown hospital {hid}")
        for col, v in cols.items():
            h.loc[h.hospital_id == hid, col] = v
    return h


def one_hot_risk(sid: str, severity: int) -> FusedRisk:
    p = [0.0] * 4
    p[severity] = 1.0
    return FusedRisk.from_probs(p, incident_id=sid, patient_id=f"{sid}-P1", timestamp=NOW,
                                modalities_used=["text"], model_version="benchmark: scenario severity")


def appropriateness(h: pd.Series, sc) -> dict:
    need = ha.blood_needed(int(sc.severity_class), sc.injury_type, bool(sc.bleeding_heavy))
    group = sc.blood_group if sc.blood_group in ha.BLOOD_GROUPS else "O-"
    stock = float(h[f"blood_{group}"])
    out = {"trauma_ok": int(h.trauma_level) >= int(sc.severity_class),
           "beds_ok": float(h.beds_available) > 0,
           "blood_needed": need, "blood_ok": (stock >= 1) if need else True,
           "specialty": ha.specialty_match(sc.injury_type, h.specialties)}
    out["appropriate"] = out["trauma_ok"] and out["beds_ok"] and out["blood_ok"]
    return out


def _times(router: StraightLineRouter, amb_pos, scene, hosp_pos) -> dict:
    to_scene = router.eta_min(amb_pos, scene)[0]
    to_hosp = router.eta_min(scene, hosp_pos)[0]
    return {"response_min": round(DISPATCH_OVERHEAD_MIN + to_scene, 2),
            "transport_min": round(to_hosp, 2),
            "total_min": round(DISPATCH_OVERHEAD_MIN + to_scene + ON_SCENE_MIN + to_hosp, 2)}


def run_scenario(sc, hospitals: pd.DataFrame, ambulances: dict) -> list[dict]:
    h = apply_overrides(hospitals, sc.hospital_overrides)
    hpos = {str(r.hospital_id): (float(r.lat), float(r.lon)) for r in h.itertuples()}
    scene = (float(sc.lat), float(sc.lon))
    router = StraightLineRouter(lambda _: scene, hpos.get)
    amb_eta = {a: router.eta_min(p, scene)[0] for a, p in ambulances.items()}
    hosp_eta = {hid: router.eta_min(scene, p)[0] for hid, p in hpos.items()}
    rows = []

    # naive: nearest ambulance, nearest hospital
    na = min(amb_eta, key=lambda a: (amb_eta[a], a))
    nh = min(hosp_eta, key=lambda k: (hosp_eta[k], k))
    rows.append(("naive", na, nh, "nearest ambulance + nearest hospital"))

    # ai: hospital agent (rule-only) + coordinator
    agent = ha.HospitalAgent(h)
    risk = one_hot_risk(sc.scenario_id, int(sc.severity_class))
    rec = agent.predict(sc.scenario_id, risk, sc.injury_type, hosp_eta, blood_group=sc.blood_group,
                        bleeding_heavy=bool(sc.bleeding_heavy), use_rf=False, top_k=len(hosp_eta))
    case = IncidentCase(incident_id=sc.scenario_id, risk=risk, injury_type=sc.injury_type,
                        reported_at=NOW, recommendation=rec, ambulance_etas_min=amb_eta,
                        bleeding_heavy=bool(sc.bleeding_heavy))
    beds = {str(r.hospital_id): int(r.beds_available) for r in h.itertuples()}
    res = coordinate([case], now=NOW, router=router, available_ambulances=list(ambulances),
                     police_units_available=0, beds_now=beds)
    d = res.decisions[0]
    rows.append(("ai", d.ambulance_id, d.hospital_id, d.explanation))

    out = []
    for method, amb, hid, why in rows:
        hrow = h.loc[h.hospital_id == hid].iloc[0]
        out.append({"scenario_id": sc.scenario_id, "name": sc.name, "method": method,
                    "ambulance_id": amb, "hospital_id": hid, "hospital": hrow["name"],
                    "trauma_level": int(hrow.trauma_level), "severity_class": int(sc.severity_class),
                    **_times(router, ambulances[amb], scene, hpos[hid]),
                    **appropriateness(hrow, sc), "expected_rule": sc.expected_best_hospital_rule,
                    "why": why})
    return out


def run(scenarios: pd.DataFrame, hospitals: pd.DataFrame, ambulances: dict) -> pd.DataFrame:
    rows = []
    for sc in scenarios.itertuples():
        rows.extend(run_scenario(sc, hospitals, ambulances))
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    g = results.groupby("method", sort=False)
    s = g.agg(scenarios=("scenario_id", "count"), appropriate=("appropriate", "sum"),
              trauma_ok=("trauma_ok", "sum"), blood_ok=("blood_ok", "sum"),
              mean_response_min=("response_min", "mean"), mean_transport_min=("transport_min", "mean"),
              mean_total_min=("total_min", "mean")).round(2)
    return s.reset_index()


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_summary(summary: pd.DataFrame) -> None:
    today, commit = dt.date.today().isoformat(), _git_commit()
    lines = [[today, commit, "system", f"benchmark_{r.method}", "{}",
              f"{r.appropriate / r.scenarios:.4f}", "", f"{r.mean_total_min:.2f}",
              "benchmark-8+2 (hospital beds/blood SIMULATED)", 42,
              f"metric=appropriate_rate test_score=mean_total_min scenarios={r.scenarios} "
              f"appropriate={r.appropriate} mean_response={r.mean_response_min} "
              f"mean_transport={r.mean_transport_min} straight-line ETAs; fair baseline D15"]
             for r in summary.itertuples()]
    pd.DataFrame(lines).to_csv(LOG_PATH, mode="a", header=False, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="AI vs naive dispatch benchmark (8 + 2 scenarios)")
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()
    results = run(load_scenarios(), ha.load_hospitals(HOSPITALS_PATH), load_ambulances())
    summary = summarize(results)
    REPORT_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORT_DIR / "benchmark_results.csv", index=False)
    summary.to_csv(REPORT_DIR / "benchmark_summary.csv", index=False)
    cols = ["scenario_id", "method", "ambulance_id", "hospital_id", "total_min", "appropriate"]
    print(results[cols].to_string(index=False))
    print(summary.to_string(index=False))
    if not a.no_log:
        log_summary(summary)
        print(f"appended {len(summary)} rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
