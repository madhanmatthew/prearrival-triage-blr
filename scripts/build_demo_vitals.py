"""Pick real VitalDB minute-by-minute vitals for the "Demo vitals" button (live mode).

    python -m scripts.build_demo_vitals        # reads data/raw/vitaldb, writes frontend/demo_vitals_vitaldb.json

Two windows of N_READINGS consecutive minutes, chosen by the DATA only (labels docs/08 §2b,
cleaning backend/data/vitaldb_windows.py), never by the model's output:
  * stable        : all 5 vitals present every minute, no event in the window or the next 30 min,
                    SBP 100-140, HR 55-100, SpO2 >= 96 throughout
  * deteriorating : all 5 vitals present, no event in the window, a NEW event (sustained SBP < 90
                    or SpO2 < 90) starting 3-8 min after the last reading, SBP falling >= 15 mmHg
                    across the window
First match in case-id order (deterministic). The deployed HGB was trained on all VitalDB cases
(--train-final), so these replays illustrate the pipeline; they are NOT a test result.
AVPU and supplemental O2 are not recorded in VitalDB (anaesthetised patients) -> sent as null.
Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.data.vitaldb_windows import RAW_DIR, VITALS, clean_case, event_minutes, label_known

OUT = Path("frontend/demo_vitals_vitaldb.json")
N_READINGS = 10          # minutes replayed (the 10-min model uses the last 5)
STABLE_AHEAD = 30        # minutes with no event after a "stable" window
DET_ONSET = (3, 8)       # first event this many minutes after the last reading
DET_SBP_DROP = 15.0


def _readings(c: pd.DataFrame, t: int) -> list[dict]:
    w = c.iloc[t - N_READINGS + 1:t + 1]
    return [{"hr": int(round(r.hr)), "rr": int(round(r.rr)), "spo2": int(round(r.spo2)),
             "sbp": int(round(r.sbp)), "temp": round(float(r.temp), 1), "avpu": None,
             "supplemental_o2": None} for r in w.itertuples()]


def find_windows(c: pd.DataFrame) -> dict[str, tuple[int, str]]:
    """{kind: (end index, outcome text)} for the first stable / deteriorating window in a case."""
    vals = c[VITALS].to_numpy(float)
    ev, known = event_minutes(c), label_known(c)
    full = ~np.isnan(vals).any(axis=1)
    sbp, hr, spo2 = c.sbp.to_numpy(), c.hr.to_numpy(), c.spo2.to_numpy()
    found: dict[str, tuple[int, str]] = {}
    for t in range(N_READINGS - 1, len(c) - 1):
        win = slice(t - N_READINGS + 1, t + 1)
        if not full[win].all() or ev[win].any():
            continue
        if "stable" not in found and t + STABLE_AHEAD < len(c):
            ahead = slice(t + 1, t + 1 + STABLE_AHEAD)
            if (known[ahead].all() and not ev[ahead].any() and sbp[win].min() >= 100
                    and sbp[win].max() <= 140 and hr[win].min() >= 55 and hr[win].max() <= 100
                    and spo2[win].min() >= 96):
                found["stable"] = (t, f"Real outcome: no SBP or SpO2 below 90 in the next {STABLE_AHEAD} min.")
        if "deteriorating" not in found and t + DET_ONSET[1] < len(c):
            onset = next((k for k in range(1, DET_ONSET[1] + 1) if ev[t + k]), None)
            if onset is not None and onset >= DET_ONSET[0] and sbp[t - N_READINGS + 1] - sbp[t] >= DET_SBP_DROP:
                e = t + onset
                found["deteriorating"] = (t, f"Real outcome: sustained low reading from +{onset - 1} min "
                                             f"(SBP {sbp[e]:.0f} mmHg, SpO2 {spo2[e]:.0f} %).")
        if len(found) == 2:
            break
    return found


def build(raw_dir: Path = RAW_DIR, out: Path = OUT) -> dict:
    cases = pd.read_csv(raw_dir / "cases.csv", usecols=["caseid"])
    picked: dict[str, dict] = {}
    for cid in sorted(cases.caseid):
        f = raw_dir / f"case_{cid}.csv.gz"
        if not f.exists():
            continue
        c = clean_case(pd.read_csv(f))
        for kind, (t, outcome) in find_windows(c).items():
            if kind not in picked:              # one window per case -> two different patients
                picked[kind] = {"id": f"vitaldb_{kind}", "name": f"VitalDB case {cid}: {kind}",
                                "caseid": int(cid), "end_minute": int(c.index[t]), "outcome": outcome,
                                "readings": _readings(c, t)}
                break
        if len(picked) == 2:
            break
    doc = {"_note": "REAL intra-operative vitals from the VitalDB open dataset (1-min medians), one reading per "
                    "minute, picked by data rules in scripts/build_demo_vitals.py. Anaesthetised surgical patients, "
                    "not pre-hospital; the deployed model saw these cases in training, so replays are an "
                    "illustration, not a test result. AVPU/O2 not recorded in VitalDB.",
           "sets": [picked[k] for k in ("deteriorating", "stable") if k in picked]}
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def _model_check(doc: dict) -> None:
    """Print the deployed model's estimate at the last reading (information only, not used to pick)."""
    models = sorted(Path("models").glob("vitaldb_hgb_v*.joblib"))
    if not models:
        return
    from backend.ml.vitaldb_model import load_hgb, predict_hgb
    b = load_hgb(models[-1])
    for s in doc["sets"]:
        X = np.array([[[r[v] for v in VITALS] for r in s["readings"][-5:]]], np.float32)
        print(f"  {s['id']}: P(event in 10 min) = {predict_hgb(b, X, np.ones((1, 5), bool))[0]:.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Real VitalDB windows for the live 'Demo vitals' replay")
    p.add_argument("--raw", type=Path, default=RAW_DIR)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    doc = build(a.raw, a.out)
    for s in doc["sets"]:
        print(f"{s['id']}: case {s['caseid']} ending minute {s['end_minute']} - {s['outcome']}")
    _model_check(doc)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
