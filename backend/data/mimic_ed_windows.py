"""MIMIC-IV-ED Demo v2.2 -> hourly vitals + ED outcomes, same feature layout as mimic_windows.

Used only for the external check in backend/ml/ed_external_eval.py (frozen ICU-trained BiGRU
applied to ED stays). Nothing here is used for training.

Tables (MIMIC-IV-ED v2.2, open-access demo; columns checked at load, [TODO-VERIFY] against
the downloaded files):
    edstays   subject_id, stay_id, intime, outtime, disposition
    triage    stay_id, temperature, heartrate, resprate, o2sat, sbp, acuity
    vitalsign stay_id, charttime, temperature, heartrate, resprate, o2sat, sbp

[ASSUMPTION] choices (logged in docs/progress/madhan.md):
* Triage vitals have no charttime; they are placed at ED `intime` (bin 0).
* Temperature is charted in Fahrenheit; values in 25-45 are taken as already Celsius,
  values in 77-113 are converted, anything else is dropped.
* The ED tables record no GCS/AVPU and no O2 device: avpu_bin = 0 (alert) and
  supplemental_o2 = 0 (room air), and NEWS2 is computed with the same two assumptions.
* Outcomes: `admitted` = disposition ADMITTED/EXPIRED vs HOME (others excluded);
  `high_acuity` = triage ESI 1-2 vs 3-5; `icu_24h` = an ICU stay (MIMIC-IV Demo icustays,
  same subject) starting between ED intime and ED outtime + 24 h.

Research prototype, not a medical device.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.data.mimic_windows import FFILL_LIMIT, PLAUSIBLE, T, make_windows
from backend.ml.news2 import news2_score

ED_RAW_DIR = Path("data/raw/mimic_ed_demo")
VITAL_COLS = {"heartrate": "hr", "resprate": "rr", "o2sat": "spo2", "sbp": "sbp", "temperature": "temp_raw"}
REQUIRED = {
    "edstays": ["subject_id", "stay_id", "intime", "outtime", "disposition"],
    "triage": ["stay_id", "acuity", *VITAL_COLS],
    "vitalsign": ["stay_id", "charttime", *VITAL_COLS],
}
ADMITTED, DISCHARGED = {"ADMITTED", "EXPIRED"}, {"HOME"}
ICU_HORIZON_H = 24


def _find(raw_dir: Path, name: str) -> Path:
    hits = sorted(raw_dir.rglob(f"{name}.csv*"))
    if not hits:
        raise FileNotFoundError(f"{name}.csv(.gz) not found under {raw_dir}")
    return hits[0]


def load_ed_tables(raw_dir: Path = ED_RAW_DIR) -> dict[str, pd.DataFrame]:
    out = {}
    for name, cols in REQUIRED.items():
        df = pd.read_csv(_find(raw_dir, name))
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}; found {list(df.columns)}")
        out[name] = df
    for col in ("intime", "outtime"):
        out["edstays"][col] = pd.to_datetime(out["edstays"][col])
    out["vitalsign"]["charttime"] = pd.to_datetime(out["vitalsign"]["charttime"])
    return out


def temp_to_celsius(t: pd.Series) -> pd.Series:
    t = pd.to_numeric(t, errors="coerce")
    return t.where(t.between(25, 45), ((t - 32) * 5 / 9).where(t.between(77, 113)))


def ed_hourly(edstays: pd.DataFrame, triage: pd.DataFrame, vitalsign: pd.DataFrame) -> pd.DataFrame:
    """One row per (stay_id, hour) from ED intime, same columns make_windows() needs."""
    stays = edstays[["subject_id", "stay_id", "intime", "outtime"]].copy()
    stays["n_hours"] = np.maximum(
        np.ceil((stays.outtime - stays.intime) / pd.Timedelta(hours=1)), 1).astype(int)
    tri = triage[["stay_id", *VITAL_COLS]].merge(stays[["stay_id", "intime"]], on="stay_id")
    tri = tri.rename(columns={"intime": "charttime"})
    r = pd.concat([tri, vitalsign[["stay_id", "charttime", *VITAL_COLS]]]).rename(columns=VITAL_COLS)
    for col in ("hr", "rr", "spo2", "sbp"):
        r[col] = pd.to_numeric(r[col], errors="coerce")
    r["temp"] = temp_to_celsius(r.temp_raw)
    r = r.merge(stays[["stay_id", "intime", "n_hours"]], on="stay_id")
    r["hour"] = np.floor((r.charttime - r.intime) / pd.Timedelta(hours=1)).astype(int)
    r = r[(r.hour >= 0) & (r.hour < r.n_hours)]
    for col, (lo, hi) in PLAUSIBLE.items():
        r.loc[(r[col] < lo) | (r[col] > hi), col] = np.nan
    vit = ["hr", "rr", "spo2", "sbp", "temp"]
    med = r.groupby(["stay_id", "hour"])[vit].median()

    grid = pd.MultiIndex.from_tuples(
        [(s, h) for s, n in zip(stays.stay_id, stays.n_hours) for h in range(n)], names=["stay_id", "hour"])
    out = pd.DataFrame(index=grid).join(med).reindex(columns=vit)
    out[vit] = out.groupby(level="stay_id")[vit].ffill(limit=FFILL_LIMIT)
    out["avpu_bin"] = 0.0        # [ASSUMPTION] not recorded in ED tables -> alert
    out["supplemental_o2"] = 0.0  # [ASSUMPTION] not recorded -> room air
    out["has_reading"] = out[vit].notna().any(axis=1)
    out["news2"] = pd.array([
        news2_score(rr=x.rr, spo2=x.spo2, supplemental_o2=False, sbp=x.sbp, hr=x.hr, avpu="A",
                    temp=x.temp).total for x in out.itertuples()], dtype="Int64")
    out = out.reset_index().merge(stays[["stay_id", "subject_id"]], on="stay_id")
    # make_windows() needs label columns; ED stays have no ICU labels
    return out.rename(columns={"subject_id": "patient_id"}).assign(severity_label=-1, trend_label=np.nan)


def ed_outcomes(edstays: pd.DataFrame, triage: pd.DataFrame,
                icustays: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per ED stay: admitted, high_acuity, icu_24h (float, NaN = excluded / unavailable)."""
    o = edstays[["subject_id", "stay_id", "intime", "outtime", "disposition"]].merge(
        triage[["stay_id", "acuity"]], on="stay_id", how="left")
    disp = o.disposition.astype(str).str.upper().str.strip()
    o["admitted"] = np.where(disp.isin(ADMITTED), 1.0, np.where(disp.isin(DISCHARGED), 0.0, np.nan))
    acuity = pd.to_numeric(o.acuity, errors="coerce")
    o["high_acuity"] = (acuity <= 2).astype(float).where(acuity.between(1, 5))
    o["icu_24h"] = np.nan
    if icustays is not None:
        icu = icustays[["subject_id", "intime"]].rename(columns={"intime": "icu_intime"})
        icu["icu_intime"] = pd.to_datetime(icu.icu_intime)
        m = o[["stay_id", "subject_id", "intime", "outtime"]].merge(icu, on="subject_id")
        hit = m[(m.icu_intime >= m.intime)
                & (m.icu_intime <= m.outtime + pd.Timedelta(hours=ICU_HORIZON_H))].stay_id
        if o.subject_id.isin(icustays.subject_id).any():  # only defined if the demos share patients
            o["icu_24h"] = o.stay_id.isin(hit).astype(float)
    return o[["stay_id", "subject_id", "admitted", "high_acuity", "icu_24h"]]


def first_hours_windows(hourly: pd.DataFrame, hours: int = T):
    """Per stay, the window ending at the last bin with a reading inside the first `hours` bins.

    Returns X [S,6,7], mask [S,6], stay_ids [S], news2 at the window end [S] (NaN if incomplete).
    Stays with no reading in that range are skipped.
    """
    Xs, masks, ids, n2 = [], [], [], []
    early = hourly[hourly.hour < hours]
    for stay_id, g in early.groupby("stay_id"):
        if not g.has_reading.any():
            continue
        X, mask, _, _, _, news2 = make_windows(g)
        Xs.append(X[-1])
        masks.append(mask[-1])
        ids.append(stay_id)
        n2.append(news2[-1])
    if not Xs:
        raise ValueError("no ED stay has a reading in the first hours")
    return np.stack(Xs), np.stack(masks), np.array(ids), np.array(n2, dtype=float)
