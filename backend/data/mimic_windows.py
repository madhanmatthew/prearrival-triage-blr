"""MIMIC-IV Demo v2.2 -> hourly vitals + outcome labels + 6-step windows (docs/08 §2, docs/07 §4).

    python -m backend.data.mimic_windows            # or: make data-mimic

Outputs `data/processed/mimic_hourly.csv.gz` (one row per ICU stay-hour; never committed)
and prints the class balance. Training code calls `load_hourly()` + `make_windows()`, fills
NaNs with the *training-fold* median, then calls `add_deltas()` (docs/07 §4.1, §4.3).

Pipeline (all choices below are logged in docs/progress/madhan.md):
1. Hour bins from ICU `intime`: bin h covers [intime + h, intime + h + 1); the reading
   time `t` of bin h is its end, intime + h + 1 hours. Labels look forward from `t`.
2. Numeric vitals: median per bin. Categorical/ordinal items (GCS parts, O2 device): last
   value in the bin.
3. Forward-fill at most 2 bins per stay (docs/07 §4.3); later gaps stay NaN.
4. Labels from real outcomes only (never from NEWS2, AGENTS.md rule 1).

Item IDs come from this dataset's `icu/d_items` (checked 2026-09-30, see ITEMS).
[ASSUMPTION] choices:
* SBP: non-invasive cuff preferred (ambulances measure NIBP); arterial line only when
  no cuff reading in that hour. RR: 220210, else 224690 (total).
* Temperature: Celsius item preferred, else Fahrenheit converted.
* Physically impossible values are dropped (PLAUSIBLE ranges below), not clipped.
* GCS total needs all three parts in the same hour. Intubated patients chart verbal
  "No Response-ETT" = 1, so they map to P/U (not alert).
* Supplemental O2: the demo never charts "room air", so O2 = 1 when an O2 device or
  O2 flow > 0 is charted (forward-filled <= 2 h), else 0 (room air assumed).
* Vasopressors = norepinephrine, epinephrine, vasopressin, phenylephrine, dopamine
  (inotropes dobutamine/milrinone excluded). Ventilation = procedureevents "Invasive
  Ventilation". MIMIC-IV splits infusions into rate-change rows, so "start" is ill-defined;
  `intervention_6h` = a vasopressor/ventilation interval overlaps (t, t+6h].
* Trend: windows already on vasopressor/ventilation during the bin ending at t are
  excluded (NaN); otherwise 1 if death or a new intervention in (t, t+6h].

Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backend.ml.news2 import gcs_to_avpu, news2_score

RAW_DIR = Path("data/raw/mimic_demo/mimic-iv-clinical-database-demo-2.2")
OUT_PATH = Path("data/processed/mimic_hourly.csv.gz")

ITEMS = {
    "hr": [220045],
    "rr": [220210, 224690],
    "spo2": [220277],
    "sbp_nibp": [220179],
    "sbp_art": [220050],
    "temp_c": [223762],
    "temp_f": [223761],
    "gcs_eye": [220739],
    "gcs_verbal": [223900],
    "gcs_motor": [223901],
    "o2_device": [226732],
    "o2_flow": [223834],
}
VASOPRESSOR_ITEMS = [221906, 221289, 222315, 221749, 221662]
INVASIVE_VENT_ITEM = 225792
PLAUSIBLE = {"hr": (1, 300), "rr": (1, 100), "spo2": (1, 100), "sbp": (1, 300), "temp": (25, 45)}

FEATURES = ["hr", "rr", "spo2", "sbp", "temp", "avpu_bin", "supplemental_o2"]  # docs/07 §4.1
DELTA_FEATURES = ["hr", "rr", "spo2", "sbp", "temp"]
T = 6
FFILL_LIMIT = 2
TREND_HORIZON_H = 6


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read(raw_dir: Path, rel: str, **kw) -> pd.DataFrame:
    return pd.read_csv(raw_dir / rel, **kw)


def load_tables(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    item_ids = [i for ids in ITEMS.values() for i in ids]
    ce = _read(raw_dir, "icu/chartevents.csv.gz",
               usecols=["stay_id", "charttime", "itemid", "value", "valuenum"],
               parse_dates=["charttime"])
    ie = _read(raw_dir, "icu/inputevents.csv.gz",
               usecols=["stay_id", "starttime", "endtime", "itemid"],
               parse_dates=["starttime", "endtime"])
    pe = _read(raw_dir, "icu/procedureevents.csv.gz",
               usecols=["stay_id", "starttime", "endtime", "itemid"],
               parse_dates=["starttime", "endtime"])
    return {
        "stays": _read(raw_dir, "icu/icustays.csv.gz",
                       usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
                       parse_dates=["intime", "outtime"]),
        "admissions": _read(raw_dir, "hosp/admissions.csv.gz",
                            usecols=["hadm_id", "dischtime", "deathtime", "hospital_expire_flag"],
                            parse_dates=["dischtime", "deathtime"]),
        "chart": ce[ce.itemid.isin(item_ids)],
        "interventions": pd.concat([
            ie[ie.itemid.isin(VASOPRESSOR_ITEMS)],
            pe[pe.itemid == INVASIVE_VENT_ITEM],
        ])[["stay_id", "starttime", "endtime"]],
    }


# ---------------------------------------------------------------------------
# Hourly features
# ---------------------------------------------------------------------------

def _item_name_map() -> dict[int, str]:
    return {i: name for name, ids in ITEMS.items() for i in ids}


def hourly_features(chart: pd.DataFrame, stays: pd.DataFrame) -> pd.DataFrame:
    """One row per (stay_id, hour) on a complete grid, features after <= 2-bin forward fill."""
    stays = stays.assign(n_hours=np.ceil((stays.outtime - stays.intime) / pd.Timedelta(hours=1)).astype(int))
    c = chart.merge(stays[["stay_id", "intime", "n_hours"]], on="stay_id")
    c["hour"] = np.floor((c.charttime - c.intime) / pd.Timedelta(hours=1)).astype(int)
    c = c[(c.hour >= 0) & (c.hour < c.n_hours)]
    c["name"] = c.itemid.map(_item_name_map())
    # rr has two items: keep the preferred one when both exist in the same hour
    c["rank"] = c.itemid.map({ITEMS["rr"][0]: 0, ITEMS["rr"][1]: 1}).fillna(0)

    num = c[~c.name.isin(["o2_device"])].dropna(subset=["valuenum"])
    rr = num[num.name == "rr"]
    rr = rr[rr["rank"] == rr.groupby(["stay_id", "hour"])["rank"].transform("min")]
    num = pd.concat([num[num.name != "rr"], rr])
    med = num[~num.name.str.startswith("gcs")].pivot_table(
        index=["stay_id", "hour"], columns="name", values="valuenum", aggfunc="median")
    last = c.sort_values("charttime").groupby(["stay_id", "hour", "name"]).last()
    gcs = last["valuenum"].unstack("name").reindex(columns=["gcs_eye", "gcs_verbal", "gcs_motor"])
    device = last["value"].unstack("name").reindex(columns=["o2_device"])

    grid = pd.MultiIndex.from_tuples(
        [(s, h) for s, n in zip(stays.stay_id, stays.n_hours) for h in range(n)],
        names=["stay_id", "hour"])
    h = pd.DataFrame(index=grid).join(med).join(gcs).join(device)
    for col in ["hr", "rr", "spo2", "sbp_nibp", "sbp_art", "temp_c", "temp_f", "o2_flow", "o2_device"]:
        if col not in h:
            h[col] = np.nan

    out = pd.DataFrame(index=grid)
    out["hr"] = h.hr
    out["rr"] = h.rr
    out["spo2"] = h.spo2
    out["sbp"] = h.sbp_nibp.fillna(h.sbp_art)
    out["temp"] = h.temp_c.fillna((h.temp_f - 32) * 5 / 9)
    for col, (lo, hi) in PLAUSIBLE.items():
        out.loc[(out[col] < lo) | (out[col] > hi), col] = np.nan
    gcs_total = h[["gcs_eye", "gcs_verbal", "gcs_motor"]].sum(axis=1, min_count=3)
    out["gcs_total"] = gcs_total.where(gcs_total.between(3, 15))
    out["o2_charted"] = (h.o2_device.notna() | (h.o2_flow > 0)).astype(float).where(
        h.o2_device.notna() | h.o2_flow.notna())

    # forward fill <= 2 bins within each stay (docs/07 §4.3)
    ffill_cols = ["hr", "rr", "spo2", "sbp", "temp", "gcs_total", "o2_charted"]
    out[ffill_cols] = out.groupby(level="stay_id")[ffill_cols].ffill(limit=FFILL_LIMIT)

    avpu = out.gcs_total.map(lambda g: gcs_to_avpu(g) if pd.notna(g) else None)
    out["avpu"] = avpu
    out["avpu_bin"] = avpu.map({"A": 0.0, "V": 1.0, "P": 1.0, "U": 1.0})
    out["supplemental_o2"] = out.o2_charted.fillna(0.0)  # [ASSUMPTION] not charted = room air
    out["has_reading"] = out[["hr", "rr", "spo2", "sbp", "temp", "avpu_bin"]].notna().any(axis=1)
    out["news2"] = [
        news2_score(rr=r.rr, spo2=r.spo2, supplemental_o2=bool(r.supplemental_o2), sbp=r.sbp,
                    hr=r.hr, avpu=r.avpu, temp=r.temp).total
        for r in out.itertuples()
    ]
    out["news2"] = out["news2"].astype("Int64")
    return out.drop(columns=["gcs_total", "o2_charted", "avpu"]).reset_index()


# ---------------------------------------------------------------------------
# Labels (docs/08 §2)
# ---------------------------------------------------------------------------

def _overlaps(intervals: pd.DataFrame, stay_id: int, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    iv = intervals.get(stay_id)
    if iv is None:
        return False
    return bool(((iv[:, 0] < end) & (iv[:, 1] > start)).any())


def severity_class(death_24h: bool, intervention_6h: bool, los_remaining_h: float) -> int:
    """docs/08 §2 table, first match top to bottom."""
    if death_24h or intervention_6h:
        return 3
    if los_remaining_h > 72:
        return 2
    if los_remaining_h >= 24:
        return 1
    return 0


def add_labels(hourly: pd.DataFrame, stays: pd.DataFrame, admissions: pd.DataFrame,
               interventions: pd.DataFrame) -> pd.DataFrame:
    adm = admissions.copy()
    for col in ("deathtime", "dischtime"):
        adm[col] = pd.to_datetime(adm[col])
    # in-hospital death; if flagged but no deathtime, use discharge time [ASSUMPTION]
    adm["death_time"] = adm.deathtime.where(
        adm.deathtime.notna(), adm.dischtime.where(adm.hospital_expire_flag == 1))
    s = stays.merge(adm[["hadm_id", "death_time"]], on="hadm_id", how="left").set_index("stay_id")
    ivs = {k: g[["starttime", "endtime"]].to_numpy() for k, g in interventions.dropna().groupby("stay_id")}

    df = hourly.join(s[["subject_id", "intime", "outtime", "death_time"]], on="stay_id")
    one_h = pd.Timedelta(hours=1)
    df["t"] = df.intime + (df.hour + 1) * one_h
    df["icu_los_remaining_h"] = (df.outtime - df.t) / one_h
    dt = (df.death_time - df.t) / one_h
    df["death_24h"] = (dt > 0) & (dt <= 24)
    death_6h = (dt > 0) & (dt <= TREND_HORIZON_H)
    df["intervention_6h"] = [
        _overlaps(ivs, s_id, t, t + TREND_HORIZON_H * one_h) for s_id, t in zip(df.stay_id, df.t)]
    active_now = pd.Series([_overlaps(ivs, s_id, t - one_h, t) for s_id, t in zip(df.stay_id, df.t)],
                           index=df.index)
    df["severity_label"] = [severity_class(d, i, l) for d, i, l in
                            zip(df.death_24h, df.intervention_6h, df.icu_los_remaining_h)]
    df["trend_label"] = (df.intervention_6h | death_6h).astype(float).where(~active_now)
    return df.rename(columns={"subject_id": "patient_id"}).drop(
        columns=["intime", "outtime", "death_time"])


def build(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    tb = load_tables(raw_dir)
    hourly = hourly_features(tb["chart"], tb["stays"])
    return add_labels(hourly, tb["stays"], tb["admissions"], tb["interventions"])


# ---------------------------------------------------------------------------
# Windows (docs/07 §4.1)
# ---------------------------------------------------------------------------

def load_hourly(path: Path = OUT_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["t"])


def make_windows(df: pd.DataFrame):
    """Windows of T=6 bins ending at every bin with a reading, left-padded within the stay.

    Returns X float32 [N, 6, 7] (FEATURES, NaN where missing), mask bool [N, 6],
    y_severity int [N], y_trend float [N] (NaN = excluded), groups (patient_id) [N],
    news2 [N] (float, NaN if incomplete). Rows whose `has_reading` is False are masked.
    """
    df = df.sort_values(["stay_id", "hour"])
    Xs, masks, ys, yt, groups, n2 = [], [], [], [], [], []
    for _, g in df.groupby("stay_id", sort=False):
        feats = g[FEATURES].to_numpy(dtype=np.float32)
        has = g.has_reading.to_numpy(dtype=bool)
        pad_f = np.full((T - 1, len(FEATURES)), np.nan, dtype=np.float32)
        feats_p = np.vstack([pad_f, feats])
        has_p = np.concatenate([np.zeros(T - 1, dtype=bool), has])
        for i in np.flatnonzero(has):
            Xs.append(feats_p[i:i + T])
            masks.append(has_p[i:i + T])
        idx = np.flatnonzero(has)
        ys.append(g.severity_label.to_numpy()[idx])
        yt.append(g.trend_label.to_numpy(dtype=float)[idx])
        groups.append(g.patient_id.to_numpy()[idx])
        n2.append(pd.to_numeric(g.news2, errors="coerce").to_numpy(dtype=float)[idx])
    if not Xs:
        raise ValueError("no windows: no bin has a reading")
    return (np.stack(Xs), np.stack(masks), np.concatenate(ys).astype(int),
            np.concatenate(yt), np.concatenate(groups), np.concatenate(n2))


def add_deltas(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[N,6,7] -> [N,6,12]: append current - previous for hr, rr, spo2, sbp, temp.

    Call after NaN filling. Delta is 0 at the first real step and wherever the previous
    step is padding (docs/07 §4.1).
    """
    idx = [FEATURES.index(f) for f in DELTA_FEATURES]
    v = X[:, :, idx]
    d = np.zeros_like(v)
    d[:, 1:] = v[:, 1:] - v[:, :-1]
    valid = mask.copy()
    valid[:, 1:] &= mask[:, :-1]
    valid[:, 0] = False
    d[~valid] = 0.0
    return np.concatenate([X, d], axis=2)


def main() -> None:
    p = argparse.ArgumentParser(description="Build MIMIC-IV Demo hourly table + labels")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    a = p.parse_args()
    df = build(a.raw_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    w = df[df.has_reading]
    print(f"wrote {a.out}: {len(df)} stay-hours, {len(w)} windows, "
          f"{w.stay_id.nunique()} stays, {w.patient_id.nunique()} patients")
    print("severity class balance (windows):")
    print(w.severity_label.value_counts(normalize=True).sort_index().round(3).to_string())
    tr = w.trend_label.dropna()
    print(f"trend: {len(tr)} eligible windows, positive rate {tr.mean():.3f}")
    print(f"news2 complete in {w.news2.notna().mean():.1%} of windows")


if __name__ == "__main__":
    main()
