"""VitalDB 1-minute vitals -> 5-minute windows + "new deterioration within 10 min" labels.

    python -m backend.data.vitaldb_windows        # reads data/raw/vitaldb, writes data/processed/

Spec: docs/08 §2b, decision docs/15 D19. Input files come from scripts/download_vitaldb.py
(case_<id>.csv.gz with minute, hr, spo2, rr, temp, sbp_nibp, sbp_art; cases.csv with subjectid).

Event minute = SBP < 90 or SpO2 < 90 for this AND the previous minute (sustained). A window ends
at minute t, covers t-4..t, and is kept only if there is no event at t or t-1 (we predict a NEW
episode). Label = any event minute in (t, t+10]. Labels never use NEWS2 (AGENTS rule 1).
Cleaning ranges and fill limits are [ASSUMPTION]s, listed below.
Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/vitaldb")
OUT_PATH = Path("data/processed/vitaldb_windows.npz")
DATA_VERSION = "vitaldb-open-2026-10-21"

VITALS = ["hr", "rr", "spo2", "sbp", "temp"]          # model input order
RAW_COLS = ["hr", "spo2", "rr", "temp", "sbp_nibp", "sbp_art"]
# [ASSUMPTION] plausible ranges; values outside -> missing (monitor artefacts, probe off).
RANGES = {"hr": (20, 250), "spo2": (50, 100), "rr": (2, 60), "temp": (30, 43), "sbp": (40, 300)}
FILL_MIN = 2           # forward-fill limit (minutes) for continuous signals
NIBP_FILL_MIN = 5      # non-invasive cuff cycles every few minutes
SBP_LOW, SPO2_LOW = 90.0, 90.0
SUSTAIN_MIN = 2
T = 5                  # window length (minutes)
STRIDE = 5             # one window every 5 minutes
HORIZON = 10           # label horizon (minutes)
N_FEATURES = len(VITALS) * 2 + 1                      # vitals + deltas + time gap


def clean_case(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw per-minute medians -> complete minute grid with cleaned hr, rr, spo2, sbp, temp."""
    d = raw.set_index("minute").reindex(columns=RAW_COLS)
    d = d[~d.index.duplicated()].sort_index()
    grid = range(int(d.index.min()), int(d.index.max()) + 1) if len(d) else range(0)
    d = d.reindex(grid)

    def rng(s: pd.Series, key: str) -> pd.Series:
        lo, hi = RANGES[key]
        return s.where(s.between(lo, hi))

    art = rng(d.sbp_art, "sbp").ffill(limit=FILL_MIN)
    nibp = rng(d.sbp_nibp, "sbp").ffill(limit=NIBP_FILL_MIN)
    out = pd.DataFrame(index=d.index)
    for v in ("hr", "rr", "spo2", "temp"):
        out[v] = rng(d[v], v).ffill(limit=FILL_MIN)
    out["sbp"] = art.fillna(nibp)
    out.index.name = "minute"
    return out[VITALS]


def event_minutes(c: pd.DataFrame) -> np.ndarray:
    """Bool per minute: SBP < 90 or SpO2 < 90, sustained for SUSTAIN_MIN minutes (ending here)."""
    low = ((c.sbp < SBP_LOW) | (c.spo2 < SPO2_LOW)).to_numpy()
    out = low.copy()
    for k in range(1, SUSTAIN_MIN):
        prev = np.zeros_like(low)
        prev[k:] = low[:-k]
        out &= prev
    return out


def label_known(c: pd.DataFrame) -> np.ndarray:
    """A minute can be judged (event or not) only if SBP or SpO2 is present."""
    return (c.sbp.notna() | c.spo2.notna()).to_numpy()


def case_windows(c: pd.DataFrame, t_len: int = T, stride: int = STRIDE, horizon: int = HORIZON):
    """Windows for one cleaned case. Returns X [n,T,5] (NaN missing), mask [n,T], y [n], end minute [n]."""
    vals = c[VITALS].to_numpy(np.float32)
    ev, known = event_minutes(c), label_known(c)
    present = ~np.isnan(vals).all(axis=1)
    X, M, Y, E = [], [], [], []
    for t in range(t_len - 1, len(c) - 1, stride):
        if ev[t] or (t >= 1 and ev[t - 1]):
            continue                                  # already in an episode
        if not present[t]:
            continue                                  # nothing measured at prediction time
        ahead = slice(t + 1, min(t + 1 + horizon, len(c)))
        if not known[ahead].any():
            continue                                  # label unknown
        X.append(vals[t - t_len + 1:t + 1])
        M.append(present[t - t_len + 1:t + 1])
        Y.append(int(ev[ahead].any()))
        E.append(int(c.index[t]))
    if not X:
        return (np.zeros((0, t_len, len(VITALS)), np.float32), np.zeros((0, t_len), bool),
                np.zeros(0, np.int64), np.zeros(0, np.int64))
    return np.stack(X), np.stack(M), np.array(Y, np.int64), np.array(E, np.int64)


def add_features(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[N,T,5] -> [N,T,11]: vitals, deltas vs the previous real step (0 at the first real step
    or after padding, NaN if either value is missing), time gap in minutes since the previous
    real step (1 for VitalDB; kept for irregular ED vitals in the joint model, D19)."""
    n, t_len, f = X.shape
    deltas = np.zeros_like(X)
    gap = np.zeros((n, t_len, 1), np.float32)
    for t in range(1, t_len):
        both = mask[:, t] & mask[:, t - 1]
        d = X[:, t] - X[:, t - 1]
        deltas[:, t] = np.where(both[:, None], d, 0.0)
        gap[:, t, 0] = np.where(both, 1.0, 0.0)
    return np.concatenate([X, deltas, gap], axis=2)


def build(raw_dir: Path = RAW_DIR, out: Path = OUT_PATH, limit: int | None = None) -> dict:
    cases = pd.read_csv(raw_dir / "cases.csv", usecols=["caseid", "subjectid"])
    subj = dict(zip(cases.caseid, cases.subjectid))
    Xs, Ms, Ys, Gs, Cs, Es = [], [], [], [], [], []
    ids = sorted(subj)[:limit] if limit else sorted(subj)
    for cid in ids:
        f = raw_dir / f"case_{cid}.csv.gz"
        if not f.exists():
            continue
        X, M, Y, E = case_windows(clean_case(pd.read_csv(f)))
        if not len(Y):
            continue
        Xs.append(X); Ms.append(M); Ys.append(Y); Es.append(E)
        Gs.append(np.full(len(Y), subj[cid], np.int64)); Cs.append(np.full(len(Y), cid, np.int64))
    data = {"X": np.concatenate(Xs), "mask": np.concatenate(Ms), "y": np.concatenate(Ys),
            "groups": np.concatenate(Gs), "caseid": np.concatenate(Cs), "minute": np.concatenate(Es)}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    return data


def load(path: Path = OUT_PATH) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    p = argparse.ArgumentParser(description="VitalDB windows + 10-min deterioration labels")
    p.add_argument("--raw", type=Path, default=RAW_DIR)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    p.add_argument("--limit", type=int, help="first N cases only (smoke test)")
    a = p.parse_args()
    d = build(a.raw, a.out, a.limit)
    print(f"wrote {a.out}: {len(d['y'])} windows, {len(np.unique(d['caseid']))} cases, "
          f"{len(np.unique(d['groups']))} subjects, positive rate {d['y'].mean():.3f}")
    miss = np.isnan(d["X"][:, -1, :]).mean(axis=0)
    print("missing at prediction step: " + ", ".join(f"{v} {m:.1%}" for v, m in zip(VITALS, miss)))


if __name__ == "__main__":
    main()
