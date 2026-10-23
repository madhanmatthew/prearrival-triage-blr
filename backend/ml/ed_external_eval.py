"""External check: frozen ICU-trained BiGRU applied to MIMIC-IV-ED Demo stays.

    python -m backend.ml.ed_external_eval                 # make eval-ed
    python -m backend.ml.ed_external_eval --model models/bigru_v1_20260930.pt

No training or tuning on ED data. Per ED stay, the model sees the window ending at the last
reading inside the first 6 hours of the stay (triage vitals at bin 0). Risk score = expected
severity class sum_k k * p_k. NEWS2 at the same time point is the baseline. Metric = AUROC
per outcome (`backend/data/mimic_ed_windows.py`), 95 % CI from a patient-grouped bootstrap.

The ED tables lack GCS/AVPU and O2 device, and the model was trained on ICU labels, so this is
a domain-transfer check on ~100 patients, not a validation. Research prototype, not a medical
device.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backend.data.mimic_ed_windows import (
    ED_RAW_DIR, ed_hourly, ed_outcomes, first_hours_windows, load_ed_tables,
)
from backend.data.mimic_windows import RAW_DIR as ICU_RAW_DIR
from backend.ml import seq_model
from backend.ml.baselines import LOG_PATH, MODEL_DIR, REPORT_DIR, SEED, _git_commit

DATA_VERSION = "mimic-iv-ed-demo-2.2"
OUTCOMES = ("admitted", "high_acuity", "icu_24h")
N_BOOT = 1000
MIN_CLASS = 5  # fewer positives or negatives than this -> AUROC flagged as unreliable


def latest_model(model_dir: Path = MODEL_DIR) -> Path:
    hits = sorted(model_dir.glob("bigru_v*.pt"))
    if not hits:
        raise FileNotFoundError(f"no bigru_v*.pt in {model_dir}; run `python -m backend.ml.seq_model --save-model`")
    return hits[-1]


def expected_severity(probs: np.ndarray) -> np.ndarray:
    return probs @ np.arange(probs.shape[1])


def grouped_bootstrap_auroc(y: np.ndarray, score: np.ndarray, groups: np.ndarray,
                            n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """AUROC and a 95 % percentile CI resampling patients (all stays of a drawn patient)."""
    auc = roc_auc_score(y, score)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[g] for g in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], score[idx]))
    if not boots:
        return auc, np.nan, np.nan
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def score_table(scores: pd.DataFrame) -> pd.DataFrame:
    """scores: per stay with subject_id, model_risk, news2 and outcome columns -> one row per
    (outcome, scorer). NEWS2 and model are compared on the stays where NEWS2 is complete."""
    rows = []
    for outcome in OUTCOMES:
        d = scores.dropna(subset=[outcome])
        if d[outcome].nunique() < 2:
            rows.append({"outcome": outcome, "scorer": "-", "n": len(d), "pos": int(d[outcome].sum()),
                         "auroc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "subset": "unavailable"})
            continue
        both = d.dropna(subset=["news2"])
        for subset, frame, scorers in (("all", d, ["model_risk"]),
                                       ("news2_complete", both, ["model_risk", "news2"])):
            y = frame[outcome].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            for s in scorers:
                auc, lo, hi = grouped_bootstrap_auroc(y, frame[s].to_numpy(float), frame.subject_id.to_numpy())
                pos = int(y.sum())
                rows.append({"outcome": outcome, "scorer": s, "subset": subset, "n": len(y), "pos": pos,
                             "auroc": auc, "ci_lo": lo, "ci_hi": hi,
                             "reliable": min(pos, len(y) - pos) >= MIN_CLASS})
    return pd.DataFrame(rows)


def evaluate(raw_dir: Path = ED_RAW_DIR, model_path: Path | None = None,
             icu_dir: Path | None = ICU_RAW_DIR):
    """Returns (per-stay scores, summary table)."""
    tb = load_ed_tables(raw_dir)
    hourly = ed_hourly(tb["edstays"], tb["triage"], tb["vitalsign"])
    X, mask, stay_ids, news2 = first_hours_windows(hourly)
    model_path = model_path or latest_model()
    probs, trend = seq_model.predict(seq_model.load(model_path), X, mask)

    icu = None
    icu_file = icu_dir / "icu/icustays.csv.gz" if icu_dir else None
    if icu_file is not None and icu_file.exists():
        icu = pd.read_csv(icu_file, usecols=["subject_id", "intime"])
    out = ed_outcomes(tb["edstays"], tb["triage"], icu)
    scores = pd.DataFrame({"stay_id": stay_ids, "model_risk": expected_severity(probs),
                           "p_critical": probs[:, 3], "trend_prob": trend, "news2": news2})
    scores = scores.merge(out, on="stay_id")
    print(f"model {model_path.name}: {len(tb['edstays'])} ED stays, {len(scores)} with a reading in the "
          f"first 6 h, {scores.subject_id.nunique()} patients, NEWS2 complete in {scores.news2.notna().mean():.0%}")
    return scores, score_table(scores), model_path


def log_results(table: pd.DataFrame, model_path: Path) -> None:
    today, commit = dt.date.today().isoformat(), _git_commit()
    params = json.dumps({"model": model_path.name, "window": "first 6h", "score": "expected severity"})
    rows = [[today, commit, "vitals", "bigru_frozen" if r.scorer == "model_risk" else r.scorer,
             params, "", "", f"{r.auroc:.4f}", DATA_VERSION, SEED,
             f"external check task={r.outcome} metric=auroc subset={r.subset} n={r.n} pos={r.pos} "
             f"ci95={r.ci_lo:.3f}-{r.ci_hi:.3f} reliable={r.reliable}"]
            for r in table.itertuples() if r.scorer != "-"]
    pd.DataFrame(rows).to_csv(LOG_PATH, mode="a", header=False, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Frozen BiGRU on MIMIC-IV-ED Demo (external check)")
    p.add_argument("--raw-dir", type=Path, default=ED_RAW_DIR)
    p.add_argument("--model", type=Path, default=None, help="default: newest models/bigru_v*.pt")
    p.add_argument("--icu-dir", type=Path, default=ICU_RAW_DIR, help="MIMIC-IV Demo, for icu_24h")
    p.add_argument("--no-log", action="store_true", help="do not append to experiments/log.csv")
    a = p.parse_args()
    scores, table, model_path = evaluate(a.raw_dir, a.model, a.icu_dir)
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    REPORT_DIR.mkdir(exist_ok=True)
    scores.to_csv(REPORT_DIR / "ed_demo_scores.csv", index=False)
    table.to_csv(REPORT_DIR / "ed_demo_auroc.csv", index=False)
    print(f"wrote {REPORT_DIR / 'ed_demo_scores.csv'}, {REPORT_DIR / 'ed_demo_auroc.csv'}")
    if not a.no_log:
        log_results(table, model_path)
        print(f"appended rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
