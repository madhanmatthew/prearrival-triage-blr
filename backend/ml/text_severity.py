"""Text severity classifier (docs/15 D2, docs/08 §4, docs/07 §3 slots 15-18).

TF-IDF character n-grams (2-5, word-boundary aware) + multinomial Logistic Regression, 4 classes
(0 stable, 1 moderate, 2 severe, 3 critical). Char n-grams work across EN/HI/KN without a
tokenizer and tolerate ASR spelling noise.

Data: `data/text_reports/reports.csv` (format in `data/text_reports/README.md`), team-built,
LLM-drafted + hand-reviewed, tagged SIM. Only rows with reviewed == 1 are used.
Protocol: a fixed 20% held-out test split stored in the CSV `split` column (created once with
`--assign-split`, stratified by language x label, seed 42). C is tuned by 5-fold stratified CV
on the train split only; the test split is scored once, never used for tuning.

At inference the input is the transcript plus the raw dialogue answers (`compose_text`),
per docs/07 §3. [ASSUMPTION] training reports are single free-text reports; the answers are
appended at inference only, so there is a train/inference input gap (stated limitation).
Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from backend.ml.baselines import _git_commit

DATA_PATH = Path("data/text_reports/reports.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
LOG_PATH = Path("experiments/log.csv")
SEED = 42
N_CLASSES = 4
LANGUAGES = ("en", "hi", "kn")
REQUIRED_COLS = ("report_id", "language", "text", "label", "reviewed")
C_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
DATA_VERSION = "text-reports-SIM"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_reports(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the report CSV; keep reviewed rows only."""
    df = pd.read_csv(path, dtype={"report_id": str, "text": str, "language": str})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    bad_lang = set(df["language"]) - set(LANGUAGES)
    if bad_lang:
        raise ValueError(f"unknown languages {sorted(bad_lang)}; expected {LANGUAGES}")
    if not df["label"].isin(range(N_CLASSES)).all():
        raise ValueError("label must be an integer in 0..3")
    if df["report_id"].duplicated().any():
        raise ValueError("duplicate report_id")
    if "split" in df.columns and not df["split"].dropna().isin(["train", "test"]).all():
        raise ValueError("split must be 'train' or 'test'")
    df = df[df["reviewed"] == 1].copy()
    df["text"] = df["text"].fillna("").str.strip()
    return df[df["text"] != ""].reset_index(drop=True)


def assign_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED) -> pd.DataFrame:
    """Fixed held-out split, stratified by language x label. Existing splits are kept."""
    df = df.reset_index(drop=True)
    df["split"] = df["split"].astype(object) if "split" in df.columns else None
    new = df["split"].isna()
    if new.any():
        strata = df.loc[new, "language"] + "_" + df.loc[new, "label"].astype(str)
        counts = strata.map(strata.value_counts())
        strat = strata.where(counts >= 2, "rare")   # tiny strata cannot be stratified
        n_test = int(np.ceil(test_size * new.sum()))
        if strat.nunique() > n_test:                 # small batch: fall back to label only
            strat = df.loc[new, "label"]
        _, test_idx = train_test_split(df.index[new], test_size=test_size, random_state=seed,
                                       stratify=strat if 1 < strat.nunique() <= n_test else None)
        df.loc[new, "split"] = "train"
        df.loc[test_idx, "split"] = "test"
    return df


def compose_text(transcript: str, answers_raw: list[str] | None = None) -> str:
    """Inference input: transcript + raw dialogue answers (docs/07 §3)."""
    parts = [transcript or ""] + [a for a in (answers_raw or []) if a]
    return " . ".join(p.strip() for p in parts if p.strip())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                                  min_df=1, lowercase=True)),
        ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                                  random_state=SEED)),
    ])


def train(df: pd.DataFrame, c_grid: list[float] = C_GRID, n_splits: int = 5) -> tuple[Pipeline, dict]:
    """Tune C by stratified CV on the given (train-split) rows, refit on all of them."""
    y = df["label"].to_numpy()
    k = min(n_splits, int(np.bincount(y, minlength=N_CLASSES)[np.unique(y)].min()))
    if k < 2 or len(c_grid) == 1:
        model = build_pipeline(c_grid[0]).fit(df["text"], y)
        return model, {"C": c_grid[0], "cv_f1_macro": None}
    gs = GridSearchCV(build_pipeline(), {"lr__C": c_grid}, scoring="f1_macro",
                      cv=StratifiedKFold(k, shuffle=True, random_state=SEED), n_jobs=1)
    gs.fit(df["text"], y)
    return gs.best_estimator_, {"C": gs.best_params_["lr__C"],
                                "cv_f1_macro": float(gs.best_score_), "cv_folds": k}


def predict_proba(model: Pipeline, texts: list[str] | str) -> np.ndarray:
    """(n, 4) probabilities in class order 0..3, even if a class was absent in training."""
    single = isinstance(texts, str)
    p = model.predict_proba([texts] if single else list(texts))
    out = np.zeros((p.shape[0], N_CLASSES))
    out[:, model.classes_.astype(int)] = p
    return out[0] if single else out


def predict(model: Pipeline, texts: list[str]) -> np.ndarray:
    return predict_proba(model, texts).argmax(axis=1)


def evaluate(model: Pipeline, test: pd.DataFrame) -> dict:
    """Score the held-out split once: macro-F1 overall and per language, confusion matrix."""
    y, yhat = test["label"].to_numpy(), predict(model, test["text"].tolist())
    res = {"n_test": len(test), "f1_macro": float(f1_score(y, yhat, average="macro",
                                                           labels=range(N_CLASSES), zero_division=0)),
           "accuracy": float((y == yhat).mean()),
           "confusion": confusion_matrix(y, yhat, labels=range(N_CLASSES)).tolist(),
           "per_language": {}}
    for lang in LANGUAGES:
        m = (test["language"] == lang).to_numpy()
        if m.any():
            res["per_language"][lang] = {
                "n": int(m.sum()),
                "f1_macro": float(f1_score(y[m], yhat[m], average="macro",
                                           labels=range(N_CLASSES), zero_division=0))}
    return res


def save(model: Pipeline, meta: dict, version: int = 1) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"text_severity_v{version}_{dt.date.today():%Y%m%d}.joblib"
    joblib.dump({"model": model, "meta": meta, "data_version": DATA_VERSION}, path)
    return path


def load(path: Path) -> Pipeline:
    return joblib.load(path)["model"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def log_result(meta: dict, res: dict) -> None:
    per_lang = " ".join(f"{k}={v['f1_macro']:.4f}(n={v['n']})" for k, v in res["per_language"].items())
    note = (f"task=text_severity metric=f1_macro held-out 20% test (n={res['n_test']}) "
            f"cv_f1={meta.get('cv_f1_macro')} per_lang: {per_lang} data=SIM LLM-drafted+reviewed")
    row = [dt.date.today().isoformat(), _git_commit(), "reporting", "tfidf_char_lr",
           json.dumps({"C": meta["C"]}), meta.get("cv_f1_macro") or "", "",
           f"{res['f1_macro']:.4f}", DATA_VERSION, SEED, note]
    pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=False, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Text severity classifier (TF-IDF char + LR)")
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--assign-split", action="store_true",
                   help="write a fixed 20%% test split into the CSV (rows without one), then exit")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--no-log", action="store_true")
    a = p.parse_args()

    if a.assign_split:
        raw = pd.read_csv(a.data, dtype={"report_id": str})
        load_reports(a.data)                                   # validate first
        out = assign_split(raw[raw["reviewed"] == 1]).set_index("report_id")["split"]
        raw["split"] = raw["report_id"].map(out).combine_first(raw.get("split", pd.Series(dtype=str)))
        raw.to_csv(a.data, index=False)
        print(raw["split"].value_counts(dropna=False).to_string())
        return

    df = load_reports(a.data)
    if "split" not in df.columns or df["split"].isna().any():
        raise SystemExit("rows without a split: run with --assign-split first (fixes the test set)")
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    model, meta = train(tr)
    res = evaluate(model, te)
    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(res["confusion"], index=[f"true_{i}" for i in range(N_CLASSES)],
                 columns=[f"pred_{i}" for i in range(N_CLASSES)]).to_csv(
        REPORT_DIR / "text_severity_confusion.csv")
    print(json.dumps({**meta, **{k: v for k, v in res.items() if k != "confusion"}}, indent=2))
    print("NOTE: reports are SIM (LLM-drafted, team-reviewed, not clinician-labelled).")
    if not a.no_log:
        log_result(meta, res)
    if a.save_model:
        print("saved", save(model, meta))


if __name__ == "__main__":
    main()
