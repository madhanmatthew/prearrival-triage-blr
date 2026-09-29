"""Tabular baselines for the vitals tasks with nested patient-grouped CV (docs/15 D5).

    python -m backend.ml.baselines            # full grids (make train-baselines)
    python -m backend.ml.baselines --quick    # 1-point grids, smoke test only

Models: majority class, NEWS2 (rule), Logistic Regression, Random Forest, Gradient Boosting.
Tasks: `severity` (4 classes, macro-F1 + confusion matrix) and `trend` (binary on eligible
windows, AUROC + PR-AUC; tuned on PR-AUC because positives are rare).

Protocol (docs/15 D5, AGENTS.md rules 3-4):
* Outer StratifiedGroupKFold(5, shuffle=True, random_state=42) grouped by patient_id and
  stratified by severity. Each patient's outer fold is reused for the trend task, so every
  model and both tasks share identical outer folds.
* Inner StratifiedGroupKFold(3) grid search on the outer-training patients only.
* Missing values: median imputer fitted inside the pipeline, i.e. on training folds only.
* Features: the window's last reading, 7 vitals/flags + 5 deltas (docs/07 §4.1 order).
* NEWS2 baseline: scored from the same fold-imputed last-reading features, so it is
  evaluated on every window (NEWS2 needs all 7 inputs). For trend, the NEWS2 total is used
  as a risk score; NEWS2 has no native trend output (docs/08 §3).

Every run appends one row per (task, model) to experiments/log.csv and writes
reports/baselines_results.csv + reports/baselines_confusion.csv.

Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from backend.data.mimic_windows import FEATURES, OUT_PATH, add_deltas, load_hourly, make_windows
from backend.ml.news2 import news2_class, news2_score

SEED = 42
N_OUTER, N_INNER = 5, 3
DATA_VERSION = "mimic-iv-demo-2.2"
FEATURE_NAMES = FEATURES + [f"{f}_delta" for f in ("hr", "rr", "spo2", "sbp", "temp")]
LOG_PATH = Path("experiments/log.csv")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")
TASKS = ("severity", "trend")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def tabular_features(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[N,6,7] windows -> [N,12]: last reading's 7 features + its 5 deltas (NaN if unknown).

    Deltas use the raw values (NaN propagates), so no information from other folds is used;
    the pipeline's imputer fills them from the training fold.
    """
    # add_deltas sets delta 0 where the previous step is padding ("no change seen")
    return add_deltas(X, mask)[:, -1, :].astype(np.float64)


# ---------------------------------------------------------------------------
# NEWS2 as an sklearn classifier (no fitting; imputer is fitted on the training fold)
# ---------------------------------------------------------------------------

class News2Classifier(ClassifierMixin, BaseEstimator):
    """Predicts the docs/08 §3 class from NEWS2 computed on (imputed) last-reading features.

    `predict_proba` for the binary trend task returns total/20 as the positive-class score.
    """

    def __init__(self, task: str = "severity"):
        self.task = task

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        return self

    @staticmethod
    def totals(X: np.ndarray) -> np.ndarray:
        idx = {f: i for i, f in enumerate(FEATURE_NAMES)}
        out = []
        for r in X:
            out.append(news2_score(
                hr=r[idx["hr"]], rr=r[idx["rr"]], spo2=min(r[idx["spo2"]], 100.0), sbp=r[idx["sbp"]],
                temp=r[idx["temp"]], avpu="A" if r[idx["avpu_bin"]] < 0.5 else "U",
                supplemental_o2=bool(r[idx["supplemental_o2"]] >= 0.5)).total)
        return np.array(out, dtype=float)

    def predict(self, X):
        t = self.totals(X)
        if self.task == "severity":
            return np.array([news2_class(int(v)) for v in t])
        return (t >= 5).astype(int)  # docs/08 §3: >= 5 is class 2+ ("severe")

    def predict_proba(self, X):
        s = self.totals(X) / 20.0
        return np.column_stack([1 - s, s])


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------

def _pipe(clf, scale: bool = False) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps + [("clf", clf)])


def model_specs(task: str, quick: bool = False) -> dict[str, tuple[Pipeline, dict]]:
    """name -> (pipeline, grid). Grids follow docs/06 §4; `quick` keeps one point each."""
    specs = {
        "majority": (_pipe(DummyClassifier(strategy="prior")), {}),
        "news2": (_pipe(News2Classifier(task)), {}),
        "logreg": (_pipe(LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True),
                   {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
        "rf": (_pipe(RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)),
               {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [4, 6, 8, None],
                "clf__min_samples_split": [2, 5, 10]}),
        "gb": (_pipe(GradientBoostingClassifier(random_state=SEED)),
               {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3],
                "clf__learning_rate": [0.05, 0.1]}),
    }
    if quick:
        specs = {k: (p, {kk: vv[:1] for kk, vv in g.items()}) for k, (p, g) in specs.items()}
    return specs


def _scoring(task: str) -> str:
    return "f1_macro" if task == "severity" else "average_precision"


def _fit(pipe: Pipeline, grid: dict, X, y, groups, task: str):
    """Grid search with inner grouped CV (or plain fit when there is nothing to tune)."""
    sw = compute_sample_weight("balanced", y)  # only used by GB (others use class_weight)
    fit_kw = {"clf__sample_weight": sw} if isinstance(pipe.named_steps["clf"], GradientBoostingClassifier) else {}
    if not grid:
        return clone(pipe).fit(X, y, **fit_kw), {}
    search = GridSearchCV(clone(pipe), grid, scoring=_scoring(task), n_jobs=-1, refit=True,
                          cv=StratifiedGroupKFold(N_INNER, shuffle=True, random_state=SEED))
    search.fit(X, y, groups=groups, **fit_kw)
    return search.best_estimator_, search.best_params_


# ---------------------------------------------------------------------------
# Data + folds
# ---------------------------------------------------------------------------

def load_task_data(path: Path = OUT_PATH):
    X, mask, y_sev, y_trend, groups, _ = make_windows(load_hourly(path))
    return tabular_features(X, mask), y_sev, y_trend, groups


def outer_fold_of_patient(y_sev: np.ndarray, groups: np.ndarray) -> dict:
    """Patient -> outer fold id from the severity split; reused for every task and model."""
    cv = StratifiedGroupKFold(N_OUTER, shuffle=True, random_state=SEED)
    fold = {}
    for k, (_, te) in enumerate(cv.split(np.zeros(len(y_sev)), y_sev, groups)):
        for g in np.unique(groups[te]):
            fold[g] = k
    return fold


def _task_arrays(task, F, y_sev, y_trend, groups):
    if task == "severity":
        return F, y_sev, groups
    keep = ~np.isnan(y_trend)
    return F[keep], y_trend[keep].astype(int), groups[keep]


# ---------------------------------------------------------------------------
# evaluate / train / predict / save / load
# ---------------------------------------------------------------------------

def evaluate(path: Path = OUT_PATH, quick: bool = False, models: list[str] | None = None):
    """Nested grouped CV for every (task, model). Returns (results_df, confusion_df)."""
    F, y_sev, y_trend, groups = load_task_data(path)
    fold_of = outer_fold_of_patient(y_sev, groups)
    rows, conf = [], []
    for task in TASKS:
        X, y, g = _task_arrays(task, F, y_sev, y_trend, groups)
        folds = np.array([fold_of[p] for p in g])
        for name, (pipe, grid) in model_specs(task, quick).items():
            if models and name not in models:
                continue
            scores, params, cm = [], [], None
            for k in range(N_OUTER):
                tr, te = folds != k, folds == k
                if task == "trend" and len(np.unique(y[te])) < 2:
                    continue  # AUROC undefined on a fold with one class; recorded in notes
                est, best = _fit(pipe, grid, X[tr], y[tr], g[tr], task)
                params.append(best)
                if task == "severity":
                    pred = est.predict(X[te])
                    scores.append({"f1_macro": f1_score(y[te], pred, average="macro")})
                    c = confusion_matrix(y[te], pred, labels=[0, 1, 2, 3])
                    cm = c if cm is None else cm + c
                else:
                    p = est.predict_proba(X[te])[:, 1]
                    scores.append({"auroc": roc_auc_score(y[te], p),
                                   "pr_auc": average_precision_score(y[te], p)})
            s = pd.DataFrame(scores)
            row = {"task": task, "model": name, "n_windows": len(y), "n_folds": len(s),
                   "prevalence": float(np.mean(y)) if task == "trend" else None,
                   "best_params": json.dumps(params)}
            for m in s.columns:
                row[f"{m}_mean"], row[f"{m}_std"] = s[m].mean(), s[m].std(ddof=0)
            rows.append(row)
            if cm is not None:
                conf.append(pd.DataFrame(cm, index=[f"true_{i}" for i in range(4)],
                                         columns=[f"pred_{i}" for i in range(4)]).assign(model=name))
            print(f"{task:8s} {name:8s} " + "  ".join(
                f"{m}={s[m].mean():.3f}+/-{s[m].std(ddof=0):.3f}" for m in s.columns))
    return pd.DataFrame(rows), pd.concat(conf) if conf else pd.DataFrame()


def train(name: str, task: str = "severity", path: Path = OUT_PATH, quick: bool = False):
    """Fit one model on all windows (grid-searched with grouped CV) for XAI / the demo.

    Not a result: reported numbers come only from `evaluate()`.
    """
    F, y_sev, y_trend, groups = load_task_data(path)
    X, y, g = _task_arrays(task, F, y_sev, y_trend, groups)
    pipe, grid = model_specs(task, quick)[name]
    est, _ = _fit(pipe, grid, X, y, g, task)
    return est


def predict(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def save(model, name: str, task: str, version: int = 1) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{name}_{task}_v{version}_{dt.date.today():%Y%m%d}.joblib"
    joblib.dump({"model": model, "features": FEATURE_NAMES, "data_version": DATA_VERSION}, path)
    return path


def load(path: Path):
    return joblib.load(path)["model"]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_results(results: pd.DataFrame, quick: bool) -> None:
    today, commit = dt.date.today().isoformat(), _git_commit()
    lines = []
    for r in results.itertuples():
        metric = "f1_macro" if r.task == "severity" else "pr_auc"
        auroc = f" auroc={r.auroc_mean:.4f}+/-{r.auroc_std:.4f}" if r.task == "trend" else ""
        note = (f"task={r.task} metric={metric} nested grouped CV {N_OUTER}x{N_INNER} "
                f"folds_used={r.n_folds} n={r.n_windows}{auroc}" + (" QUICK-GRID smoke test" if quick else ""))
        lines.append([today, commit, "vitals", r.model, r.best_params,
                      f"{getattr(r, metric + '_mean'):.4f}", f"{getattr(r, metric + '_std'):.4f}",
                      "", DATA_VERSION, SEED, note])
    pd.DataFrame(lines).to_csv(LOG_PATH, mode="a", header=False, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Tabular baselines, nested grouped CV")
    p.add_argument("--data", type=Path, default=OUT_PATH)
    p.add_argument("--quick", action="store_true", help="1-point grids (smoke test, not a result)")
    p.add_argument("--models", nargs="*", help="subset, e.g. --models majority news2 logreg")
    a = p.parse_args()
    results, conf = evaluate(a.data, a.quick, a.models)
    REPORT_DIR.mkdir(exist_ok=True)
    suffix = "_quick" if a.quick else ""
    results.to_csv(REPORT_DIR / f"baselines_results{suffix}.csv", index=False)
    conf.to_csv(REPORT_DIR / f"baselines_confusion{suffix}.csv")
    if not a.quick:
        log_results(results, a.quick)
        print(f"appended {len(results)} rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
