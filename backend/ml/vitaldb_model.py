"""10-minute deterioration on VitalDB (docs/15 D19, labels docs/08 §2b) with nested grouped CV.

    python -m backend.ml.vitaldb_model                 # all models, logs to experiments/log.csv
    python -m backend.ml.vitaldb_model --quick         # tiny grids, 2 epochs: smoke test, not a result
    python -m backend.ml.vitaldb_model --train-final   # fit the deployed model (HGB) -> models/

Models (all share the outer folds):
  majority   predicts the training prevalence
  news2      partial NEWS2 on the last minute (HR, RR, SpO2 scale 1, SBP, temp; AVPU and O2 flag
             are not recorded in VitalDB). A score, not fitted. Baseline only, never the label.
  logreg     L2 logistic regression on 25 window summaries (last, mean, min, max, change per vital)
  hgb        HistGradientBoosting on the same summaries
  bigru      BiGRU(11 features: vitals, deltas, time gap) -> 1 logit, BCE with pos_weight
Protocol (docs/15 D5): outer StratifiedGroupKFold(5) by subjectid; inner 3-fold grouped CV picks
hyper-parameters (and the BiGRU epoch count, median best epoch); refit on the outer training set;
score once on the outer test fold. PR-AUC primary, AUROC secondary. Preprocessing is fitted on
training data only. Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from backend.data.vitaldb_windows import DATA_VERSION, OUT_PATH, VITALS, add_features, load
from backend.ml import news2 as n2

SEED = 42
N_OUTER, N_INNER = 5, 3
LOG_PATH = Path("experiments/log.csv")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")
TASK_NOTE = "task=deterioration_10min metric=pr_auc"

HIDDEN, DROPOUT, LR, BATCH = 64, 0.3, 1e-3, 512
MAX_EPOCHS, PATIENCE = 15, 3


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Tabular summaries + NEWS2 baseline
# ---------------------------------------------------------------------------

def tabular_features(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """[N,T,5] -> [N,25]: per vital last, mean, min, max, change (last - first real value)."""
    Xm = np.where(mask[:, :, None], X, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            last = Xm[:, -1, :]
            mean, lo, hi = np.nanmean(Xm, 1), np.nanmin(Xm, 1), np.nanmax(Xm, 1)
            first_idx = np.argmax(~np.isnan(Xm), axis=1)            # first non-NaN per vital
            first = np.take_along_axis(Xm, first_idx[:, None, :], 1)[:, 0, :]
    return np.concatenate([last, mean, lo, hi, last - first], axis=1).astype(np.float64)


TAB_NAMES = [f"{s}_{v}" for s in ("last", "mean", "min", "max", "change") for v in VITALS]


def news2_partial(last: np.ndarray) -> np.ndarray:
    """Partial NEWS2 per row from last-minute [hr, rr, spo2, sbp, temp]; missing parameter -> 0."""
    fns = {"hr": n2.score_hr, "rr": n2.score_rr, "spo2": n2.score_spo2, "sbp": n2.score_sbp,
           "temp": n2.score_temp}
    out = np.zeros(len(last))
    for j, v in enumerate(VITALS):
        col = last[:, j]
        out += np.array([0 if np.isnan(x) else (fns[v](float(x)) or 0) for x in col])
    return out


class News2Scorer(ClassifierMixin, BaseEstimator):
    """Ranks windows by partial NEWS2 on the last minute (no fitting)."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        s = news2_partial(X[:, :len(VITALS)]) / 15.0
        return np.column_stack([1 - s, s])


def model_specs(quick: bool = False) -> dict[str, tuple[Pipeline, dict]]:
    imp = lambda: SimpleImputer(strategy="median", keep_empty_features=True)
    specs = {
        "majority": (Pipeline([("impute", imp()), ("clf", DummyClassifier(strategy="prior"))]), {}),
        "news2": (Pipeline([("clf", News2Scorer())]), {}),
        "logreg": (Pipeline([("impute", imp()), ("scale", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
                   {"clf__C": [0.01, 0.1, 1.0]}),
        "hgb": (Pipeline([("clf", HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced"))]),
                {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31]}),
    }
    if quick:
        specs = {k: (p, {kk: vv[:1] for kk, vv in g.items()}) for k, (p, g) in specs.items()}
    return specs


# ---------------------------------------------------------------------------
# BiGRU
# ---------------------------------------------------------------------------

class Scaler:
    """z-score over real steps of the training windows; NaN -> 0 (= training mean) after scaling."""

    def fit(self, F: np.ndarray, mask: np.ndarray) -> "Scaler":
        real = F[mask]
        self.mean = np.nanmean(real, 0)
        self.std = np.nanstd(real, 0)
        self.std[~np.isfinite(self.std) | (self.std < 1e-6)] = 1.0
        self.mean[~np.isfinite(self.mean)] = 0.0
        return self

    def transform(self, F: np.ndarray, mask: np.ndarray) -> np.ndarray:
        Z = (F - self.mean) / self.std
        Z = np.nan_to_num(Z, nan=0.0)
        Z[~mask] = 0.0
        return Z.astype(np.float32)


class BiGRU(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, mask):
        out, _ = self.gru(x)
        last = mask.shape[1] - 1 - torch.flip(mask, [1]).float().argmax(dim=1)
        return self.head(out[torch.arange(out.shape[0]), last]).squeeze(-1)


@torch.no_grad()
def _predict(model, Z, mask, batch=8192) -> np.ndarray:
    model.eval()
    return np.concatenate([torch.sigmoid(model(torch.from_numpy(Z[i:i + batch]),
                                               torch.from_numpy(mask[i:i + batch]))).numpy()
                           for i in range(0, len(Z), batch)])


def fit_bigru(Z, mask, y, epochs=MAX_EPOCHS, val=None, seed=SEED):
    """Train; with val=(Z, mask, y) early-stop on PR-AUC -> (model, best_epoch, best_score)."""
    set_seed(seed)
    model = BiGRU(Z.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    pos = max(float(y.sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - pos) / pos))
    tz, tm, ty = torch.from_numpy(Z), torch.from_numpy(mask), torch.from_numpy(y.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    best, best_ep, best_state, bad = -1.0, 0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(tz), generator=g).split(BATCH):
            opt.zero_grad()
            loss_fn(model(tz[idx], tm[idx]), ty[idx]).backward()
            opt.step()
        if val is None:
            continue
        s = average_precision_score(val[2], _predict(model, val[0], val[1]))
        if s > best:
            best, best_ep, bad = s, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if val is not None:
        model.load_state_dict(best_state)
        return model, best_ep, best
    return model, epochs, None


def tune_bigru(F, mask, y, groups, max_epochs):
    """Inner grouped CV -> median best epoch (the only tuned BiGRU setting)."""
    eps, scores = [], []
    for tr, va in StratifiedGroupKFold(N_INNER, shuffle=True, random_state=SEED).split(F, y, groups):
        sc = Scaler().fit(F[tr], mask[tr])
        _, ep, s = fit_bigru(sc.transform(F[tr], mask[tr]), mask[tr], y[tr], max_epochs,
                             val=(sc.transform(F[va], mask[va]), mask[va], y[va]))
        eps.append(ep)
        scores.append(s)
    return max(int(np.median(eps)), 1), float(np.mean(scores))


# ---------------------------------------------------------------------------
# Nested CV
# ---------------------------------------------------------------------------

def outer_splits(y, groups):
    return list(StratifiedGroupKFold(N_OUTER, shuffle=True, random_state=SEED).split(y, y, groups))


def _scores(y, p) -> dict:
    return {"pr_auc": average_precision_score(y, p), "auroc": roc_auc_score(y, p)}


def evaluate(path: Path = OUT_PATH, quick: bool = False, models: list[str] | None = None,
             max_epochs: int = MAX_EPOCHS) -> pd.DataFrame:
    d = load(path)
    X, mask, y, groups = d["X"], d["mask"], d["y"], d["groups"]
    T = tabular_features(X, mask)
    F = add_features(X, mask)
    specs = model_specs(quick)
    names = models or [*specs, "bigru"]
    rows = []
    for k, (tr, te) in enumerate(outer_splits(y, groups)):
        for name in names:
            if name == "bigru":
                ep, inner = tune_bigru(F[tr], mask[tr], y[tr], groups[tr], 2 if quick else max_epochs)
                sc = Scaler().fit(F[tr], mask[tr])
                model, _, _ = fit_bigru(sc.transform(F[tr], mask[tr]), mask[tr], y[tr], epochs=ep)
                p = _predict(model, sc.transform(F[te], mask[te]), mask[te])
                params = {"epochs": ep, "inner_pr_auc": round(inner, 4)}
            else:
                pipe, grid = specs[name]
                if grid:
                    gs = GridSearchCV(clone(pipe), grid, scoring="average_precision", n_jobs=-1,
                                      cv=StratifiedGroupKFold(N_INNER, shuffle=True, random_state=SEED))
                    gs.fit(T[tr], y[tr], groups=groups[tr])
                    est, params = gs.best_estimator_, gs.best_params_
                else:
                    est, params = clone(pipe).fit(T[tr], y[tr]), {}
                p = est.predict_proba(T[te])[:, 1]
            r = {"fold": k, "model": name, "n_test": len(te), "pos_rate": float(y[te].mean()),
                 "params": json.dumps(params), **_scores(y[te], p)}
            rows.append(r)
            print(f"fold {k} {name:8s} pr_auc {r['pr_auc']:.4f} auroc {r['auroc']:.4f} {params}", flush=True)
    return pd.DataFrame(rows)


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    g = folds.groupby("model", sort=False)
    out = g.agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"), n_folds=("fold", "count"))
    out["params"] = g["params"].apply(lambda s: "[" + ", ".join(s) + "]")
    return out.reset_index()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_results(summary: pd.DataFrame, n: int) -> None:
    today, commit = dt.date.today().isoformat(), _git_commit()
    lines = [[today, commit, "vitals", f"vitaldb_{r.model}", r.params, f"{r.pr_auc_mean:.4f}",
              f"{r.pr_auc_std:.4f}", "", DATA_VERSION, SEED,
              f"{TASK_NOTE} nested grouped CV {N_OUTER}x{N_INNER} by subjectid folds_used={r.n_folds} "
              f"n={n} auroc={r.auroc_mean:.4f}+/-{r.auroc_std:.4f} intra-operative data (D19)"]
             for r in summary.itertuples()]
    pd.DataFrame(lines).to_csv(LOG_PATH, mode="a", header=False, index=False)


# ---------------------------------------------------------------------------
# Final model (train / predict / save / load)
# ---------------------------------------------------------------------------

def train(path: Path = OUT_PATH, epochs: int | None = None) -> dict:
    """Fit the BiGRU on all windows; epochs default = inner-CV median over all data."""
    d = load(path)
    X, mask, y, groups = d["X"], d["mask"], d["y"], d["groups"]
    F = add_features(X, mask)
    ep = epochs or tune_bigru(F, mask, y, groups, MAX_EPOCHS)[0]
    sc = Scaler().fit(F, mask)
    model, _, _ = fit_bigru(sc.transform(F, mask), mask, y, epochs=ep)
    return {"model": model, "scaler": sc, "epochs": ep, "data_version": DATA_VERSION}


def predict(bundle: dict, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """X [N,T,5] raw vitals (hr, rr, spo2, sbp, temp; NaN missing) -> P(new deterioration in 10 min)."""
    F = add_features(X.astype(np.float32), mask)
    return _predict(bundle["model"], bundle["scaler"].transform(F, mask), mask)


def save(bundle: dict, version: int = 1) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"vitaldb_bigru_v{version}_{dt.date.today():%Y%m%d}.pt"
    sc = bundle["scaler"]
    torch.save({"state": bundle["model"].state_dict(), "mean": sc.mean, "std": sc.std,
                "epochs": bundle["epochs"], "data_version": bundle["data_version"],
                "n_features": bundle["model"].gru.input_size}, path)
    return path


def load_model(path: Path) -> dict:
    ck = torch.load(path, weights_only=False)
    model = BiGRU(ck["n_features"])
    model.load_state_dict(ck["state"])
    sc = Scaler()
    sc.mean, sc.std = ck["mean"], ck["std"]
    return {"model": model, "scaler": sc, "epochs": ck["epochs"], "data_version": ck["data_version"]}


# ---------------------------------------------------------------------------
# Deployed model: HGB (best PR-AUC in the nested CV), isotonic-calibrated
# ---------------------------------------------------------------------------
# class_weight="balanced" inflates scores, so the deployed model is wrapped in isotonic calibration
# fitted on grouped held-out folds; the API then shows a probability, not a raw score.

def train_hgb(path: Path = OUT_PATH, quick: bool = False) -> dict:
    """Grid search (inner grouped CV) on all windows, then calibrated refit."""
    d = load(path)
    X, mask, y, groups = d["X"], d["mask"], d["y"], d["groups"]
    T = tabular_features(X, mask)
    pipe, grid = model_specs(quick)["hgb"]
    gs = GridSearchCV(clone(pipe), grid, scoring="average_precision", n_jobs=-1,
                      cv=StratifiedGroupKFold(N_INNER, shuffle=True, random_state=SEED))
    gs.fit(T, y, groups=groups)
    splits = list(StratifiedGroupKFold(N_INNER, shuffle=True, random_state=SEED + 1).split(T, y, groups))
    cal = CalibratedClassifierCV(clone(gs.best_estimator_), method="isotonic", cv=splits).fit(T, y)
    return {"model": cal, "params": gs.best_params_, "inner_pr_auc": float(gs.best_score_),
            "data_version": DATA_VERSION, "kind": "hgb_isotonic", "n_windows": int(len(y))}


def predict_hgb(bundle: dict, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """X [N,T,5] raw vitals (hr, rr, spo2, sbp, temp; NaN missing) -> calibrated P(new event in 10 min)."""
    return bundle["model"].predict_proba(tabular_features(X.astype(np.float32), mask))[:, 1]


def save_hgb(bundle: dict, version: int = 1) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"vitaldb_hgb_v{version}_{dt.date.today():%Y%m%d}.joblib"
    joblib.dump(bundle, path)
    return path


def load_hgb(path: Path) -> dict:
    return joblib.load(path)


def main() -> None:
    p = argparse.ArgumentParser(description="VitalDB 10-min deterioration, nested grouped CV")
    p.add_argument("--data", type=Path, default=OUT_PATH)
    p.add_argument("--quick", action="store_true", help="tiny grids, 2 epochs (smoke test, not a result)")
    p.add_argument("--models", nargs="*", help="subset, e.g. --models majority news2 logreg")
    p.add_argument("--train-final", action="store_true", help="fit the deployed HGB model and save")
    p.add_argument("--final-model", choices=["hgb", "bigru"], default="hgb")
    a = p.parse_args()
    if a.train_final:
        if a.final_model == "bigru":
            print(f"saved {save(train(a.data))}")
        else:
            b = train_hgb(a.data, a.quick)
            print(f"params {b['params']} inner PR-AUC {b['inner_pr_auc']:.4f}; saved {save_hgb(b)}")
        return
    folds = evaluate(a.data, a.quick, a.models)
    summary = summarize(folds)
    REPORT_DIR.mkdir(exist_ok=True)
    tag = "_quick" if a.quick else ""
    folds.to_csv(REPORT_DIR / f"vitaldb_folds{tag}.csv", index=False)
    summary.to_csv(REPORT_DIR / f"vitaldb_results{tag}.csv", index=False)
    print(summary.drop(columns="params").round(4).to_string(index=False))
    if not a.quick:
        log_results(summary, len(load(a.data)["y"]))
        print(f"appended {len(summary)} rows to {LOG_PATH}")


if __name__ == "__main__":
    main()
