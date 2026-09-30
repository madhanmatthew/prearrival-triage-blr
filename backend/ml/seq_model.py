"""BiGRU two-head vitals model (vitals-only variant of docs/07 §4.2) with nested grouped CV.

    python -m backend.ml.seq_model            # make train-seq
    python -m backend.ml.seq_model --quick    # 1 lambda, few epochs: smoke test, not a result

Architecture (docs/07 §4.2, text branch dropped):
    x[B,6,12] -> BiGRU(hidden=64, 1 layer) -> last valid step -> h[B,128]
    -> Linear(128,64)+ReLU+Dropout(0.3) -> severity Linear(64,4) | trend Linear(64,1)
Loss (docs/07 §4.4): CE(severity, balanced class weights) + lambda * BCE(trend), trend
loss masked where the window is not trend-eligible. [ASSUMPTION] BCE uses
pos_weight = neg/pos from the training fold because trend positives are ~3 %.
Adam lr 1e-3, batch 256, early stopping on validation macro-F1, patience 10, max 100 epochs.

Protocol (docs/15 D5): identical outer folds to backend/ml/baselines.py. Inside each outer
training set, inner 3-fold grouped CV picks lambda and the epoch count (median best epoch);
the model is then refit on the whole outer training set for that many epochs and scored
once on the outer test fold. Preprocessing is fitted on training data only: per-feature
median fill of the 7 raw features, deltas (docs/07 §4.1), then z-score over real steps.

Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from backend.data.mimic_windows import OUT_PATH, add_deltas, load_hourly, make_windows
from backend.ml.baselines import (
    DATA_VERSION, LOG_PATH, MODEL_DIR, N_INNER, N_OUTER, REPORT_DIR, SEED, _git_commit,
    outer_fold_of_patient,
)

LAMBDAS = [0.25, 0.5, 1.0]
HIDDEN, DROPOUT, LR, BATCH = 64, 0.3, 1e-3, 256
MAX_EPOCHS, PATIENCE = 100, 10


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Preprocessing (fitted on training windows only)
# ---------------------------------------------------------------------------

@dataclass
class Preprocessor:
    median: np.ndarray = field(default=None)  # [7]
    mean: np.ndarray = field(default=None)    # [12]
    std: np.ndarray = field(default=None)     # [12]

    def fit(self, X: np.ndarray, mask: np.ndarray) -> "Preprocessor":
        real = X[mask]                                   # [n_steps, 7]
        self.median = np.nanmedian(real, axis=0)
        self.median = np.where(np.isnan(self.median), 0.0, self.median)
        Z = self._fill_and_delta(X, mask)[mask]
        self.mean = Z.mean(axis=0)
        self.std = np.where(Z.std(axis=0) < 1e-6, 1.0, Z.std(axis=0))
        return self

    def _fill_and_delta(self, X, mask):
        Xf = np.where(np.isnan(X), self.median, X)
        Xf[~mask] = 0.0
        return add_deltas(Xf.astype(np.float32), mask)

    def transform(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        Z = (self._fill_and_delta(X, mask) - self.mean) / self.std
        Z[~mask] = 0.0
        return Z.astype(np.float32)

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ("median", "mean", "std")}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: np.asarray(v, dtype=np.float32) for k, v in d.items()})


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BiGRUTwoHead(nn.Module):
    def __init__(self, n_features: int = 12, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.trunk = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Dropout(dropout))
        self.severity_head = nn.Linear(64, 4)
        self.trend_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        out, _ = self.gru(x)                                     # [B, T, 2H]
        # windows are left-padded and always end on a real reading -> last valid step = T-1
        last = mask.shape[1] - 1 - torch.flip(mask, [1]).float().argmax(dim=1)
        h = out[torch.arange(out.shape[0]), last]
        z = self.trunk(h)
        return self.severity_head(z), self.trend_head(z).squeeze(-1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _tensors(Z, mask, y_sev, y_trend):
    return (torch.from_numpy(Z), torch.from_numpy(mask), torch.from_numpy(y_sev.astype(np.int64)),
            torch.from_numpy(np.nan_to_num(y_trend, nan=-1.0).astype(np.float32)))


def _loss_fn(y_sev, y_trend):
    counts = np.bincount(y_sev, minlength=4).astype(np.float32)
    w = torch.tensor(np.where(counts > 0, len(y_sev) / (4 * np.maximum(counts, 1)), 0.0), dtype=torch.float32)
    elig = y_trend[~np.isnan(y_trend)]
    pos = max(float(elig.sum()), 1.0)
    pos_w = torch.tensor((len(elig) - pos) / pos if len(elig) else 1.0)
    ce = nn.CrossEntropyLoss(weight=w)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="sum")

    def loss(logits, trend_logit, ys, yt, lam):
        m = yt >= 0
        lt = bce(trend_logit[m], yt[m]) / m.sum().clamp(min=1)
        return ce(logits, ys) + lam * lt
    return loss


@torch.no_grad()
def _predict_tensors(model, Z, mask, batch=4096):
    model.eval()
    probs, trend = [], []
    for i in range(0, len(Z), batch):
        lg, tl = model(torch.from_numpy(Z[i:i + batch]), torch.from_numpy(mask[i:i + batch]))
        probs.append(torch.softmax(lg, 1).numpy())
        trend.append(torch.sigmoid(tl).numpy())
    return np.concatenate(probs), np.concatenate(trend)


def fit_model(Z, mask, y_sev, y_trend, lam, epochs=MAX_EPOCHS, val=None, seed=SEED):
    """Train; with `val=(Z, mask, y_sev)` early-stop on macro-F1 and return (model, best_epoch)."""
    set_seed(seed)
    model = BiGRUTwoHead(Z.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss = _loss_fn(y_sev, y_trend)
    tz, tm, ts, tt = _tensors(Z, mask, y_sev, y_trend)
    g = torch.Generator().manual_seed(seed)
    best_f1, best_epoch, best_state, bad = -1.0, 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(tz), generator=g).split(BATCH):
            opt.zero_grad()
            lg, tl = model(tz[idx], tm[idx])
            loss(lg, tl, ts[idx], tt[idx], lam).backward()
            opt.step()
        if val is None:
            continue
        p, _ = _predict_tensors(model, val[0], val[1])
        f1 = f1_score(val[2], p.argmax(1), average="macro")
        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if val is not None:
        model.load_state_dict(best_state)
        return model, best_epoch, best_f1
    return model, epochs, None


def _tune(X, mask, y_sev, y_trend, groups, lambdas, max_epochs):
    """Inner grouped CV -> (best lambda, median best epoch, per-lambda scores)."""
    inner = StratifiedGroupKFold(N_INNER, shuffle=True, random_state=SEED)
    splits = list(inner.split(X, y_sev, groups))
    scores = {}
    for lam in lambdas:
        f1s, eps = [], []
        for tr, va in splits:
            pp = Preprocessor().fit(X[tr], mask[tr])
            _, ep, f1 = fit_model(pp.transform(X[tr], mask[tr]), mask[tr], y_sev[tr], y_trend[tr], lam,
                                  epochs=max_epochs, val=(pp.transform(X[va], mask[va]), mask[va], y_sev[va]))
            f1s.append(f1)
            eps.append(ep)
        scores[lam] = (float(np.mean(f1s)), int(np.median(eps)))
    best = max(scores, key=lambda k: scores[k][0])
    return best, max(scores[best][1], 1), scores


# ---------------------------------------------------------------------------
# evaluate / train / predict / save / load
# ---------------------------------------------------------------------------

def load_data(path: Path = OUT_PATH):
    X, mask, y_sev, y_trend, groups, _ = make_windows(load_hourly(path))
    return X, mask, y_sev, y_trend, groups


def evaluate(path: Path = OUT_PATH, quick: bool = False):
    """Nested grouped CV on the baselines' outer folds. Returns (summary_row, confusion)."""
    X, mask, y_sev, y_trend, groups = load_data(path)
    fold_of = outer_fold_of_patient(y_sev, groups)
    folds = np.array([fold_of[p] for p in groups])
    lambdas, max_epochs = ([0.5], 5) if quick else (LAMBDAS, MAX_EPOCHS)
    sev_f1, aurocs, praucs, chosen, cm = [], [], [], [], np.zeros((4, 4), dtype=int)
    for k in range(N_OUTER):
        tr, te = folds != k, folds == k
        lam, n_ep, scores = _tune(X[tr], mask[tr], y_sev[tr], y_trend[tr], groups[tr], lambdas, max_epochs)
        pp = Preprocessor().fit(X[tr], mask[tr])
        model, _, _ = fit_model(pp.transform(X[tr], mask[tr]), mask[tr], y_sev[tr], y_trend[tr], lam, epochs=n_ep)
        probs, trend = _predict_tensors(model, pp.transform(X[te], mask[te]), mask[te])
        pred = probs.argmax(1)
        sev_f1.append(f1_score(y_sev[te], pred, average="macro"))
        cm += confusion_matrix(y_sev[te], pred, labels=[0, 1, 2, 3])
        elig = ~np.isnan(y_trend[te])
        yt = y_trend[te][elig]
        if len(np.unique(yt)) == 2:
            aurocs.append(roc_auc_score(yt, trend[elig]))
            praucs.append(average_precision_score(yt, trend[elig]))
        chosen.append({"lambda": lam, "epochs": n_ep, "inner": {str(a): b for a, b in scores.items()}})
        print(f"fold {k}: lambda={lam} epochs={n_ep} f1={sev_f1[-1]:.3f}"
              + (f" auroc={aurocs[-1]:.3f}" if len(np.unique(yt)) == 2 else ""))
    summary = {
        "severity": (np.mean(sev_f1), np.std(sev_f1), len(sev_f1)),
        "trend_auroc": (np.mean(aurocs), np.std(aurocs), len(aurocs)),
        "trend_pr_auc": (np.mean(praucs), np.std(praucs), len(praucs)),
        "params": chosen, "n": len(y_sev), "n_trend": int((~np.isnan(y_trend)).sum()),
    }
    print(f"severity bigru f1_macro={summary['severity'][0]:.3f}+/-{summary['severity'][1]:.3f}")
    print(f"trend    bigru auroc={summary['trend_auroc'][0]:.3f}+/-{summary['trend_auroc'][1]:.3f}"
          f"  pr_auc={summary['trend_pr_auc'][0]:.3f}+/-{summary['trend_pr_auc'][1]:.3f}")
    conf = pd.DataFrame(cm, index=[f"true_{i}" for i in range(4)], columns=[f"pred_{i}" for i in range(4)])
    return summary, conf


def train(path: Path = OUT_PATH, quick: bool = False):
    """Fit on all windows (lambda/epochs from grouped 3-fold CV) for XAI and the demo.

    Not a result: reported numbers come only from `evaluate()`.
    """
    X, mask, y_sev, y_trend, groups = load_data(path)
    lambdas, max_epochs = ([0.5], 5) if quick else (LAMBDAS, MAX_EPOCHS)
    lam, n_ep, _ = _tune(X, mask, y_sev, y_trend, groups, lambdas, max_epochs)
    pp = Preprocessor().fit(X, mask)
    model, _, _ = fit_model(pp.transform(X, mask), mask, y_sev, y_trend, lam, epochs=n_ep)
    return {"model": model, "preprocessor": pp, "lambda": lam, "epochs": n_ep}


def predict(bundle, X: np.ndarray, mask: np.ndarray):
    """Raw windows [N,6,7] + mask -> (severity_probs [N,4], trend_prob [N])."""
    return _predict_tensors(bundle["model"], bundle["preprocessor"].transform(X, mask), mask)


def save(bundle, version: int = 1) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"bigru_v{version}_{dt.date.today():%Y%m%d}.pt"
    torch.save({"state_dict": bundle["model"].state_dict(), "preprocessor": bundle["preprocessor"].to_dict(),
                "lambda": bundle["lambda"], "epochs": bundle["epochs"], "data_version": DATA_VERSION}, path)
    return path


def load(path: Path):
    ckpt = torch.load(path, weights_only=False)
    model = BiGRUTwoHead()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "preprocessor": Preprocessor.from_dict(ckpt["preprocessor"]),
            "lambda": ckpt["lambda"], "epochs": ckpt["epochs"]}


def log_results(summary) -> None:
    today, commit = dt.date.today().isoformat(), _git_commit()
    a_m, a_s, a_n = summary["trend_auroc"]
    rows = [
        [today, commit, "vitals", "bigru", json.dumps(summary["params"]), f"{summary['severity'][0]:.4f}",
         f"{summary['severity'][1]:.4f}", "", DATA_VERSION, SEED,
         f"task=severity metric=f1_macro nested grouped CV {N_OUTER}x{N_INNER} folds_used={summary['severity'][2]} n={summary['n']}"],
        [today, commit, "vitals", "bigru", json.dumps(summary["params"]), f"{summary['trend_pr_auc'][0]:.4f}",
         f"{summary['trend_pr_auc'][1]:.4f}", "", DATA_VERSION, SEED,
         f"task=trend metric=pr_auc nested grouped CV {N_OUTER}x{N_INNER} folds_used={a_n} n={summary['n_trend']} "
         f"auroc={a_m:.4f}+/-{a_s:.4f}"],
    ]
    pd.DataFrame(rows).to_csv(LOG_PATH, mode="a", header=False, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="BiGRU two-head, nested grouped CV")
    p.add_argument("--data", type=Path, default=OUT_PATH)
    p.add_argument("--quick", action="store_true", help="1 lambda, 5 epochs (smoke test, not a result)")
    p.add_argument("--save-model", action="store_true", help="also fit on all data and save to models/")
    a = p.parse_args()
    summary, conf = evaluate(a.data, a.quick)
    if not a.quick:
        REPORT_DIR.mkdir(exist_ok=True)
        conf.to_csv(REPORT_DIR / "bigru_confusion.csv")
        log_results(summary)
        print(f"appended 2 rows to {LOG_PATH}")
    if a.save_model:
        print(f"saved {save(train(a.data, a.quick))}")


if __name__ == "__main__":
    main()
