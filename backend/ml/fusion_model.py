"""Fusion model (docs/07 §4.2) + 3-way ablation + text-agreement sensitivity (docs/08 §5).

    python -m backend.ml.fusion_model            # make train-fusion
    python -m backend.ml.fusion_model --quick    # smoke test, not a result

Variants (same nested grouped CV, same outer folds as baselines / BiGRU, docs/15 D5):
    vitals_only : BiGRU branch only      text_only : text MLP branch only      fused : both
Text features are SYNTHETIC: generated from the severity label only (never from vitals) by
data/text_feature_generator.yaml. The ablation measures the generator's assumed
informativeness as well as the model; it demonstrates the pipeline, not clinical performance.
Modality dropout (fused, training only): text zeroed w.p. 0.2, vitals w.p. 0.1, never both.

Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from backend.data.mimic_windows import OUT_PATH
from backend.ml.baselines import (
    DATA_VERSION, LOG_PATH, MODEL_DIR, N_INNER, N_OUTER, REPORT_DIR, SEED, _git_commit,
    outer_fold_of_patient,
)
from backend.ml.seq_model import (
    BATCH, DROPOUT, HIDDEN, LAMBDAS, LR, MAX_EPOCHS, PATIENCE, Preprocessor, _loss_fn, _tensors,
    load_data, set_seed,
)

GEN_PATH = Path("data/text_feature_generator.yaml")
VARIANTS = ("vitals_only", "text_only", "fused")
P_DROP_TEXT, P_DROP_VITALS = 0.2, 0.1
N_TEXT = 19
CAVEAT = ("Text modality is synthetic; the fusion gain reflects the assumed informativeness of "
          "text and demonstrates the pipeline, not clinical performance.")


# ---------------------------------------------------------------------------
# Synthetic text-feature generator (docs/08 §5): label in, 19-d vector out
# ---------------------------------------------------------------------------

def load_generator(path: Path = GEN_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def effective_class(y: int, agreement: float, adjacent_share: float, rng) -> int:
    """Class the text is 'typical of': true class, an adjacent class, or a random other one."""
    u = rng.random()
    if u < agreement:
        return int(y)
    if u < agreement + (1 - agreement) * adjacent_share:
        opts = [c for c in (y - 1, y + 1) if 0 <= c <= 3]
        return int(rng.choice(opts))
    return int(rng.choice([c for c in range(4) if c != y]))


def make_text_features(y_sev: np.ndarray, gen: dict, agreement: float | None = None,
                       seed: int = SEED) -> np.ndarray:
    """[N,19] text vectors per docs/07 §3, drawn from the label only (independence rule)."""
    agreement = gen["agreement"] if agreement is None else agreement
    rng = np.random.default_rng(seed)
    out = np.zeros((len(y_sev), N_TEXT), dtype=np.float32)
    for i, y in enumerate(y_sev):
        c = effective_class(int(y), agreement, gen["adjacent_share"], rng)
        known = rng.random(4) < gen["p_known"]                     # conscious, breathing, bleeding, speak
        if known[0]:
            out[i, 0], out[i, 1] = float(rng.random() < gen["conscious"][c]), 1.0
        if known[1]:
            out[i, 2], out[i, 3] = float(rng.random() < gen["breathing_difficulty"][c]), 1.0
        if known[2]:
            out[i, 4], out[i, 5] = (0.0, 0.5, 1.0)[rng.choice(3, p=gen["bleeding"][c])], 1.0
        if known[0] and out[i, 0] == 0:
            out[i, 6], out[i, 7] = 0.0, 1.0            # unresponsive -> cannot speak (as the dialogue infers)
        elif known[3]:
            out[i, 6], out[i, 7] = float(rng.random() < gen["can_speak"][c]), 1.0
        out[i, 8] = min(1 + rng.poisson(gen["num_injured_extra"][c]), 5) / 5
        out[i, 9] = float(rng.random() < gen["high_energy"][c])
        out[i, 10 + rng.choice(5, p=gen["injury_type"][c])] = 1.0
        alpha = np.ones(4)
        alpha[c] += gen["prob_strength"]
        out[i, 15:19] = rng.dirichlet(alpha)
    return out


def risk_score(severity_probs: np.ndarray) -> np.ndarray:
    """docs/07 §2.3: expected severity normalised to [0, 1]."""
    return np.asarray(severity_probs) @ (np.arange(4) / 3)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FusionNet(nn.Module):
    """docs/07 §4.2. `variant` drops a branch for the ablation; masks zero a branch per row."""

    def __init__(self, variant: str = "fused", n_features: int = 12, hidden: int = HIDDEN,
                 dropout: float = DROPOUT):
        super().__init__()
        assert variant in VARIANTS
        self.variant = variant
        self.use_vit, self.use_txt = variant != "text_only", variant != "vitals_only"
        width = 0
        if self.use_vit:
            self.gru = nn.GRU(n_features, hidden, num_layers=1, batch_first=True, bidirectional=True)
            width += 2 * hidden
        if self.use_txt:
            self.txt = nn.Sequential(nn.Linear(N_TEXT, 32), nn.ReLU())
            width += 32
        self.trunk = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Dropout(dropout))
        self.severity_head = nn.Linear(64, 4)
        self.trend_head = nn.Linear(64, 1)

    def forward(self, x, mask, text, m_vit=None, m_txt=None):
        parts = []
        if self.use_vit:
            out, _ = self.gru(x)
            last = mask.shape[1] - 1 - torch.flip(mask, [1]).float().argmax(dim=1)
            h = out[torch.arange(out.shape[0]), last]
            parts.append(h if m_vit is None else h * m_vit[:, None])
        if self.use_txt:
            h = self.txt(text)
            parts.append(h if m_txt is None else h * m_txt[:, None])
        z = self.trunk(torch.cat(parts, dim=1))
        return self.severity_head(z), self.trend_head(z).squeeze(-1)


@torch.no_grad()
def _predict_tensors(model, Z, mask, T, m_txt=None, batch=4096):
    """m_txt: optional float [N] text-availability mask (0 -> text branch zeroed)."""
    model.eval()
    probs, trend = [], []
    for i in range(0, len(Z), batch):
        mt = None if m_txt is None else torch.from_numpy(m_txt[i:i + batch]).float()
        lg, tl = model(torch.from_numpy(Z[i:i + batch]), torch.from_numpy(mask[i:i + batch]),
                       torch.from_numpy(T[i:i + batch]), None, mt)
        probs.append(torch.softmax(lg, 1).numpy())
        trend.append(torch.sigmoid(tl).numpy())
    return np.concatenate(probs), np.concatenate(trend)


def fit_model(Z, mask, T, y_sev, y_trend, lam, variant="fused", epochs=MAX_EPOCHS, val=None,
              seed=SEED):
    """Train; with `val=(Z, mask, T, y_sev)` early-stop on macro-F1 -> (model, best_epoch, f1)."""
    set_seed(seed)
    model = FusionNet(variant, Z.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss = _loss_fn(y_sev, y_trend)
    tz, tm, ts, tt = _tensors(Z, mask, y_sev, y_trend)
    tx = torch.from_numpy(T)
    g = torch.Generator().manual_seed(seed)
    best_f1, best_epoch, best_state, bad = -1.0, 0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(tz), generator=g).split(BATCH):
            m_vit = m_txt = None
            if variant == "fused":                       # modality dropout, never both
                r = torch.rand(len(idx), generator=g)
                m_txt = (r >= P_DROP_TEXT).float()
                m_vit = (~((r >= P_DROP_TEXT) & (r < P_DROP_TEXT + P_DROP_VITALS))).float()
            opt.zero_grad()
            lg, tl = model(tz[idx], tm[idx], tx[idx], m_vit, m_txt)
            loss(lg, tl, ts[idx], tt[idx], lam).backward()
            opt.step()
        if val is None:
            continue
        p, _ = _predict_tensors(model, val[0], val[1], val[2])
        f1 = f1_score(val[3], p.argmax(1), average="macro")
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


def _tune(X, mask, T, y_sev, y_trend, groups, variant, lambdas, max_epochs):
    """Inner grouped CV -> (best lambda, median best epoch, per-lambda scores)."""
    inner = StratifiedGroupKFold(N_INNER, shuffle=True, random_state=SEED)
    splits = list(inner.split(X, y_sev, groups))
    scores = {}
    for lam in lambdas:
        f1s, eps = [], []
        for tr, va in splits:
            pp = Preprocessor().fit(X[tr], mask[tr])
            _, ep, f1 = fit_model(pp.transform(X[tr], mask[tr]), mask[tr], T[tr], y_sev[tr], y_trend[tr],
                                  lam, variant, max_epochs,
                                  val=(pp.transform(X[va], mask[va]), mask[va], T[va], y_sev[va]))
            f1s.append(f1)
            eps.append(ep)
        scores[lam] = (float(np.mean(f1s)), int(np.median(eps)))
    best = max(scores, key=lambda k: scores[k][0])
    return best, max(scores[best][1], 1), scores


# ---------------------------------------------------------------------------
# evaluate / train / predict / save / load
# ---------------------------------------------------------------------------

def evaluate_variant(X, mask, y_sev, y_trend, groups, T, variant, quick=False):
    """Nested grouped CV for one variant on the shared outer folds -> (summary, confusion)."""
    fold_of = outer_fold_of_patient(y_sev, groups)
    folds = np.array([fold_of[p] for p in groups])
    lambdas, max_epochs = ([0.5], 5) if quick else (LAMBDAS, MAX_EPOCHS)
    sev_f1, aurocs, praucs, chosen, cm = [], [], [], [], np.zeros((4, 4), dtype=int)
    for k in range(N_OUTER):
        tr, te = folds != k, folds == k
        lam, n_ep, scores = _tune(X[tr], mask[tr], T[tr], y_sev[tr], y_trend[tr], groups[tr], variant,
                                  lambdas, max_epochs)
        pp = Preprocessor().fit(X[tr], mask[tr])
        model, _, _ = fit_model(pp.transform(X[tr], mask[tr]), mask[tr], T[tr], y_sev[tr], y_trend[tr],
                                lam, variant, n_ep)
        probs, trend = _predict_tensors(model, pp.transform(X[te], mask[te]), mask[te], T[te])
        pred = probs.argmax(1)
        sev_f1.append(f1_score(y_sev[te], pred, average="macro"))
        cm += confusion_matrix(y_sev[te], pred, labels=[0, 1, 2, 3])
        elig = ~np.isnan(y_trend[te])
        yt = y_trend[te][elig]
        if len(np.unique(yt)) == 2:
            aurocs.append(roc_auc_score(yt, trend[elig]))
            praucs.append(average_precision_score(yt, trend[elig]))
        chosen.append({"lambda": lam, "epochs": n_ep})
        print(f"  [{variant}] fold {k}: lambda={lam} epochs={n_ep} f1={sev_f1[-1]:.3f}")
    mean_std = lambda v: (float(np.mean(v)) if v else float("nan"), float(np.std(v)) if v else float("nan"), len(v))
    summary = {"variant": variant, "severity": mean_std(sev_f1), "trend_auroc": mean_std(aurocs),
               "trend_pr_auc": mean_std(praucs), "params": chosen, "n": len(y_sev),
               "n_trend": int((~np.isnan(y_trend)).sum()), "fold_f1": sev_f1}
    conf = pd.DataFrame(cm, index=[f"true_{i}" for i in range(4)], columns=[f"pred_{i}" for i in range(4)])
    return summary, conf


def gain_table(results: list[dict]) -> pd.DataFrame:
    """Sensitivity table. Gain is paired per outer fold (all variants share the folds)."""
    by = {(r["variant"], r["agreement"]): r for r in results}
    base = by[("vitals_only", None)]
    rows = []
    for (v, a), r in by.items():
        if v != "fused":
            continue
        d = np.asarray(r["fold_f1"]) - np.asarray(base["fold_f1"])
        rows.append({"agreement": a, "text_only_f1": by[("text_only", a)]["severity"][0],
                     "vitals_only_f1": base["severity"][0], "fused_f1": r["severity"][0],
                     "fusion_gain_mean": float(d.mean()), "fusion_gain_std": float(d.std()),
                     "folds_fused_better": int((d > 0).sum())})
    return pd.DataFrame(rows)


def evaluate(path: Path = OUT_PATH, quick: bool = False, agreements: list[float] | None = None,
             gen_path: Path = GEN_PATH):
    """Ablation at each agreement level. vitals_only does not depend on text, so it runs once.

    Returns (list of summaries with an `agreement` key, dict of confusion matrices, gain table).
    """
    X, mask, y_sev, y_trend, groups = load_data(path)
    gen = load_generator(gen_path)
    agreements = agreements or gen["agreement_sweep"]
    results, confs = [], {}
    s, c = evaluate_variant(X, mask, y_sev, y_trend, groups, np.zeros((len(y_sev), N_TEXT), np.float32),
                            "vitals_only", quick)
    s["agreement"] = None
    results.append(s)
    confs["vitals_only"] = c
    for a in agreements:
        T = make_text_features(y_sev, gen, a)
        for v in ("text_only", "fused"):
            s, c = evaluate_variant(X, mask, y_sev, y_trend, groups, T, v, quick)
            s["agreement"] = a
            results.append(s)
            confs[f"{v}_agree{a}"] = c
    gain = gain_table(results)
    print(gain.to_string(index=False))
    print(CAVEAT)
    return results, confs, gain


def train(path: Path = OUT_PATH, quick: bool = False, agreement: float | None = None,
          gen_path: Path = GEN_PATH):
    """Fit the fused model on all windows (SYNTHETIC text) for XAI and the demo. Not a result."""
    X, mask, y_sev, y_trend, groups = load_data(path)
    gen = load_generator(gen_path)
    T = make_text_features(y_sev, gen, agreement)
    lambdas, max_epochs = ([0.5], 5) if quick else (LAMBDAS, MAX_EPOCHS)
    lam, n_ep, _ = _tune(X, mask, T, y_sev, y_trend, groups, "fused", lambdas, max_epochs)
    pp = Preprocessor().fit(X, mask)
    model, _, _ = fit_model(pp.transform(X, mask), mask, T, y_sev, y_trend, lam, "fused", n_ep)
    return {"model": model, "preprocessor": pp, "lambda": lam, "epochs": n_ep, "variant": "fused"}


def predict(bundle, X: np.ndarray, mask: np.ndarray, text: np.ndarray | None = None):
    """Raw windows [N,6,7] + mask + optional text [N,19] -> (severity_probs [N,4], trend_prob [N]).

    text=None zeroes the text branch (vitals-only inference through the fused net).
    """
    Z = bundle["preprocessor"].transform(X, mask)
    T = np.zeros((len(Z), N_TEXT), np.float32) if text is None else np.asarray(text, np.float32)
    m_txt = np.full(len(Z), 0.0 if text is None else 1.0, np.float32)
    return _predict_tensors(bundle["model"], Z, mask, T, m_txt)


def save(bundle, version: int = 1) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"fusion_v{version}_{dt.date.today():%Y%m%d}.pt"
    torch.save({"state_dict": bundle["model"].state_dict(), "preprocessor": bundle["preprocessor"].to_dict(),
                "lambda": bundle["lambda"], "epochs": bundle["epochs"], "variant": bundle["variant"],
                "data_version": DATA_VERSION, "text": "SYNTHETIC"}, path)
    return path


def load(path: Path):
    ckpt = torch.load(path, weights_only=False)
    model = FusionNet(ckpt["variant"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "preprocessor": Preprocessor.from_dict(ckpt["preprocessor"]),
            "lambda": ckpt["lambda"], "epochs": ckpt["epochs"], "variant": ckpt["variant"]}


def log_results(results: list[dict]) -> None:
    today, commit = dt.date.today().isoformat(), _git_commit()
    rows = []
    for r in results:
        tag = "vitals_only" if r["agreement"] is None else f"{r['variant']}_agree{r['agreement']}"
        note = (f"nested grouped CV {N_OUTER}x{N_INNER} n={r['n']} "
                + ("same net+seed as bigru (should reproduce it)" if r["agreement"] is None
                   else f"text=SYNTHETIC agreement={r['agreement']}"))
        rows.append([today, commit, "fusion", tag, json.dumps(r["params"]), f"{r['severity'][0]:.4f}",
                     f"{r['severity'][1]:.4f}", "", DATA_VERSION, SEED,
                     f"task=severity metric=f1_macro folds_used={r['severity'][2]} {note}"])
        if r["variant"] != "text_only":
            rows.append([today, commit, "fusion", tag, json.dumps(r["params"]), f"{r['trend_pr_auc'][0]:.4f}",
                         f"{r['trend_pr_auc'][1]:.4f}", "", DATA_VERSION, SEED,
                         f"task=trend metric=pr_auc folds_used={r['trend_pr_auc'][2]} {note} "
                         f"auroc={r['trend_auroc'][0]:.4f}+/-{r['trend_auroc'][1]:.4f}"])
    pd.DataFrame(rows).to_csv(LOG_PATH, mode="a", header=False, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Fusion ablation + sensitivity (SYNTHETIC text)")
    p.add_argument("--data", type=Path, default=OUT_PATH)
    p.add_argument("--generator", type=Path, default=GEN_PATH)
    p.add_argument("--agreements", type=float, nargs="+", default=None, help="default: sweep in the yaml")
    p.add_argument("--quick", action="store_true", help="1 lambda, 5 epochs (smoke test, not a result)")
    p.add_argument("--save-model", action="store_true", help="also fit fused on all data and save to models/")
    a = p.parse_args()
    results, confs, gain = evaluate(a.data, a.quick, a.agreements, a.generator)
    if not a.quick:
        REPORT_DIR.mkdir(exist_ok=True)
        gain.to_csv(REPORT_DIR / "fusion_sensitivity.csv", index=False)
        for name, conf in confs.items():
            conf.to_csv(REPORT_DIR / f"fusion_confusion_{name}.csv")
        log_results(results)
        print(f"appended rows to {LOG_PATH}")
    if a.save_model:
        print(f"saved {save(train(a.data, a.quick, gen_path=a.generator))}")


if __name__ == "__main__":
    main()
