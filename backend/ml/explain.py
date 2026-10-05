"""Explainability for the vitals models (docs/02 §7, docs/15 D13, AGENTS rule 10).

    python -m backend.ml.explain                      # make explain (trees trained, BiGRU loaded)
    python -m backend.ml.explain --skip-trees         # IG on the saved BiGRU only
    python -m backend.ml.explain --tree-model-paths models/rf_severity_v1_*.joblib ...

* Tree baselines (RF, GB): SHAP TreeExplainer on the imputed 12 tabular features.
* BiGRU: Integrated Gradients (Captum) on the z-scored [6,12] input, per head. SHAP is never
  used on the BiGRU.

Explanations are of models fit on all windows (`train()` in baselines / seq_model); they show
what the fitted model relies on, not reported performance. IG baseline [ASSUMPTION]: the all-zero
z-scored input, i.e. every real step at the training mean ("average reading"); padded steps are
zero in input and baseline, so they receive zero attribution. Windows are subsampled
(seeded) for speed. Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients

from backend.data.mimic_windows import OUT_PATH, T
from backend.ml import baselines, seq_model
from backend.ml.baselines import FEATURE_NAMES, MODEL_DIR, REPORT_DIR, SEED, tabular_features

FIG_DIR = REPORT_DIR / "figures"
N_SAMPLE = 2000
IG_STEPS = 64
CLASSES = [0, 1, 2, 3]
STEP_NAMES = [f"t-{T - 1 - i}" if i < T - 1 else "t0" for i in range(T)]


def _sample(n: int, n_sample: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(n, n_sample), replace=False))


# ---------------------------------------------------------------------------
# SHAP on tree pipelines
# ---------------------------------------------------------------------------

def shap_tree(pipe, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pipeline(impute, clf) + raw features [N,12] -> (shap [N,12,C], expected_value [C])."""
    import shap
    from sklearn.ensemble import GradientBoostingClassifier
    Xi = pipe[:-1].transform(F)
    clf = pipe.named_steps["clf"]
    if isinstance(clf, GradientBoostingClassifier) and clf.n_classes_ > 2:
        return _shap_gb_multiclass(clf, Xi)
    expl = shap.TreeExplainer(clf)
    sv = np.asarray(expl.shap_values(Xi))
    if sv.ndim == 2:                                     # binary GB: log-odds of class 1
        sv = sv[..., None]
    return sv, np.atleast_1d(expl.expected_value)


def _shap_gb_multiclass(clf, Xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact TreeSHAP for multiclass sklearn GB (unsupported by shap.TreeExplainer).

    Raw score of class k = init_k + learning_rate * sum_i tree_ik(x); SHAP is additive over
    trees, so per-tree TreeSHAP values are summed. Output is in raw (log-odds) space.
    """
    import shap
    n_cls = clf.n_classes_
    sv = np.zeros((len(Xi), Xi.shape[1], n_cls))
    ev = clf._raw_predict_init(Xi[:1])[0].astype(float)
    for k in range(n_cls):
        for tree in clf.estimators_[:, k]:
            e = shap.TreeExplainer(tree)
            sv[..., k] += clf.learning_rate * np.asarray(e.shap_values(Xi))
            ev[k] += clf.learning_rate * float(np.atleast_1d(e.expected_value)[0])
    return sv, ev


def summarize_shap(sv: np.ndarray, feature_names=FEATURE_NAMES) -> pd.DataFrame:
    """Mean |SHAP| per feature, overall and per output column, sorted by overall."""
    df = pd.DataFrame({f"class_{c}": np.abs(sv[..., c]).mean(axis=0) for c in range(sv.shape[2])},
                      index=list(feature_names))
    df.insert(0, "mean_abs_all", np.abs(sv).mean(axis=(0, 2)))
    return df.sort_values("mean_abs_all", ascending=False)


# ---------------------------------------------------------------------------
# Integrated Gradients on the BiGRU
# ---------------------------------------------------------------------------

class _Head(torch.nn.Module):
    """Expose one head of BiGRUTwoHead as a [B, K] output for Captum."""

    def __init__(self, model, head: str):
        super().__init__()
        self.model, self.head = model, head

    def forward(self, z, mask):
        logits, trend = self.model(z, mask)
        return logits if self.head == "severity" else trend.unsqueeze(1)


def ig_bigru(model, Z: np.ndarray, mask: np.ndarray, head: str = "severity",
             n_steps: int = IG_STEPS, batch: int = 256):
    """Z [N,6,12] z-scored, mask [N,6] -> (attr [N,6,12], delta [N], target [N]).

    Severity: target = predicted class (logit). Trend: target = trend logit.
    """
    model.eval()
    f = _Head(model, head)
    ig = IntegratedGradients(f)
    attrs, deltas, targets = [], [], []
    for i in range(0, len(Z), batch):
        z = torch.from_numpy(Z[i:i + batch]).requires_grad_()
        m = torch.from_numpy(mask[i:i + batch])
        with torch.no_grad():
            out = f(z, m)
        tgt = out.argmax(1) if head == "severity" else torch.zeros(len(z), dtype=torch.long)
        a, d = ig.attribute(z, baselines=torch.zeros_like(z), target=tgt,
                            additional_forward_args=(m,), n_steps=n_steps,
                            return_convergence_delta=True)
        attrs.append(a.detach().numpy())
        deltas.append(d.detach().numpy())
        targets.append(tgt.numpy())
    return np.concatenate(attrs), np.concatenate(deltas), np.concatenate(targets)


def summarize_ig(attr: np.ndarray, feature_names=FEATURE_NAMES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (per feature: mean |attr| summed over time + mean signed; heatmap step x feature)."""
    per_feat = pd.DataFrame({"mean_abs": np.abs(attr).sum(axis=1).mean(axis=0),
                             "mean_signed": attr.sum(axis=1).mean(axis=0)},
                            index=list(feature_names)).sort_values("mean_abs", ascending=False)
    heat = pd.DataFrame(np.abs(attr).mean(axis=0), index=STEP_NAMES, columns=list(feature_names))
    return per_feat, heat


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _bar(df: pd.DataFrame, col: str, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = df[col].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(s.index, s.values, color="#4C72B0")
    ax.set_xlabel(col.replace("_", " "))
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _heatmap(heat: pd.DataFrame, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(heat.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(heat.shape[1]), heat.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(heat.shape[0]), heat.index, fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="mean |IG|")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def latest(pattern: str) -> Path | None:
    hits = sorted(MODEL_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def run_trees(X, mask, y_sev, y_trend, names, tasks, paths, n_sample, save_trees, path):
    F = tabular_features(X, mask)
    given = {}
    for p in paths or []:
        stem = Path(p).name.split("_")                   # <name>_<task>_v<N>_<date>.joblib
        given[(stem[0], stem[1])] = baselines.load(Path(p))
    for task in tasks:
        keep = np.ones(len(F), bool) if task == "severity" else ~np.isnan(y_trend)
        idx = np.flatnonzero(keep)[_sample(int(keep.sum()), n_sample)]
        for name in names:
            pipe = given.get((name, task))
            if pipe is None:
                print(f"training {name}/{task} on all windows (grid search, for XAI only) ...")
                pipe = baselines.train(name, task, path)
                if save_trees:
                    print("saved", baselines.save(pipe, name, task))
            sv, _ = shap_tree(pipe, F[idx])
            summ = summarize_shap(sv)
            stem = f"xai_shap_{name}_{task}"
            summ.to_csv(REPORT_DIR / f"{stem}.csv")
            _bar(summ, "mean_abs_all", f"SHAP mean |value| - {name.upper()} {task} "
                 f"(MIMIC-IV Demo, n={len(idx)})", FIG_DIR / f"{stem}.png")
            print(f"{stem}: top = {', '.join(summ.index[:5])}")


def run_ig(X, mask, bigru_path: Path, n_sample: int, n_steps: int):
    bundle = seq_model.load(bigru_path)
    idx = _sample(len(X), n_sample)
    Z = bundle["preprocessor"].transform(X[idx], mask[idx])
    rows = []
    for head in ("severity", "trend"):
        attr, delta, tgt = ig_bigru(bundle["model"], Z, mask[idx], head, n_steps)
        per_feat, heat = summarize_ig(attr)
        stem = f"xai_ig_bigru_{head}"
        per_feat.to_csv(REPORT_DIR / f"{stem}_features.csv")
        heat.to_csv(REPORT_DIR / f"{stem}_heatmap.csv")
        _bar(per_feat, "mean_abs", f"Integrated Gradients - BiGRU {head} head "
             f"(MIMIC-IV Demo, n={len(idx)})", FIG_DIR / f"{stem}.png")
        _heatmap(heat, f"IG |attribution| by time step - BiGRU {head}", FIG_DIR / f"{stem}_heatmap.png")
        if head == "severity":                           # per predicted class
            for c in CLASSES:
                if (tgt == c).any():
                    summarize_ig(attr[tgt == c])[0].to_csv(REPORT_DIR / f"{stem}_pred{c}.csv")
        rows.append({"head": head, "n": len(idx), "ig_steps": n_steps, "model": bigru_path.name,
                     "abs_delta_mean": float(np.abs(delta).mean()),
                     "abs_delta_max": float(np.abs(delta).max())})
        print(f"{stem}: top = {', '.join(per_feat.index[:5])}  |delta| mean={rows[-1]['abs_delta_mean']:.2e}")
    pd.DataFrame(rows).to_csv(REPORT_DIR / "xai_ig_convergence.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="SHAP (trees) + Integrated Gradients (BiGRU)")
    p.add_argument("--data", type=Path, default=OUT_PATH)
    p.add_argument("--tree-models", nargs="*", default=["rf", "gb"], choices=["rf", "gb"])
    p.add_argument("--tasks", nargs="*", default=["severity", "trend"], choices=["severity", "trend"])
    p.add_argument("--tree-model-paths", nargs="*", help="saved baselines joblibs to reuse")
    p.add_argument("--save-trees", action="store_true", help="save trees trained here to models/")
    p.add_argument("--bigru", type=Path, help="default: latest models/bigru_v*.pt")
    p.add_argument("--n-sample", type=int, default=N_SAMPLE)
    p.add_argument("--ig-steps", type=int, default=IG_STEPS)
    p.add_argument("--skip-trees", action="store_true")
    p.add_argument("--skip-ig", action="store_true")
    a = p.parse_args()

    torch.manual_seed(SEED)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    X, mask, y_sev, y_trend, _ = seq_model.load_data(a.data)
    if not a.skip_trees:
        run_trees(X, mask, y_sev, y_trend, a.tree_models, a.tasks, a.tree_model_paths,
                  a.n_sample, a.save_trees, a.data)
    if not a.skip_ig:
        path = a.bigru or latest("bigru_v*.pt")
        if path is None:
            raise SystemExit("no BiGRU model: run `python -m backend.ml.seq_model --save-model` first")
        run_ig(X, mask, path, a.n_sample, a.ig_steps)
    print("NOTE: explanations of models fit on all windows; not performance results.")


if __name__ == "__main__":
    main()
