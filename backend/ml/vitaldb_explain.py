"""SHAP for the deployed VitalDB 10-min deterioration model (docs/15 D13 + D19).

    python -m backend.ml.vitaldb_explain     # global summary -> reports/vitaldb_shap_*.csv + figures

The deployed model is isotonic calibration over 3 HistGradientBoosting pipelines (one per grouped
calibration fold). SHAP TreeExplainer runs on each HGB and the values are averaged. Values are in
the HGB log-odds space BEFORE calibration; isotonic calibration is monotone, so the direction and
ranking of reasons carry over, but SHAP values do not add up to the shown probability.
`reasons()` turns the top features of one prediction into short text for the dashboards.
Temperature features are left OUT of the live reasons: globally they rank high mainly when the
probe is missing / just placed (early in surgery, around induction hypotension), a surgical
workflow artefact that would mislead in an ambulance. They stay in the global report.
Explanations describe what the fitted model relies on (intra-operative data), not causes.
Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from backend.data.vitaldb_windows import OUT_PATH, VITALS, load
from backend.ml.vitaldb_model import TAB_NAMES, load_hgb, tabular_features

REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"
SEED = 42
LABEL = {"hr": "HR", "rr": "RR", "spo2": "SpO2", "sbp": "SBP", "temp": "Temp"}
UNIT = {"hr": "bpm", "rr": "/min", "spo2": "%", "sbp": "mmHg", "temp": "°C"}


def _explainers(bundle: dict) -> list:
    if "_explainers" not in bundle:
        bundle["_explainers"] = [shap.TreeExplainer(c.estimator.named_steps["clf"])
                                 for c in bundle["model"].calibrated_classifiers_]
    return bundle["_explainers"]


def shap_values(bundle: dict, T: np.ndarray) -> np.ndarray:
    """[N,25] tabular features -> [N,25] mean SHAP (log-odds, pre-calibration)."""
    vals = [np.asarray(e.shap_values(T)) for e in _explainers(bundle)]
    vals = [v[..., 1] if v.ndim == 3 else v for v in vals]       # binary: keep positive class
    return np.mean(vals, axis=0)


def describe(name: str, value: float) -> str:
    """'change_sbp', -12.0 -> 'SBP fell 12 mmHg over 5 min'."""
    stat, vital = name.split("_", 1)
    lab, unit = LABEL[vital], UNIT[vital]
    if np.isnan(value):
        return f"{lab} not measured"
    fmt = f"{value:.1f}" if vital == "temp" else f"{value:.0f}"
    if stat == "change":
        if abs(value) < 0.5:
            return f"{lab} stable over 5 min"
        mag = f"{abs(value):.1f}" if vital == "temp" else f"{abs(value):.0f}"
        return f"{lab} {'rose' if value > 0 else 'fell'} {mag} {unit} over 5 min"
    word = {"last": "latest", "mean": "5-min mean", "min": "5-min lowest", "max": "5-min highest"}[stat]
    return f"{lab} {word} {fmt} {unit}"


LIVE_EXCLUDE = {"temp"}


def reasons(bundle: dict, X: np.ndarray, mask: np.ndarray, k: int = 3) -> list[str]:
    """Top-k reasons for ONE window X [1,T,5]: '<description> (raises risk)' / '(lowers risk)'."""
    T = tabular_features(X.astype(np.float32), mask)
    sv = shap_values(bundle, T)[0]
    keep = [j for j in np.argsort(-np.abs(sv))
            if TAB_NAMES[j].split("_", 1)[1] not in LIVE_EXCLUDE and abs(sv[j]) > 1e-6
            and not np.isnan(T[0, j])]
    return [f"{describe(TAB_NAMES[j], T[0, j])} ({'raises' if sv[j] > 0 else 'lowers'} risk)"
            for j in keep[:k]]


def global_summary(bundle: dict, X: np.ndarray, mask: np.ndarray, n_sample: int = 5000,
                   seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    idx = np.random.default_rng(seed).choice(len(X), min(n_sample, len(X)), replace=False)
    T = tabular_features(X[idx], mask[idx])
    sv = shap_values(bundle, T)
    feat = pd.DataFrame({"feature": TAB_NAMES, "mean_abs_shap": np.abs(sv).mean(0)}) \
        .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    feat["vital"] = feat.feature.str.split("_", n=1).str[1]
    by_vital = feat.groupby("vital", as_index=False).mean_abs_shap.sum() \
        .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return feat, by_vital, sv, T


def _figures(sv: np.ndarray, T: np.ndarray, feat: pd.DataFrame, tag: str) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    top = feat.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.feature, top.mean_abs_shap, color="#d9480f")
    ax.set_xlabel("mean |SHAP| (log-odds, pre-calibration)")
    ax.set_title("VitalDB 10-min deterioration: top features (HGB)")
    fig.tight_layout()
    p = FIG_DIR / f"vitaldb_shap_bar{tag}.png"
    fig.savefig(p, dpi=150); plt.close(fig); out.append(p)
    shap.summary_plot(sv, T, feature_names=TAB_NAMES, max_display=12, show=False)
    p = FIG_DIR / f"vitaldb_shap_beeswarm{tag}.png"
    plt.title("VitalDB 10-min deterioration (intra-operative data)")
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close("all"); out.append(p)
    return out


def latest_model() -> Path | None:
    files = sorted(Path("models").glob("vitaldb_hgb_v*.joblib"))
    return files[-1] if files else None


def main() -> None:
    p = argparse.ArgumentParser(description="SHAP for the deployed VitalDB HGB model")
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--data", type=Path, default=OUT_PATH)
    p.add_argument("--n-sample", type=int, default=5000)
    a = p.parse_args()
    path = a.model or latest_model()
    if path is None:
        raise SystemExit("no models/vitaldb_hgb_v*.joblib: run python -m backend.ml.vitaldb_model --train-final")
    bundle, d = load_hgb(path), load(a.data)
    feat, by_vital, sv, T = global_summary(bundle, d["X"], d["mask"], a.n_sample)
    REPORT_DIR.mkdir(exist_ok=True)
    feat.to_csv(REPORT_DIR / "vitaldb_shap_features.csv", index=False)
    by_vital.to_csv(REPORT_DIR / "vitaldb_shap_by_vital.csv", index=False)
    figs = _figures(sv, T, feat, "")
    print(f"model {path.name}, {len(T)} sampled windows")
    print(feat.head(10).round(4).to_string(index=False))
    print(by_vital.round(4).to_string(index=False))
    print("figures: " + ", ".join(str(f) for f in figs))


if __name__ == "__main__":
    main()
