"""Hospital selection agent (docs/04, docs/08 §7, decisions D10/D11).

    python -m backend.agents.hospital_agent --hospitals data/hospitals.csv

total_score = 0.8 * policy_score + 0.2 * rf_qvalue   (docs/08 §7.2)
policy_score = 0.30 proximity + 0.25 trauma + 0.20 beds + 0.10 specialty + 0.15 blood.
When no transfusion risk is flagged the blood weight moves to proximity.

The policy score is a hand-defined rule score (say so in the report). The RF is a surrogate
trained on SIMULATED dispatch episodes with a hand-defined reward (docs/08 §7.3); it is not
learned from clinical outcomes. Pass `use_rf=False` for the rule-only ablation.

ETA comes from the caller (ORS route durations, AGENTS rule 11), keyed by hospital_id.

[ASSUMPTION] trauma_level 0-3, higher = more capable; required level = severity_class.
[ASSUMPTION] injury -> specialty map below; [TODO-VERIFY] against the specialty vocabulary of
data/hospitals.csv once it is built. Blood stock in the table is simulated unless sourced.
Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from backend.schemas import FusedRisk, HospitalRecommendation, RankedHospital, ScoreBreakdown

SEED = 42
BLOOD_GROUPS = ("A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-")
BLOOD_COLS = [f"blood_{g}" for g in BLOOD_GROUPS]
W = {"proximity": 0.30, "trauma": 0.25, "beds": 0.20, "specialty": 0.10, "blood": 0.15}
POLICY_W, RF_W = 0.8, 0.2
TRAUMA_INJURIES = ("road_accident", "fall")
INJURY_SPECIALTY = {
    "road_accident": {"trauma", "orthopaedics", "neurosurgery"},
    "fall": {"trauma", "orthopaedics", "neurosurgery"},
    "burn": {"burns", "plastic_surgery"},
    "cardiac_medical": {"cardiology", "general_medicine"},
}
# reward = -eta - 15 trauma_mismatch - 20 no_bed - 10 blood_short (docs/08 §7.3)
REWARD_FLOOR = -(60 + 15 + 20 + 10)
MODEL_PATH = Path("models/hospital_rf_v1.joblib")


def load_hospitals(path: str | Path = "data/hospitals.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"hospital_id", "name", "trauma_level", "beds_available", "specialties", *BLOOD_COLS}
    if missing := need - set(df.columns):
        raise ValueError(f"hospitals table missing columns: {sorted(missing)}")
    return df


# ---- rule-based factors (all in [0, 1]) ----
def proximity(eta_min: float) -> float:
    return 1.0 - min(eta_min / 30.0, 1.0)


def trauma_match(level: int, required: int) -> float:
    if level >= required:
        return 1.0
    return 0.5 if level >= required - 1 else 0.0


def bed_avail(beds: float) -> float:
    return min(beds / 10.0, 1.0)


def specialty_match(injury_type: str, specialties: str) -> float:
    wanted = INJURY_SPECIALTY.get(injury_type)
    if not wanted:
        return 0.5  # "other": no specific specialty, neutral
    have = {s.strip().lower() for s in str(specialties).split(";")}
    return 1.0 if wanted & have else 0.0


def blood_needed(severity_class: int, injury_type: str, bleeding_heavy: bool) -> bool:
    return bool(bleeding_heavy or (severity_class >= 2 and injury_type in TRAUMA_INJURIES))


def blood_score(row: pd.Series, group: str | None) -> tuple[float, str]:
    """(score, flag). Unknown group -> O-negative stock as proxy (docs/08 §7.2)."""
    use = group if group in BLOOD_GROUPS else "O-"
    units = float(row[f"blood_{use}"])
    score = 1.0 if units >= 4 else 0.5 if units >= 1 else 0.0
    flag = "short" if units < 1 else "matched" if group in BLOOD_GROUPS else "unknown"
    return score, flag


def policy_weights(blood_need: bool) -> dict[str, float]:
    w = dict(W)
    if not blood_need:
        w["proximity"] += w["blood"]
        w["blood"] = 0.0
    return w


def _rf_features(eta, level, beds, spec, severity, need, short) -> list[float]:
    return [float(eta), float(level), float(beds), float(spec), float(severity),
            float(need), float(short)]


def simulated_reward(eta, level, beds, severity, need, short) -> float:
    return float(-eta - 15 * (level < severity) - 20 * (beds <= 0)
                 - 10 * (bool(need) and bool(short)))


def simulate_episodes(n: int = 5000, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Random (hospital, incident) pairs + realised reward. SIMULATED, hand-defined reward."""
    rng = np.random.default_rng(seed)
    eta = rng.uniform(2, 60, n)
    level = rng.integers(0, 4, n)
    beds = rng.integers(0, 15, n)
    spec = rng.choice([0.0, 0.5, 1.0], n)
    sev = rng.integers(0, 4, n)
    need = rng.random(n) < 0.35
    short = rng.random(n) < 0.3
    X = np.array([_rf_features(*t) for t in zip(eta, level, beds, spec, sev, need, short)])
    y = np.array([simulated_reward(eta[i], level[i], beds[i], sev[i], need[i], short[i])
                  for i in range(n)])
    return X, y + rng.normal(0, 1.5, n)  # realised-reward noise


class HospitalAgent:
    def __init__(self, hospitals: pd.DataFrame, rf: RandomForestRegressor | None = None):
        self.hospitals = hospitals.reset_index(drop=True)
        self.rf = rf

    # ---- model lifecycle ----
    def train(self, n_episodes: int = 5000, seed: int = SEED) -> dict:
        X, y = simulate_episodes(n_episodes, seed)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        self.rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                        random_state=seed, n_jobs=-1).fit(Xtr, ytr)
        return self.evaluate(Xte, yte)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        pred = self.rf.predict(X)
        return {"mae_reward": float(mean_absolute_error(y, pred)),
                "n": int(len(y)), "data": "simulated episodes"}

    def save(self, path: str | Path = MODEL_PATH) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.rf, path)

    def load(self, path: str | Path = MODEL_PATH) -> None:
        self.rf = joblib.load(path)

    # ---- inference ----
    def predict(self, incident_id: str, risk: FusedRisk, injury_type: str,
                etas_min: dict[str, float], blood_group: str | None = None,
                bleeding_heavy: bool = False, use_rf: bool = True,
                top_k: int = 5) -> HospitalRecommendation:
        use_rf = use_rf and self.rf is not None
        sev = risk.severity_class
        need = blood_needed(sev, injury_type, bleeding_heavy)
        w = policy_weights(need)
        rows = [r for _, r in self.hospitals.iterrows() if r["hospital_id"] in etas_min]
        if not rows:
            raise ValueError("no hospital has an ETA")
        parts = []
        for r in rows:
            eta = float(etas_min[r["hospital_id"]])
            b, flag = blood_score(r, blood_group)
            parts.append(dict(r=r, eta=eta, flag=flag, prox=proximity(eta),
                              tr=trauma_match(int(r["trauma_level"]), sev),
                              bd=bed_avail(float(r["beds_available"])),
                              sp=specialty_match(injury_type, r["specialties"]),
                              bl=b if need else 0.0))
        if use_rf:
            X = np.array([_rf_features(p["eta"], p["r"]["trauma_level"], p["r"]["beds_available"],
                                       p["sp"], sev, need, p["flag"] == "short") for p in parts])
            q = np.clip(1.0 + self.rf.predict(X) / -REWARD_FLOOR, 0.0, 1.0)
        out = []
        for i, p in enumerate(parts):
            policy = (w["proximity"] * p["prox"] + w["trauma"] * p["tr"] + w["beds"] * p["bd"]
                      + w["specialty"] * p["sp"] + w["blood"] * p["bl"])
            qv = float(q[i]) if use_rf else None
            total = POLICY_W * policy + RF_W * qv if use_rf else policy
            out.append(RankedHospital(
                hospital_id=str(p["r"]["hospital_id"]), name=str(p["r"]["name"]),
                total_score=round(float(total), 6), eta_min=p["eta"],
                breakdown=ScoreBreakdown(proximity=p["prox"], trauma_match=p["tr"], beds=p["bd"],
                                         specialty=p["sp"], blood_match=p["bl"], rf_qvalue=qv),
                blood_flag=p["flag"] if need else "unknown"))
        out.sort(key=lambda h: (-h.total_score, h.eta_min))
        return HospitalRecommendation(incident_id=incident_id, ranked=out[:top_k])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hospitals", default="data/hospitals.csv")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(MODEL_PATH))
    a = ap.parse_args()
    agent = HospitalAgent(load_hospitals(a.hospitals))
    print(agent.train(a.episodes, a.seed))
    agent.save(a.out)


if __name__ == "__main__":
    main()
