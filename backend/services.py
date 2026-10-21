"""Service container for the API: loads whatever models/data exist and records degraded modes.

Nothing here is required at start-up; each missing piece degrades as docs/07 §6 says, and the
reason is listed in `Services.notes` (shown by GET /health):
  * no Whisper model / audio unusable -> 422, client falls back to text input (never guess).
  * no text classifier                -> uniform text-severity placeholder, model_version flags it.
  * no fusion model                   -> text-severity probs only (`model_version` says so).
  * no RAG index                      -> guidance "Wait for professional help; call 108."
  * no hospitals.csv / ambulance file -> dispatch returns 503 with the reason.
  * no ORS key / ORS down             -> straight-line ETAs, route_estimated = true.

[ASSUMPTION] `data/ambulance_bases.csv` columns: ambulance_id, base_name, lat, lon (one row per
ambulance; docs/15 D11 names the file but not its columns).
[ASSUMPTION] POLICE_UNITS_AVAILABLE env default 6 and police units requested = 0 per incident
until the police alert-sizing agent exists (AGENTS status: not started).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

MODELS = Path("models")
HOSPITALS_CSV = Path("data/hospitals.csv")
AMBULANCES_CSV = Path("data/ambulance_bases.csv")


def latest(pattern: str) -> Optional[Path]:
    files = sorted(MODELS.glob(pattern))
    return files[-1] if files else None


@dataclass
class Services:
    transcriber: Any = None            # backend.asr.transcribe.Transcriber (lazy model load)
    text_model: Any = None             # sklearn Pipeline
    fusion: Optional[dict] = None      # backend.ml.fusion_model bundle
    fusion_version: str = ""
    deterioration: Optional[dict] = None   # backend.ml.vitaldb_model HGB bundle (D19)
    deterioration_version: str = ""
    rag: Any = None                    # backend.rag.pipeline.FirstAidRAG
    hospital_agent: Any = None         # backend.agents.hospital_agent.HospitalAgent
    hospitals: Optional[pd.DataFrame] = None
    ambulances: dict[str, tuple[float, float]] = field(default_factory=dict)
    police_units_available: int = 6
    beds_now: dict[str, int] = field(default_factory=dict)
    cordons: list = field(default_factory=list)
    router_factory: Any = None         # (locate_incident, locate_hospital) -> Router
    notes: list[str] = field(default_factory=list)

    def hospital_location(self, hospital_id: str):
        if self.hospitals is None:
            return None
        row = self.hospitals.loc[self.hospitals["hospital_id"] == hospital_id]
        return None if row.empty else (float(row.iloc[0]["lat"]), float(row.iloc[0]["lon"]))


def _try(notes: list[str], what: str, fn):
    try:
        return fn()
    except Exception as e:  # any failure = degraded mode, never a crash at start-up
        notes.append(f"{what}: {type(e).__name__}: {e}")
        return None


def load_services() -> Services:
    """Real services from `models/` and `data/`, with every gap recorded in `notes`."""
    from dotenv import load_dotenv
    load_dotenv()
    s = Services(police_units_available=int(os.getenv("POLICE_UNITS_AVAILABLE", "6")))
    n = s.notes

    from backend.asr.transcribe import Transcriber
    s.transcriber = _try(n, "asr", Transcriber)

    p = latest("text_severity_v*.joblib")
    if p is None:
        n.append("text classifier: no models/text_severity_v*.joblib, using uniform placeholder")
    else:
        from backend.ml import text_severity
        s.text_model = _try(n, "text classifier", lambda: text_severity.load(p))

    p = latest("fusion_v*.pt")
    if p is None:
        n.append("fusion: no models/fusion_v*.pt, risk = text severity only")
    else:
        from backend.ml import fusion_model
        s.fusion = _try(n, "fusion", lambda: fusion_model.load(p))
        s.fusion_version = p.stem

    from backend.rag import pipeline as rp
    if (rp.INDEX_DIR / "index.faiss").exists():
        s.rag = _try(n, "rag", lambda: rp.load(nli=False))
    else:
        n.append("rag: no index (run make build-rag), guidance falls back to 'call 108'")

    p = latest("vitaldb_hgb_v*.joblib")
    if p is None:
        n.append("deterioration: no models/vitaldb_hgb_v*.joblib (run vitaldb_model --train-final)")
    else:
        from backend.ml import vitaldb_model
        s.deterioration = _try(n, "deterioration model", lambda: vitaldb_model.load_hgb(p))
        s.deterioration_version = p.stem

    if HOSPITALS_CSV.exists():
        from backend.agents.hospital_agent import MODEL_PATH, HospitalAgent, load_hospitals
        s.hospitals = _try(n, "hospitals", lambda: load_hospitals(HOSPITALS_CSV))
        if s.hospitals is not None:
            s.hospital_agent = HospitalAgent(s.hospitals)
            if Path(MODEL_PATH).exists():
                _try(n, "hospital RF", lambda: s.hospital_agent.load(MODEL_PATH))
            else:
                n.append("hospital RF: not trained, rule-only scores")
            s.beds_now = {str(r.hospital_id): int(r.beds_available) for r in s.hospitals.itertuples()}
    else:
        n.append(f"hospitals: {HOSPITALS_CSV} missing (docs/15 D11), dispatch unavailable")

    if AMBULANCES_CSV.exists():
        df = pd.read_csv(AMBULANCES_CSV, dtype={"ambulance_id": str})
        s.ambulances = {r.ambulance_id: (float(r.lat), float(r.lon)) for r in df.itertuples()}
    else:
        n.append(f"ambulances: {AMBULANCES_CSV} missing (docs/15 D11), dispatch unavailable")

    from backend.routing import OrsRouter, StraightLineRouter
    s.router_factory = lambda inc, hosp: OrsRouter.from_env(StraightLineRouter(inc, hosp))
    if not os.getenv("ORS_API_KEY", "").strip():
        n.append("routing: no ORS_API_KEY, straight-line ETAs (route_estimated=true)")
    return s
