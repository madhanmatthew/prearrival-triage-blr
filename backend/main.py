"""FastAPI app (docs/07 §5): intake -> dialogue -> guidance -> vitals/fusion -> dispatch.

Agents are in-process Python modules (docs/07 D5); only clients use HTTP/WebSocket.
State is in memory (one process, demo scale). Degraded modes follow docs/07 §6; GET /health
lists which services are live. Research prototype, not a medical device: every response is
decision support for a human dispatcher.

    uvicorn backend.main:app --reload        (make run-api)
"""
from __future__ import annotations

import datetime as dt
import itertools
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.agents.coordinator import IncidentCase, coordinate
from backend.agents.police_agent import (AlertBoard, depart_time, load_corridor_junctions,
                                         plan_junction_posts, size_alert)
from backend.dialogue.state_machine import DialogueManager
from backend.ml.news2 import score_reading
from backend.rag.pipeline import LLM_DOWN_MSG
from backend.schemas import (GPS, DispatchDecision, FusedRisk, IncidentReport, Language,
                             VitalsReading, WSMessage)
from backend.services import Services, load_services

WINDOW_T = 6  # docs/07 §4.1


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class TextIntake(BaseModel):
    text: str
    gps: GPS
    language: Language = "en"


class AnswerIn(BaseModel):
    answer: str


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class Incident:
    incident_id: str
    created_at: str
    gps: GPS
    dialogue: DialogueManager
    report: Optional[IncidentReport] = None
    vitals: list[VitalsReading] = field(default_factory=list)
    risk: Optional[FusedRisk] = None
    decision: Optional[DispatchDecision] = None
    dispatch_notes: list[str] = field(default_factory=list)

    @property
    def patient_id(self) -> str:
        return f"{self.incident_id}-P1"


class Hub:
    """WebSocket channels: police, map, hospital/<id>."""

    def __init__(self):
        self.channels: dict[str, set[WebSocket]] = {}

    async def serve(self, channel: str, ws: WebSocket):
        await ws.accept()
        self.channels.setdefault(channel, set()).add(ws)
        try:
            while True:
                await ws.receive_text()        # clients only listen; keep the socket open
        except WebSocketDisconnect:
            self.channels[channel].discard(ws)

    async def send(self, channel: str, type_: str, data: BaseModel):
        msg = WSMessage(type=type_, data=data.model_dump(mode="json")).model_dump(mode="json")
        for ws in list(self.channels.get(channel, ())):
            try:
                await ws.send_json(msg)
            except Exception:
                self.channels[channel].discard(ws)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _question(dm: DialogueManager) -> Optional[dict]:
    q = dm.next_question()
    return None if q is None else {"q_id": q[0], "text": q[1]}


def _text_probs(svc: Services, dm: DialogueManager) -> Optional[list[float]]:
    if svc.text_model is None:
        return None                                    # dialogue uses its uniform placeholder
    from backend.ml.text_severity import compose_text, predict_proba
    raw = [a.raw for a in dm.answers.values()]
    return [float(p) for p in predict_proba(svc.text_model, compose_text(dm.transcript, raw))[0]]


def _finish_dialogue(svc: Services, inc: Incident, now: str) -> None:
    probs = _text_probs(svc, inc.dialogue)
    inc.report = inc.dialogue.to_incident_report(inc.incident_id, now, inc.gps, probs)
    inc.risk = compute_risk(svc, inc, now)


def _window(readings: list[VitalsReading]) -> tuple[np.ndarray, np.ndarray]:
    """Last 6 readings -> X [1,6,7] (NaN = missing), mask [1,6]; left-padded like training."""
    X = np.full((1, WINDOW_T, 7), np.nan, np.float32)
    mask = np.zeros((1, WINDOW_T), bool)
    recent = readings[-WINDOW_T:]
    for i, r in enumerate(recent):
        t = WINDOW_T - len(recent) + i
        avpu = None if r.avpu is None else (0.0 if r.avpu == "A" else 1.0)
        o2 = None if r.supplemental_o2 is None else float(r.supplemental_o2)
        X[0, t] = [np.nan if v is None else v for v in (r.hr, r.rr, r.spo2, r.sbp, r.temp, avpu, o2)]
        mask[0, t] = not r.all_null
    return X, mask


DETERIORATION_T = 5   # last 5 readings, treated as 1-minute steps (docs/15 D19) [ASSUMPTION]


def deterioration_prob(svc: Services, readings: list[VitalsReading]) -> Optional[float]:
    """P(new SBP < 90 or SpO2 < 90 within 10 min) from the last 5 readings, or None."""
    if svc.deterioration is None or not readings:
        return None
    from backend.ml.vitaldb_model import predict_hgb
    X = np.full((1, DETERIORATION_T, 5), np.nan, np.float32)
    mask = np.zeros((1, DETERIORATION_T), bool)
    recent = readings[-DETERIORATION_T:]
    for i, r in enumerate(recent):
        t = DETERIORATION_T - len(recent) + i
        X[0, t] = [np.nan if v is None else v for v in (r.hr, r.rr, r.spo2, r.sbp, r.temp)]
        mask[0, t] = not np.isnan(X[0, t]).all()
    if not mask.any():
        return None
    return float(predict_hgb(svc.deterioration, X, mask)[0])


def compute_risk(svc: Services, inc: Incident, now: str) -> FusedRisk:
    rep = inc.report
    X, mask = _window(inc.vitals)
    has_vitals = bool(mask.any())
    trend = None
    if svc.fusion is not None:
        from backend.ml.fusion_model import predict
        probs, tr = predict(svc.fusion, X, mask, np.array([rep.text_features], np.float32),
                            vitals=has_vitals)
        probs = probs[0].astype(float)
        trend = float(tr[0]) if has_vitals else None
        version = f"{svc.fusion_version} (text branch trained on SYNTHETIC pairs)"
        modalities = ["text", "vitals"] if has_vitals else ["text"]
    else:
        probs = np.asarray(rep.text_severity_probs, float)
        version = "text_severity_only"
        modalities = ["text"]
    if svc.text_model is None:
        version += "; text severity = uniform placeholder"
    probs = (probs / probs.sum()).tolist()
    news2 = score_reading(inc.vitals[-1]).total if inc.vitals else None
    det = deterioration_prob(svc, inc.vitals)
    return FusedRisk.from_probs(probs, incident_id=inc.incident_id, patient_id=inc.patient_id,
                                timestamp=now, trend_prob=trend, modalities_used=modalities,
                                news2_score=news2, model_version=version,
                                deterioration_10min_prob=det,
                                deterioration_model=(f"{svc.deterioration_version} (intra-operative data)"
                                                     if det is not None else None))


class PoliceAck(BaseModel):
    junction_id: Optional[str] = None


def _require(store: dict, incident_id: str) -> Incident:
    inc = store.get(incident_id)
    if inc is None:
        raise HTTPException(404, f"unknown incident {incident_id}")
    return inc


def _incident_view(inc: Incident) -> dict:
    return {"incident_id": inc.incident_id, "created_at": inc.created_at,
            "gps": inc.gps.model_dump(), "dialogue_done": inc.report is not None,
            "next_question": None if inc.report else _question(inc.dialogue),
            "report": inc.report.model_dump() if inc.report else None,
            "vitals_count": len(inc.vitals),
            "risk": inc.risk.model_dump() if inc.risk else None,
            "decision": inc.decision.model_dump() if inc.decision else None,
            "dispatch_notes": inc.dispatch_notes}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app(services: Optional[Services] = None,
               clock: Callable[[], str] = utc_now) -> FastAPI:
    app = FastAPI(title="Pre-arrival triage (research prototype)",
                  description="Decision support only; not a medical device.")
    svc = services if services is not None else load_services()
    store: dict[str, Incident] = {}
    hub = Hub()
    counter = itertools.count(1)
    busy_ambulances: set[str] = set()
    board = AlertBoard()
    try:
        junctions = load_corridor_junctions()
    except (OSError, KeyError, ValueError):
        junctions = {}  # no corridor file: alerts without junction posts
    app.state.services, app.state.store, app.state.hub = svc, store, hub
    app.state.police_board = board

    def new_incident(dm: DialogueManager, gps: GPS) -> Incident:
        now = clock()
        iid = f"INC-{now[:10].replace('-', '')}-{next(counter):04d}"
        inc = Incident(iid, now, gps, dm)
        store[iid] = inc
        if dm.done:
            _finish_dialogue(svc, inc, now)
        return inc

    def intake_response(inc: Incident) -> dict:
        return {"incident_id": inc.incident_id, "done": inc.report is not None,
                "question": None if inc.report else _question(inc.dialogue),
                "report": inc.report.model_dump() if inc.report else None}

    @app.get("/health")
    def health():
        return {"status": "ok", "degraded": svc.notes,
                "dispatch_ready": svc.hospital_agent is not None and bool(svc.ambulances)}

    @app.post("/incident/text")
    def incident_text(body: TextIntake):
        if not body.text.strip():
            raise HTTPException(422, "empty report")
        dm = DialogueManager(language=body.language)
        dm.start(body.text)
        return intake_response(new_incident(dm, body.gps))

    @app.post("/incident/voice")
    async def incident_voice(audio: UploadFile = File(...), lat: float = Form(...),
                             lon: float = Form(...), language: Optional[Language] = Form(None)):
        if svc.transcriber is None:
            raise HTTPException(422, "speech recognition unavailable; please type the report")
        from backend.asr.intake import intake
        suffix = Path(audio.filename or "a.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(await audio.read())
            path = Path(f.name)
        try:
            dm, result = intake(path, svc.transcriber, language=language)
        except ValueError as e:                     # unsupported language / no speech
            raise HTTPException(422, f"{e}; please type the report")
        finally:
            path.unlink(missing_ok=True)
        out = intake_response(new_incident(dm, GPS(lat=lat, lon=lon)))
        out["transcript"] = result.text
        return out

    @app.post("/incident/{incident_id}/answer")
    def answer(incident_id: str, body: AnswerIn):
        inc = _require(store, incident_id)
        if inc.report is not None:
            raise HTTPException(409, "dialogue already finished")
        if inc.dialogue.next_question() is None:
            raise HTTPException(409, "no pending question")
        inc.dialogue.answer(body.answer)
        if inc.dialogue.done:
            _finish_dialogue(svc, inc, clock())
        return intake_response(inc)

    @app.get("/incident/{incident_id}/guidance")
    def guidance(incident_id: str):
        inc = _require(store, incident_id)
        if svc.rag is None:
            return {"incident_id": incident_id, "status": "unavailable", "message": LLM_DOWN_MSG,
                    "items": []}
        if inc.report is not None:
            items = [{"problem": p, **g.to_dict()} for p, g in svc.rag.guide_report(inc.report)]
            if any(i["status"] == "ok" for i in items):     # hide "not covered" next to real steps
                items = [i for i in items if i["status"] == "ok"]
        else:                                      # dialogue unfinished: free-text fallback
            items = [{"problem": "report", **svc.rag.guide(inc.dialogue.transcript).to_dict()}]
        return {"incident_id": incident_id, "status": "ok", "items": items}

    @app.post("/incident/{incident_id}/vitals")
    async def vitals(incident_id: str, reading: VitalsReading):
        inc = _require(store, incident_id)
        if reading.incident_id != incident_id:
            raise HTTPException(422, "reading.incident_id does not match the URL")
        if inc.report is None:
            raise HTTPException(409, "finish the dialogue first")
        inc.vitals.append(reading)
        inc.risk = compute_risk(svc, inc, clock())
        if inc.decision is not None:
            await hub.send(f"hospital/{inc.decision.hospital_id}", "FusedRisk", inc.risk)
        return inc.risk

    @app.post("/incident/{incident_id}/dispatch")
    async def dispatch(incident_id: str):
        inc = _require(store, incident_id)
        if inc.report is None:
            raise HTTPException(409, "finish the dialogue first")
        if inc.decision is not None:
            return inc.decision
        if svc.hospital_agent is None or not svc.ambulances:
            raise HTTPException(503, "dispatch unavailable: " + "; ".join(
                x for x in svc.notes if x.startswith(("hospitals", "ambulances"))))
        now = clock()
        pending = [i for i in store.values() if i.report is not None and i.decision is None]
        locate_inc = lambda iid: (store[iid].gps.lat, store[iid].gps.lon) if iid in store else None
        router = svc.router_factory(locate_inc, svc.hospital_location)
        free = [a for a in svc.ambulances if a not in busy_ambulances]
        cases = []
        for i in pending:
            scene = (i.gps.lat, i.gps.lon)
            etas = {str(h.hospital_id): router.eta_min(scene, (float(h.lat), float(h.lon)))[0]
                    for h in svc.hospitals.itertuples()}
            bleeding = i.dialogue.answers.get("Q_BLEEDING")
            rec = svc.hospital_agent.predict(
                i.incident_id, i.risk, i.report.injury_type, etas,
                bleeding_heavy=bool(bleeding and (bleeding.bleeding_level or 0) >= 1.0))
            police = size_alert(i.incident_id, i.gps, i.risk.severity_class, i.report.injury_type,
                                i.report.num_injured)
            cases.append(IncidentCase(
                incident_id=i.incident_id, risk=i.risk, injury_type=i.report.injury_type,
                reported_at=i.created_at, recommendation=rec,
                ambulance_etas_min={a: router.eta_min(svc.ambulances[a], scene)[0] for a in free},
                police_units_required=police.units_required,
                bleeding_heavy=bool(bleeding and (bleeding.bleeding_level or 0) >= 1.0)))
        committed = sum(i.decision.police_units for i in store.values() if i.decision)
        result = coordinate(cases, now=now, router=router, available_ambulances=free,
                            police_units_available=max(svc.police_units_available - committed, 0),
                            beds_now=svc.beds_now, cordons=svc.cordons)
        for d in result.decisions:
            store[d.incident_id].decision = d
            busy_ambulances.add(d.ambulance_id)
            await hub.send("map", "DispatchDecision", d)
            inc = store[d.incident_id]
            route = router.route(d.incident_id, d.hospital_id)
            alert = size_alert(d.incident_id, inc.gps, inc.risk.severity_class,
                               inc.report.injury_type, inc.report.num_injured,
                               route.polyline if route else ())
            posts = []
            if route is not None and d.police_units > 0:   # no units allocated -> no posts
                to_scene = router.eta_min(svc.ambulances[d.ambulance_id],
                                          (inc.gps.lat, inc.gps.lon))[0]
                posts = plan_junction_posts(route.polyline, depart_time(now, to_scene),
                                            route.eta_min, junctions)
            alert = board.open(alert.model_copy(update={"units_required": d.police_units,
                                                        "junction_posts": posts}),
                               route.eta_min if route else 0.0)
            await hub.send("police", "PoliceAlert", alert)
            await hub.send(f"hospital/{d.hospital_id}", "FusedRisk", store[d.incident_id].risk)
        for iid, r in result.resolutions.items():
            store[iid].dispatch_notes = r.needs_dispatcher
        if inc.decision is None:
            raise HTTPException(409, "not dispatched: " + ("; ".join(inc.dispatch_notes)
                                                           or "no decision"))
        return inc.decision

    @app.get("/incidents")
    def incidents():
        """Summary list for the dispatcher console, newest first."""
        out = []
        for inc in sorted(store.values(), key=lambda i: i.created_at, reverse=True):
            out.append({"incident_id": inc.incident_id, "created_at": inc.created_at,
                        "gps": inc.gps.model_dump(), "dialogue_done": inc.report is not None,
                        "injury_type": inc.report.injury_type if inc.report else None,
                        "num_injured": inc.report.num_injured if inc.report else None,
                        "severity_class": inc.risk.severity_class if inc.risk else None,
                        "risk_score": inc.risk.risk_score if inc.risk else None,
                        "vitals_count": len(inc.vitals),
                        "decision": inc.decision.model_dump() if inc.decision else None,
                        "dispatch_notes": inc.dispatch_notes})
        return out

    @app.get("/resources")
    def resources():
        """Static map layers: hospitals, ambulances (base positions), corridor junctions."""
        hosp = []
        if svc.hospitals is not None:
            for h in svc.hospitals.itertuples():
                hosp.append({"hospital_id": str(h.hospital_id), "name": str(getattr(h, "name", "")),
                             "lat": float(h.lat), "lon": float(h.lon)})
        return {"hospitals": hosp,
                "ambulances": [{"ambulance_id": a, "lat": p[0], "lon": p[1], "busy": a in busy_ambulances}
                               for a, p in svc.ambulances.items()],
                "junctions": [{"junction_id": j, "lat": p[0], "lon": p[1]} for j, p in junctions.items()]}

    @app.get("/incident/{incident_id}")
    def incident(incident_id: str):
        return _incident_view(_require(store, incident_id))

    def _alert(incident_id: str):
        _require(store, incident_id)
        if incident_id not in board:
            raise HTTPException(404, f"no police alert for {incident_id} (not dispatched)")

    @app.post("/incident/{incident_id}/position")
    async def position(incident_id: str, gps: GPS):
        _alert(incident_id)
        alert = board.update_position(incident_id, gps.lat, gps.lon, clock())
        await hub.send("police", "PoliceAlert", alert)
        return alert

    @app.get("/police/{incident_id}")
    def police_alert(incident_id: str):
        _alert(incident_id)
        return board.get(incident_id)

    @app.post("/police/{incident_id}/ack")
    async def police_ack(incident_id: str, body: Optional[PoliceAck] = None):
        _alert(incident_id)
        try:
            alert = board.acknowledge(incident_id, body.junction_id if body else None)
        except KeyError:
            raise HTTPException(404, f"junction {body.junction_id} is not on this alert")
        except ValueError as e:
            raise HTTPException(409, str(e))
        await hub.send("police", "PoliceAlert", alert)
        return alert

    @app.post("/police/{incident_id}/close")
    async def police_close(incident_id: str):
        _alert(incident_id)
        alert = board.close(incident_id)
        await hub.send("police", "PoliceAlert", alert)
        return alert

    @app.post("/benchmark/run")
    def benchmark():
        """8 + 2 scenarios, AI vs fair naive baseline, on the loaded hospitals/ambulances."""
        if svc.hospitals is None or not svc.ambulances:
            raise HTTPException(503, "benchmark needs data/hospitals.csv and data/ambulance_bases.csv")
        from backend.benchmarks import benchmark as bm
        try:
            results = bm.run(bm.load_scenarios(), svc.hospitals, dict(svc.ambulances))
        except (OSError, KeyError) as e:
            raise HTTPException(503, f"benchmark unavailable: {e}")
        return {"summary": bm.summarize(results).to_dict(orient="records"),
                "results": results.to_dict(orient="records"),
                "note": "Hospital beds/blood are SIMULATED; straight-line ETAs; same inputs for both "
                        "methods (fair baseline, docs/15 D15). Decision logic only, not patient outcomes."}

    @app.websocket("/ws/police")
    async def ws_police(ws: WebSocket):
        await hub.serve("police", ws)

    @app.websocket("/ws/map")
    async def ws_map(ws: WebSocket):
        await hub.serve("map", ws)

    @app.websocket("/ws/hospital/{hospital_id}")
    async def ws_hospital(ws: WebSocket, hospital_id: str):
        await hub.serve(f"hospital/{hospital_id}", ws)

    ui = Path(__file__).resolve().parents[1] / "frontend"
    if ui.is_dir():
        app.mount("/ui", StaticFiles(directory=ui, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        def root():
            return RedirectResponse("/ui/")

    return app


def __getattr__(name: str):
    # `uvicorn backend.main:app` builds the real app lazily, so importing this module (tests)
    # does not load models.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
