"""API tests with fake/fixture services (no model downloads, no network)."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from backend.agents import hospital_agent as ha
from backend.asr.transcribe import TranscriptResult
from backend.main import create_app
from backend.ml.fusion_model import FusionNet
from backend.ml.seq_model import Preprocessor
from backend.rag.pipeline import LLM_DOWN_MSG, Guidance
from backend.routing import StraightLineRouter
from backend.services import Services

GPS_ = {"lat": 12.9166, "lon": 77.6231}
ANSWERS = {"Q_CONSCIOUS": "no", "Q_BREATHING": "no", "Q_BLEEDING": "yes, a lot",
           "Q_SPEAK": "no", "Q_NUM_INJURED": "1", "Q_MECHANISM": "bike accident"}
READING = dict(patient_id="p", timestamp="2026-10-06T10:01:00Z", hr=120, rr=26, spo2=91, sbp=95,
               temp=37.0, avpu="V", supplemental_o2=False)


def hospitals():
    rows = []
    for hid, lat, lon, beds in [("H1", 12.92, 77.62, 5), ("H2", 12.95, 77.70, 8)]:
        rows.append({"hospital_id": hid, "name": f"Hosp {hid}", "lat": lat, "lon": lon,
                     "trauma_level": 3, "beds_available": beds, "specialties": "trauma",
                     **{c: 10 for c in ha.BLOOD_COLS}})
    return pd.DataFrame(rows)


def tiny_fusion():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.normal(80, 10, (20, 6, 7)).astype(np.float32)
    mask = np.ones((20, 6), bool)
    model = FusionNet("fused")
    model.eval()
    return {"model": model, "preprocessor": Preprocessor().fit(X, mask), "lambda": 0.5,
            "epochs": 1, "variant": "fused"}


class FakeRAG:
    def __init__(self):
        self.calls = []

    def guide_report(self, report):
        self.calls.append("report")
        return [("cpr", Guidance(query="q", status="ok", mode="extractive", message="Push hard."))]

    def guide(self, query, topic_file=None):
        self.calls.append("text")
        return Guidance(query=query, status="ok", mode="extractive", message="Press the wound.")


class FakeTranscriber:
    def __init__(self, text="bike accident, he is bleeding", lang="en"):
        self.text, self.lang = text, lang

    def transcribe(self, path, language=None):
        return TranscriptResult(self.text, language or self.lang, 0.99, 3.0)


def services(dispatch=False, **kw):
    s = Services(notes=["test services"], **kw)
    if dispatch:
        s.hospitals = hospitals()
        s.hospital_agent = ha.HospitalAgent(s.hospitals)
        s.beds_now = {"H1": 5, "H2": 8}
        s.ambulances = {"A1": (12.917, 77.624), "A2": (12.99, 77.75)}
        s.router_factory = lambda inc, hosp: StraightLineRouter(inc, hosp)
    return s


def client(svc=None):
    return TestClient(create_app(svc or services(), clock=lambda: "2026-10-06T10:00:00Z"))


def finish(c, text="bike accident on ORR, he is bleeding"):
    r = c.post("/incident/text", json={"text": text, "gps": GPS_}).json()
    iid = r["incident_id"]
    while not r["done"]:
        r = c.post(f"/incident/{iid}/answer", json={"answer": ANSWERS[r["question"]["q_id"]]}).json()
    return iid, r


def test_health_lists_degraded_modes():
    h = client().get("/health").json()
    assert h["degraded"] == ["test services"] and h["dispatch_ready"] is False


def test_text_intake_dialogue_to_report_and_text_only_risk():
    c = client()
    r = c.post("/incident/text", json={"text": "bike accident, he is bleeding", "gps": GPS_}).json()
    assert r["incident_id"] == "INC-20261006-0001" and not r["done"] and r["question"]["q_id"]
    iid, r = finish(c)
    assert r["done"] and r["report"]["gps"] == GPS_ and len(r["report"]["text_features"]) == 19
    inc = c.get(f"/incident/{iid}").json()
    risk = inc["risk"]
    assert risk["modalities_used"] == ["text"] and risk["trend_prob"] is None
    assert "placeholder" in risk["model_version"]
    assert c.post(f"/incident/{iid}/answer", json={"answer": "yes"}).status_code == 409


def test_errors():
    c = client()
    assert c.get("/incident/NOPE").status_code == 404
    assert c.post("/incident/text", json={"text": "  ", "gps": GPS_}).status_code == 422
    assert c.post("/incident/text", json={"text": "x", "gps": {"lat": 99, "lon": 0}}).status_code == 422
    assert c.post("/benchmark/run").status_code == 503          # no hospitals in test services


def test_guidance_fallback_and_routed():
    c = client()
    iid, _ = finish(c)
    g = c.get(f"/incident/{iid}/guidance").json()
    assert g["status"] == "unavailable" and g["message"] == LLM_DOWN_MSG
    rag = FakeRAG()
    c = client(services(rag=rag))
    r = c.post("/incident/text", json={"text": "bike accident, he is bleeding", "gps": GPS_}).json()
    g = c.get(f"/incident/{r['incident_id']}/guidance").json()   # unfinished -> free text
    assert g["items"][0]["problem"] == "report" and rag.calls == ["text"]
    iid, _ = finish(c)
    g = c.get(f"/incident/{iid}/guidance").json()
    assert g["items"][0]["problem"] == "cpr" and rag.calls[-1] == "report"


def test_vitals_with_fusion_uses_both_modalities():
    c = client(services(fusion=tiny_fusion(), fusion_version="fusion_test"))
    iid, _ = finish(c)
    before = c.get(f"/incident/{iid}").json()["risk"]
    assert before["modalities_used"] == ["text"] and before["trend_prob"] is None
    r = c.post(f"/incident/{iid}/vitals", json={"incident_id": iid, **READING})
    assert r.status_code == 200
    risk = r.json()
    assert risk["modalities_used"] == ["text", "vitals"] and 0 <= risk["trend_prob"] <= 1
    assert risk["news2_score"] is not None and "SYNTHETIC" in risk["model_version"]
    assert abs(sum(risk["severity_probs"]) - 1) < 1e-6


def test_vitals_rules():
    c = client()
    r = c.post("/incident/text", json={"text": "bike accident, he is bleeding", "gps": GPS_}).json()
    iid = r["incident_id"]
    assert c.post(f"/incident/{iid}/vitals", json={"incident_id": iid, **READING}).status_code == 409
    iid, _ = finish(c)
    assert c.post(f"/incident/{iid}/vitals", json={"incident_id": "OTHER", **READING}).status_code == 422
    risk = c.post(f"/incident/{iid}/vitals", json={"incident_id": iid, **READING}).json()
    assert risk["modalities_used"] == ["text"]        # no fusion model -> text severity only


def test_window_left_padded_and_masked():
    from backend.main import _window
    from backend.schemas import VitalsReading
    rs = [VitalsReading(incident_id="i", **READING), VitalsReading(incident_id="i", patient_id="p",
                                                                    timestamp="t")]
    X, mask = _window(rs)
    assert mask.tolist() == [[False, False, False, False, True, False]]
    assert X[0, 4, 0] == 120 and X[0, 4, 5] == 1.0 and np.isnan(X[0, 5]).all()


def test_dispatch_unavailable_without_data():
    c = client()
    iid, _ = finish(c)
    r = c.post(f"/incident/{iid}/dispatch")
    assert r.status_code == 503


def test_dispatch_end_to_end_and_ws():
    c = client(services(dispatch=True))
    iid, _ = finish(c)
    with c.websocket_connect("/ws/map") as ws:
        r = c.post(f"/incident/{iid}/dispatch")
        assert r.status_code == 200
        d = r.json()
        msg = ws.receive_json()
    assert msg["type"] == "DispatchDecision" and msg["data"]["incident_id"] == iid
    assert d["ambulance_id"] == "A1" and d["hospital_id"] in {"H1", "H2"}
    assert d["route_estimated"] is True and "Rule-based" in d["explanation"]
    assert c.post(f"/incident/{iid}/dispatch").json() == d          # idempotent
    iid2, _ = finish(c)
    assert c.post(f"/incident/{iid2}/dispatch").json()["ambulance_id"] == "A2"
    iid3, _ = finish(c)
    r3 = c.post(f"/incident/{iid3}/dispatch")
    assert r3.status_code == 409 and "no free ambulance" in r3.json()["detail"]


def test_hospital_ranking_endpoint():
    c = client(services(dispatch=True))
    iid, _ = finish(c)
    assert c.get(f"/incident/{iid}/hospitals").status_code == 409     # before dispatch
    d = c.post(f"/incident/{iid}/dispatch").json()
    r = c.get(f"/incident/{iid}/hospitals")
    assert r.status_code == 200
    rec = r.json()
    scores = [h["total_score"] for h in rec["ranked"]]
    assert rec["incident_id"] == iid and scores == sorted(scores, reverse=True)
    assert d["hospital_id"] in {h["hospital_id"] for h in rec["ranked"]}
    assert c.get("/incident/nope/hospitals").status_code == 404


def test_hospital_ws_gets_risk_updates():
    c = client(services(dispatch=True))
    iid, _ = finish(c)
    hid = c.post(f"/incident/{iid}/dispatch").json()["hospital_id"]
    with c.websocket_connect(f"/ws/hospital/{hid}") as ws:
        c.post(f"/incident/{iid}/vitals", json={"incident_id": iid, **READING})
        msg = ws.receive_json()
    assert msg["type"] == "FusedRisk" and msg["data"]["incident_id"] == iid


def test_voice_intake():
    c = client()
    files = {"audio": ("a.wav", b"RIFF0000", "audio/wav")}
    assert c.post("/incident/voice", files=files, data={"lat": 12.9, "lon": 77.6}).status_code == 422
    c = client(services(transcriber=FakeTranscriber()))
    r = c.post("/incident/voice", files=files, data={"lat": 12.9, "lon": 77.6}).json()
    assert r["transcript"].startswith("bike accident") and r["question"]
    c = client(services(transcriber=FakeTranscriber(text="")))
    r = c.post("/incident/voice", files=files, data={"lat": 12.9, "lon": 77.6})
    assert r.status_code == 422 and "type the report" in r.json()["detail"]


def test_police_alert_lifecycle_over_api():
    c = client(services(dispatch=True))
    iid, _ = finish(c)
    assert c.get(f"/police/{iid}").status_code == 404                 # not dispatched yet
    with c.websocket_connect("/ws/police") as ws:
        d = c.post(f"/incident/{iid}/dispatch").json()
        msg = ws.receive_json()
    assert msg["type"] == "PoliceAlert" and msg["data"]["incident_id"] == iid
    assert msg["data"]["units_required"] == d["police_units"]
    alert = c.get(f"/police/{iid}").json()
    assert alert["alert_status"] == "active" and len(alert["route_polyline"]) >= 2
    assert c.post(f"/police/{iid}/ack").json()["alert_status"] == "acknowledged"
    assert c.post(f"/police/{iid}/ack", json={"junction_id": "nowhere"}).status_code == 404
    end = alert["route_polyline"][-1]
    a = c.post(f"/incident/{iid}/position", json={"lat": end[0], "lon": end[1]}).json()
    assert all(p["status"] == "released" for p in a["junction_posts"])
    assert c.post(f"/police/{iid}/close").json()["alert_status"] == "closed"
    assert c.post(f"/police/{iid}/ack").status_code == 409


def test_incidents_resources_and_ui():
    c = client(services(dispatch=True))
    iid, _ = finish(c)
    lst = c.get("/incidents").json()
    assert lst[0]["incident_id"] == iid and lst[0]["severity_class"] is not None
    r = c.get("/resources").json()
    assert {h["hospital_id"] for h in r["hospitals"]} == {"H1", "H2"}
    assert {a["ambulance_id"] for a in r["ambulances"]} == {"A1", "A2"}
    assert {j["junction_id"] for j in r["junctions"]} >= {"silkboard", "krpuram"}
    ui = c.get("/ui/")
    assert ui.status_code == 200 and "Pre-arrival" in ui.text
    for page in ("report.html", "dispatch.html", "police.html", "hospital.html", "common.js"):
        assert c.get(f"/ui/{page}").status_code == 200
    assert c.get("/", follow_redirects=False).status_code in (302, 307)


def test_benchmark_endpoint():
    assert client().post("/benchmark/run").status_code == 503             # no hospitals loaded
    s = services(dispatch=True)
    s.hospitals = ha.load_hospitals("data/hospitals.csv")
    s.ambulances = {"A1": (12.93, 77.62), "A2": (13.03, 77.59)}
    body = client(s).post("/benchmark/run").json()
    assert {r["method"] for r in body["summary"]} == {"naive", "ai"}
    assert len(body["results"]) == 20 and "SIMULATED" in body["note"]


class FakeDeterioration:
    """Stands in for the HGB bundle: probability rises as SBP falls (test only)."""

    def predict_proba(self, T):
        sbp_last = np.nan_to_num(T[:, 3], nan=120.0)
        p = np.clip((130 - sbp_last) / 60, 0, 1)
        return np.column_stack([1 - p, p])


def test_vitals_endpoint_returns_10min_deterioration():
    c = client()
    iid, _ = finish(c)
    assert c.get(f"/incident/{iid}").json()["risk"]["deterioration_10min_prob"] is None
    r = c.post(f"/incident/{iid}/vitals", json={"incident_id": iid, **READING}).json()
    assert r["deterioration_10min_prob"] is None                          # model not loaded
    s = services()
    s.deterioration = {"model": FakeDeterioration()}
    s.deterioration_version = "vitaldb_hgb_test"
    c2 = client(s)
    iid2, _ = finish(c2)
    probs = []
    for sbp in (120, 105, 90):
        r = c2.post(f"/incident/{iid2}/vitals", json={"incident_id": iid2, **{**READING, "sbp": sbp}}).json()
        probs.append(r["deterioration_10min_prob"])
    assert probs == sorted(probs) and probs[-1] > probs[0]
    assert "intra-operative" in r["deterioration_model"]


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "data" / "prompts_audio").is_dir(),
                    reason="no prompt recordings")
def test_prompt_recordings_served():
    c = client()
    for key in ("q1_en", "q6_hi", "s3_kn"):
        assert c.get(f"/prompts/{key}.mp3").status_code == 200
    assert c.get("/prompts/nope.mp3").status_code == 404


def test_dialogue_finishes_with_a_real_text_model():
    """Regression 2026-10-23: a trained text model crashed _text_probs (1-D row indexed twice)."""
    import pandas as pd
    from backend.ml import text_severity as ts
    df = pd.DataFrame({"text": ["man unconscious not breathing", "minor scratch, fine", "deep cut bleeding a lot",
                                "twisted ankle, talking"] * 6, "label": [3, 0, 2, 1] * 6, "language": ["en"] * 24})
    model, _ = ts.train(df, c_grid=[1.0], n_splits=2)
    s = services()
    s.text_model = model
    c = client(s)
    _, r = finish(c)
    probs = r["report"]["text_severity_probs"]
    assert len(probs) == 4 and abs(sum(probs) - 1) < 1e-6
