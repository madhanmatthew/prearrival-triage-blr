# 07 — Interfaces, Schemas, and Tensor Contracts

**Purpose:** the single contract between components. Resolves the "TBD" interface between
Component 1 (reporting) and Component 2 (vitals/fusion). Items marked `[DECISION]` are
fixed here; change them only by editing this file first.

---

## 1. Resolved decisions

| # | Decision |
|---|---|
| D1 `[DECISION]` | Component 1 → fusion interface is a **fixed-length structured vector (19 dims)**, defined in §3, produced by the dialogue state machine plus the text severity classifier. Raw text is *not* fed to the fusion model. |
| D2 `[DECISION]` | Severity classes are 4: `0 stable, 1 moderate, 2 severe, 3 critical`, everywhere in the system. |
| D3 `[DECISION]` | Fusion handles a missing modality via **modality masks + modality dropout training**. Text-only or vitals-only inference is a supported mode (ambulance vitals not yet available at report time). |
| D4 `[DECISION]` | All timestamps are ISO-8601 UTC strings; all coordinates `{lat, lon}` in WGS84. |
| D5 `[DECISION]` | Backend is FastAPI; agents are Python modules called in-process by the coordinator. Only the frontend/police dashboard use HTTP/WebSocket. |

## 2. Core message schemas (JSON)

### 2.1 IncidentReport (Component 1 → Coordinator)
```json
{
  "incident_id": "INC-20260928-0001",
  "timestamp": "2026-09-28T09:15:00Z",
  "gps": {"lat": 12.9166, "lon": 77.6231},
  "language": "kn",
  "transcript": "string (raw or ASR output)",
  "dialogue": [
    {"q_id": "Q_CONSCIOUS", "question": "string", "answer_raw": "string", "answer_value": "yes|no|unknown"}
  ],
  "text_features": [0.0],
  "text_severity_probs": [0.1, 0.3, 0.4, 0.2],
  "injury_type": "road_accident",
  "num_injured": 2
}
```
`text_features` = the 19-dim vector from §3. `language` ∈ `en|hi|kn`.

### 2.2 VitalsReading (Ambulance → Component 2)
```json
{
  "incident_id": "INC-20260928-0001",
  "patient_id": "P-1",
  "timestamp": "2026-09-28T09:24:00Z",
  "hr": 112, "rr": 24, "spo2": 93, "sbp": 98, "temp": 36.4,
  "avpu": "A",
  "supplemental_o2": false
}
```
Allowed `avpu`: `A|V|P|U` (V/P/U all score 3 in NEWS2; the model encodes `A=0`, else `1`). Any field may be `null` (sensor dropout) → handled per §4.3.

### 2.3 FusedRisk (Component 2/5 → Hospital Agent, Coordinator)
```json
{
  "incident_id": "INC-20260928-0001",
  "patient_id": "P-1",
  "timestamp": "2026-09-28T09:24:05Z",
  "severity_probs": [0.02, 0.18, 0.55, 0.25],
  "severity_class": 2,
  "risk_score": 0.68,
  "trend_prob": 0.41,
  "trend_horizon_readings": 6,
  "modalities_used": ["text", "vitals"],
  "news2_score": 7,
  "model_version": "fusion_v1_20260915"
}
```
`risk_score = Σ p_i · (i/3)` (expected severity normalised to 0–1). Updated on every new
`VitalsReading` (live-update requirement).

### 2.4 HospitalRecommendation (Hospital Agent → Coordinator)
```json
{
  "incident_id": "INC-20260928-0001",
  "ranked": [
    {
      "hospital_id": "H-07", "name": "string", "total_score": 0.81, "eta_min": 14.2,
      "breakdown": {"proximity": 0.7, "trauma_match": 1.0, "beds": 0.8, "specialty": 1.0, "blood_match": 1.0, "rf_qvalue": 0.74},
      "blood_flag": "matched|unknown|short"
    }
  ]
}
```

### 2.5 SignalState / CorridorStatus (Traffic Agent → Coordinator)
```json
{
  "incident_id": "INC-20260928-0001",
  "route_id": "ORR-SB-KR",
  "junctions": [
    {"id": "silkboard", "phase": "main_green", "ambulance_eta_s": 210, "preempted": true}
  ],
  "conflicts": [{"type": "cordon_on_route", "detail": "string"}],
  "estimated_transit_s": 1180
}
```

### 2.6 PoliceAlert (Coordinator → Police dashboard)
```json
{
  "incident_id": "INC-20260928-0001",
  "gps": {"lat": 12.9166, "lon": 77.6231},
  "severity_class": 2,
  "route_polyline": [[12.91, 77.62]],
  "units_required": 3,
  "perimeter_m": 150
}
```

### 2.7 DispatchDecision (Coordinator → all clients)
```json
{
  "incident_id": "INC-20260928-0001",
  "ambulance_id": "AMB-12",
  "hospital_id": "H-07",
  "route_id": "ORR-SB-KR",
  "police_units": 3,
  "conflicts_resolved": [{"type": "hospital_bed_shortfall", "resolution": "switched to rank-2 hospital"}],
  "explanation": "string (human-readable, includes top reasons)",
  "timestamp": "2026-09-28T09:24:10Z"
}
```

## 3. Text feature vector (19 dims) — Component 1 → fusion

Produced by the dialogue state machine. Unknown answers → value `0`, known-flag `0`.

| Idx | Name | Values |
|---|---|---|
| 0 | `conscious` | 1 = conscious, 0 = not/unknown |
| 1 | `conscious_known` | 0/1 |
| 2 | `breathing_difficulty` | 0/1 |
| 3 | `breathing_known` | 0/1 |
| 4 | `bleeding_level` | 0 none, 0.5 minor, 1.0 heavy |
| 5 | `bleeding_known` | 0/1 |
| 6 | `can_speak` | 0/1 |
| 7 | `can_speak_known` | 0/1 |
| 8 | `num_injured_norm` | min(n,5)/5 |
| 9 | `high_energy_mechanism` | 0/1 (road accident at speed, fall from height, etc.) |
| 10–14 | `injury_type` one-hot | road_accident, fall, burn, cardiac_medical, other |
| 15–18 | `text_severity_probs` | 4-class probabilities from the text severity classifier (TF-IDF+LR on transcript + dialogue answers) |

Bounded question set (state machine): `Q_CONSCIOUS, Q_BREATHING, Q_BLEEDING, Q_SPEAK, Q_NUM_INJURED, Q_MECHANISM`. Max 6 questions; one asked per turn; skip a question if already answered in the initial report.

## 4. Model tensor contracts

### 4.1 Vitals sequence encoder input
- `x`: `float32 [B, T, F]` with **T = 6** readings, **F = 12** features.
- `mask`: `bool [B, T]` (True = real reading; sequences shorter than 6 are left-padded).
- Feature order: `hr, rr, spo2, sbp, temp, avpu_bin, supplemental_o2, hr_delta, rr_delta, spo2_delta, sbp_delta, temp_delta`.
- `*_delta` = current − previous reading (0 at first step).

### 4.2 Architecture `[DECISION]`
```
x[B,T,12] → BiGRU(hidden=64, layers=1) → last valid step → h_vit [B,128]
text_features[B,19] → Linear(19,32)+ReLU → h_txt [B,32]
concat(h_vit·m_vit, h_txt·m_txt) [B,160] → Linear(160,64)+ReLU+Dropout(0.3) → z [B,64]
severity head: Linear(64,4) → logits [B,4]
trend head:    Linear(64,1) → logit  [B,1]
```
`m_vit`, `m_txt` ∈ {0,1} are modality masks (zero the missing modality's branch).
Standalone variants for the ablation: vitals-only (drop text branch), text-only (drop vitals branch).

### 4.3 Missing values
- Forward-fill a missing reading for at most 2 steps; beyond that set to train-median and add nothing else (mask covers whole missing timestep only if *all* fields null).
- Normalisation: z-score with mean/std computed on the **training fold only**; saved with the model artifact.

### 4.4 Loss and training
- `loss = CE(severity, class_weights) + 0.5 · BCE(trend)` `[DECISION: λ=0.5, tune on validation]`.
- Modality dropout during training: zero text branch w.p. 0.2 and vitals branch w.p. 0.1 (never both).
- Optimiser Adam, lr 1e-3, early stopping on validation macro-F1, patience 10.

## 5. API endpoints (FastAPI) `[DECISION]`

| Method | Path | Purpose |
|---|---|---|
| POST | `/incident/voice` | multipart audio + gps → transcribe → start dialogue → returns first question |
| POST | `/incident/text` | text + gps → start dialogue |
| POST | `/incident/{id}/answer` | answer to current question → next question or `done` with `IncidentReport` |
| GET | `/incident/{id}/guidance` | RAG first-aid guidance + source chunks + groundedness flag |
| POST | `/incident/{id}/vitals` | push a `VitalsReading` → returns updated `FusedRisk` |
| POST | `/incident/{id}/dispatch` | run coordinator → `DispatchDecision` |
| GET | `/incident/{id}` | full incident state |
| POST | `/benchmark/run` | run the 8-scenario benchmark, return table |

### WebSocket channels
| Channel | Direction | Payload |
|---|---|---|
| `/ws/police` | server → dashboard | `PoliceAlert`, updates |
| `/ws/hospital/{hospital_id}` | server → hospital view | `FusedRisk` stream + ETA + prep advice |
| `/ws/map` | server → frontend | ambulance position, route, `SignalState` |

Every WS message: `{"type": "<PoliceAlert|FusedRisk|SignalState|DispatchDecision>", "data": {...}}`.

## 6. Errors and degraded modes

| Situation | Behaviour |
|---|---|
| ASR confidence low / empty transcript | Fall back to text input prompt; never guess |
| No vitals yet | Fusion runs with `m_vit=0`, `modalities_used=["text"]` |
| Vitals stream drops | Keep last `FusedRisk`, mark `stale=true` after 3 missed intervals |
| LLM API unreachable | Dialogue uses template phrasing for questions; guidance returns "Wait for professional help; call 108" |
| RAG has no relevant chunk (similarity below threshold) | Return "not covered", no improvised advice |
| ORS API down | Coordinator uses cached/straight-line ETA and flags `route_estimated=true` |
