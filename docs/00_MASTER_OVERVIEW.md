# AI-Assisted Emergency Response & Coordination System — Master Project Document

**Status:** Planning/build-in-progress documentation. Distribute alongside the 4
component-specific `.md` files (`01_reporting_agent.md` through
`04_hospital_agent.md`) and `05_coordinator_fusion_evaluation.md`.

**Audience:** HOD, project teammates, and any coding agent (Claude Code, Cursor,
etc.) picking up implementation work. This document is the single source of truth
for what the project is, why it's designed this way, and how the pieces fit
together. Component files go deep on each piece; read this one first for context.

---

## 1. Project Title

**Pre-Arrival Multimodal AI Triage & Emergency Coordination System for Bengaluru**

Working alternate title used in earlier reviews: "AI-Assisted Emergency Response &
Smart Traffic Clearance System."

## 2. Problem Statement

Emergency response in Indian metro cities loses critical time at four specific
points:

1. Reporting is slow, unclear, and language-constrained (bystanders may not speak
   English, panic affects the quality of a verbal report).
2. The hospital receives an advance alert (108 already does this), but the alert
   carries no live physiological picture, no deterioration trend and no AI risk
   score while the ambulance is in transit.
3. Traffic is not cleared proactively for the ambulance's route — signals run on
   fixed timers regardless of an approaching emergency vehicle.
4. Hospital selection is typically driven by distance alone, not by trauma
   capability, real bed availability, or blood-type stock.

## 3. Real-World Motivation (cite this in the report's Introduction)

As publicly reported (ETV Bharat, 25 May 2026), Karnataka's **108 Arogya Kavacha**
command centre already performs GPS ambulance tracking, nearest-ambulance
identification, ETA computation, **advance alerts to hospitals**, and **continuous
vitals recording** inside the ambulance via MDT tablets. The same reporting says
the government **plans to introduce** connected ambulances with IoT biotelemetry
that transmit real-time HR/BP/SpO2/ECG to hospitals during transit — described as
a future plan, not a deployed feature.

**This project targets the intelligence layer on top of that planned stream:** a
live vitals-trend risk score fused with the bystander report, corridor signal
clearance, and capability-aware hospital ranking. It does not duplicate the
state's existing 108 infrastructure.

Source: ETV Bharat, "Karnataka Government Launches Centralised 108 Emergency
System In State", 25 May 2026 (see `13` §10 for URL). Say "as publicly reported".

## 4. Positioning on Novelty (say this exact framing in the viva — do not overclaim)

Multimodal fusion of vitals + text for clinical triage is an **established,
active 2024-2026 research area** — not invented here. Precedents to cite directly
in the report's Related Work section:

- **DeepTriager** — BiLSTM + self-attention text encoder fused with structured
  vitals for ED acuity prediction; concatenation and bilinear fusion variants
  reported AUC 0.956-0.959 (secondary citation, `[TODO-VERIFY]` in the original
  paper before putting on a slide).
- **VitalML** — predicts clinical decompensation (tachycardia, hypoxia,
  hypotension) up to 90 minutes ahead from continuous monitoring.
- 2026 multimodal attention triage work (arXiv 2607.16662: TabNet for vitals +
  self-attention fusion with text `[TODO-VERIFY]`).
- **DeepTriage-CN** (Scientific Reports, 2026) — text added little over a strong
  tabular vitals model except when vitals were degraded; supports our
  modality-dropout design.
- Prehospital AI narrative review (Cureus, 2025) — prehospital ML exists, so our
  claim is integration and context, not "first prehospital ML".
- Pediatric modality-dropout paper (2026) — addresses models over-relying on one
  modality; relevant if our fusion model shows the same failure mode.
- **SIGMA** (2026, arXiv preprint) — RL-based emergency vehicle signal
  prioritization, evaluated in SUMO on four Kolkata intersection models
  calibrated to municipal counts.
- **EMVLight** — multi-agent RL for joint emergency vehicle routing + signal
  control.

**Our actual contribution is context, not architecture:**
1. Pre-arrival (ambulance-to-hospital), not in-hospital EHR-scale triage.
2. Multilingual voice intake (Kannada/Hindi/English), not clean English clinical
   text.
3. Low-resource/connectivity-constrained deployment target, not hospital-grade
   compute.
4. RAG-grounded first-aid **action guidance**, not prediction-only.
5. RL-based signal control applied to a real, named Bengaluru corridor.

## 5. System Architecture — 4 Components + Coordinator

```
                    ┌─────────────────────┐
   Voice/Text   ──► │  1. REPORTING AGENT  │──┐
   Report            └─────────────────────┘  │
                                                │   Text severity
                    ┌─────────────────────┐    │
   Ambulance    ──► │ 2. VITALS/TRIAGE AGENT│──┼──► Fused Emergency Risk Score
   Vitals Stream     └─────────────────────┘    │      (multimodal fusion)
                                                │
                    ┌─────────────────────┐    │
   Traffic Sensors─►│ 3. TRAFFIC/POLICE    │────┤
   (simulated)       │    AGENT (RL)        │    │
                    └─────────────────────┘    │
                                                │
                    ┌─────────────────────┐    │
   Hospital Data ──►│  4. HOSPITAL AGENT   │◄───┘   (receives fused risk score)
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │     COORDINATOR      │──► Final dispatch decision
                    │ (conflict resolution)│     (map, alerts, hospital pick)
                    └─────────────────────┘
```

Each component is documented in its own file (see filenames in section 8). This
document covers only the cross-cutting concerns: architecture, novelty,
evaluation strategy, AIML-concept mapping, timeline, and honest current status.

## 6. Current Status — MVP/Prototype (say this honestly to HOD)

| Component | Current State | ML Type |
|---|---|---|
| Reporting | Keyword/regex-based report parser | Not ML |
| Vitals/Triage | **Does not exist yet** — this is new work | N/A currently |
| Text severity classifier | Working, TF-IDF + LogisticRegression | Real ML, synthetic training data |
| Traffic delay/congestion | Working, RandomForest | Real ML, synthetic training data |
| Traffic signal control (RL) | **Does not exist yet** — this is new work | N/A currently |
| Hospital scoring | Working, RandomForest Q-value model | Real ML, synthetic training data |
| ETA prediction | Working, GradientBoostingRegressor | Real ML, synthetic training data |
| Coordinator | Working, hand-tuned weighted bidding | Rule-based, not ML |
| Blood-bank integration | Does not exist yet | N/A |
| Benchmark vs. naive dispatch | Working — 8 Bengaluru scenarios | Quantitative comparison script |

**Bottom line for HOD/team:** the current MVP is a well-engineered rule-based
simulation with 4 real-but-synthetically-trained ML models. The final project
closes this gap by (a) replacing synthetic data with real datasets everywhere,
and (b) adding two genuinely new ML components that don't exist in the MVP at
all — the vitals/fusion triage model and the RL traffic controller.

## 7. AIML Core Concepts — Explicit Mapping (for HOD / viva defense)

| Concept | Where |
|---|---|
| Supervised learning (classification/regression) | Severity classifier, traffic models, hospital scoring |
| Sequential deep learning (BiLSTM/GRU) | Vitals time-series encoder |
| Multimodal fusion | Text severity embedding + vitals encoder → joint Emergency Risk Score |
| NLP — ASR | Whisper (multilingual voice intake) |
| NLP — LLM dialogue / prompt engineering | Structured conversational reporting |
| Retrieval-Augmented Generation (RAG) via LangChain | First-aid guidance grounded in curated protocol documents; LCEL retrieval chain, groundedness verification |
| Reinforcement Learning | Traffic signal control for ambulance green-corridor priority |
| Explainable AI (XAI) | SHAP TreeExplainer on tree/tabular models; Integrated Gradients (Captum) on the BiGRU |
| Evaluation & training methodology | 3-way baselines, ablation studies, k-fold cross-validation, hyperparameter search, RL training curves + checkpointing — see `06_ml_engineering_practices.md` |

**Explicitly rule-based, not ML (be upfront about this, do not disguise it):**
Coordinator conflict-resolution weights, ambulance/police scoring formulas, route
selection (OpenRouteService API reuse), blood-stock lookup.

## 8. Component Documentation Files

| File | Covers |
|---|---|
| `01_reporting_agent.md` | Voice intake, conversational AI, RAG first-aid guidance |
| `02_ambulance_vitals_agent.md` | Core ML — vitals sequential model, fusion, NEWS2 benchmark |
| `03_traffic_police_agent.md` | RL traffic signal control, SUMO corridor, police alerting |
| `04_hospital_agent.md` | Hospital scoring, blood-bank integration |
| `05_coordinator_fusion_evaluation.md` | Fused risk score, coordinator logic, full evaluation plan |
| `06_ml_engineering_practices.md` | Training methodology, hyperparameter tuning, cross-validation, RL training discipline, reproducibility — applies across all components |
| `07_interfaces_and_schemas.md` | Message schemas, API, tensor contracts |
| `08_data_and_labels.md` | Datasets, label definitions, synthetic-data rules |
| `09_setup_and_runbook.md` | Repo layout, environment, commands, RL spec, demo flow |
| `10_risks_ethics_scope.md` | Priority tiers, risks, ethics |
| `12_required_edits_to_existing_files.md` | Corrections to 00–06 |
| `15_DECISIONS_FINAL.md` | **Final decisions; highest precedence** |
| `11`, `13`, `14`, `16`, `17`, `18` | Presentation, novelty brief, viva Q&A, scripts (not build specs) |
| `../AGENTS.md`, `../CLAUDE.md` | Rules for coding agents |

## 9. Tech Stack Summary

- **Backend:** Python, FastAPI, WebSockets
- **ML/DL:** scikit-learn, PyTorch (sequential/fusion models), SHAP
- **NLP:** Whisper (ASR), LLM API (dialogue), LangChain (RAG pipeline —
  document loading/chunking, retrieval, LCEL grounded-generation chain),
  sentence-transformers embeddings + FAISS/Chroma vector store
- **RL:** SUMO (traffic simulator), `sumo-rl`, `stable-baselines3` (DQN/PPO)
- **Frontend:** Leaflet/OpenStreetMap, HTML/JS
- **Routing:** OpenRouteService API
- **Data:** MIMIC-III Demo / Kaggle Human Vital Signs Dataset (vitals), real
  Bengaluru traffic/accident open data, eRaktKosh-style blood-stock data

## 10. Datasets — Consolidated List

| Purpose | Dataset | Access |
|---|---|---|
| Vitals/triage | MIMIC-III Clinical Database Demo | physionet.org/content/mimiciii-demo/1.4/ — no credentialing needed |
| Vitals/triage (alternative/supplement) | Kaggle "Human Vital Signs Dataset" | kaggle.com/datasets/nasirayub2/human-vital-sign-dataset |
| Traffic/congestion | Real Bengaluru traffic/accident open data | BBMP/Karnataka open data portals, traffic police reports |
| Blood stock | eRaktKosh-style data | Live if accessible, else realistic simulated dataset |
| First-aid protocols (RAG corpus) | WHO/Red Cross first-aid guidelines, standard trauma triage protocols | Curated/compiled manually |
| RL traffic network | Real Bengaluru ORR corridor geometry (approximated) | Modeled in SUMO using real inter-junction distances (see `03_traffic_police_agent.md`) |

## 11. 2-Month Build Timeline (reference — see individual files for detail)

| Weeks | Focus |
|---|---|
| 1-2 | Data sourcing across all components |
| 3-5 | Core ML build — vitals sequential model + NEWS2 benchmark, RL corridor training, retrain existing models on real data |
| 6-7 | Integration — fused risk score, blood-bank scoring, RAG pipeline, voice intake |
| 8 | Polish — SHAP, benchmark visuals, deployment, IEEE paper |

## 12. What Is Explicitly Out of Scope / Future Work Only

- Full RL/MARL-based coordinator (kept rule-based for this submission)
- Image-based human injury classification (dropped — accuracy ceiling ~68-85% in
  literature even on clean data, real ethical concerns, no viable dataset)
- Transformer/attention-based fusion (concatenation fusion is proven sufficient
  per DeepTriager precedent and is the achievable choice for this timeline)
- Full LangGraph agentic rebuild of the coordinator (optional polish only)
- Edge/offline model quantization (mention as future work unless time allows)

## 13. Citations to Use in the Report

- Karnataka 108 Arogya Kavacha — ETV Bharat (2026)
- DeepTriager — multimodal ED acuity prediction
- VitalML — decompensation prediction from continuous monitoring
- SIGMA — RL emergency vehicle signal prioritization, Kolkata intersections
- EMVLight — multi-agent RL emergency routing + signal control
- NEWS2 / MEWS — Royal College of Physicians early warning score standard
- eRaktKosh — national blood-stock tracking system (Ministry of Health, India)
