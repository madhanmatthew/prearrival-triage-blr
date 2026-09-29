# 10 — Risks, Fallbacks, Ethics, Priorities, Roles

---

## 1. Priority tiers (build in this order; stop when time runs out)

| Tier | Deliverable | "Done" means |
|---|---|---|
| 1 | Vitals pipeline: MIMIC windows + labels, NEWS2, LR/RF/GB baselines, BiGRU two-head, patient-grouped CV | Metrics table (5 baselines) + confusion matrices + AUC trend + model card |
| 2 | RL corridor: SUMO net + routes, fixed-time baseline, DQN, ≥3 seeds, curves | Transit-time comparison with general-delay reported honestly |
| 3 | RAG first-aid: corpus, FAISS, LCEL chain, groundedness eval | Groundedness % + retrieval hit@k on eval set |
| 4 | Fusion + coordinator glue | Ablation (text/vitals/fused) with synthetic-pairing caveat + live-update demo |
| 5 | Voice (Whisper), multilingual dialogue polish, blood factor, real-data retrain of traffic/hospital RFs, UI | Demo runs end to end |

## 2. Risk register and fallbacks

| Risk | Likelihood | Impact | Fallback |
|---|---|---|---|
| MIMIC demo too small → unstable BiGRU | High | High | Report CV mean±std; use smaller model (hidden 32); tabular models as primary result; state limitation |
| Class imbalance makes 4-class labels unusable | Medium | High | Merge to 3-class or binary (critical vs not); update `07` D2 |
| MIMIC item-ID mapping errors (CareVue vs MetaVision) | Medium | Medium | Restrict to one system; validate ranges (SpO2 0–100, HR 20–250) |
| Trend label too rare | Medium | Medium | Lengthen horizon H to 12 readings; report AUC-PR too |
| Fusion shows no gain over vitals-only | Medium | Low (if reported honestly) | Report as finding; discuss modality dominance / dropout literature; show sensitivity table |
| RL does not converge or beats baseline only marginally | Medium | Medium | Show training curves, seeds, reward-weight sweep; add compressed-spacing variant; report honestly |
| SUMO/sumo-rl install or version issues | Medium | High | Pin versions early; use `eclipse-sumo` pip package; keep a Dockerfile |
| LLM API latency/outage or cost | Medium | Medium | Template question phrasing; cache; use local small model as backup |
| Whisper poor on Kannada | Medium | Medium | Use `small`; measure WER; allow text fallback; state limits |
| RAG hallucination/ungrounded | Medium | High | Similarity threshold → "not covered"; NLI/LLM-judge groundedness; human-check a sample |
| Real traffic/hospital data not obtainable | High | Low | Use SIM data and label it; document search attempts |
| Citations wrong/unverifiable | Medium | High | Verify each against primary source before report; remove any that cannot be verified |
| Scope too large for 2 months | High | High | Follow priority tiers; tier 5 is optional |

## 3. Ethics, safety, legal `[required section for the report]`

- **Not a medical device.** Research prototype. Not clinically validated; must not be used for real patient care or dispatch. Show this disclaimer in the UI, README, and report.
- **First-aid guidance risk.** Restricted to retrieved corpus text from public guideline documents; always includes "call 108 / follow professional instructions"; logs source chunk; refuses when not covered.
- **Data and privacy.** Voice recordings and health data are personal data under India's Digital Personal Data Protection Act, 2023 `[TODO-VERIFY current rules]`. For the prototype: no real patient data collection; team-recorded consenting volunteers only; delete audio after transcription unless needed for WER evaluation; no real names or phone numbers in logs.
- **MIMIC data use.** Follow PhysioNet data-use terms; do not redistribute; cite properly.
- **Bias/fairness.** MIMIC skews toward a US ICU population; not representative of Indian pre-hospital patients or age/sex subgroups. State in model card; do not claim generalisation.
- **Language equity.** Kannada/Hindi ASR and dialogue may perform worse than English; report per-language metrics.
- **Automation risk.** System is decision *support*; a human dispatcher/paramedic remains responsible. Coordinator outputs include an explanation string.
- **Claims discipline.** No "first-ever", "clinically proven", "reduces mortality". Speak only in measured simulation/benchmark terms.

## 4. Verification tasks before submission

- [ ] Verify every citation (title, authors, venue, year, numbers): DeepTriager, VitalML, SIGMA, EMVLight, NEWS2, eRaktKosh, and the Karnataka 108 Arogya Kavacha report (main motivation claim).
- [ ] Confirm MIMIC-III Demo access terms and table availability.
- [ ] Confirm current eRaktKosh access options.
- [ ] Reconcile `07`/`08` formulas with the existing code (`hospital_agent.py`, `coordinator.py`).

## 5. Team roles (fill in names)

| Area | Owner | Backup |
|---|---|---|
| Component 1: ASR, dialogue, RAG | ____ | ____ |
| Component 2: vitals + sequence model | ____ | ____ |
| Component 3: SUMO + RL + police | ____ | ____ |
| Component 4: hospital + blood factor | ____ | ____ |
| Fusion, coordinator, benchmark | ____ | ____ |
| Frontend + demo | ____ | ____ |
| Report/paper + slides + citations | ____ | ____ |

## 6. Timeline mapped to tiers (8 weeks)

| Weeks | Work |
|---|---|
| 1–2 | Data: MIMIC windows/labels, corpus, SUMO net, text set, hospital table; environment pinned |
| 3–4 | Tier 1 (baselines + BiGRU), RL baseline + first DQN runs |
| 5 | RL seeds/eval, RAG build + eval, text classifier |
| 6 | Fusion + ablation, coordinator glue, blood factor |
| 7 | Whisper + dialogue integration, frontend, benchmark re-run |
| 8 | XAI figures, model card, report/paper, slides, demo rehearsal |
