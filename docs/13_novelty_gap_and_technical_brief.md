# 13 — Real-World Gap, Novelty Case, and Full Technical Brief

**Purpose:** one document to convince the HOD that this project (a) targets a documented
real-world gap and (b) has a defensible novelty, with all technical details.

**How claims are handled here:** every external claim below was checked against a source
(listed in §10). Claims I could not verify are marked `[VERIFY]`. The novelty case is
written to survive a viva: it is strong *because* it does not overclaim.

---

## 1. The pitch in one paragraph

Karnataka has just built a state-run 108 Arogya Kavacha command centre that tracks
ambulances, computes ETAs, records patient vitals in the ambulance and alerts hospitals in
advance. The government itself lists the next step as future work: connected ambulances that
stream real-time patient data (heart rate, BP, oxygen, ECG) to hospitals during transit.
Nobody has yet built the intelligence layer on top of that stream: a model that turns the
vitals trend into a live risk score, combined with the bystander report, the traffic on the
route, and the hospital's real capability. **This project prototypes that layer for
Bengaluru**, end to end, and evaluates it against clinical and operational baselines.

## 2. The real-world gap (with evidence)

### 2.1 What Karnataka has already built (do not claim to duplicate this)
Source: ETV Bharat, 25 May 2026 (also Devdiscourse / Medical Buyer coverage).
- Centralised Command and Control Centre for 108 Arogya Kavacha, integrated with 112, 104, Tele-MANAS, eSanjeevani and other helplines.
- GPS-enabled ambulance tracking, GIS-based hospital mapping, caller location detection, real-time dispatch.
- Mobile Data Terminals (tablets) in 108 ambulances in all districts; nearest-ambulance identification, ETA, SMS alerts, digital patient care records.
- Hospitals receive advance alerts to prepare for incoming cases.
- Mandatory digital workflow and **continuous recording of patient vitals**.

### 2.2 What the government itself lists as not yet done
Same source: the government "plans to introduce" connected ambulances with IoT and
biotelemetry that transmit real-time heart rate, blood pressure, oxygen level and ECG to
doctors and hospitals during transit. This is stated as a future plan, not a deployed
feature.

> **Important precision (fixes an earlier wording in `00`).** Hospitals *do* already get
> an advance alert. So the gap is **not** "the hospital gets no warning". The gap is
> that the alert carries no live physiological picture and no predicted trend, and there
> is no decision-support layer (risk score, hospital matching, corridor clearing) built on
> the vitals that are already being recorded. Say it this way in the review.

### 2.3 Documented performance problems the system targets
- **Golden-hour failure:** a 2020 CAG audit, reported by Deccan Herald, found that from 2014-15 to 2018-19 the service could not reach about half of trauma cases inside the golden hour.
- **Hospital denial / no bed:** ETV Bharat carried a report (May 2026) of an accident victim who spent about two days in an ambulance after hospitals refused admission citing no ICU bed or specialist. `[VERIFY: only the headline was seen; read the article before quoting]`. This supports scoring hospitals on bed availability and specialty rather than distance alone.
- **Time matters:** the Chief Minister himself framed the golden hour as decisive for road accidents and cardiac emergencies (ETV Bharat launch coverage).
- **Bengaluru corridor congestion:** the ORR junction list in `03` comes from BBMP/traffic police data. `[VERIFY: attach the exact source before presenting]`.

### 2.4 The four gaps, stated precisely

| # | Gap | Evidence | What this project builds |
|---|---|---|---|
| G1 | Reports are slow, unstructured, language-constrained | Multilingual state; caller-operator model relies on human call takers | Voice + text intake (Kannada/Hindi/English), guided triage dialogue, GPS location |
| G2 | Vitals are recorded but not turned into a live, predictive picture for the hospital | ETV Bharat: vitals recorded now; real-time transmission is a future plan | BiGRU severity + deterioration model, fused live risk score |
| G3 | Signals are not proactively cleared for the ambulance's route | Fixed-time signals; no signal preemption mentioned in the 108 launch coverage `[VERIFY]` | RL signal control on a simulated ORR corridor, benchmarked against fixed-time |
| G4 | Hospital choice is distance-driven; capacity mismatch causes delays | Hospital denial reports; GIS mapping exists but no capability-aware ranking stated | Hospital scoring on ETA, trauma level, beds, specialty, blood stock |

## 3. Where the research literature stands (verified)

Being explicit about prior work is what makes the novelty claim credible.

| Work | What it does | Setting | How we relate |
|---|---|---|---|
| **DeepTriager** (Wang et al., IEEE, 2019) | BiLSTM + self-attention over clinical text and structured vitals to predict ED acuity | In-hospital ED, EHR-scale | Same idea (text + vitals fusion). We use it as precedent for concatenation fusion. Reported AUCs: 0.98 for acuity level I in the original abstract; 0.956 (concatenation) and 0.9594 (bilinear) as quoted by a 2026 follow-up paper `[VERIFY the numbers in the original paper]` |
| **VitalML** (Sundrani et al., npj Digital Medicine, 2023) | Predicts new tachycardia, hypotension or hypoxia within 90 min from 15 min of continuous monitoring plus triage data | ED, 19,847 visits; AUROC 0.836 / 0.802 / 0.713 | Precedent for predicting deterioration from vitals trends. We do not claim to invent it |
| **Multimodal attention triage** (arXiv 2607.16662, 2026) | TabNet for vitals + self-attention fusion with text | ED, 11,102 records, Malaysia | Shows the area is active; more complex fusion than we need |
| **DeepTriage-CN** (Scientific Reports, 2026) | Late-fusion text + vitals for older ED patients | ED, 8,000 visits | Finding relevant to us: text added no significant gain over a strong tabular model overall, but helped when vitals were degraded. Supports our modality-dropout design and is a risk we discuss honestly |
| **AI in prehospital emergency care** (narrative review, Cureus, Sept 2025) | Reviews ML for prehospital triage and destination decisions | Prehospital, mostly high-resource settings | ML generally beat traditional early-warning scores; trauma models AUC 0.75-0.93. Confirms prehospital ML is real, so our claim must be about integration and context, not "first prehospital ML" |
| **SIGMA** (arXiv 2608.18263, 2026) | RL + LLM-guided multi-objective signal control for emergency priority; evaluated in SUMO on four Kolkata intersection models calibrated to municipal counts | Simulation, India | Direct precedent for Indian-context RL emergency signal control. Preprint |
| **EMVLight** (AAAI 2022) | Multi-agent RL for joint emergency-vehicle routing and signal control | Simulation, grid/real maps | Precedent for RL signal preemption without ignoring general traffic |

**Consequence for the novelty claim:** each *ingredient* exists. Do not say otherwise.

## 4. The novelty case (defensible wording)

### 4.1 What we claim
> "To our knowledge, and in the literature we reviewed, we did not find a single system that
> integrates multilingual voice intake, RAG-grounded first-aid guidance, sequence-based
> vitals severity and deterioration modelling, RL signal control on a named Indian corridor,
> and capability-aware hospital selection into one pre-arrival pipeline, evaluated against
> NEWS2, fixed-time signals and naive dispatch. We address the specific gap the Karnataka
> government lists as future work for its 108 service."

(Note "to our knowledge / in the literature we reviewed". Our search was not exhaustive.)

### 4.2 The five contribution points

| # | Contribution | Why it is not just "reusing a paper" |
|---|---|---|
| C1 | **Pre-arrival framing on the ambulance-to-hospital link** | Most cited triage work is in-hospital ED. Prehospital ML exists but is typically single-purpose (triage or destination), not tied to signals and hospital ranking |
| C2 | **Multilingual, bystander-driven intake feeding a fusion model** | Published triage text is clean clinical text from nurses/clinicians. Ours is panicked bystander speech in Kannada/Hindi/English, converted into a bounded structured vector via a state machine |
| C3 | **Safety-constrained first-aid guidance (RAG) with a measured groundedness metric** | Not prediction-only. Guidance is restricted to curated protocol text, cited, and its groundedness is evaluated rather than assumed |
| C4 | **Indian-context RL signal control on a named Bengaluru corridor, with ambulance-aware state and a general-traffic penalty** | Extends Kolkata-based SIGMA-style work to ORR geometry; reports the trade-off honestly against fixed-time and always-green baselines |
| C5 | **Integrated, live-updating decision pipeline with honest ablations** | Text-only vs vitals-only vs fused, sensitivity to text quality, modality dropout for missing data, and system-level before/after benchmark on 8+2 Bengaluru scenarios |

### 4.3 What we do NOT claim
- Not the first multimodal triage model, not the first deterioration predictor, not the first RL signal controller.
- Not clinical validity. MIMIC is ICU data; text pairing for fusion is synthetic.
- Not a replacement for 108 or eRaktKosh. We plug into their gaps.
- No "novel architecture". The novelty is context, integration, and evaluation.

### 4.4 Why this still counts as project-level novelty
1. **Problem-driven, not method-driven.** The gap comes from a live government programme with an explicit future-work statement.
2. **Systems integration is itself an engineering contribution.** Four AI paradigms (sequence deep learning, NLP/RAG, RL, supervised ranking) connected through defined interfaces.
3. **Rigour is the differentiator.** Baselines, patient-grouped cross-validation, ablations, seeds, and stated limitations are what most student projects lack.
4. **A falsifiable hypothesis set** (see §7). The project can fail informatively.

### 4.5 One-line answers for likely challenges
- *"Prehospital AI already exists."* Yes; we cite the review. Ours is the integrated pipeline for the Karnataka 108 gap, with multilingual intake, signal clearance, and hospital matching, not a standalone triage score.
- *"Fusion is known."* Yes (DeepTriager). We reuse concatenation deliberately and contribute the bystander-text interface and the missing-modality handling.
- *"RL for signals is known."* Yes (SIGMA, EMVLight). We apply it to a named Bengaluru corridor with an ambulance-aware state and honest trade-off reporting.
- *"Your data is synthetic."* Partly; each synthetic piece is labelled, and sensitivity analysis shows how conclusions depend on it.

## 5. System architecture (technical)

```
Voice/Text ─► [1 Reporting Agent] ─► text feature vector (19-d) ──┐
  (kn/hi/en)   Whisper → state-machine dialogue → classifier        │
               └─► RAG first-aid guidance (LangChain, FAISS, cited) │
                                                                    ▼
Ambulance vitals stream ─► [2 Vitals Agent: BiGRU, 2 heads] ─► Fusion (concat) ─► FusedRisk
                                                                    │  (live-updating)
Traffic sim (SUMO) ─► [3 Traffic Agent: DQN signal control] ────────┤
Hospital data ─► [4 Hospital Agent: weighted score + RF surrogate] ◄┘
                                     ▼
                         [Coordinator: rule-based conflict resolution]
                                     ▼
              DispatchDecision → map, police dashboard, hospital view
```

## 6. Technical details per component

### 6.1 Reporting agent
- **ASR:** Whisper (`small`), Kannada/Hindi/English; metric WER on a team-recorded set (≥30 utterances per language).
- **Location:** device GPS only; never parsed from speech.
- **Dialogue:** bounded state machine with 6 questions (`conscious, breathing, bleeding, speak, num_injured, mechanism`); the LLM only phrases each question. This keeps the flow testable and auditable.
- **Output:** 19-dim vector (known-flags for unanswered items, injury-type one-hot, 4-class text severity probabilities from a TF-IDF + logistic regression classifier).
- **RAG:** LangChain LCEL chain; corpus from IFRC first-aid guidelines and public protocols; chunk ≈500 chars/50 overlap, tuned; `all-MiniLM-L6-v2` embeddings; FAISS; k=4; strict "answer only from context" prompt; similarity threshold returns "not covered"; per-answer source logging.
- **Groundedness evaluation:** entailment/LLM-judge check per answer sentence; report % fully grounded; retrieval hit@k on a 30-50 query set.

### 6.2 Vitals agent (core ML)
- **Baselines:** majority class, NEWS2 (implemented exactly per RCP bands; 4-class banding is a project convention), logistic regression, RandomForest, Gradient Boosting, then BiGRU.
- **Data:** MIMIC-III Clinical Database Demo (100 ICU patients, about hourly vitals, no notes).
- **Labels:** outcome-based, never NEWS2-derived. Severity classes from death within 24 h, vasopressor/ventilation start within 6 h, and remaining ICU length of stay; trend = new deterioration event within the next 6 readings.
- **Input:** `[B, 6, 12]` (7 vitals/flags + 5 deltas), mask `[B, 6]`, z-score using train-fold statistics.
- **Model:** BiGRU (hidden 64) → 128-d; two heads: severity (4 classes), trend (1 logit). Loss = class-weighted CE + 0.5 · BCE.
- **Validation:** `StratifiedGroupKFold` by patient; 5 folds; mean ± std; test set used once.
- **Explainability:** SHAP TreeExplainer on RF/GB; Integrated Gradients on the BiGRU.
- **Metrics:** accuracy, macro-F1, confusion matrix (severity); AUROC (+AUPRC if rare) for trend.

### 6.3 Fusion
- Concatenate vitals embedding (128) with text branch (Linear 19→32) → Dense 64 → severity + trend heads.
- Modality masks and modality dropout (text 0.2, vitals 0.1) so text-only, vitals-only, and fused inference all work.
- Fused risk score = expected severity normalised to [0,1]; recomputed for every new reading.
- **Ablation:** text-only vs vitals-only vs fused, plus a sensitivity table over text-generator agreement (0.5 / 0.65 / 0.8). Text pairs are synthetic, generated from labels only (independent of vitals values) with noise levels fixed in advance.

### 6.4 Traffic agent (RL)
- **Simulator:** SUMO with `sumo-rl`; corridor Silk Board → Bellandur → Marathahalli → KR Puram on the ORR, junction spacing 6/5/5 km, 3-lane main road, 2-lane cross streets.
- **Agent:** single DQN controlling all 4 junctions jointly (2⁴ = 16 actions); 5 s decision step, 10 s min green, 3 s yellow.
- **State (24-d):** per-junction main/cross queues, phase one-hot, ambulance-within-1 km flag, normalised ambulance distance (custom observation class).
- **Reward:** `−1.0·ambulance wait − 0.1·mean waiting of others − 0.05·max cross-queue penalty` (initial weights, tuned and logged).
- **Baselines:** SUMO fixed-time and naive "always green for ambulance".
- **Protocol:** ≥3 seeds, checkpoints, deterministic evaluation, ≥20 eval episodes per demand level, 3 demand levels; report ambulance transit time and general delay together.
- **Known risk:** 5-6 km spacing means weak coupling, so gains may be modest; an optional compressed-spacing variant is a labelled secondary experiment.
- **Police alerts:** WebSocket push (incident, severity, route, units, perimeter); deployment sizing stays rule-based.

### 6.5 Hospital agent
- Weighted score: proximity 0.30, trauma match 0.25, beds 0.20, specialty 0.10, blood match 0.15, blended 80/20 with a RandomForest surrogate trained on simulated dispatch rewards. `[reconcile weights with existing code]`
- Blood stock is a scoring factor only (eRaktKosh already exists nationally; no separate app). Live eRaktKosh access is uncertain, so blood stock is simulated and labelled.
- Benchmark: 8 existing Bengaluru scenarios plus 2 where blood availability should change the pick.

### 6.6 Coordinator
- Rule-based weighted priority bidding; conflict table covers cordon-on-route, bed shortfall, ambulance contention, police-unit shortage, junction preemption overlap, blood shortage. Deliberately not learned; declared as scope decision.

### 6.7 Stack

| Layer | Tech |
|---|---|
| Backend | Python, FastAPI, WebSockets, Pydantic |
| ASR/NLP | Whisper, LLM API, LangChain (RAG only), sentence-transformers, FAISS |
| ML | scikit-learn, PyTorch, SHAP, Captum |
| RL | SUMO, sumo-rl, Stable-Baselines3 |
| Routing | OpenRouteService |
| Frontend | Leaflet / OpenStreetMap |
| Practice | seeds fixed, versions pinned, versioned model artifacts, experiment log |

## 7. Falsifiable hypotheses (what "success" and "honest failure" look like)

| H | Hypothesis | Support if | Still valuable if it fails |
|---|---|---|---|
| H1 | Sequence model beats NEWS2 and tabular baselines on outcome labels | Higher macro-F1 and AUROC across patient-grouped folds | Report that tabular models suffice on small data; discuss data size |
| H2 | Fused risk beats text-only and vitals-only | Fused > both in ablation | Cite DeepTriage-CN finding; show benefit only under degraded vitals (missing-modality test) |
| H3 | RL corridor control reduces ambulance transit time vs fixed-time without large general-delay cost | Lower transit time, bounded delay change, consistent over seeds | Show training curves and the coupling limitation of 5-6 km spacing |
| H4 | RAG guidance stays grounded | High % fully grounded answers; low ungrounded rate | Tighten threshold; show refusal behaviour |
| H5 | Capability-aware hospital selection changes picks in the right direction | Blood/trauma-matched picks in the designed scenarios | Adjust weights; report which scenarios flip |

## 8. Limitations to state before being asked
- MIMIC-III Demo: ICU, in-hospital, about 100 patients, about hourly data; horizon is "next 6 readings", not minutes.
- Fusion text is synthetic; result demonstrates the pipeline and assumptions, not clinical performance.
- Simulation-only traffic results on an approximated corridor.
- Hospital beds and blood stock simulated unless real data obtained.
- Not a medical device; not for real patient care.

## 9. Deliverables for the paper/report
Problem and gap section (this file §2), related work (§3), method (§5-6), experiments and ablations (§7), model card, limitations, future work (real ambulance data, attention fusion, multi-agent RL, edge deployment, drift monitoring).

## 10. Sources (all opened or found via search; re-check before citing)

- ETV Bharat, "Karnataka Government Launches Centralised 108 Emergency System In State", 25 May 2026: https://www.etvbharat.com/en/state/karnataka-government-on-sunday-launched-state-owned-108-arogya-kavacha-centralised-command-and-control-centre-enn26052503708
- Devdiscourse coverage of the same launch: https://www.devdiscourse.com/article/science-environment/3921155-karnataka-launches-centralised-command-and-control-centre-for-emergency-ambulance-services
- Deccan Herald, CAG 2020 findings on golden-hour performance: https://www.deccanherald.com/amp/story/india%2Fkarnataka%2Fkarnatakas-108-ambulances-to-get-faster-fitter-more-efficient-1072699.html
- DeepTriager (IEEE): https://ieeexplore.ieee.org/abstract/document/8983093/
- Multimodal attention triage (arXiv 2607.16662, quotes DeepTriager AUCs): https://arxiv.org/abs/2607.16662
- DeepTriage-CN, Scientific Reports 2026: https://www.nature.com/articles/s41598-026-61270-7
- VitalML, npj Digital Medicine 2023: https://www.nature.com/articles/s41746-023-00803-0
- Prehospital AI review, Cureus 2025 (PMC12494111): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12494111/
- SIGMA, arXiv 2608.18263: https://arxiv.org/abs/2608.18263
- EMVLight (AAAI 2022): https://cdn.aaai.org/ojs/20383/20383-13-24396-1-2-20220628.pdf
- MIMIC-III Demo: https://physionet.org/content/mimiciii-demo/
- sumo-rl: https://github.com/LucasAlegre/sumo-rl
- eRaktKosh: https://eraktkosh.mohfw.gov.in/BLDAHIMS/bloodbank/about.cnt

## 11. Corrections to earlier project files found while verifying

1. `00` §2 point 2 and §3: hospitals already receive advance alerts from 108; rephrase the gap as "no live vitals trend or AI risk score in that alert" (see §2.2).
2. `00` §4 and `05` §8: SIGMA is best described as validated in SUMO on four Kolkata intersection models calibrated to municipal counts (arXiv preprint), not "real intersections".
3. `00` §4: the 2026 multimodal triage paper (arXiv 2607.16662) uses TabNet with self-attention fusion; the doc says "BERT for text". `[VERIFY and fix]`.
4. DeepTriager AUC numbers (0.956 / 0.9594) come from a secondary citation; confirm in the original before putting them on a slide.
5. Add DeepTriage-CN and the prehospital AI review to related work; they are the most likely sources of viva challenges.
6. `00` §3 says the state "explicitly stated" real-time transmission is future; the source says the government "plans to introduce" connected ambulances. Use that wording.
