# Project Explainer — Read This First, Then Present

**Project:** Pre-Arrival Multimodal AI Triage & Emergency Coordination System for Bengaluru
**Purpose of this file:** one document you can read end to end and then explain to
your HOD, faculty, and teammates without notes. Simple words first, technical
terms attached, honest about what is built and what is planned.

> Honesty rule used throughout: every claim below is either (a) already built and
> checked in the repo, (b) clearly labelled "planned", or (c) labelled "to be
> verified". Do not repeat any number that is not in this file.

---

## 1. The 60-Second Version (memorise this)

When there is a road accident in Bengaluru, minutes decide outcomes. Today the
weak points are: the report is slow and language-limited, the hospital does not
know how sick the patient is until arrival, traffic signals do not help the
ambulance, and the hospital choice is mostly "nearest".

We are building one connected AI system that (1) takes the report by voice or text
in Kannada, Hindi or English and asks smart follow-up questions while giving safe,
document-grounded first-aid advice, (2) reads the patient's vitals during transport
and predicts both how serious they are now and whether they are getting worse,
(3) uses reinforcement learning to control traffic signals along a real
Bengaluru corridor so the ambulance is delayed less, and (4) picks a hospital that
can actually handle the case, including blood availability. A fused "risk score"
ties the text and the vitals together and updates as new data arrives.

We already have a working prototype (rule-based, with small ML models trained on
synthetic data). The final project adds the parts that are genuinely machine
learning research-level: a time-series vitals model, RAG, multimodal fusion and
RL, each compared against a real baseline.

---

## 2. The Problem

### 2.1 In plain words
In an emergency, four things go wrong, and each one costs minutes or quality of
care:

| # | What goes wrong today | Why it matters |
|---|---|---|
| 1 | Reporting is slow/unclear: a panicked bystander, language barriers, no structured questions | Wrong or late severity estimate, wrong resources sent |
| 2 | The hospital is blind until arrival: no live picture of the patient's condition | ER cannot prepare team, blood, OT in advance |
| 3 | Signals run on fixed timers: the ambulance sits in the same queues as everyone | Every minute lost in traffic is a minute lost for the patient |
| 4 | Hospital choice is "nearest": ignores trauma capability, free beds, blood stock | Patient may be taken somewhere that cannot treat them and re-transferred |

### 2.2 Why Bengaluru
Bengaluru is known for severe corridor congestion (Silk Board, Bellandur,
Marathahalli, KR Puram on the Outer Ring Road are among the worst). The project
grounds itself in these real junctions and in the city's real hospitals and
ambulance bases (20 hospitals and 16 ambulance bases are in the repo datasets).

### 2.3 The real-world hook (use carefully)
Public reporting on Karnataka's 108 "Arogya Kavacha" ambulance platform says
ambulances already record patient vitals on tablets, while live transmission of
those vitals to the receiving hospital is described as a planned capability.
Our project prototypes the intelligence that would sit on top of that feed.
**Before you present, re-open the news source and confirm the wording; say "as
publicly reported", not "the government confirmed".**

---

## 3. The Solution

### 3.1 One picture

```
 Bystander (voice/text, kn/hi/en)                Ambulance (vitals every few minutes)
        │                                                   │
        ▼                                                   ▼
 [1 REPORTING]  Whisper ASR → guided questions      [2 VITALS MODEL]  GRU time-series
   + RAG first-aid advice (grounded)                   → severity + "getting worse?" 
        │  text severity probabilities                       │  vitals embedding + trend
        └──────────────► [FUSION] one live Emergency Risk Score ◄──────────────┘
                                   │
            ┌──────────────────────┼─────────────────────────┐
            ▼                      ▼                         ▼
   [3 TRAFFIC / POLICE]     [4 HOSPITAL SELECTOR]       [COORDINATOR]
   RL signal control on     trauma + beds + distance    resolves conflicts
   ORR corridor (SUMO)      + blood availability        → final dispatch plan
```

### 3.2 A story you can tell (timeline)
1. **t = 0** A bystander calls in Kannada: "bike accident near Silk Board, one man
   is not responding." Whisper turns speech into text; the phone gives GPS.
2. The AI asks one question at a time (conscious? breathing? bleeding?). Each
   answer sharpens a text-based severity estimate. Meanwhile it tells the caller
   what to do (for example, how to control bleeding), but **only** from a
   curated first-aid library and never medication advice.
3. **Dispatch** An ambulance is chosen using real road distances (OpenRouteService).
4. **In transit** Paramedics enter or stream vitals. The GRU model reads the last
   few readings and outputs (a) current severity and (b) probability that the
   patient deteriorates soon. It is compared with NEWS2, the standard hospital score.
5. **Risk score** Text severity + vitals result are fused into one number that
   changes as new readings arrive.
6. **Road** The RL agent controls the signals along the corridor so the
   ambulance waits less, while a penalty stops it from simply blocking everyone else.
7. **Hospital** The hospital agent ranks hospitals using trauma capability, free
   beds, distance and blood availability; the ER is told what is coming.
8. **Coordinator** If two agents clash (e.g., the ambulance route passes a police
   cordon), it resolves the conflict and issues the final plan.

---

## 4. How Each Part Works (Approach → Technique → Data → How we measure)

### 4.1 Reporting Agent (Component 1)
- **Approach:** replace keyword parsing with a guided conversation plus grounded advice.
- **Speech:** Whisper (open speech-recognition model). Measured by word error
  rate on our own recordings in Kannada/Hindi/English (Kannada is expected to be
  weaker; we measure and report it).
- **Dialogue:** a small state machine decides which question is next; the LLM
  only phrases the question and extracts answers. So the conversation is guided,
  not free chat. This makes behaviour predictable and safe.
- **Text severity:** multilingual sentence embeddings + a simple classifier. No
  public Kannada incident dataset exists, so we build a small labelled corpus
  (600–1000 utterances, several annotators, agreement measured). It is partly
  synthetic and we say so.
- **RAG first aid (LangChain):** first-aid documents are split into chunks,
  embedded, stored in FAISS. For a question we retrieve the best chunks and force
  the LLM to answer only from them, or decline. We then check that each sentence
  is supported by the retrieved text (groundedness) and report the percentage.
- **Why not just ask an LLM?** It can invent medical advice. Retrieval plus a
  verification step plus a refusal path is how we reduce that risk.

### 4.2 Ambulance Vitals Model (Component 2) — the core ML piece
- **Input:** a short window (last ~6 readings) of heart rate, respiratory rate,
  SpO2, systolic BP, temperature, consciousness (AVPU), oxygen use, plus
  changes between readings (deltas) and missing-value flags.
- **Model:** one bidirectional GRU (a recurrent neural network for sequences)
  that produces an embedding, followed by two output heads: **severity** (4
  classes) and **deterioration probability**. Two heads share one encoder
  (multi-task learning).
- **Baselines it must beat:** majority class, **NEWS2** (the clinical score),
  logistic regression, and RandomForest/GradientBoosting on the same features.
- **Labels:** real future outcomes from ICU data (death, sustained low BP or
  SpO2, etc.), **not** derived from NEWS2. (Deriving them from NEWS2 makes the
  comparison circular; we caught and avoided this.)
- **Data:** MIMIC (public de-identified ICU database). The free demo has only 100
  patients, so we are applying for the full MIMIC-IV; results use patient-level
  splits and bootstrap confidence intervals.
- **Explainability:** SHAP for tree models; Integrated Gradients for the GRU.
- **Honest caveat:** MIMIC is charted roughly hourly. So "deterioration soon"
  means "within the next reading(s)". The live demo compresses time. We do not
  claim a literal 10–15 minute prediction unless we obtain minute-level data.

### 4.3 Traffic / Police (Component 3) — Reinforcement Learning
- **Idea:** an RL agent learns which signal phase to turn green so the ambulance
  spends less time waiting.
- **Environment:** SUMO traffic simulator via `sumo-rl`. A schematic model of
  the ORR corridor Silk Board → Bellandur → Marathahalli → KR Puram, using real
  approximate distances between junctions (~6, 5, 5 km), with cross-streets so
  the decision is non-trivial. (Already built and simulated in the repo work.)
- **State / action / reward:** queue lengths, current phase and ambulance
  position; choose the next green phase; reward penalises ambulance waiting
  **and** general queues, which prevents the lazy "always green for the ambulance" answer.
- **Algorithm:** DQN (stable-baselines3), several random seeds.
- **Compared with:** fixed-time signals and SUMO's actuated (vehicle-responsive)
  control. Metrics: ambulance transit time, other vehicles' delay, throughput.
- **Police part** stays rule-based (unit count, perimeter). We say that openly.
- **Status:** network, routes and a fixed-time baseline run exist; RL training and
  its results are still to be produced.

### 4.4 Hospital Selector (Component 4)
- Already exists: score = 0.80 × hand-designed policy (distance, beds, trauma,
  load, specialty) + 0.20 × RandomForest score; hard penalties if beds are
  insufficient or trauma capability is missing.
- Upgrades: takes the real fused risk score, and adds a blood-availability term.
- **Blood data:** India has a national system (eRaktKosh); we do **not** rebuild
  it. If we cannot get live stock we use a clearly labelled simulated file.
- **Honest note:** hospital beds are simulated today, and no public data shows
  which hospital "should" have been chosen, so this remains a documented
  decision policy, not a model that has learned from real outcomes.

### 4.5 Fusion (the "connecting" idea)
- Inputs: 5 text-severity probabilities + the vitals embedding + deterioration
  probability. A small neural network outputs one risk score.
- Before vitals exist the vitals part is masked, so the score is text-only, then
  it updates as readings arrive ("live-updating").
- **Ablation:** we train text-only, vitals-only and fused versions on the same
  data to prove fusion helps. **Modality dropout** stops the model from
  ignoring one input.
- **Honest caveat:** no public dataset has matched (report text, vitals) pairs,
  so pairs are built semi-synthetically. Fusion results show the mechanism
  works; they are not clinical accuracy claims.

### 4.6 Coordinator
Rule-based on purpose: two conflict types (route vs police perimeter, hospital
bed shortfall), agents "bid" with weighted scores, the winner keeps priority.
A learned/RL coordinator was considered and deferred as future work.

---

## 5. How the Models Are Trained (so you can defend "not trained once")
1. Split **by patient** into train / validation / test (test used once, at the end).
2. Try many settings (hyperparameter search) with cross-validation on the training
   part; compare several model types, not just one.
3. Pick the best on validation performance, retrain, then evaluate once on test.
4. Repeat with several random seeds; report mean ± spread and confidence intervals.
5. RL is trained over thousands of simulated steps; we plot learning curves and
   evaluate with exploration switched off.
6. Every run is logged (settings, seed, metrics) and models are saved with versions.
(Example: a grid of 3×4×3 = 36 settings with 5-fold CV means 180 training runs
for the search; the exact numbers in your final report should be whatever you
actually ran.)

---

## 6. Novelty and Uniqueness — Stated Honestly

### 6.1 What is NOT new (say this before they say it)
- Vitals/text fusion for triage and vitals-based deterioration prediction are
  established research (e.g., DeepTriager, VitalML, other 2024–26 work).
- RL for traffic signals and emergency-vehicle priority is established (e.g.,
  SIGMA, EMVLight). RAG and Whisper are standard tools.

### 6.2 What is our contribution
1. **A different setting:** pre-arrival (ambulance to hospital) rather than
   inside-hospital triage, for an Indian metro, with Kannada/Hindi/English input
   and low-connectivity thinking.
2. **Safety-first guidance:** RAG with an explicit groundedness check and a
   refusal path, measured, instead of free LLM advice.
3. **Integration:** one pipeline linking speech → risk → routing/signals →
   hospital, with a live-updating fused score and proper ablations.
4. **Local grounding:** RL on a corridor with real Bengaluru junction spacing;
   real Bengaluru hospitals and ambulance bases.
5. **Evaluation discipline:** real future-outcome labels, patient-level splits,
   NEWS2 and other baselines, confidence intervals, honest limitation statements.

### 6.3 One-sentence novelty statement
"We apply established multimodal-triage and RL-signal-control ideas to the
pre-arrival Bengaluru emergency pathway, add safety-checked multilingual
guidance, and evaluate the integrated system against clinical and rule-based baselines."

---

## 7. Tech Stack

| Layer | Tools | Used for |
|---|---|---|
| Backend | Python, FastAPI, WebSockets, Pydantic | APIs, live streaming |
| Classical ML | scikit-learn (RandomForest, GradientBoosting, LogisticRegression) | baselines, existing models |
| Deep learning | PyTorch (GRU), Captum | vitals + fusion models, explanations |
| Explainability | SHAP, Integrated Gradients | why a patient was flagged |
| Speech/NLP | Whisper, multilingual sentence embeddings | ASR, text severity |
| RAG | LangChain (LCEL), FAISS, an LLM API | grounded first-aid guidance |
| RL | SUMO, sumo-rl, stable-baselines3 (DQN) | signal control |
| Routing | OpenRouteService API (+ physics fallback) | road distance/time |
| Data | MIMIC (ICU vitals), team-built text corpus, hospital/ambulance CSVs, OSM hospital geojson, blood-stock file | training/evaluation |
| Frontend | HTML/JS, Leaflet + OpenStreetMap | map and dashboards |
| Experiment hygiene | seeds, version pinning, run logs (MLflow optional) | reproducibility |

---

## 8. Where the Project Stands Today (be exact)

| Item | Status |
|---|---|
| FastAPI prototype with 5 agents, live map, WebSocket streaming | **Built and working** |
| ML models: severity (TF-IDF + LogReg), traffic (2× RandomForest), hospital (RandomForest), ETA (GradientBoosting) | Built, but trained on **synthetic, formula-generated data** |
| Ambulance/police/coordinator logic | Built, **rule-based** |
| NEWS2 scoring + tabular vitals baseline scaffold | Written (`vitals_model.py`); placeholder data only |
| Benchmark vs naive dispatch (8 scenarios) | **Built; see honest reading below** |
| SUMO ORR corridor network, routes, config, fixed-time baseline run | **Built and running** |
| Voice/dialogue/RAG, GRU vitals model, fusion, RL training, blood factor, real-data retraining | **Planned (final build)** |

**Honest reading of the current benchmark:** across 8 Bengaluru scenarios the AI
dispatch was faster in 3 and slower in 5; average ETA 11.0 min vs 9.9 min for
the baseline; conflicts occurred in all 8; the hospital choice differed from
"nearest" in 5. The baseline is deliberately optimistic (flat speed, no dispatch
overhead) while the AI ETA includes overhead, congestion and a +2 min conflict
penalty. So today's value is smarter hospital choice, not faster ETA. The final
evaluation redesigns the baseline to be fair.

**Known prototype issues you should know about:** the trained ETA model is
not actually used by the ambulance agent; a routing API key is hard-coded in two
files and must be rotated and moved to an environment variable; hospital beds
are simulated.

---

## 9. Limitations and Ethics (say them yourself)
- Academic prototype; **not clinically validated; not a medical device**.
- MIMIC demo is small; hourly sampling limits the time claim.
- Fusion pairs are semi-synthetic; the text corpus is partly synthetic.
- The RL corridor is a schematic simulation, not the live road.
- Whisper Kannada accuracy may be limited; the system asks for confirmation.
- Data handling: MIMIC used under its data-use agreement, never committed to git;
  voice recordings are our own with consent.
- Guidance never includes medication or dosing.

---

## 10. How to Present (3 to 5 minutes)
1. Problem in one breath (four gaps). 2. The picture in Section 3.1. 3. Story in
Section 3.2 in 4–5 lines. 4. Say which parts are ML and which are rules. 5. Show
the honest status table. 6. Novelty statement (Section 6.3). 7. Plan and how you
will prove it (baselines, ablations, real labels). Finish by inviting questions.

---

## 11. Q & A — Practice These

### A. Big-picture questions (HOD level)
**Q1. What problem are you solving in one line?**
Reducing avoidable delay and mismatch in emergency response by connecting
reporting, patient condition, traffic and hospital choice with AI, in Bengaluru.

**Q2. Who benefits?** The patient (faster, better-matched care), the ER (advance
warning), ambulance crews and police (clearer plans), and the city (fewer
avoidable delays).

**Q3. What is the single most important part?** The vitals model, because it
supplies the patient-condition signal that today's system lacks, and it is the
piece most clearly machine learning.

**Q4. Is this deployable in real life?** Not as is. It is an academic prototype
without clinical validation. Real deployment needs partnership with the 108
service and hospitals, regulatory approval, and prospective testing.

**Q5. How is it different from existing dispatch/traffic apps?** Those handle
dispatch or navigation. We add patient-condition prediction, safety-checked
multilingual guidance, ambulance-aware signal control and condition-aware hospital
choice as one connected pipeline.

**Q6. Why these four components?** They map to the four failure points in the
emergency chain: reporting, patient condition, road, destination.

**Q7. What did you build already and what is left?** See Section 8: prototype and
corridor simulation are done; the ML-heavy components are the final build.

### B. AI/ML questions
**Q8. Is this really AI or just rules?** Both, and we label which is which:
severity, traffic, hospital and ETA models are ML (currently on synthetic data);
ambulance/police scoring, coordinator, routing and blood lookup are rule-based.
The final build adds a GRU, multilingual classifier, RAG, fusion and RL.

**Q9. Why a GRU instead of a simple RandomForest for vitals?** Patient
deterioration is about the trend over time, which a sequence model captures
natively. But we do not assume it wins: RandomForest/GB with delta features are
included as baselines and we report which is better.

**Q10. What is NEWS2 and why compare with it?** A standard clinical early-warning
score built from seven vitals. If our model cannot beat it on future outcomes,
there is no reason to use it.

**Q11. What are the two outputs of the vitals model?** Current severity band and
probability of deterioration; they share one encoder.

**Q12. How do you avoid data leakage?** Labels are future outcomes; splits are by
patient; the test set is used once; NEWS2 is a baseline predictor, never the label.

**Q13. Why is accuracy not enough?** Severe cases are rare, so a model that
always predicts "stable" looks accurate. We use macro-F1, AUROC, AUPRC,
sensitivity at fixed specificity, and calibration.

**Q14. Explain fusion simply.** Each source gives a probability-like summary; we
join them and let a small network decide the final risk, learning how much to
trust each. If vitals are missing, that part is masked.

**Q15. Why concatenation and not attention/transformers?** Published triage work
shows simple concatenation performs well, it is easier to train on limited data
and easier to explain; heavier fusion is future work.

**Q16. What is modality dropout?** Randomly hiding one input during training so
the model learns to work with either and does not over-rely on one.

**Q17. What is RAG and why use it?** Retrieval-Augmented Generation: fetch
relevant passages from a trusted library and make the LLM answer only from them.
It reduces made-up medical advice; we still verify and allow refusal.

**Q18. How do you know the RAG answer is safe?** We measure retrieval accuracy,
check every sentence for support in retrieved text, test that out-of-scope
questions get declined, and forbid medication advice.

**Q19. Why is LangChain used?** For standard RAG plumbing (loaders, splitters,
retriever, prompt-model pipeline). Safety comes from our prompt rules and checks,
not from the library.

**Q20. What is reinforcement learning here?** The agent tries signal timings in a
simulator, gets a reward for reducing ambulance waiting and queues, and improves
by trial and error. DQN uses a neural network to estimate how good each action is.

**Q21. Why not just give the ambulance permanent green?** It would trap cross
traffic and is unrealistic; the reward penalises general queues, and we report
the trade-off.

**Q22. Why simulation and not real signals?** Real signal control needs city
authority and safety approval; SUMO is the standard research environment.

**Q23. Which baselines for RL?** Fixed-time and vehicle-actuated control.

**Q24. How do you show it is not overfitted?** Cross-validation, patient-level
held-out test, several seeds, bootstrap confidence intervals, and for RL,
evaluation on unseen traffic seeds.

**Q25. What does "trained many times" mean?** Hyperparameter search with cross-
validation over several model types and seeds; then one final test evaluation.

### C. Data questions
**Q26. What data do you use?** MIMIC ICU vitals (demo now, full MIMIC-IV if
approved), a team-built multilingual incident text corpus, real Bengaluru
hospital and ambulance-base lists, and a simulated (or real if obtainable) blood-stock file.

**Q27. Is your current ML trained on real data?** No. The prototype's models use
synthetic formula-generated data; replacing them with real data where a real
target exists (and stating where none exists) is part of the final work.

**Q28. Only 100 patients in the demo?** Correct; that is why we seek MIMIC-IV
access, use grouped CV and confidence intervals, and state the limitation.

**Q29. Are the text/vitals pairs real?** No public paired data exists, so the
pairs are semi-synthetic; we report fusion as a mechanism test, not clinical accuracy.

**Q30. Why can't the hospital model learn from real data?** No public dataset
records which hospital was the best choice for a case; so it stays a transparent
scoring policy.

**Q31. Privacy?** MIMIC is de-identified and used under its agreement, not
redistributed; no personal patient data in the repo; voice samples are consented.

### D. Novelty and honesty questions
**Q32. Isn't this already done in research?** The building blocks are; we say so
and cite them. Our contribution is the pre-arrival Indian-context integration,
safety-checked multilingual guidance, local corridor RL and a rigorous
end-to-end evaluation.

**Q33. Your benchmark shows the AI slower on average — why?** The baseline is
optimistic (no overhead) while the AI includes dispatch overhead, congestion and
a conflict penalty; the demonstrated benefit so far is hospital suitability. We
are redesigning the comparison to be fair in the final version.

**Q34. What is the weakest part?** Data realism: small real vitals set,
synthetic paired data, and a simulated corridor. We state these openly.

**Q35. What would you do with more time/data?** Minute-level vitals, real 108
data, OSM-based network of the real corridor, and a learned coordinator.

**Q36. Did you drop any ideas?** Yes: image-based injury analysis (weak accuracy
in literature, ethical and data problems) and a fully learned multi-agent
coordinator (too large for the timeline).

**Q37. Is the Karnataka 108 claim verified?** It is based on public news
reporting; we quote it as "as publicly reported" and will re-verify before submission.

### E. Team/implementation questions
**Q38. Who does what?** Fill in your team split: Reporting/RAG, Vitals/fusion,
Traffic/RL/police, Hospital/blood/integration.

**Q39. What if the internet/LLM is unavailable in the ambulance?** Whisper can run
locally; the vitals model and hospital scoring run without internet; RAG needs a
local model or falls back to showing retrieved protocol text directly.

**Q40. How will you test the whole system?** Scripted scenarios end to end,
unit tests for scoring functions, the redesigned benchmark, and a live demo with
simulated vitals streamed over WebSocket.

---

## 12. Cheat Sheet
- **Vitals (7):** HR, RR, SpO2, systolic BP, temperature, AVPU, supplemental O2.
- **Severity scales:** text 1–5; vitals bands 0–3; fused risk 0–1 (mapped to 1–5).
- **Corridor:** Silk Board → Bellandur → Marathahalli → KR Puram (~6, 5, 5 km).
- **Baselines:** NEWS2 (triage), fixed-time and actuated (signals), nearest-only
  (hospital), naive dispatch (system).
- **Metrics:** macro-F1, AUROC, AUPRC, calibration (ECE, Brier), WER, retrieval
  hit@k, groundedness %, ambulance transit time, general delay.
- **Sentences to memorise:** "established components, new setting"; "real future
  outcomes, patient-level splits"; "ML where labelled, rules where honest".
- **Documents:** master overview `00`, components `01–04`, integration `05`,
  training rules `06`, codebase `07`, API `08`, data/model specs `09`, theory
  `10`, claims guardrails `11`.
