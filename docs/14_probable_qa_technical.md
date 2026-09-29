# 14 — Probable Questions & Answers (Datasets, Models, Parameters, Training, Evaluation)

**How to use:** answers are short and speakable. Values marked **(initial)** are starting
settings that will be tuned; say "we start with X and tune it with cross-validation", never
"X is optimal". **Never quote result numbers until the experiment has actually run.** If
asked for results you don't have yet, use the script in §10.

Consistent with: `07` (schemas/tensors), `08` (data/labels), `09` (RL spec), `13` (novelty).

---

## Section A — Datasets

**A1. Which dataset will you use for the vitals model?**
MIMIC-III Clinical Database Demo from PhysioNet. It has 100 de-identified ICU patients, real vital-sign time series (about one reading per hour), and outcome tables. It is open access and needs no credentialing, unlike the full MIMIC-III.

**A2. Why MIMIC and not ambulance data?**
Real Indian ambulance vitals with outcomes are not openly available. MIMIC is the best open source of sequential vitals linked to real outcomes. We state the ICU-vs-prehospital domain gap as a limitation, and the pipeline is built so real ambulance data can replace it.

**A3. Only 100 patients. Isn't that too small?**
Yes, it is small, and we say so. That is why we (i) compare against simple baselines, (ii) use patient-grouped cross-validation and report mean ± standard deviation, (iii) keep the neural model small (about 41k parameters), and (iv) treat tabular models as a fallback primary result if the sequence model does not win.

**A4. How are the labels created?**
From outcomes, not from NEWS2. Severity class 3 (critical) = death within 24 h or vasopressor/ventilation start within 6 h; class 2 = more than 72 h of ICU stay remaining; class 1 = 24-72 h; class 0 = less than 24 h. The trend label = a new deterioration event (death, vasopressor or ventilation start) within the next 6 readings. Thresholds are a team convention; we check class balance and merge classes if any class is under about 5%.

**A5. Why not label with NEWS2?**
Because then NEWS2 would trivially predict its own labels and the benchmark would be circular. Independent outcome labels make the comparison fair.

**A6. Where does the text (bystander report) data come from?**
We build it. About 400-600 short incident reports in English, Hindi and Kannada, drafted with an LLM from templates, then every sample is reviewed by hand (native speakers for Hindi/Kannada) and labelled with a written 4-class rubric. It is synthetic and tagged as such. A 20% test split is never used for tuning.

**A7. Where does the fusion training data come from? MIMIC has no text.**
The text features are synthetic and paired to MIMIC windows through a class-conditional generator. Text is generated from the label only, not from the vitals values, so it carries independent information. Noise levels are fixed before looking at results, and we report a sensitivity table (text agreement 0.5, 0.65, 0.8). We state that the fusion result demonstrates the pipeline, not clinical performance.

**A8. What data does the RAG use?**
A curated first-aid corpus from public guideline documents (IFRC/Red Cross international first-aid guidelines, plus public trauma-care protocols), scoped to bleeding, CPR, fractures, shock and burns. About 20-60 pages, converted to text.

**A9. What data does the traffic model use?**
Two things. (1) The RL environment is a SUMO network built from real inter-junction distances on the Outer Ring Road, with generated demand (main road 900-1500 vehicles/hour, cross streets 300-600/hour, three demand levels). (2) The congestion/delay random-forest models are retrained on real Bengaluru traffic/accident open data if we can obtain it; otherwise on labelled simulated data. `[state which one was used]`

**A10. Hospital and blood data?**
A hospital table for 15-30 Bengaluru hospitals with real attributes (location, trauma level, specialties) from public directories. Bed availability and blood stock are simulated and labelled, because eRaktKosh exposes a public search but we found no open API.

**A11. How do you handle Kannada/Hindi speech evaluation data?**
Team members record at least 30 emergency-style utterances per language with reference transcripts. We measure word error rate per language.

---

## Section B — Vitals model (sequence model)

**B1. Which model and why?**
A bidirectional GRU over a window of up to 6 hourly readings, with two output heads: current severity (4 classes) and deterioration trend (probability). A GRU handles ordered data and captures trends; it is lighter than an LSTM and suits the small dataset. The same encoder feeds both heads (multi-task learning).

**B2. Why GRU and not LSTM or Transformer?**
GRU has fewer parameters than LSTM, so it overfits less on 100 patients, and it performs comparably on short sequences. A Transformer needs far more data and adds complexity without evidence of benefit at this scale.

**B3. What are the inputs?**
A tensor of shape `[batch, 6, 12]` plus a mask `[batch, 6]`. The 12 features: heart rate, respiratory rate, SpO₂, systolic BP, temperature, consciousness (binary from AVPU), supplemental O₂, plus five delta features (change from the previous reading for HR, RR, SpO₂, SBP, temperature).

**B4. What are the hyperparameters?** *(initial values, tuned on validation)*
Hidden size 64 per direction (128 after concatenation), 1 layer, dropout 0.3, Adam, learning rate 1e-3, weight decay 1e-4, batch size 32, max 100 epochs, early stopping on validation macro-F1 with patience 10, gradient clipping 1.0, seed 42. Loss = class-weighted cross-entropy + 0.5 × binary cross-entropy for the trend head.

**B5. How many parameters?**
About 41,000 for the fused network (BiGRU ≈ 30k, text branch and fusion layers ≈ 11k). It is deliberately small.

**B6. How do you handle missing readings?**
Forward-fill up to 2 steps, then fill with the training-set median; shorter windows are left-padded and masked. Normalisation (z-score) uses statistics from the training fold only, saved with the model.

**B7. How do you handle class imbalance?**
Class weights in the loss and macro-F1 as the selection metric. We avoid oversampling sequences unless done strictly inside the training fold.

**B8. What are the baselines?**
Five: majority class, NEWS2, logistic regression, RandomForest / Gradient Boosting on single-snapshot features, and the BiGRU. All use the same patient-level splits.

**B9. Tabular baseline settings?**
Logistic regression with balanced class weights and C tuned by search. RandomForest grid: trees 100/200/300, max depth 4/6/8/None, min samples split 2/5/10, balanced class weights. Gradient Boosting: trees 100/200, learning rate 0.05/0.1, depth 2/3/4 **(initial grid)**. Search by grid with 5-fold grouped CV scored on macro-F1.

**B10. What is NEWS2?**
National Early Warning Score 2 from the Royal College of Physicians. It scores respiratory rate, SpO₂, supplemental oxygen, systolic BP, heart rate, consciousness and temperature from 0 to 3 each. We implement the published bands and use the total as a clinical baseline. Our four-class banding of the total is a project convention.

**B11. How do you deal with GCS vs AVPU?**
MIMIC records Glasgow Coma Scale, not AVPU. We map GCS 15 → Alert, 13-14 → Voice, 9-12 → Pain, ≤8 → Unresponsive, and encode Alert as 0 and anything else as 1. New confusion (also scored 3 in NEWS2) is not available in MIMIC; a limitation.

**B12. How do you predict "trend"?**
A binary target: will a new deterioration event happen within the next 6 readings. The trend head takes the encoder summary of the whole window. Patients already on vasopressors or ventilation at the window end are excluded from the trend task.

**B13. Your trend horizon says 6 readings, not minutes. Why?**
MIMIC vitals are about hourly, so we cannot honestly claim a 10-15 minute horizon. In the demo we replay a patient's sequence on a compressed timeline.

**B14. What metrics?**
Severity: accuracy, macro-F1, confusion matrix. Trend: ROC-AUC, plus AUPRC if positives are rare. We report mean ± standard deviation across folds.

**B15. Will the neural network beat random forest?**
Not guaranteed. On small tabular-like data, tree models are often as good or better. The plan is to report whichever wins and say why; the sequence model has to earn its place, and its unique advantage is the trend head.

---

## Section C — Fusion

**C1. How do you fuse text and vitals?**
Concatenation. The vitals encoder gives a 128-d vector, the text features (19-d) go through a small linear layer to 32-d, the two are concatenated (160-d), and pass through a dense layer of 64 units with ReLU and dropout 0.3, then into the severity and trend heads.

**C2. Why concatenation and not attention?**
Published work (DeepTriager) reports strong results with concatenation, and it is simple to implement, debug and explain in our timeline. Attention-based fusion is future work.

**C3. What if only one modality is available?**
We use modality masks and modality dropout during training (text dropped with probability 0.2, vitals 0.1, never both). So text-only (before the ambulance has vitals) and vitals-only inference both work.

**C4. What is the 19-dimensional text vector?**
Known-flags and values for conscious, breathing difficulty, bleeding level and ability to speak; normalised number injured; a high-energy-mechanism flag; a 5-way injury-type one-hot; and four probabilities from a text severity classifier.

**C5. How do you prove fusion helps?**
An ablation: text-only vs vitals-only vs fused on the same patient-level folds, reporting accuracy, macro-F1, AUC. Plus a sensitivity table for text quality. If fused does not beat vitals-only, we report that honestly. A 2026 study found text added little over strong vitals models except when vitals were degraded, so this outcome is plausible and we test the missing-vitals case explicitly.

**C6. What does the "live update" mean?**
Each new vitals reading triggers a recomputation of the fused risk score over the latest window, so the score changes during transit rather than being computed once.

---

## Section D — Text, ASR, dialogue, RAG

**D1. Which speech model?**
OpenAI Whisper, `small` size (about 244M parameters), running locally or via API depending on connectivity. It supports Kannada, Hindi and English. Measured by word error rate.

**D2. Why not extract the location from speech?**
GPS is far more reliable than a spoken address in a panicked call. It's a deliberate design decision.

**D3. How does the dialogue work?**
A state machine with six bounded questions (conscious, breathing, bleeding, ability to speak, number injured, mechanism). The LLM only phrases the question in the caller's language. Decision logic stays in our code, so the flow is predictable and testable.

**D4. Which LLM?**
An LLM API for phrasing and RAG generation, with template fallback if the API is unreachable. Final provider is a cost/latency decision. `[state chosen provider]`

**D5. Text severity classifier?**
TF-IDF features with logistic regression. For Kannada and Hindi we use character n-grams (2-5) so that it works across scripts without language-specific tokenisers **(initial)**; regularisation strength tuned by cross-validation.

**D6. What is RAG and why?**
Retrieval-Augmented Generation: retrieve relevant passages from a trusted corpus, then make the LLM answer only from them. It reduces the risk of hallucinated first-aid advice.

**D7. RAG parameters?**
Chunks of about 500 characters with 50 overlap (tuned; procedure steps are short), embeddings `all-MiniLM-L6-v2` (384-d, about 22M parameters), FAISS index, top-k = 4, a similarity threshold below which the system says "not covered", and a strict "answer only from context" prompt.

**D8. Why LangChain?**
It provides ready abstractions for loading, splitting, retrieval and chains; RAG is its natural use case. We use it only there, not for dialogue logic.

**D9. How do you know the answers are grounded?**
We measure it: an entailment or LLM-judge check per answer sentence against the retrieved text, reported as % fully grounded on a 30-50 query test set, plus retrieval hit@k. Every answer logs its source chunks.

**D10. Is it safe to give medical advice?**
Guidance is limited to curated public guidelines, always says to call emergency services, refuses when the corpus does not cover the situation, and is labelled a research prototype. It supplements, not replaces, professional help.

---

## Section E — Reinforcement learning

**E1. Which RL algorithm?**
DQN from Stable-Baselines3 first, PPO as a second run if time allows. DQN suits discrete signal-phase choices and works directly with the `sumo-rl` Gymnasium wrapper.

**E2. State, action, reward?**
State (24-d): for each of 4 junctions, normalised queue on main and cross approaches, current phase, an ambulance-within-1-km flag, and normalised ambulance distance. Action: each junction picks main-green or cross-green, giving 16 joint actions. Reward: minus ambulance waiting time, minus 0.1 × mean waiting time of other vehicles, minus 0.05 × a cross-street queue penalty (initial weights, tuned and logged).

**E3. Why penalise other traffic?**
Without it the trivial policy "always green for the ambulance" wins. The penalty forces a realistic trade-off.

**E4. DQN hyperparameters?** *(initial)*
Network 2 × 64 hidden layers, learning rate 1e-3, discount 0.99, replay buffer 50k, batch size 64, learning starts at 1,000 steps, target network update every 500 steps, epsilon decayed from 1.0 to 0.05 over the first 30% of training, 100,000 timesteps per seed. With a 5-second decision step and 1,800 s episodes, that is about 360 steps per episode, so roughly 280 episodes.

**E5. How do you train and evaluate?**
At least 3 random seeds; checkpoints saved periodically; periodic deterministic evaluation (no exploration) of 5 episodes; final comparison over at least 20 evaluation episodes per demand level, at three demand levels (low / medium / peak).

**E6. What do you compare against?**
SUMO's default fixed-time signals and a naive "always green for ambulance" preemption. Metrics: ambulance transit time, mean waiting time of other vehicles, throughput and maximum cross-street queue.

**E7. What if the RL doesn't improve?**
We report the training curves, seeds and reward-weight sweep, and discuss why. One known reason: junctions 5-6 km apart interact weakly, so gains may be modest. A compressed-spacing variant is included as a labelled secondary experiment.

**E8. Why a single agent instead of multi-agent?**
Stable-Baselines3 has no native multi-agent support, and four junctions give only 16 joint actions, which is manageable. Multi-agent RL (as in EMVLight) is future work.

**E9. Is the simulation realistic?**
It uses real inter-junction distances on the ORR but approximated geometry and generated demand. Results are simulation results, not field results.

**E10. Do you change the route?**
No. Routing stays with OpenRouteService; RL only controls signals.

---

## Section F — Hospital and coordinator

**F1. How is the hospital chosen?**
A weighted score: proximity (ETA) 0.30, trauma match 0.25, bed availability 0.20, specialty match 0.10, blood match 0.15, blended 80/20 with a RandomForest surrogate. `[reconcile with existing code]`

**F2. Where does the random forest's ground truth come from?**
There is no real "best hospital" ground truth. The RF is trained on simulated dispatch episodes with a defined reward (minus ETA, minus penalties for trauma mismatch, no bed and missing blood). It is a surrogate, and if it adds little we report the rule-based score alone.

**F3. Why rule-based parts?**
Because a weighted formula is transparent and adequate. We do not present it as learned.

**F4. What does the coordinator do?**
Resolves conflicts (route through a police cordon, hospital without beds, two incidents competing for one ambulance, police shortage, junction preemption overlap, blood shortage) with weighted priority bidding. It is rule-based on purpose; a learned coordinator is future work.

**F5. Blood bank?**
Only a scoring factor. eRaktKosh already provides the national system, so we do not build a blood-bank app.

---

## Section G — Training methodology & evaluation

**G1. How do you split the data?**
By patient, never by row. With only 100 patients a single 20-patient test set is noisy, so our main reporting is nested grouped cross-validation: an outer 5-fold `StratifiedGroupKFold` for reporting (each patient is tested exactly once) and an inner 3-fold loop for tuning. Test folds are never used for tuning.

**G2. Why grouped by patient?**
Windows from the same patient are highly correlated. Putting them in both train and validation would leak information and inflate the scores.

**G3. How do you tune hyperparameters?**
Grid search for the small tree-model grids, randomized search if the space grows, and a small manual sweep (hidden size 32/64/128, dropout 0.2/0.3/0.5, learning rate 1e-3/3e-4) for the neural network, all scored on inner-fold macro-F1.

**G4. How do you avoid overfitting?**
Small model, dropout, weight decay, early stopping, patient-grouped validation, and comparison to simpler baselines.

**G5. How do you ensure reproducibility?**
Seeds fixed for NumPy, sklearn, PyTorch and RL; library versions pinned in `requirements.txt`; models saved with version tags; every run logged to an experiment table (model, parameters, CV mean ± std, date, git commit).

**G6. Which metric is primary and why?**
Macro-F1 for severity (rare severe classes count equally) and AUROC for trend. Accuracy alone is misleading with imbalance.

**G7. What about missing a critical patient?**
The confusion matrix shows under-triage. We report per-class recall for the critical class and discuss the cost asymmetry: under-triage is worse than over-triage.

**G8. How do you explain predictions?**
SHAP (TreeExplainer) for the RF/GB models; Integrated Gradients for the BiGRU (SHAP TreeExplainer does not apply to neural networks).

---

## Section H — Deployment and practicality

**H1. Can this run in a real ambulance?**
Not as is. It is a research prototype validated only in simulation and on historical ICU data. Real deployment needs clinical validation, regulatory approval, integration with 108's tablets, and reliable connectivity.

**H2. What about poor connectivity?**
Degraded modes: text-only or vitals-only inference, cached ETA if the routing API is down, template dialogue if the LLM is unreachable, and Whisper can run locally. Edge quantisation is future work.

**H3. Latency?**
The vitals model is tiny (about 41k parameters) and runs in milliseconds on a CPU. ASR and LLM calls dominate latency. We measure end-to-end time for the demo scenario `[fill in after measuring]`.

**H4. Privacy?**
Voice and health data are personal data (India's DPDP Act 2023). The prototype uses no real patient data, deletes audio after transcription, and logs no names or phone numbers. `[verify current rules]`

**H5. Cost?**
Mostly open-source tooling; LLM/ORS API usage is small at demo scale.

---

## Section I — Tough or trap questions

**I1. "Isn't this just four existing models stapled together?"**
The integration is the contribution: defined interfaces, a live fused score feeding hospital ranking and coordination, and an evaluation that measures each part against a baseline. We do not claim new architectures.

**I2. "Your fusion data is synthetic, so the result is meaningless."**
It demonstrates the pipeline and shows how gain depends on text quality (sensitivity table). We say it is not clinical evidence.

**I3. "Why should we believe an ICU model for ambulance patients?"**
We don't claim it transfers; that is why it's a stated limitation and why the model card lists it.

**I4. "What if a baseline beats your model?"**
We report it. A finding that a logistic regression suffices on 100 patients is a legitimate result.

**I5. "Which part is actually yours?"**
Name the owner for each component (fill in from the roles table). Be ready to explain your own part in detail: data, model, parameters, results.

**I6. "What is the accuracy?"**
Use the script in §10 unless you have real numbers.

**I7. "Why 4 severity classes?"**
It matches common triage bands and NEWS2 risk tiers. If class balance is poor we merge to 3 classes or binary; the choice is data-driven and documented.

**I8. "Why not deep learning everywhere?"**
Because rule-based or simple models are appropriate where data is small or transparency matters (coordinator, hospital formula, NEWS2 baseline).

**I9. "What happens if the ASR mishears?"**
The dialogue confirms key facts with targeted questions, and low-confidence transcripts fall back to text input; nothing is guessed.

**I10. "Is the RL safe in the real world?"**
No claim of safety. It is a simulated study. Real signal preemption needs traffic-engineering and legal approval.

---

## Section J — Numbers cheat sheet (memorise these)

| Item | Value |
|---|---|
| Vitals dataset | MIMIC-III Demo: 100 ICU patients, ~hourly vitals, 25 CSV files, no notes |
| Severity classes | 4 (stable / moderate / severe / critical) |
| Trend horizon | next 6 readings |
| Window | 6 readings × 12 features |
| Vitals encoder | BiGRU, hidden 64/direction → 128-d |
| Text vector | 19-d |
| Fusion | concat 160-d → Dense 64 → severity (4) + trend (1) |
| Parameters (fused net) | ≈ 41k |
| Optimiser | Adam, lr 1e-3, batch 32, early stop patience 10 |
| Loss | weighted CE + 0.5 × BCE |
| Validation | nested grouped CV: outer 5-fold, inner 3-fold, by patient |
| Baselines (vitals) | majority, NEWS2, LR, RF/GB, BiGRU |
| ASR | Whisper small (~244M params); WER on ≥30 utterances/language |
| RAG | MiniLM (384-d), FAISS, chunk ≈500/50, top-k 4 |
| RL | DQN, 16 actions, 24-d state, 100k steps × ≥3 seeds |
| RL corridor | Silk Board → Bellandur → Marathahalli → KR Puram, 6/5/5 km |
| RL baselines | fixed-time; always-green-for-ambulance |
| Hospital score weights | 0.30 / 0.25 / 0.20 / 0.10 / 0.15 (then 80/20 with RF) |
| Benchmark | 8 existing + 2 blood-match scenarios |

## §10 What to say if results don't exist yet
> "We have finished the design, data plan and baselines scaffold. The numbers will come from
> patient-grouped cross-validation, and we will report the mean and standard deviation, not
> one lucky split. I would rather not quote a number that we have not measured. What I can
> show now is the evaluation protocol and the baselines the model must beat."

## §11 What to have ready on screen
1. Architecture diagram.
2. Label-definition table (from `08` §2).
3. The tensor shapes and model diagram (from `07` §4.2).
4. The RL state/action/reward table.
5. The baseline comparison table (empty template is fine if results aren't ready; say so).
