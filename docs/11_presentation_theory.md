# 11 — Presentation Theory, Slide Outline, and Viva Prep

Use this for slides, speaking notes, and viva. Numbers cited from literature come from the
project documents and must be verified against primary sources before presenting
(`[TODO-VERIFY]`). Do not present unverified numbers as fact.

---

## Part A — Suggested slide outline (≈18 slides)

| # | Slide | Key message |
|---|---|---|
| 1 | Title | Pre-Arrival Multimodal AI Triage & Emergency Coordination System for Bengaluru |
| 2 | Problem | Four points where emergency time is lost: reporting, hospital blind spot, traffic, hospital choice |
| 3 | Motivation | Karnataka 108 records vitals but pre-arrival transmission to hospital is a stated future goal; we prototype that gap |
| 4 | Related work & positioning | Fusion triage, deterioration prediction, and RL signal control exist; our contribution is context and integration |
| 5 | System architecture | 4 agents + coordinator diagram |
| 6 | Component 1: reporting | Whisper → guided dialogue → RAG guidance |
| 7 | Component 2: vitals model | NEWS2 baseline → BiGRU with severity + trend heads |
| 8 | Fusion | Concatenate text + vitals, live-updating risk score |
| 9 | Component 3: RL traffic | SUMO ORR corridor, state/action/reward |
| 10 | Component 4: hospital | Distance + trauma + beds + specialty + blood |
| 11 | Coordinator | Rule-based conflict resolution (deliberately) |
| 12 | Data | What is real vs synthetic (honesty table) |
| 13 | Evaluation design | Baselines, cross-validation, ablation |
| 14 | Results — vitals & fusion | Table + confusion matrix + ablation |
| 15 | Results — RL & dispatch | Transit time, general delay, benchmark |
| 16 | Explainability | SHAP for trees, Integrated Gradients for sequence model |
| 17 | Limitations & ethics | Synthetic pairing, ICU domain gap, small data, not a medical device |
| 18 | Future work & conclusion | Real ambulance data, edge deployment, learned coordinator |

---

## Part B — Theory explained simply (speaker notes)

### B1. The problem in plain words
Every minute counts in emergencies. Time is lost when a panicked bystander cannot explain clearly, when the hospital knows nothing about the patient until arrival, when the ambulance sits at red lights, and when the hospital chosen is merely the closest rather than the best equipped.

### B2. Automatic Speech Recognition (ASR) — Whisper
ASR converts speech to text. Whisper is an encoder-decoder Transformer trained on large multilingual audio; it supports many languages, including Kannada and Hindi. We use small variants for speed. Quality metric: **Word Error Rate (WER)** = (substitutions + deletions + insertions) / words in reference. Lower is better.
*Why GPS instead of speech for location:* addresses spoken in panic are error-prone; GPS is precise.

### B3. Guided dialogue (LLM + state machine)
An LLM generates language, but we do not let it run the conversation freely. A **state machine** holds a fixed set of triage questions (conscious? breathing? bleeding? can they speak?) and decides which is next; the LLM only phrases the question naturally in the user's language. This mimics trained emergency call operators and keeps behaviour predictable and testable.

### B4. Retrieval-Augmented Generation (RAG)
- **Problem:** LLMs can hallucinate; wrong first-aid advice is dangerous.
- **Idea:** do not ask the model from memory; first *retrieve* relevant passages from a trusted corpus, then instruct the model to answer only from them.
- **Steps:** split documents into chunks → convert each chunk into an **embedding** (a vector capturing meaning; we use all-MiniLM-L6-v2) → store in a **vector database** (FAISS) → for a query, embed it and fetch the top-k nearest chunks by similarity → feed chunks + question to the LLM with a strict prompt.
- **LangChain / LCEL:** a library that composes these steps as a pipeline (`retriever | prompt | llm | parser`). Used only here.
- **Groundedness check:** retrieval does not guarantee the answer stayed inside the context, so we verify with an entailment/judge check and report the % of grounded answers.

### B5. NEWS2 — the clinical baseline
National Early Warning Score 2 (Royal College of Physicians, UK). Scores seven parameters (respiratory rate, SpO₂, supplemental oxygen, systolic BP, heart rate, consciousness, temperature) 0–3 each; the total indicates deterioration risk. It is a hand-designed rule, not learned. We benchmark against it because "our model gets X%" means nothing without a real standard to compare with. Our 4-class banding from the total is a project convention.

### B6. Why sequences? RNN, LSTM, GRU, "Bi"
A single vitals reading says how the patient is *now*. The **trend** (falling BP, rising heart rate) says where the patient is *going*. A **recurrent neural network** reads readings in order and carries a memory. **LSTM/GRU** add gates that decide what to remember or forget, fixing the vanishing-gradient problem of plain RNNs; GRU is the lighter version. **Bidirectional** processing reads the window forward and backward for richer context (valid because the whole window is already observed). We add **delta features** (change between readings) to make trends explicit.
*Two heads, one encoder:* the encoder's summary vector feeds (1) a severity classifier and (2) a trend predictor. Sharing the encoder is **multi-task learning**: both tasks regularise each other.

### B7. Multimodal fusion
Different data types (text-derived features, vitals) are encoded separately, then combined. We use **concatenation fusion**: join the two vectors and pass them through dense layers. Chosen over attention/Transformer fusion because published work reports strong results with simple concatenation (`[TODO-VERIFY DeepTriager numbers]`) and it is far easier to implement, debug and explain. **Modality dropout** randomly hides one modality during training so the model works when only text (before vitals arrive) or only vitals is available and does not over-rely on one input. **Live update:** the score is recomputed as each new reading arrives.

### B8. Reinforcement learning for traffic signals
RL learns by trial and error inside a simulator.
- **Agent:** the signal controller. **Environment:** the SUMO-simulated corridor.
- **State:** queue lengths, current phase, ambulance distance to each junction.
- **Action:** which approach gets green at each junction.
- **Reward:** negative ambulance waiting time, minus a penalty on other vehicles' delay (prevents the trivial "always green for ambulance" policy).
- **DQN:** learns Q(s,a), the expected future reward of an action, with a neural network, using a replay buffer and target network. **PPO** is a policy-gradient alternative.
- **Why cross-streets matter:** without cross traffic, "always green" would be optimal and RL would be pointless.
- **Training discipline:** learning curves, multiple seeds (RL is unstable), checkpoints, deterministic evaluation separate from training.
- **Simulator:** SUMO (Simulation of Urban MObility); `sumo-rl` wraps it as a Gymnasium environment.

### B9. Hospital selection
A weighted score over proximity (ETA), trauma capability, bed availability, specialty match, and blood stock; blended with a RandomForest that learned expected reward from simulated dispatches. The weighted part is rule-based on purpose and we say so. Blood availability is a factor only; India's eRaktKosh already provides the national blood-stock system, so we do not rebuild it.

### B10. Coordinator
Resolves conflicts (route crosses a police perimeter, hospital out of beds, two incidents competing for one ambulance) via weighted priority bidding. Rule-based by design: a learned coordinator was scoped out as too ambitious to validate in the time available.

### B11. Explainable AI
- **SHAP** assigns each feature a contribution to a prediction (Shapley values from game theory). `TreeExplainer` is exact and fast for tree models, so we use it on RF/GB.
- **Integrated Gradients** attributes a neural network's output to its inputs by integrating gradients along a path from a baseline input; suited to the BiGRU. Attention visualisation is an alternative.
- Why it matters: clinicians need to see *why* a patient is flagged.

### B12. Evaluation concepts (know these cold)
- **Accuracy:** fraction correct; misleading with imbalance.
- **Precision / recall / F1:** F1 is their harmonic mean; **macro-F1** averages per class equally, so rare severe classes count.
- **Confusion matrix:** rows = true class, columns = predicted; shows *which* mistakes happen (missing a critical patient is worse than over-triaging).
- **ROC-AUC:** probability the model ranks a random positive above a random negative; used for the trend task. Add AUC-PR when positives are rare.
- **Baselines:** majority-class (dumbest), NEWS2 (clinical), LR (simple), RF/GB (strong tabular). The sequence model must *earn* its place.
- **Cross-validation (k-fold):** average over k train/validation splits; report mean ± std. **Grouped by patient** so one patient's readings never sit in both train and validation (otherwise leakage inflates scores).
- **Ablation:** remove a component to prove it matters (text-only vs vitals-only vs fused).
- **Overfitting:** memorising training data; countered by validation, early stopping, dropout, small models.
- **Data leakage:** information from test/validation entering training; the classic example here is deriving labels from NEWS2 then comparing against NEWS2.

### B13. Real vs synthetic (the honesty slide)
| Part | Status |
|---|---|
| Vitals model | Real MIMIC ICU data (small; in-hospital) |
| Text ↔ vitals pairing | Synthetic (no dataset pairs bystander speech with vitals) |
| Text severity data | Team-built, reviewed |
| Traffic RL | Simulation on approximated real geometry |
| Hospital beds / blood | Simulated unless real data obtained |
| Coordinator | Rule-based |

---

## Part C — Viva Q&A

**Q1. What is novel here?**
Not the architecture. Multimodal triage fusion and RL signal control both exist in the literature. Our contribution is applying and integrating them for a pre-arrival, multilingual, low-resource, Bengaluru context with grounded first-aid guidance, and prototyping the ambulance-to-hospital vitals link that Karnataka's 108 system lists as a future goal.

**Q2. Why not just use NEWS2?**
NEWS2 is a snapshot rule with fixed thresholds and no trend or text input. We use it as the benchmark; the sequence model is judged on whether it beats it on real outcome labels.

**Q3. Isn't comparing to NEWS2 circular?**
It would be if labels came from NEWS2. Ours come from outcomes (death, vasopressor/ventilation start, ICU length of stay), independent of NEWS2 formulas.

**Q4. Your text data is synthetic; is the fusion result meaningful?**
It demonstrates the pipeline and how fusion behaves under assumed text informativeness. We generated text features from labels only (independent of vitals), fixed noise levels beforehand, and report a sensitivity table. We do not claim clinical performance.

**Q5. MIMIC is ICU data. Is it valid for ambulance patients?**
No; that is a stated limitation (domain shift). It is the best openly accessible sequential vitals-with-outcomes source; the pipeline is designed so real ambulance data could replace it.

**Q6. Only ~100 patients. Can a neural net learn from that?**
Risky; that is why we compare with simpler models, use patient-grouped cross-validation, report variance, and keep the model small. If the BiGRU does not beat tabular baselines, we report that.

**Q7. Trend "in the next 10–15 minutes"?**
Data is hourly, so we define the horizon as the next 6 readings. The demo replays this on a compressed timeline.

**Q8. Why RAG instead of fine-tuning or just prompting?**
Safety and traceability: answers come only from curated protocol text, can be cited, and we measure groundedness. Fine-tuning would not guarantee this and needs far more data.

**Q9. Why LangChain, and why only in RAG?**
It gives ready abstractions for loading, splitting, retrieval and chains. The dialogue logic stays in our own state machine so the conversation remains bounded and testable.

**Q10. How do you know the RAG answers are not hallucinated?**
We measure it: a groundedness check over a test set, retrieval hit@k, and a similarity threshold that returns "not covered" instead of improvising.

**Q11. Why concatenation, not attention fusion?**
Literature reports strong results with concatenation; it is simpler, easier to debug in our timeline, and sufficient. Attention fusion is future work.

**Q12. What does the RL actually beat?**
SUMO's fixed-time program and a naive "always green" preemption baseline, on the same demand, reporting ambulance transit time and general delay together, across multiple seeds.

**Q13. Junctions are 5–6 km apart, so is RL useful?**
The benefit may be modest because junctions interact weakly; we report it honestly and include a compressed-spacing variant as a secondary experiment.

**Q14. Why is the coordinator rule-based?**
A learned coordinator would need a joint multi-agent training environment and validation beyond the time available. It's a deliberate scoping choice.

**Q15. Where's the ground truth for hospital selection?**
There is none. The RF is a surrogate trained on a hand-defined reward in simulated dispatches; the rule-based score does the primary work. We state this openly.

**Q16. Why not build a blood-bank app?**
eRaktKosh already exists nationally. We use blood stock as one scoring factor and simulate the data if live access isn't available.

**Q17. Why SHAP for trees but not the GRU?**
TreeExplainer is designed for tree ensembles. For a neural sequence model we use Integrated Gradients or attention.

**Q18. Real-world deployment challenges?**
Connectivity, regulatory approval, sensor reliability, ASR on noisy dialects, integration with 108's MDT tablets, privacy under DPDP Act, and clinical validation on real ambulance data.

**Q19. What would you do with more time?**
Real ambulance vitals and outcomes, clinician-labelled text data, attention fusion, multi-agent RL over more junctions, edge deployment with quantised models, periodic retraining and drift monitoring.

**Q20. What failed or surprised you?** *(prepare your own honest answer from actual results, e.g., class merging, RL reward tuning, fusion gain smaller than expected.)*

---

## Part D — One-paragraph abstract (draft)

We present a prototype pre-arrival emergency triage and coordination system for Bengaluru that connects four agents. A multilingual voice-and-text reporting agent uses guided dialogue and retrieval-grounded first-aid guidance. A vitals agent uses a bidirectional GRU over ambulance vital-sign sequences to estimate current severity and short-term deterioration, benchmarked against the NEWS2 early-warning score. A reinforcement-learning traffic agent controls signals on a simulated Outer Ring Road corridor to reduce ambulance transit time, and a hospital agent ranks destinations by trauma capability, beds, and blood availability. A rule-based coordinator fuses these outputs into a dispatch decision. We report baselines, ablations, and limitations, positioning the contribution as context-specific integration of established techniques rather than a new architecture.
