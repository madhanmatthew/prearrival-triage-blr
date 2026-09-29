# 16 — Speaking Script (10–15 Minutes)

**How to use:** read it aloud twice, then speak it in your own words. At a natural pace
(about 130–140 words per minute) the full script runs about **14 minutes**. Each section has a
time box. `[Slide: ...]` tells you what should be on screen. *Italic lines* are stage notes,
not to be read out.

**If you must cut to 10 minutes:** skip the bracketed "(optional)" paragraphs.
**If you have 15 minutes:** slow down in sections 4 and 5, the core ML parts.

| # | Section | Time |
|---|---|---|
| 1 | Opening and problem | 0:00–1:15 |
| 2 | The real-world gap | 1:15–2:30 |
| 3 | Our solution in one picture | 2:30–3:30 |
| 4 | Reporting agent | 3:30–5:00 |
| 5 | Vitals model (core ML) | 5:00–7:00 |
| 6 | Fusion and risk score | 7:00–8:00 |
| 7 | Traffic signal AI (RL) | 8:00–9:30 |
| 8 | Hospital selection and coordinator | 9:30–10:30 |
| 9 | Novelty | 10:30–11:30 |
| 10 | Tools and data | 11:30–12:30 |
| 11 | How we are building it, and status | 12:30–13:30 |
| 12 | Limitations and close | 13:30–14:30 |

---

## 1. Opening and problem (0:00–1:15)
`[Slide: title + one photo of Bengaluru traffic or an ambulance]`

"Good morning, sir/ma'am. Our project is called the **Pre-Arrival AI Triage and Emergency
Coordination System for Bengaluru**.

Let me start with a simple fact: in a road accident, minutes decide outcomes. When we looked at
what actually goes wrong between the accident and the hospital, we found four weak points.

**One**, the report itself. A bystander is panicking, may not speak English, and gives unclear
information, so the wrong resources may be sent.

**Two**, the hospital does not get a live picture of how sick the patient is while the ambulance
is on the way.

**Three**, traffic signals run on fixed timers. The ambulance waits in the same queue as
everyone else.

**Four**, hospitals are usually chosen because they are nearest, not because they can actually
treat this patient. That can mean a second transfer and lost time.

Our project connects these four steps with AI."

---

## 2. The real-world gap (1:15–2:30)
`[Slide: "What exists" vs "What is missing" — two columns]`

"This is not an imaginary problem. As publicly reported in May 2026, Karnataka launched a
**108 Arogya Kavacha** command centre. It tracks ambulances by GPS, finds the nearest one,
computes arrival time, records patient vitals on tablets in the ambulance, and alerts hospitals
in advance.

The same reporting says the government *plans* to introduce connected ambulances that send live
heart rate, blood pressure, oxygen and ECG to the hospital during transit. So that step is
described as a future plan, not a deployed feature.

Notice what this means. The alert to the hospital exists. The vitals are being recorded. What is
missing is the **intelligence on top**: something that reads the vitals trend, estimates how
serious the patient is and whether they are getting worse, helps clear the road, and helps pick
the right hospital.

We are **not** duplicating the state system. We are prototyping the missing layer that sits on top of it."

*Stage note: say "as publicly reported", not "the government confirmed". Have the ETV Bharat article open in case you're asked for the source.*

---

## 3. Our solution in one picture (2:30–3:30)
`[Slide: the architecture diagram — 4 boxes + coordinator]`

"Here is the system in one picture. There are four parts and a coordinator.

The **reporting agent** takes the emergency report by voice or text and gives safe first-aid advice.
The **vitals agent** watches the patient's vitals in the ambulance.
The **traffic agent** controls traffic signals along the ambulance's route.
The **hospital agent** chooses the best receiving hospital.

The reporting agent and the vitals agent feed one shared **risk score** that updates as new
information arrives. That score goes to the hospital agent, and finally a **coordinator** combines everything
and resolves conflicts, for example if the fastest route passes through a police cordon.

Let me explain each part briefly."

---

## 4. Reporting agent (3:30–5:00)
`[Slide: voice → text → questions → first-aid advice]`

"The first part handles the report.

The caller can speak in **Kannada, Hindi or English**. We use **Whisper**, an open speech
recognition model, to turn speech into text. For location we use the phone's **GPS**, not the
caller's words, because a panicked person cannot describe an address reliably.

Then the system asks **guided questions**, one at a time: 'Is the person conscious? Is there
heavy bleeding? Can they speak?' This works like a trained emergency call operator. It is not an
open chatbot. The order of questions is controlled by fixed logic, and the language model only
helps phrase them naturally. That keeps behaviour predictable and safe.

While help is on the way, the system gives **first-aid advice**. Here we use a technique called
**RAG, Retrieval-Augmented Generation**. The problem with a normal AI is that it can invent medical
advice. RAG solves that: the system first searches a trusted first-aid library, for example the Red
Cross guidelines, and answers **only** from what it finds. If the library does not cover the situation,
the system says so and tells the caller to wait for professionals. And we measure how many answers
stay within the source text."

*(Optional, if time allows: "It never gives medicine or dosage advice.")*

---

## 5. Vitals model — the core ML part (5:00–7:00)
`[Slide: vitals over time graph → model → severity + "getting worse?"]`

"The second part is our main machine learning contribution.

Inside the ambulance we get vitals: heart rate, breathing rate, oxygen level, blood pressure,
temperature and alertness. Hospitals normally use a scoring system called **NEWS2** to turn these
into a warning level. It is a standard clinical tool, but it looks at one moment in time.

Our idea is that what matters is not just **where the patient is, but where they are heading**. A
blood pressure of 100 that is stable is very different from a 100 that was 130 ten minutes ago.

So we use a **GRU**, a type of neural network that reads a series of readings in order. It gives
two outputs: **how serious the patient is now**, and **the chance that the patient is going to
get worse**.

To prove it is useful, we compare it with simpler methods: a 'guess the most common answer'
baseline, NEWS2 itself, and standard machine-learning models such as logistic regression and
random forest. If our model cannot beat them, we say so.

We train it on **MIMIC**, a public de-identified hospital ICU database. Two important details. First,
the labels come from **real outcomes** such as deterioration events, not from NEWS2. Otherwise we
would be testing NEWS2 against itself. Second, we split the data **by patient**, so the model
is always tested on patients it has never seen.

We also show **why** a patient was flagged, using explainability tools, because in healthcare a
number without a reason is not enough."

---

## 6. Fusion and the risk score (7:00–8:00)
`[Slide: text signal + vitals signal → one live risk score]`

"Now, how do we connect the first two parts? This is **multimodal fusion**.

The reporting agent gives a severity estimate from what the caller said. The vitals model gives
one from the ambulance data. We combine the two into a **single Emergency Risk Score**.

Early on, before the ambulance has any vitals, the score depends only on the report. As readings
come in, the score **updates live**, like a doctor changing their assessment as new
information arrives.

To prove the combination helps, we run an **ablation**: text only, vitals only, and both together,
on the same data. If both together is not better, we report that honestly too.

One honest note: no public dataset pairs bystander reports with vitals, so the pairs are
partly synthetic. That means this result shows the method works, not clinical accuracy."

---

## 7. Traffic signal AI (8:00–9:30)
`[Slide: corridor map Silk Board → Bellandur → Marathahalli → KR Puram + reward idea]`

"The third part is **reinforcement learning** for traffic signals.

In reinforcement learning, an AI learns by trial and error inside a simulator. It tries signal timings,
sees the result, and gets rewarded for good outcomes.

We built a simulation of a real Bengaluru route on the **Outer Ring Road**: Silk Board, Bellandur,
Marathahalli and KR Puram, using the real approximate distances between junctions. The agent
controls the signals. It is rewarded when the ambulance waits less.

But there is one important design point. If we only rewarded the ambulance, the AI would learn the
lazy answer: 'always green for the ambulance and stop everyone else.' So we also
penalise delay to other vehicles. That is why the problem is meaningful.

We compare against normal fixed-timer signals, and we report both the ambulance's travel time and the
effect on other traffic, even if the trade-off is not perfect.

Current status: the simulated corridor and the fixed-timer baseline are built and running. Training the
AI and producing the results is the next step. Police alerts remain simple rule-based logic, which we say openly."

---

## 8. Hospital selection and coordinator (9:30–10:30)
`[Slide: hospital score factors + conflict example]`

"The fourth part chooses the hospital. Instead of 'nearest', it scores hospitals on distance,
trauma capability, free beds, specialty match, and **blood availability**. It uses the risk score,
so a critical patient is matched with a hospital that can really handle them.

On blood: India already has a national system called **eRaktKosh**. We do not rebuild it. We use blood
availability as one more factor in the choice, and where live data is not accessible we use a clearly labelled
simulated file.

Finally, the **coordinator** puts everything together. If two agents disagree, for example the
best route crosses a police cordon, or the top hospital has no free beds, it resolves the conflict with
a weighted priority scheme. This part is **rule-based on purpose**, and we say that openly. A
learned coordinator is future work."

---

## 9. Novelty (10:30–11:30)
`[Slide: "What is not new" / "What is our contribution"]`

"Let me be clear about novelty, because we do not want to overclaim.

Combining text and vitals for triage, predicting deterioration from vitals, and using reinforcement
learning for traffic signals are all **established research**. Examples include DeepTriager, VitalML,
SIGMA and EMVLight. We cite them and we build on them.

Our contribution is the **context and integration**:
- first, the **pre-arrival** setting, from the ambulance to the hospital, rather than triage inside a hospital;
- second, **multilingual voice input** for Kannada, Hindi and English;
- third, **safety-checked first-aid guidance**, where we measure how grounded the answers are;
- fourth, reinforcement learning on a **named Bengaluru corridor**;
- and fifth, one **connected pipeline** with proper baselines, so every claim is tested.

In one sentence: we apply established ideas to a real gap in Bengaluru's emergency pathway
and evaluate the whole system honestly."

---

## 10. Tools and data (11:30–12:30)
`[Slide: tech stack table]`

"For tools: the backend is **Python with FastAPI**, with live updates over WebSockets. Speech uses
**Whisper**. First-aid search uses **LangChain with a vector database, FAISS**. The vitals and fusion
models use **PyTorch**, with scikit-learn for the baselines. Traffic uses the **SUMO simulator with
Stable-Baselines3**. Routing uses **OpenRouteService**, and the map uses Leaflet with OpenStreetMap.

For data: vitals come from **MIMIC**, the public ICU database. The first-aid library comes from public
guidelines such as the Red Cross. Traffic uses real junction distances with generated demand.
Hospitals are a real Bengaluru list, and bed and blood numbers are simulated and labelled."

---

## 11. How we are building it and where we stand (12:30–13:30)
`[Slide: status table + timeline]`

"On status: we already have a **working prototype** with a live map, small ML models trained on
synthetic data, a rule-based coordinator, a benchmark on eight Bengaluru scenarios, and the traffic
simulation with a fixed-timer baseline.

What we are building now is the real-data vitals model, fusion, the RL training, the first-aid guidance
system, voice input, and the blood factor.

One honest result from our current benchmark: our AI dispatch was faster in three of the eight
scenarios and slower in five. The main benefit so far is a **smarter hospital choice**, not a
faster arrival, because our baseline is deliberately optimistic. For the final version we are redesigning the
comparison to be fair, including whether the patient reaches a hospital that can actually treat them.

Our plan runs over eight weeks: data and setup, then the vitals model, the RL and the first-aid guidance,
then fusion, voice and the interface, and finally explainability, the report and the demo. At each step we
test against a simple baseline before moving on."

---

## 12. Limitations and close (13:30–14:30)
`[Slide: limitations + one-sentence summary]`

"Some limitations we want to say ourselves. The vitals data is ICU data, not ambulance data, and the free
version is small. Part of the fusion data is synthetic. The traffic results come from simulation, not real roads.
Hospital beds and blood stock are simulated. And this is a **research prototype, not a medical device**.

To summarise: emergency response loses time in reporting, patient information, traffic and hospital choice. We are
building one connected AI system for Bengaluru that addresses all four, built on established methods, tested honestly against
clinical and rule-based baselines, and aimed at the exact gap the state's own 108 programme lists as its next step.

Thank you. We are happy to take questions."

---

## Quick reference for questions (keep this open)

| If asked | Answer in one line |
|---|---|
| What's new? | Context and integration, not architecture (Section 9) |
| Is the data real? | Vitals: real ICU data. Text pairs, beds, blood: simulated and labelled |
| Why a GRU? | Deterioration is about the trend over time, but we compare with simpler models and report the winner |
| Is it safe to give medical advice? | Only from a trusted library, refuses if not covered, no medicine advice |
| Why not always-green for the ambulance? | It would block other traffic, so the reward penalises that |
| Can it be deployed? | Not yet; it needs clinical validation, approval and partnership with 108 |
| Which parts are AI and which are rules? | AI: vitals, fusion, RL, text classifier, RAG. Rules: coordinator, hospital formula, police sizing, routing |
| Who did what? | *(fill in your team split)* |

## Delivery tips
- Practise with a timer; aim to finish at 13–14 minutes so questions have room.
- Do not read the slides; the slides show the diagram, you tell the story.
- Say "planned" and "built" clearly. The honesty is what makes it credible.
- If you forget a detail, fall back to the one-sentence version in Section 12.
