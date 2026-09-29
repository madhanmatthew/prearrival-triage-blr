# 15-Minute Presentation Script
## Pre-Arrival Multimodal AI Triage & Emergency Coordination System

### PPT flow

1. **Title / Introduction**
2. **The Emergency Gap**
3. **System Architecture**
4. **Core ML Architecture**
5. **Traffic & Hospital Selection**
6. **Current Status & Evaluation**
7. **Novelty, Limitations & Conclusion**

---

# SLIDE 1 — Introduction
### Time: 0:00–1:15

**PPT:** *Pre-Arrival Multimodal AI Triage & Emergency Coordination System*

### Say:

> Good morning sir/ma'am.
>
> Our project is titled **Pre-Arrival Multimodal AI Triage and Emergency Coordination System for Bengaluru**.
>
> The main idea behind this project is very simple:
>
> **When a road accident happens, the problem is not only reaching the patient. We also need to understand the patient's condition, reduce the ambulance's delay, and make sure the ambulance reaches a hospital that can actually treat the patient.**
>
> So our system tries to connect these different parts of emergency response into one pipeline.
>
> We have four major components.
>
> First, **multilingual emergency reporting**, where a bystander can communicate in Kannada, Hindi or English.
>
> Second, a **vitals-based AI model** that monitors the patient's condition during transport.
>
> Third, **reinforcement learning for traffic signal control**, so that the ambulance can move through a simulated Bengaluru corridor with reduced delay.
>
> And fourth, **hospital selection**, where we consider factors such as trauma capability, available beds and blood availability rather than simply selecting the nearest hospital.
>
> The important word here is **pre-arrival**.
>
> We are trying to make useful decisions **before the ambulance reaches the hospital**, instead of waiting until the patient arrives.

### Transition:

> So before explaining how our system works, let me first explain the problem we are trying to solve.

---

# SLIDE 2 — The Emergency Gap
### Time: 1:15–2:45

**PPT:** *4 Critical Failure Points in Urban Pre-Arrival Transit*

### Say:

> We identified **four major failure points** in the current pre-arrival emergency process.
>
> **The first is reporting delay.**
>
> During an accident, the person reporting the incident may be panicking. They may not know medical terminology, and there can also be language barriers.
>
> So instead of getting structured information about the patient, the emergency service may initially receive incomplete information.
>
> **The second problem is what we call ER blindness in transit.**
>
> Once the ambulance picks up the patient, the receiving hospital may not have a continuous picture of how the patient's condition is changing.
>
> A patient may look relatively stable at one point and then deteriorate during transport.
>
> **Third is corridor congestion.**
>
> Normal traffic signals operate according to their existing timing or traffic-control policies. An ambulance can therefore get stuck in the same traffic queues as other vehicles.
>
> And finally, there is the **nearest-hospital problem**.
>
> The nearest hospital is not necessarily the most suitable hospital.
>
> For example, a patient with serious trauma may require a hospital with appropriate trauma capability, available beds, and potentially the required blood supply.
>
> So these four problems give us the structure for our solution:
>
> **reporting, patient condition, road movement, and hospital selection.**
>
> Our system connects all four.

### Transition:

> Now I'll show how we connect these four components into one system.

---

# SLIDE 3 — System Architecture
### Time: 2:45–4:30

**PPT:** *Connected Multimodal AI Triage & Dispatch Pipeline*

### Say:

> This slide shows the complete architecture of our system.
>
> We have **five stages**.
>
> The first is the **Bystander Report**.
>
> The bystander provides information through voice or text. For voice input, we use Whisper for speech recognition. The system then asks guided questions and produces a text-based severity estimate.
>
> The second stage is the **Vitals Stream**.
>
> Once the patient is inside the ambulance, vitals such as heart rate, respiratory rate, oxygen saturation, blood pressure and other measurements become available.
>
> Our time-series model looks at these measurements over time rather than treating each measurement independently.
>
> The third stage is **Risk Fusion**.
>
> This is where the information from the report and the information from the patient's vitals are combined.
>
> Instead of having two completely separate assessments, we create a **unified emergency risk score** that can update as new information arrives.
>
> The fourth stage is **Traffic RL**.
>
> Based on the ambulance's route, the traffic component decides which signal phase should be prioritised in the simulation.
>
> Finally, we have **Hospital Dispatch**.
>
> The hospital selection considers factors such as trauma capability, free beds, distance and blood availability.
>
> At the bottom we have the **System Coordinator**.
>
> Its job is not currently another AI model. It is a rule-based coordinator that resolves conflicts between the different components.
>
> For example, if the fastest route conflicts with a police restriction, or if the preferred hospital does not have an available bed, the coordinator resolves that conflict and produces the final plan.

### Good line to emphasize:

> **So the novelty is not that every individual component is new. The important part is connecting them into one pre-arrival emergency pipeline.**

### Transition:

> Among these components, the most important machine-learning component is the patient's vitals model.

---

# SLIDE 4 — Core ML Architecture
### Time: 4:30–7:15

**PPT:** *GRU Time-Series Modeling & Multimodal Risk Fusion*

### Say:

> This is the core machine-learning part of our project.
>
> The main question here is:
>
> **How do we determine whether a patient is currently serious and whether their condition is getting worse?**
>
> We use a **GRU, or Gated Recurrent Unit**, because patient vitals are naturally a time-series problem.
>
> We don't just want to look at one blood-pressure reading.
>
> For example, suppose a patient's blood pressure is currently 100.
>
> If it was 100 consistently, that tells us one thing.
>
> But if it was 130 earlier and has gradually dropped to 100, that trend can indicate something different.
>
> So the model receives a sequence of recent readings rather than a single snapshot.
>
> The model uses a **bidirectional GRU encoder**.
>
> The input contains measurements such as:
>
> - heart rate,
> - respiratory rate,
> - SpO2,
> - blood pressure,
> - temperature,
> - consciousness level,
> - and oxygen usage.
>
> We also include changes between readings and missing-value indicators.
>
> The GRU converts this sequence into a representation of the patient's condition.
>
> From this shared representation, we have **two output heads**.
>
> The first output estimates the **current severity** of the patient.
>
> The second estimates the **probability of deterioration**.
>
> This is a form of **multi-task learning**, because both tasks share the same GRU representation.
>
> We don't simply build a GRU and claim that it is better.
>
> We compare it against several baselines.
>
> One important baseline is **NEWS2**, which is a standard clinical early-warning scoring system.
>
> We also compare against simpler machine-learning approaches such as logistic regression and Random Forest or Gradient Boosting.
>
> This is important because if a complex neural network cannot provide an improvement over a simpler method or the clinical baseline, then there is no strong reason to use the complex model.
>
> For training, the intended real-data source is **MIMIC**, which is a de-identified ICU database.
>
> One important issue is that MIMIC is not ambulance data.
>
> Its measurements are generally much more widely spaced than real-time ambulance measurements.
>
> So we have to be careful with the meaning of "deterioration soon."
>
> We cannot honestly claim that the current dataset proves a literal 10 or 15-minute deterioration prediction.
>
> Instead, the model is evaluated based on future readings and future outcomes available in the dataset.
>
> Since this is a healthcare-related system, another important part is **explainability**.
>
> We don't want the model to simply output a risk score without giving us an indication of what contributed to that prediction.
>
> For the models we use techniques such as **SHAP and Integrated Gradients** to investigate the important inputs.
>
> The next question is: what happens to the information from the bystander?
>
> This is where **multimodal fusion** comes in.
>
> The reporting system gives us a text-based severity probability.
>
> The vitals model gives us information from the physiological measurements.
>
> We combine these into one **Emergency Risk Score**.
>
> At the beginning, we may only have the bystander's report, so the risk score is based on the text information.
>
> As soon as vitals start arriving, the system updates the score.
>
> So the assessment is not static.
>
> It evolves as more information becomes available.

### Strong explanation if faculty asks "Why fusion?"

> **Text tells us what happened. Vitals tell us what is happening to the patient. Fusion combines both perspectives.**

### Transition:

> Once we know how serious the patient is, the next problem is making sure the ambulance can actually reach the destination efficiently.

---

# SLIDE 5 — Traffic & Hospital Selection
### Time: 7:15–9:45

### Part A — Traffic RL

> The third component is **reinforcement learning for traffic signal control**.
>
> We use the **SUMO traffic simulator** to model an approximate Bengaluru Outer Ring Road corridor:
>
> **Silk Board → Bellandur → Marathahalli → KR Puram.**
>
> Instead of directly controlling real traffic signals, which would require government authority and safety approval, we first test the idea inside a simulator.
>
> The reinforcement-learning agent observes things such as traffic queues, the current signal phase and the ambulance's position.
>
> It then chooses the next signal phase.
>
> The important part is the **reward function**.
>
> If we simply rewarded the system for reducing ambulance travel time, the easiest solution would be to keep the ambulance direction green all the time.
>
> But that would create a huge problem for everyone else.
>
> So our reward has two objectives:
>
> **reduce ambulance waiting while also penalising excessive delay to other traffic.**
>
> This makes the optimisation more realistic.
>
> We compare against normal fixed-timer signals, and we report both the ambulance's travel time and the effect on other traffic, even if the trade-off is not perfect.
>
> Current status: the simulated corridor and the fixed-timer baseline are built and running. Training the AI and producing the results is the next step. Police alerts remain simple rule-based logic, which we say openly.

### Part B — Hospital Selection

> The fourth component is hospital selection.
>
> Instead of using a simple "nearest hospital" rule, we consider multiple factors.
>
> These include:
>
> **distance, trauma capability, free beds, specialty requirements and blood availability.**
>
> The system can therefore make a decision such as:
>
> "This hospital is slightly farther away, but it has the required trauma capability and available resources."
>
> For blood availability, the project is designed to use **eRaktKosh** information where available. If live information is unavailable, we use clearly labelled simulated data.
>
> The important point is that the hospital model is currently a **decision policy**, not a model trained on real historical outcomes, because there is no public dataset telling us which hospital was objectively the correct destination for each emergency case.
>
> Finally, the coordinator connects these decisions.
>
> Suppose the traffic system suggests one route, but that route has a police restriction.
>
> Or suppose the highest-ranked hospital no longer has a free bed.
>
> The coordinator resolves these conflicts using weighted priorities.
>
> At the moment, this coordinator is deliberately **rule-based**, rather than learned.

### Transition:

> So that is the complete pipeline. The next question is: how much of this system is actually implemented today?

---

# SLIDE 6 — Current Build Status & Evaluation
### Time: 9:45–11:45

### Say:

> This slide shows the current status of the project.
>
> We have separated the project into what is already implemented and what is part of the final ML research build.
>
> On the **prototype side**, we already have a FastAPI-based system, WebSocket streaming, the SUMO Bengaluru corridor setup, fixed-time traffic control and an initial benchmark.
>
> We also have prototype ML models for components such as severity, hospital selection and ETA, but these current models are trained using synthetic or formula-generated data.
>
> The more research-oriented components are shown as the **final ML engine**.
>
> These include the MIMIC-based GRU model, multilingual Whisper and RAG pipeline, multimodal fusion, DQN traffic training and the blood-availability factor.
>
> We also performed an initial benchmark across eight Bengaluru scenarios.
>
> One result we want to be transparent about is that the AI dispatch was faster in **3 out of 8 scenarios and slower in 5**.
>
> The average ETA was also higher than the baseline.
>
> So we are **not claiming that the current prototype is already faster than conventional dispatch**.
>
> In fact, the current result suggests that the demonstrated benefit is more about **smarter hospital selection** than faster ETA.
>
> One reason is that our current baseline is deliberately optimistic, while the AI pipeline includes additional overhead and conflict penalties.
>
> Therefore, the final evaluation needs a fairer baseline before making a performance claim.

### Important sentence:

> **"We are presenting the current result as a prototype evaluation, not as proof of clinical or real-world superiority."**

---

# SLIDE 7 — Novelty, Limitations & Conclusion
### Time: 11:45–14:30

## Novelty

> Now coming to the contribution of the project.
>
> We don't claim that GRUs, RAG, multimodal fusion or reinforcement learning are individually new.
>
> These are established techniques.
>
> Our contribution is primarily in **the application and integration of these techniques in the pre-arrival emergency setting**.
>
> There are five main aspects.
>
> First, we focus on the **pre-arrival stage**, from the accident and ambulance to the hospital.
>
> Second, we support **Kannada, Hindi and English** reporting.
>
> Third, we use **safety-checked RAG** for first-aid guidance rather than allowing an LLM to freely generate medical instructions.
>
> Fourth, we apply traffic reinforcement learning to a **Bengaluru-specific simulated corridor**.
>
> And fifth, we connect reporting, patient risk, traffic and hospital selection into one integrated pipeline.

## Limitations

> At the same time, there are important limitations.
>
> First, the MIMIC data is **ICU data**, not actual ambulance data.
>
> Second, the speech-vitals paired data required for multimodal fusion is partly semi-synthetic because there is no public dataset containing those exact pairs.
>
> Third, our traffic environment is a **SUMO simulation**, not live Bengaluru traffic.
>
> Fourth, hospital bed and blood availability are currently simulated where live data is unavailable.
>
> And finally, this is an **academic research prototype, not a clinically validated medical device**.
>
> Real deployment would require clinical validation, regulatory approval, partnerships with emergency services and prospective testing.

---

# FINAL CONCLUSION
### ~14:00–14:45

> So, to conclude:
>
> The problem we identified is that emergency response can lose valuable time at four different stages:
>
> **reporting, understanding the patient's condition, travelling through traffic, and selecting the destination hospital.**
>
> Our project proposes one connected AI pipeline to address these four stages before the patient reaches the hospital.
>
> The key machine-learning component is the **time-series vitals model**, which looks not only at the patient's current state but also at how that state is changing.
>
> The multimodal fusion then combines that information with the initial emergency report.
>
> Reinforcement learning addresses the traffic component, while the hospital policy considers whether the destination can actually handle the patient.
>
> Most importantly, we are separating what is already built from what is still being developed, and we are evaluating each component against appropriate baselines rather than assuming that AI automatically performs better.
>
> **Our goal is not to replace emergency professionals. It is to provide better information and coordination before the patient reaches the hospital.**
>
> Thank you. We are happy to take questions.

---

# 15-Minute Timing

| PPT Slide | Topic | Time |
|---|---|---:|
| **1** | Introduction | 1:15 |
| **2** | Emergency Gap | 1:30 |
| **3** | Architecture | 1:45 |
| **4** | GRU + Fusion | 2:45 |
| **5** | Traffic RL + Hospital | 2:30 |
| **6** | Status + Evaluation | 2:00 |
| **7** | Novelty + Limitations + Conclusion | 2:30 |
| | **Total** | **14:15–14:45** |

---

# The 5 Things You Absolutely Need to Understand

If your HOD/faculty starts asking technical questions, these are the parts you should be able to explain without memorizing the script:

1. **Why GRU?**  
   Because vitals are sequential; the trend matters, not just the current value.

2. **Why fusion?**  
   Bystander report tells us about the incident; vitals tell us about the patient's physiological state.

3. **Why RAG?**  
   To constrain first-aid answers to a trusted knowledge base and allow refusal instead of allowing an LLM to freely invent medical advice.

4. **Why RL for traffic?**  
   Because signal control is a sequential decision problem where actions affect future traffic states.

5. **What is actually built vs planned?**  
   **Prototype:** FastAPI, WebSockets, rule-based coordination, existing synthetic-data ML models, SUMO corridor and fixed-time baseline.  
   **Final/research build:** GRU, RAG/Whisper, fusion, DQN training and real-data retraining.

---

# One Sentence to Memorize

> **"We are building a pre-arrival emergency coordination layer that combines the initial incident report, live patient condition, traffic conditions and hospital resources to make better decisions before the patient reaches the hospital."**
