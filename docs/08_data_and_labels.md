# 08 — Data, Labels, and Synthetic-Data Rules

**Purpose:** for every model in the project: what data, where from, what the label is, what is
real vs synthetic. Also resolves the biggest methodological gaps in the earlier docs.

Legend: **REAL** = collected from a real source. **SIM** = simulated/generated (must be labelled as such in every result).
`[TODO-VERIFY]` = confirm access/details before relying on it.

---

## 1. Master data table

| Model / component | Data source | Label | Real/Sim | Approx size | Notes |
|---|---|---|---|---|---|
| Vitals severity + trend (BiGRU) | MIMIC-III Clinical Database Demo (PhysioNet) `[TODO-VERIFY access terms]` | Outcome-based, §2 | REAL (ICU) | ~100 patients | Primary source. Tables: CHARTEVENTS, D_ITEMS, ADMISSIONS, ICUSTAYS, PATIENTS, plus INPUTEVENTS/PROCEDUREEVENTS for interventions |
| Vitals tabular baselines (LR/RF/GB) | Same windows, last-step features only | Same as above | REAL | same | Baselines must use identical patient split |
| Vitals snapshot supplement | Kaggle "Human Vital Signs Dataset" | Its `Risk Category` | SIM (clinically structured) | — | Snapshot task only, cannot support trend. Never mix with MIMIC in one test set |
| NEWS2 baseline | Computed from the same windows | — (predictor, not label) | — | — | Predicts a class via banding in §3 |
| Text severity classifier | Team-built incident-report set, §4 | 4-class by written rubric | SIM | 400–600 reports | EN/HI/KN |
| Fusion model | MIMIC windows paired with **synthetic** text features, §5 | Same outcome label as vitals | SIM pairing | same as MIMIC | Ablation caveat mandatory |
| Whisper ASR eval | Team-recorded utterances | Reference transcript | REAL (recorded) | ≥30 per language | Metric: WER |
| RAG corpus | WHO, Red Cross, public trauma triage guidance | — | REAL docs | 20–60 pages | Scope: bleeding, CPR, fractures, shock, burns |
| RAG eval set | Team-written first-aid queries + expected source doc | expected source(s) | SIM | 30–50 queries | Metrics: retrieval hit@k, groundedness % |
| Traffic delay/congestion RF | Bengaluru open traffic/accident data `[TODO-VERIFY]`: candidates = Karnataka/OGD India open data, Bengaluru traffic police published reports | delay / congestion level | REAL if found, else SIM (labelled) | — | Retrain; document exact source |
| ETA model (GradientBoosting) | ORS route durations sampled across times of day + traffic features | actual/simulated travel time | SIM-derived | 5–20k rows | State honestly that time-of-day multipliers are modelled, not measured |
| RL corridor | SUMO network from real inter-junction distances; demand per §6 | reward | SIM | — | Approximated geometry |
| Hospital scoring | Bengaluru hospital directory/trauma listings `[TODO-VERIFY]` | — | REAL attributes + SIM beds/blood | 15–30 hospitals | See §7 for schema |
| Hospital RF Q-value | Simulated dispatch episodes | Realised reward, §7.3 | SIM | 5–20k episodes | Surrogate, see caveat |
| Blood stock | eRaktKosh-style; live only if accessible `[TODO-VERIFY]` | — | SIM (default) | per hospital × 8 groups | Document which was used |

## 2. Vitals label definitions (MIMIC) `[DECISION]`

**Unit of learning:** a *window* = up to 6 consecutive charted readings for one ICU stay ending at time `t`. Hourly resolution is typical in CHARTEVENTS; readings are aggregated to hourly bins (median per hour per parameter).

**Outcome events (all independent of NEWS2 thresholds):**
- `death_24h`: in-hospital death (ADMISSIONS.DEATHTIME / HOSPITAL_EXPIRE_FLAG) within 24 h after `t`.
- `intervention_6h`: start of vasopressor infusion or invasive mechanical ventilation within 6 h after `t` (from INPUTEVENTS_* / PROCEDUREEVENTS_MV / ventilator chart items).
- `icu_los_remaining_h`: remaining ICU length of stay from `t` to ICU out-time.

**Severity class at `t` (label for the severity head):**

| Class | Rule (first match, top to bottom) |
|---|---|
| 3 critical | `death_24h` OR `intervention_6h` |
| 2 severe | `icu_los_remaining_h` > 72 |
| 1 moderate | `icu_los_remaining_h` between 24 and 72 |
| 0 stable | `icu_los_remaining_h` < 24 and none of the above |

**Trend label (trend head):** `trend = 1` if a **deterioration event** occurs within the next **H = 6 readings (≈6 h)**: `death` or `intervention_6h`-type start (vasopressor/ventilation) *newly starting after `t`* in that horizon, otherwise 0. Exclude windows where the patient was already on vasopressor/ventilation at `t` from the trend task (already deteriorated).

**Caveats to state in the report:**
- Thresholds (24 h / 72 h) are a team convention; check class balance first. If any class < 5% of windows, merge classes (fallback: 3-class, then binary critical vs non-critical) and update `07` D2.
- The claim "10–15 minutes ahead" is **not** supported by hourly MIMIC data. State the horizon as "next 6 readings". Demo replays a compressed timeline.
- MIMIC = in-hospital ICU patients. Pre-hospital ambulance patients differ → domain-shift limitation.
- ~100 patients → wide CV variance; report mean ± std; do not over-claim.

**Feature mapping from MIMIC item IDs:** build a `d_items` lookup for HR, RR, SpO2, systolic BP (arterial and non-invasive variants), temperature (convert °F → °C where needed), GCS components, supplemental O2 / FiO2. Save mapping in `data/mimic_itemid_map.json`. `[TODO-VERIFY item IDs against D_ITEMS; CareVue vs MetaVision differ]`.

**GCS → AVPU mapping `[DECISION]`:** total GCS 15 → `A`; GCS 13–14 → `V`; GCS 9–12 → `P`; GCS ≤ 8 → `U`. Model input `avpu_bin = 0 if A else 1`. NEWS2 scoring: `A = 0`, `V/P/U = 3` (also 3 for new confusion; not available in MIMIC, note as limitation).

## 3. NEWS2 baseline output → class
NEWS2 scoring per `02_ambulance_vitals_agent.md` §3. Class banding used here is a **project convention**, not official NEWS2 banding: total ≥7 → 3, 5–6 → 2, 1–4 → 1, 0 → 0. Official NEWS2 clinical response tiers (low / low-medium / medium / high) differ; mention this in the report. The NEWS2 baseline only predicts *severity*; it has no native trend output.

## 4. Text severity dataset (Component 1) `[DECISION]`
- Build 400–600 short incident reports across EN/HI/KN (≈150 each). Generate candidates with an LLM from templates (injury type × severity × phrasing style × noise like panic, incomplete sentences), then **manually review every sample** and fix.
- Label with a written rubric: 3 = unconscious/not breathing/heavy uncontrolled bleeding/major trauma; 2 = conscious but serious injury or heavy bleeding controllable; 1 = minor-moderate injury, stable; 0 = no injury/low concern.
- Keep a held-out test set of 20% never used for prompt or rubric tuning.
- Tag as **SIM**. Kannada/Hindi reports: have native-speaker teammates review.
- Limitation: not labelled by clinicians.

## 5. Building fusion training pairs (text ↔ vitals) `[DECISION]`
MIMIC contains no bystander reports. For each MIMIC window with severity label `y`:
1. Sample text features from a **class-conditional generator** using a confusion matrix, e.g. with prob 0.65 draw features typical of class `y`, 0.25 of an adjacent class, 0.10 of a random class (adjust; document the values).
2. Draw the dialogue answers (conscious, bleeding, etc.) from class-conditional probabilities defined in a table you write down in `data/text_feature_generator.yaml`.
3. **Independence rule:** text features are generated from the label only, *not* from the vitals values, so text carries information independent of vitals (otherwise fusion gain is fake).
4. Set text-generator noise levels **before** looking at fusion results; do not tune them to make fusion win.
5. Report the ablation (text-only / vitals-only / fused) with the mandatory caveat: *"Text modality is synthetic; the fusion gain reflects the assumed informativeness of text and demonstrates the pipeline, not clinical performance."*
6. Additionally report a sensitivity table: fusion gain vs. text agreement (0.5, 0.65, 0.8).

## 6. RL traffic demand `[DECISION]`
- Main ORR line (each direction): 900–1500 veh/h, Poisson arrivals, mixed vehicle types (car 60%, two-wheeler 30%, bus/truck 10%).
- Cross streets: 300–600 veh/h.
- Run 3 demand levels (low / medium / peak) for evaluation.
- Ambulance: spawn at `entry_west` at random offsets, one per evaluation episode; `vClass=emergency`, elevated speed factor/impatience.
- Episode length: enough for the ambulance to clear the corridor (≥ 1800 s sim time).

## 7. Hospital data

### 7.1 Hospitals table (`data/hospitals.csv`)
`hospital_id, name, lat, lon, trauma_level (0–3), beds_total, beds_available, icu_beds_available, specialties (semicolon list), open_24x7, blood_A+, blood_A-, blood_B+, blood_B-, blood_O+, blood_O-, blood_AB+, blood_AB-` (units).

### 7.2 Scoring formula `[DECISION — reconcile with existing hospital_agent.py; if the code differs, code wins and this table must be updated]`
```
policy_score = 0.30·proximity + 0.25·trauma_match + 0.20·bed_avail
             + 0.10·specialty_match + 0.15·blood_match
total_score  = 0.8·policy_score + 0.2·rf_qvalue
```
- `proximity = 1 − min(eta_min / 30, 1)`
- `trauma_match = 1` if hospital trauma level ≥ required level for `severity_class`, else scaled down (≥ level−1 → 0.5, else 0)
- `bed_avail = min(beds_available / 10, 1)`
- `blood_match`: only weighted when transfusion risk is flagged (bleeding heavy OR severity ≥ 2 with trauma injury type); otherwise the weight redistributes to proximity. 1 if required group stock ≥ 4 units, 0.5 if 1–3, 0 if 0; `unknown` group → O-negative stock used as proxy.

### 7.3 RF Q-value target `[DECISION]`
There is no real "best hospital" ground truth. The RF is trained on **simulated dispatch episodes** with realised reward:
`reward = −eta_min − 15·[trauma_mismatch] − 20·[no_bed] − 10·[blood_short_when_needed]`
The RF learns to predict that reward from (distance, trauma level, beds, specialty match, severity, blood flag). State plainly: it is a surrogate trained on a hand-defined reward, not on clinical outcomes. If this feels weak in review, drop the RF blend and report the rule-based score alone plus an ablation.

### 7.4 Benchmark scenarios
Extend the existing 8 scenarios in `benchmark.py` with columns: `scenario_id, location, injury_type, severity_class, blood_need, expected_best_hospital_rule`. Add 2 scenarios where two hospitals are equally close and only one has matching blood.

## 8. Dataset provenance log
Maintain `data/PROVENANCE.md` with one entry per dataset: source URL, access date, licence, preprocessing steps, real/sim flag. Do not redistribute restricted data in the repo.

## 9. Data-leakage checklist
- [ ] Patient-level splits (train/val/test), same split reused across all baselines
- [ ] Normalisation stats from train folds only
- [ ] No NEWS2-derived labels
- [ ] Trend-window features use only readings ≤ `t`
- [ ] Text generator not conditioned on vitals values
- [ ] Held-out text test set untouched during prompt/rubric tuning
