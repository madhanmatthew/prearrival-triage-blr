# Results summary (for the report, slides and viva)

Compiled 2026-10-22 from `experiments/log.csv` and `reports/`. **Every number below is a logged run**; the
source column names the log `model` (+ date) or the report file. Simulation and synthetic results are
labelled as such. Research prototype, not a medical device (AGENTS rule 14).

Novelty framing (AGENTS §1, `05` §8): *context + integration* — pre-arrival, multilingual (en/hi/kn),
low-resource, Bengaluru — **not** a new architecture. Never "first-ever".

---

## 1. Voice intake (ASR) — real team recordings

**Dev set:** 90 clips (30 per language), language forced, repetition penalty 1.1, GPU (RTX 3050 4 GB).

| Model | en WER / CER | hi WER / CER | kn WER / CER | Source |
|---|---|---|---|---|
| Whisper small (no guard) | 3.6 % / 1.9 % | 51.7 % / 31.6 % | 105.9 % / 130.6 % (loops) | `faster-whisper-small` 2026-10-07 |
| Whisper medium | 4.1 / 1.8 | 40.5 / 20.1 | 103.1 / 133.1 (loops) | `faster-whisper-medium` 2026-10-07 |
| Whisper large-v3 (VAD on) | **3.6 / 1.7** | 30.5 / 15.9 | 80.4 / 23.9 | `faster-whisper-large-v3` 2026-10-17 |
| Whisper large-v3-turbo (VAD on) | 3.6 / 1.7 | 33.6 / 16.5 | 82.3 / 24.6 | `faster-whisper-large-v3-turbo` 2026-10-17 |
| Whisper large-v3 (VAD off) | — | 30.8 / 15.7 | 80.1 / 24.2 | 2026-10-22 rows |
| **Hindi fine-tune** (vasista22 whisper-hindi-medium, VAD off) | — | **25.7 / 11.7** | — | `faster-whisper-whisper-hindi-medium` 2026-10-22 |
| **Kannada fine-tune** (vasista22 whisper-kannada-medium, VAD off) | — | — | **47.8 / 15.4** | `faster-whisper-whisper-kannada-medium` 2026-10-22 |

* The Kannada fine-tune with VAD **on** was 62.4 % / 35.7 %: VAD cut sentence endings (e.g. "not breathing").
  VAD is now configurable per language (`WHISPER_VAD_<LANG>`). CER is the fairer metric for kn/hi (code-mixed
  spelling, D18).
* **Held-out** (10 new clips per language, recorded after the lexicon was written; `reports/asr_wer_heldout_*`):

| | en | hi | kn |
|---|---|---|---|
| large-v3-turbo, VAD on (WER / CER) | 6.9 % / 3.2 % | 33.5 % / 29.1 % | 84.3 % / 48.4 % |
| deployed per-language models, VAD off | 8.3 % / 4.5 % (turbo) | **24.3 %** / 26.6 % | **58.0 %** / 44.2 % |

* **Latency** (median real-time factor, 10 clips, GPU; `asr_latency` 2026-10-17): turbo en 0.09 / hi 0.12 /
  kn 0.18 (~0.9–1.4 s per ~9 s clip); large-v3 0.14 / 0.30 / 0.51. CPU large-v3 ≈ 57 s per clip → live intake
  needs the GPU (D18).
* **Noise** (`asr_noise`, turbo, noise mixed into real speech, synthetic): en stays ≤ 4.7 % WER down to 5 dB SNR;
  hi 34 % → 47 %; kn 85 % → 99 % (clean → 5 dB, VAD off); at 0 dB en 7.3 %, hi 57 %, kn ~100 %.

## 2. Dialogue robustness (ASR → triage fields)

All-field agreement of the 6-question cue extraction from the ASR transcript vs the English reference of the
same scenario (`asr_triage`):

| Split | en | hi | kn | Note |
|---|---|---|---|---|
| Dev (90 clips; lexicon written on them) | 1.00 | 0.83 | 0.83 | 2026-10-08 |
| **Held-out**, turbo VAD on | 1.00 | 0.50 | 0.60 | 2026-10-22 (other session) |
| **Held-out**, deployed fine-tunes | 1.00 | 0.60 | 0.60 | `reports/asr_triage_summary_heldout_finetuned.csv` |

Honest reading: the dev 83 % was optimistic; the clean held-out number for hi/kn is ~60 %. Tap buttons
(yes/no/don't know) make the six answers independent of ASR quality; the gap is in free-text report parsing.

## 2b. Text severity classifier (SIM data, D2)

450 template-drafted bystander reports (150 per language, balanced over 4 classes), all human-reviewed
(1 relabel, 0 text edits), fixed 20 % held-out test split (90 reports): **macro-F1 0.978** (en 1.00, hi 1.00,
kn 0.93). Honest reading: the test set shares the templates, so this mostly measures template recovery. On the
30 **real** held-out recordings (native speakers' own words) the predictions are inconsistent (mean confidence
0.5; e.g. "rider not responding" → class 1 in en/hi). In the deployed pipeline the fusion network also uses the
dialogue answers (conscious, breathing, bleeding, ...), which lifts clear cases correctly (end-to-end check:
critical → class 3 at 0.74, moderate → 1, stable → 0). Limitation: not clinician-labelled, template-based.

## 3. First-aid guidance (RAG, IFRC 2020 corpus, 18 topics)

| Version | Test queries | hit@k | answer rate (in scope) | refusal rate (out of scope) | fully grounded | Source |
|---|---|---|---|---|---|---|
| v1 extractive | 40 | 0.900 | 0.933 | 0.600 | 100 % | `rag_extractive` 2026-10-06 (1st) |
| **v2** (guard + routing + bge-small) | 40 | **0.964** | **1.000** | 0.667 | **100 %** | `rag_extractive` 2026-10-06 (2nd) |

Every sentence shown comes from retrieved corpus text (rule 6); 0 medication sentences removed.

## 4. Vitals models

### 4a. MIMIC-IV Demo (ICU, hourly; nested patient-grouped CV 5×3; 12,299 windows)

| Model | Severity macro-F1 (4 classes) | Trend PR-AUC | Trend AUROC |
|---|---|---|---|
| Majority | 0.164 ± 0.005 | 0.028 | 0.500 |
| NEWS2 | 0.278 ± 0.029 | 0.061 | 0.622 |
| Logistic regression | 0.352 ± 0.034 | 0.158 | 0.750 |
| Random forest | 0.368 ± 0.037 | 0.112 | 0.712 |
| Gradient boosting | 0.368 ± 0.039 | 0.094 | 0.694 |
| **BiGRU two-head** | **0.386 ± 0.037** | **0.162 ± 0.074** | 0.708 |

Source: `vitals` rows 2026-09-30. Labels from real outcomes, never NEWS2 (rule 1). Limits: 100 patients,
ICU not pre-hospital.

**Explainability** (D13; models fit on all windows, explanations not performance; `reports/xai_shap_*.csv`,
`reports/figures/xai_shap_*.png`, IG figures for the BiGRU in `reports/figures/`): SHAP top features — severity:
RF avpu_bin, supplemental_o2, temp, spo2, sbp; GB avpu_bin, sbp, supplemental_o2, temp, hr. Trend: RF sbp, hr, spo2,
rr, sbp_delta; GB sbp, hr, supplemental_o2, rr, sbp_delta. Consciousness/oxygen dominate severity, blood pressure
and heart rate dominate deterioration — clinically plausible.

### 4b. VitalDB — new deterioration within 10 min (intra-operative, 1-min vitals; D19)

203,294 windows, 5,643 subjects, positive rate 13.8 %; nested CV 5×3 grouped by subject.

| Model | PR-AUC | AUROC |
|---|---|---|
| Majority | 0.138 | 0.500 |
| NEWS2 (partial) | 0.242 ± 0.002 | 0.660 |
| Logistic regression | 0.346 ± 0.006 | 0.749 |
| **Gradient boosting (HGB)** | **0.512 ± 0.008** | **0.833** |
| BiGRU | 0.497 ± 0.006 | 0.826 |

Source: `vitaldb_*` rows 2026-10-21. Deployed: isotonic-calibrated HGB; explanations: SHAP (low/falling SBP and
rising HR raise risk; temperature excluded from live reasons as a surgical artefact). Limit: anaesthetised
surgical patients, not pre-hospital; label vital-sign-defined.

### 4c. Fusion (vitals + **synthetic** text; ablation)

| Text agreement (synthetic generator) | Text only F1 | Fused F1 | Vitals only F1 |
|---|---|---|---|
| 0.50 | 0.472 | 0.473 | 0.386 |
| 0.65 | 0.600 | 0.575 | 0.386 |
| 0.80 | 0.753 | 0.721 | 0.386 |

Fused trend PR-AUC 0.20 / 0.26 / 0.31 vs vitals-only 0.16. **Mandatory caveat:** text is synthetic; the fusion
gain reflects the assumed informativeness of text and demonstrates the pipeline, not clinical performance.
Fused does not beat text-only at higher agreement — report as found.

## 5. Traffic: real-data calibration of the ORR corridor (SUMO)

* Real OSM network (D21); TomTom typical (672 eastbound rows, 24 h × 7) + live flow (7.4 days); published counts
  (BTP Oct 2025, Silk Board Jul 2024; D20); Silk Board on the reported 2024 signal program.
* Demand-only calibration **failed at peak** (needed 14× local demand, gridlocked;
  `reports/rl_calibration_demand_v2_failed.md`). Final = **speed-only calibration** (per-leg per-slot lane limits,
  stated as imposed): **12/12 rows within 15 % of TomTom typical**, full corridor error low +0.3 %, medium +4.4 %,
  peak +3.5 % (5 seeds; `osm_speed_*` last three rows 2026-10-22, `reports/rl_calibration.md`).
* Finding: the eastbound ORR is signal-free from Bellandur to KR Puram (flyovers/underpasses); the ambulance meets
  its signals in the first 6 km (Silk Board, pedestrian crossing, 19th Main, 6.0 km junction).

## 6. Traffic RL (DQN) — ambulance transit time

### 6a. Schematic corridor (4 junctions, real spacing; 20 episodes per demand level, 3 seeds × 100k steps)

| Demand | Fixed-time (s) | Always-green (s) | **DQN, mean of 3 seeds (s)** | DQN vs fixed |
|---|---|---|---|---|
| Low | 946.6 ± 56.4 | 843.2 ± 28.8 | **808.9** | **−14.5 %** |
| Medium | 962.4 ± 64.5 | 871.0 ± 40.8 | **830.0** | **−13.8 %** |
| Peak | 1017.8 ± 63.0 | 931.6 ± 50.8 | **872.4** | **−14.3 %** |

General traffic mean wait: fixed 1.13–1.21 s, DQN 0.10–0.14 s; max cross-street queue fixed 15.6–26.1 vs DQN
7.5–16.8 vehicles (`traffic_rl` rows; `reports/rl_comparison.csv`). DQN also beats always-green by 4–7 %.

### 6b. Real OSM corridor (calibrated slots) — baselines done, DQN in progress

Ambulance over the 7.7 km controlled stretch (Silk Board → 1 km past the 6.0 km junction), 6 paired episodes
per slot, all completed (`reports/rl_osm_baselines_episodes.csv`):

| Slot | Fixed-time (real Silk Board 2024 program) | Always-green | Always-green vs fixed |
|---|---|---|---|
| Low | **520 s** (wait 39 s) | 625 s (wait 155 s) | +20 % (worse) |
| Medium | 1463 s (wait 403 s) | **1081 s** (wait 187 s) | −26 % |
| Peak | 1515 s (wait 365 s) | **1244 s** (wait 230 s) | −18 % |

General traffic mean wait: fixed 0.9 / 2.1 / 2.0 s, always-green 0.5 / 1.0 / 1.2 s; max cross queue fixed
3.8 / 17.3 / 17.5, always-green 2.0 / 7.8 / 8.7 vehicles. At low traffic always-green is slower because outside
preemption it runs a generic 41 s cycle instead of the real programs — report as found.

DQN: 3 seeds, 9 h wall-clock budget each (randomized demand ±20 %, 10 % lane-incident episodes), evaluated on
the same paired episodes → `reports/rl_osm_dqn_s*_episodes.csv`, `reports/rl_osm_comparison.csv` (filled in
2026-10-23). Training length is set by the deadline, not convergence: say so.

## 7. Hospital selection + dispatch (system benchmark, D15)

10 scenarios (8 Bengaluru + 2 blood-match), fair naive baseline (same ETA model and overheads):

| | Naive (nearest) | AI (coordinator + hospital agent) |
|---|---|---|
| Appropriate hospital (trauma level, beds, blood) | 5 / 10 | **9 / 10** |
| Blood OK when needed | 7 / 10 | **10 / 10** |
| Mean total time (min) | 25.9 | 27.5 (+1.6) |

Source: `benchmark_*` 2026-10-21. The AI trades ~1.6 min for capable hospitals; one honest miss (Hebbal,
severity 3 → level-2 hospital because bed availability outweighed the trauma gap). Hospital beds/blood are
SIMULATED; ETAs straight-line.

## 8. Other built components (no metric, describe as features)

Police alerting (units, perimeter, timed junction posts, lifecycle); FastAPI + WebSockets; frontend (reporter,
dispatcher, police, hospital pages) with native-speaker prompt audio; per-language ASR routing.

## 9. Honest limits (put on one slide)

* Fusion text is synthetic; MIMIC is ICU, VitalDB is intra-operative — neither is pre-hospital.
* Held-out hi/kn triage ~60 % (dev 83 % was optimistic); Kannada ASR still weak (WER 58 % held-out).
* RL is simulation; corridor speeds imposed per slot (speed-only calibration); TomTom is one provider.
* Hospital beds/blood simulated; ambulance bases approximate; benchmark ETAs straight-line.
* Pending at time of writing: OSM RL results (6b); MIMIC-IV-ED access (PhysioNet approval).
