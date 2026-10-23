# Demo runbook (live demo, docs/09 §6)

Machine: Harish's laptop (RTX 3050). Repo: `C:\Users\haris\Madhan\prearrival-triage-blr`. All commands from the
repo folder in a terminal. Research prototype — say "decision support, not a medical device" once at the start.

## 0. The day before (once)

- [ ] `git pull` and `.venv\Scripts\python -m pytest -q` → all pass.
- [ ] These local files exist (gitignored, not on GitHub — **back them up to a USB stick**):
  `models\text_severity_v1_20261023.joblib`, `models\vitaldb_hgb_v1_20261021.joblib`, `models\fusion_v1_20261006.pt`,
  `models\bigru_v1_20260930.pt`, `models\rag\`, `models\whisper\` (incl. `ct2\whisper-hindi-medium`,
  `ct2\whisper-kannada-medium`), `models\dqn_osm_s1_*.zip`, `.env`.
- [ ] **Screen-record the fallbacks** (Win + Alt + R): (a) one full reporter flow in Kannada, (b) `demo_sumo`
  fixed-time vs DQN at peak (step 7). Keep the videos on the desktop.
- [ ] Charger, power plan "Best performance", Windows updates paused, notifications off (Focus assist).

## 1. Start (10 min before)

```bash
.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
Wait until the log says it is running (first voice request loads Whisper on the GPU: say one test sentence
before the audience arrives so the model is warm).

Pre-flight: open `http://127.0.0.1:8000/` (home page). **Backend status** must say `API ok · dispatch ready` and
the degraded list should only show: `hospital RF: not trained, rule-only scores` and
`routing: no ORS_API_KEY, straight-line ETAs`. Anything else (text classifier, fusion, rag, deterioration,
hospitals, ambulances) → a model file is missing: copy it back from the USB stick.

Open 4 browser tabs: **Report** (`/ui/report.html`), **Dispatcher** (`/ui/dispatch.html`), **Police**
(`/ui/police.html`), **Hospital** (`/ui/hospital.html` → pick the hospital after dispatch).
Allow the microphone on the Report tab.

## 2. Live script (~8 min) — scenario: road accident near Silk Board, 2 injured, heavy bleeding

| # | Do | Say / point at |
|---|---|---|
| 1 | Report tab → **ಕನ್ನಡ** → *Demo: set location manually* → **Silk Board (demo)** → hold 🎤 and speak the accident in Kannada (or Hindi) | Location comes from GPS only; speech never sets the location. Language chosen by the user; Kannada/Hindi use their own fine-tuned Whisper models (held-out WER kn 58 % vs 84 % for stock Whisper). |
| 2 | Answer the questions with the tap buttons (No / No / heavy bleeding / 2 / yes); the native-speaker audio plays each question | Bounded 6-question state machine; taps make answers robust to speech errors. |
| 3 | Result → severity badge; guidance steps with **grounded ✓** and chunk ids; ▶ next / repeat | Every sentence is retrieved IFRC text (rule 6), never generated advice; no medication. |
| 4 | Dispatcher tab → click the incident → **Dispatch** | Rule-based coordinator: nearest free ambulance + hospital agent (trauma level, beds, blood); explanation text; route on the map. |
| 5 | **▶ Vitals replay (demo)** | Fused risk + NEWS2 + **"New drop in BP/SpO2 within 10 min"** with the top-3 reasons (SHAP). Say: replay values are hand-written demo values; the 10-min model is trained on intra-operative VitalDB data. |
| 6 | Hospital tab → select the dispatched hospital | Live risk stream with history and reasons. Blood-match effect: show `reports/benchmark_results.csv` rows B1/B2 (naive picks the hospital without O+, AI the one with stock). |
| 7 | Terminal: `.venv/Scripts/python -m scripts.demo_sumo --policy fixed_time --slot peak` then `--policy dqn --seed 2` | Real OSM Silk Board → Iblur corridor, calibrated to TomTom (±15 %); RL controls 4 signals; prints ambulance transit time. Say: seed 2 is the best of 3 seeds (−34 % at peak); 2 seeds were unstable after deadline-limited training. (If slow on stage: play the recorded video.) |
| 8 | Dispatcher → **🚑 Simulate drive**; Police tab | Police alert: units, perimeter, timed junction posts that release as the ambulance passes. |
| 9 | Slides: results table + limits | `docs/RESULTS_SUMMARY.md` §1–§9; be explicit about the limits slide. |

## 3. If something fails

| Problem | Fix on stage |
|---|---|
| Microphone / voice fails | Type the report in the text box (same flow), or play the recorded video. |
| `dispatch unavailable` | `data/hospitals.csv` / `data/ambulance_bases.csv` missing → `python -m scripts.build_hospitals`. |
| Severity always "Stable" | text model missing → copy `models\text_severity_v1_*.joblib` back, restart the API. |
| sumo-gui slow or hangs | Close it; play the recorded fixed-time vs DQN video. |
| Browser shows old page | Ctrl + F5. |
| Port 8000 busy | add `--port 8001` and use that port in the tabs. |

## 4. Numbers to have ready (all logged; `docs/RESULTS_SUMMARY.md`)

ASR (held-out, deployed models): en 8 % / hi 24 % / kn 58 % WER. RAG: hit@4 0.96, 100 % grounded.
Vitals: VitalDB HGB PR-AUC 0.51 vs NEWS2 0.24; MIMIC BiGRU F1 0.39 vs NEWS2 0.28. Calibration: 12/12 within 15 %
of TomTom. Benchmark: appropriate hospital 9/10 AI vs 5/10 naive. RL (schematic): DQN −14 % ambulance transit vs
fixed-time (3/3 seeds); OSM corridor: best seed −39 % (medium) / −34 % (peak) vs fixed-time, 2 of 3 seeds unstable.
