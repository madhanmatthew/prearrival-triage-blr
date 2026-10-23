# Triage console (React UI)

Frontend for the Pre-Arrival Triage & Emergency Coordination System. Research prototype, not a medical device.
Replaces the plain HTML pages in `../frontend/` as the demo UI (those stay as a fallback at `/ui/`).

**Stack:** Vite + React 18 + TypeScript (strict) + Tailwind (tokens in `src/styles/index.css`) + Framer Motion +
react-leaflet (CARTO/OSM tiles, free) + Recharts + Zustand + lucide icons. Fonts self-hosted via @fontsource.
No paid services, no API keys in the bundle.

## Run

```bash
cd web
npm install
npm run dev          # http://localhost:5173/app/  (proxies API + WebSockets to uvicorn on :8000)
npm run build        # -> web/dist, served by FastAPI at http://localhost:8000/app/
npm test             # vitest: replay engine, API client, severity mapping
node scripts/shots.mjs   # Playwright screenshots (desktop 1440x900 + mobile 390x844) -> web/screenshots
node scripts/a11y.mjs    # axe-core WCAG 2.1 AA audit of every screen, dark + light
```

From the repo root: `make frontend-install frontend-build run-api`, then open `http://localhost:8000/` (redirects to `/app/`
once `web/dist` exists, otherwise to the old `/ui/`).

## Modes

| Mode | When | Pill |
|---|---|---|
| Live | `/health` answers | **Live** (WebSockets `/ws/map`, `/ws/police`; polls `/incidents`) |
| Replay | `?mock=1`, `VITE_MOCK=1`, or backend unreachable | **Replay (mock)** |

Force with `?mock=1` / `?mock=0`. Click the pill to switch. Replay URL helpers: `?s=B1` (scenario id), `?t=95` (pause at demo second).
`VITE_API_BASE` / `VITE_WS_BASE` point the app at a remote backend.

The replay (`src/mocks/`, deletable as one folder) is a deterministic scripted case: 10 benchmark scenarios from
`data/benchmark_scenarios.csv` + an ORR corridor demo. Hospital ranking mirrors the weights in `backend/agents/hospital_agent.py`,
coordinator rules 2 and 6 mirror `coordinator.py`, police sizing mirrors `police_agent.py`, NEWS2 mirrors `news2.py`.
Vitals trajectories, class probabilities, attributions and signal phases are **scripted** and tagged "Scripted" / "Simulated" in the UI.

## Data honesty

* The Evaluation screen reads **only** `reports/*.csv` and `experiments/log.csv`, imported at build time by
  `scripts/sync-reports.mjs` into `src/data/generated/repo.json` (git-ignored). Missing results render as "Pending: not yet run".
* Every simulated element (beds, blood, ETAs, signals, vitals) carries a "Simulated" tag; footer on every screen states
  "Research prototype. Not a medical device. Simulated data."
* Severity is exactly 4 classes (0-3); `risk_score` is a float in [0, 1].

## Keep types in sync

`src/api/types.ts` mirrors `backend/schemas.py` (and the dict responses in `backend/main.py`).
Change a schema -> `docs/07` -> `schemas.py` -> `src/api/types.ts` (+ `src/mocks/scenario.ts` if the shape changes).

## Add a screen

1. `src/pages/MyScreen.tsx` (default export; read data with `useWorld()`; it works in live and replay mode).
2. Register the route in `src/App.tsx` (lazy) and add a link in `src/components/Shell.tsx` `NAV`.
3. Use tokens (`bg-s1/s2/s3`, `text-mute`, `sev0..sev3`, `.surface`, `.card`, `.glass` only over the map), `<SeverityBadge>` for any class,
   `<SimTag>` for anything simulated, `<Pending>` for results that are not in the log.

## Known gaps (backend does not expose these yet)

See "Schema gaps" in `docs/progress/madhan.md`: vitals history, Integrated Gradients attributions,
SignalState, ambulance GPS, retrieved chunk text. In live mode these panels show a designed empty state; replay shows them fully.
