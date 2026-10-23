import { create } from "zustand";
import { useMemo } from "react";
import type {
  DispatchDecision, FusedRisk, GPS, IncidentView, JunctionId, LatLon, Phase, PoliceAlert, Resources, SeverityClass, VitalsReading, InjuryType, Language,
} from "@/api/types";
import { JUNCTIONS, HOSPITALS, JUNCTION_ORDER } from "@/mocks/data";
import { buildPlan, frameAt, listScenarios, T, CORRIDOR_DEMO_ID, type Phase4, type RlEvent, type RuleEntry } from "@/mocks/scenario";
import { asSeverity } from "@/lib/severity";

export type Source = "mock" | "live";
export type Backend = "checking" | "up" | "down";
export type WsState = "idle" | "open" | "reconnecting";

interface LiveState {
  resources: Resources | null; incidents: import("@/api/types").IncidentSummary[]; selectedId: string | null; view: IncidentView | null;
  police: Record<string, PoliceAlert>; ws: WsState; error: string | null; localVitals: Record<string, VitalsReading[]>; loaded: boolean;
  ambulancePos: Record<string, { pos: LatLon; heading: number; trail: LatLon[] }>;
  ranking: Record<string, import("@/api/types").HospitalRecommendation>;
}
interface Session {
  source: Source; backend: Backend; theme: "dark" | "light";
  mock: { scenarioId: string; t: number; playing: boolean; speed: number };
  live: LiveState;
  setSource: (s: Source) => void; setBackend: (b: Backend) => void; toggleTheme: () => void;
  setScenario: (id: string) => void; setT: (t: number) => void; play: (p?: boolean) => void; setSpeed: (s: number) => void; restart: () => void;
  patchLive: (p: Partial<LiveState>) => void;
}

const initialTheme = (): "dark" | "light" => { try { return (localStorage.getItem("theme") as "dark" | "light") ?? "dark"; } catch { return "dark"; } };
export const useSession = create<Session>((set, get) => ({
  source: "mock", backend: "checking", theme: initialTheme(),
  mock: { scenarioId: CORRIDOR_DEMO_ID, t: 0, playing: false, speed: 1 },
  live: { resources: null, incidents: [], selectedId: null, view: null, police: {}, ws: "idle", error: null, localVitals: {}, loaded: false, ambulancePos: {}, ranking: {} },
  setSource: (source) => set({ source }),
  setBackend: (backend) => set({ backend }),
  toggleTheme: () => { const theme = get().theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = theme; try { localStorage.setItem("theme", theme); } catch { /* private mode */ } set({ theme }); },
  setScenario: (scenarioId) => set({ mock: { ...get().mock, scenarioId, t: 0, playing: true } }),
  setT: (t) => set({ mock: { ...get().mock, t } }),
  play: (p) => set({ mock: { ...get().mock, playing: p ?? !get().mock.playing } }),
  setSpeed: (speed) => set({ mock: { ...get().mock, speed } }),
  restart: () => set({ mock: { ...get().mock, t: 0, playing: true } }),
  patchLive: (p) => set({ live: { ...get().live, ...p } }),
}));

/* --------------------------- unified view model used by every screen --------------------------- */
export interface RankView {
  hospital_id: string; name: string; total_score: number; eta_min: number; blood_flag: "matched" | "unknown" | "short";
  breakdown: { proximity: number; trauma_match: number; beds: number; specialty: number; blood_match: number; rf_qvalue: number | null };
  trauma: number | null; beds: number | null; bloodUnits: number | null; chosen: boolean;
}
export interface SignalView { phase: Phase; preempted: boolean; etaS: number | null; onRoute: boolean }
export interface World {
  source: Source; label: string; loading: boolean;
  incident: { id: string; gps: GPS; injury: InjuryType | null; injured: number | null; language: Language | null; transcript: string | null; name?: string } | null;
  phase: Phase4 | "live"; t: number; duration: number;
  risk: FusedRisk | null; riskHistory: { t: number; score: number }[]; vitals: VitalsReading[]; vitalsAll: VitalsReading[]; vitalsCount: number;
  ig: { feature: string; value: number }[] | null; textProbs: number[] | null;
  ranking: RankView[] | null; decision: DispatchDecision | null; rules: RuleEntry[] | null;
  signals: Record<JunctionId, SignalView> | null; rlEvents: RlEvent[]; police: PoliceAlert | null;
  ambulance: { id: string; position: LatLon; heading: number; trail: LatLon[]; etaS: number | null; speedKmh: number | null; base?: LatLon } | null;
  route: LatLon[]; hospitals: { id: string; name: string; lat: number; lon: number; trauma?: number; chosen: boolean }[]; junctions: Record<string, LatLon>;
  gaps: string[];
}

export function worldFromMock(scenarioId: string, t: number): World {
  const def = listScenarios().find((s) => s.id === scenarioId) ?? listScenarios()[0];
  const plan = buildPlan(def); const f = frameAt(def, t);
  const ranking: RankView[] = plan.ranked.map((h) => ({
    hospital_id: h.hospital_id, name: h.name, total_score: h.total_score, eta_min: h.eta_min, blood_flag: h.blood_flag, breakdown: h.breakdown,
    trauma: h.row.trauma, beds: h.row.beds, bloodUnits: h.bloodUnits, chosen: h.hospital_id === plan.chosen.hospital_id,
  }));
  return {
    source: "mock", label: "Replay (mock)", loading: false,
    incident: t >= 0 ? { id: `MOCK-${def.id}`, gps: { lat: def.lat, lon: def.lon }, injury: def.injury, injured: def.injured, language: "en", transcript: def.name, name: def.name } : null,
    phase: f.phase, t: f.t, duration: T.end, risk: f.risk, riskHistory: f.riskHistory, vitals: f.vitals, vitalsAll: f.vitalsAll, vitalsCount: f.vitalsAll.length,
    ig: f.ig, textProbs: f.t >= T.triage ? f.textProbs : null, ranking: f.t >= T.triage ? ranking : null, decision: f.decision, rules: f.rules,
    signals: f.signals, rlEvents: f.rlEvents, police: f.police,
    ambulance: { id: plan.ambulance.id, position: f.position, heading: f.heading, trail: f.trail, etaS: f.phase === "transport" || f.phase === "arrived" ? f.etaS : null, speedKmh: f.speedKmh, base: plan.ambulance.base },
    route: plan.route,
    hospitals: HOSPITALS.map((h) => ({ id: h.id, name: h.name, lat: h.lat, lon: h.lon, trauma: h.trauma, chosen: h.id === plan.chosen.hospital_id && f.t >= T.toScene })),
    junctions: JUNCTIONS, gaps: [],
  };
}

export function worldFromLive(l: LiveState): World {
  const v = l.view, id = l.selectedId;
  const police = id ? l.police[id] ?? null : null;
  const decision = v?.decision ?? null;
  const risk = v?.risk ?? null;
  const local = id ? l.localVitals[id] ?? [] : [];
  const amb = id ? l.ambulancePos[id] : undefined;
  const rec = id ? l.ranking[id] : undefined;
  const liveRanking: RankView[] | null = rec ? rec.ranked.map((h) => ({ ...h, trauma: null, beds: null, bloodUnits: null, chosen: decision?.hospital_id === h.hospital_id })) : null;
  const dispatched = decision && l.resources?.ambulances.find((a) => a.ambulance_id === decision.ambulance_id);
  const rules: RuleEntry[] | null = decision ? decision.conflicts_resolved.map((c, i) => ({ rule: i + 1, id: c.type, title: c.type.replace(/_/g, " "), fired: true, detail: c.resolution, at: 0 })) : null;
  const junctions: Record<string, LatLon> = l.resources ? Object.fromEntries(l.resources.junctions.map((j) => [j.junction_id, [j.lat, j.lon] as LatLon])) : JUNCTIONS;
  return {
    source: "live", label: "Live", loading: !l.loaded,
    incident: v ? { id: v.incident_id, gps: v.gps, injury: v.report?.injury_type ?? null, injured: v.report?.num_injured ?? null, language: v.report?.language ?? null, transcript: v.report?.transcript ?? null } : null,
    phase: "live", t: 0, duration: 0, risk, riskHistory: [], vitals: local.slice(-6), vitalsAll: local, vitalsCount: v?.vitals_count ?? 0, ig: null,
    textProbs: v?.report?.text_severity_probs ?? null, ranking: liveRanking, decision, rules, signals: null, rlEvents: [], police,
    ambulance: decision ? { id: decision.ambulance_id, position: amb?.pos ?? (dispatched ? [dispatched.lat, dispatched.lon] : v!.gps && [v!.gps.lat, v!.gps.lon]) as LatLon, heading: amb?.heading ?? 0, trail: amb?.trail ?? [], etaS: null, speedKmh: null } : null,
    route: police?.route_polyline ?? [],
    hospitals: (l.resources?.hospitals ?? []).map((h) => ({ id: h.hospital_id, name: h.name, lat: h.lat, lon: h.lon, chosen: decision?.hospital_id === h.hospital_id })),
    junctions,
    gaps: [
      "Vitals history and attributions (Integrated Gradients): not exposed by the API.",
      "Live signal phases (SignalState): no endpoint or WebSocket message yet.",
      "Ambulance GPS: only reflected in PoliceAlert posts.",
    ],
  };
}

/** The one hook every screen uses. Mock and live produce the same World shape. */
export function useWorld(): World {
  const source = useSession((s) => s.source);
  const mock = useSession((s) => s.mock);
  const live = useSession((s) => s.live);
  return useMemo(() => (source === "mock" ? worldFromMock(mock.scenarioId, mock.t) : worldFromLive(live)), [source, mock.scenarioId, mock.t, live]);
}
export { asSeverity, JUNCTION_ORDER };
export type { SeverityClass };
