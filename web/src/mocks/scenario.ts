/**
 * Deterministic scripted replay ("Replay (mock)"). Everything here is SIMULATED and tagged as such in the UI.
 * The engine is a pure function: frameAt(scenario, t) -> Frame. No timers, no randomness at call time,
 * so tests and screenshots are reproducible and the whole folder can be deleted cleanly.
 *
 * What is mirrored from the real backend (so the replay behaves like the product):
 *   hospital policy weights + factor formulas  -> backend/agents/hospital_agent.py
 *   coordinator rules 2 and 6 thresholds       -> backend/agents/coordinator.py
 *   police unit/perimeter sizing               -> backend/agents/police_agent.py
 *   NEWS2 bands                                -> backend/ml/news2.py
 * What is scripted (NOT model output): vitals trajectories, class probabilities, trend probability,
 * attributions, signal phases and the "RL" pre-emption. These are labelled mock wherever shown.
 */
import type {
  AVPU, DispatchDecision, FusedRisk, InjuryType, JunctionId, LatLon, Phase, PoliceAlert, RankedHospital, SeverityClass, VitalsReading,
} from "@/api/types";
import { AMBULANCE_BASES, HOSPITALS, JUNCTIONS, JUNCTION_ORDER, repo, type HospitalRow } from "./data";
import { clamp } from "@/lib/format";
import { cumulative, haversineM, pointAt, projectOnPolyline, densify } from "@/lib/geo";
import { news2 } from "@/lib/news2";

/* ----------------------------- timeline (demo seconds) ----------------------------- */
export const T = { intake: 0, triage: 6, dispatch: 12, toScene: 16, onScene: 30, transport: 36, end: 136, readEvery: 5, firstRead: 30 } as const;
export type Phase4 = "intake" | "triage" | "dispatch" | "to_scene" | "on_scene" | "transport" | "arrived";

export interface ScenarioDef {
  id: string; name: string; lat: number; lon: number; injury: InjuryType; sev: SeverityClass; injured: number;
  bleeding: boolean; blood: string | null; verified: boolean; demo: boolean; note?: string;
  overrides: Record<string, Record<string, number>>; bedsNow: Record<string, number>; forceCorridor?: boolean;
}
export interface RuleEntry { rule: number; id: string; title: string; fired: boolean; detail: string; chose?: string; at: number }
export interface RlEvent { at: number; junction: JunctionId; action: string }
export interface Frame {
  t: number; phase: Phase4; progress: number; scenario: ScenarioDef;
  position: LatLon; heading: number; trail: LatLon[]; etaS: number; speedKmh: number;
  vitals: VitalsReading[]; vitalsAll: VitalsReading[]; risk: FusedRisk | null; riskHistory: { t: number; score: number }[];
  textProbs: number[]; ig: { feature: string; value: number }[] | null;
  decision: DispatchDecision | null; rules: RuleEntry[]; signals: Record<JunctionId, { phase: Phase; preempted: boolean; etaS: number | null; onRoute: boolean }>;
  rlEvents: RlEvent[]; police: PoliceAlert | null; passed: JunctionId[];
}
export interface Plan {
  def: ScenarioDef; ranked: (RankedHospital & { row: HospitalRow; bloodUnits: number })[]; chosen: RankedHospital & { row: HospitalRow; bloodUnits: number };
  rules: RuleEntry[]; ambulance: { id: string; base: LatLon; name: string }; route: LatLon[]; cum: number[]; toScene: LatLon[]; toSceneCum: number[];
  junctionAt: Partial<Record<JunctionId, number>>; decision: DispatchDecision; police: PoliceAlert; transportM: number;
}

/* --------------------------------- seeded noise --------------------------------- */
function rng(seed: string) {
  let h = 1779033703 ^ seed.length;
  for (let i = 0; i < seed.length; i++) { h = Math.imul(h ^ seed.charCodeAt(i), 3432918353); h = (h << 13) | (h >>> 19); }
  return () => { h = Math.imul(h ^ (h >>> 16), 2246822507); h = Math.imul(h ^ (h >>> 13), 3266489909); h ^= h >>> 16; return (h >>> 0) / 4294967296; };
}
const smooth = (x: number) => { const t = clamp(x, 0, 1); return t * t * (3 - 2 * t); };

/* --------------------------------- scenario list --------------------------------- */
export const CORRIDOR_DEMO_ID = "C0";
function parseOverrides(s: string): Record<string, Record<string, number>> {
  try { return JSON.parse(s || "{}"); } catch { return {}; }
}
export function listScenarios(): ScenarioDef[] {
  const base = repo.scenarios.map<ScenarioDef>((r) => ({
    id: r.scenario_id, name: r.name, lat: +r.lat, lon: +r.lon, injury: r.injury_type as InjuryType, sev: Number(r.severity_class) as SeverityClass,
    injured: +r.num_injured, bleeding: r.bleeding_heavy === "True", blood: r.blood_group || null, verified: r.coords_quality.startsWith("verified"),
    demo: false, overrides: parseOverrides(r.hospital_overrides), bedsNow: {},
  }));
  const hero: ScenarioDef = {
    id: CORRIDOR_DEMO_ID, name: "ORR corridor run: Silk Board to the north (demo)", lat: JUNCTIONS.silkboard[0], lon: JUNCTIONS.silkboard[1],
    injury: "road_accident", sev: 3, injured: 2, bleeding: true, blood: null, verified: true, demo: true,
    note: "Demo scenario, not one of the 8 + 2 benchmark scenarios. Nearer hospitals are marked full (simulated) so the route uses the whole corridor.",
    overrides: {}, bedsNow: {}, forceCorridor: true,
  };
  return [hero, ...base];
}

/* ------------------------------ hospital agent (mirror) ------------------------------ */
const INJURY_SPECIALTY: Record<string, string[]> = {
  road_accident: ["trauma", "orthopaedics", "neurosurgery"], fall: ["trauma", "orthopaedics", "neurosurgery"],
  burn: ["burns", "plastic_surgery"], cardiac_medical: ["cardiology", "general_medicine"],
};
const W = { proximity: 0.30, trauma: 0.25, beds: 0.20, specialty: 0.10, blood: 0.15 };
const proximity = (eta: number) => 1 - Math.min(eta / 30, 1);
const traumaMatch = (lvl: number, req: number) => (lvl >= req ? 1 : lvl >= req - 1 ? 0.5 : 0);
const bedAvail = (b: number) => Math.min(b / 10, 1);
function specialtyMatch(inj: string, have: string[]) {
  const want = INJURY_SPECIALTY[inj];
  if (!want) return 0.5;
  return want.some((w) => have.includes(w)) ? 1 : 0;
}
export const bloodNeeded = (sev: number, inj: string, bleeding: boolean) => bleeding || (sev >= 2 && (inj === "road_accident" || inj === "fall"));
function bloodScore(h: HospitalRow, group: string | null): [number, "matched" | "unknown" | "short", number] {
  const use = group && h.blood[group] !== undefined ? group : "O-";
  const u = h.blood[use] ?? 0;
  return [u >= 4 ? 1 : u >= 1 ? 0.5 : 0, u < 1 ? "short" : group && h.blood[group] !== undefined ? "matched" : "unknown", u];
}
const etaMin = (a: LatLon, b: LatLon) => (haversineM(a, b) * 1.35) / 1000 / 30 * 60; // straight-line x1.35 at 30 km/h

/* ----------------------------------- plan builder ----------------------------------- */
const planCache = new Map<string, Plan>();
export function buildPlan(def: ScenarioDef): Plan {
  const hit = planCache.get(def.id); if (hit) return hit;
  const scene: LatLon = [def.lat, def.lon];
  // simulated stock/bed overrides: from the benchmark file, or the demo's "full" hospitals
  const hospitals = HOSPITALS.map((h) => {
    const o = def.overrides[h.id] ?? {};
    const blood = { ...h.blood };
    for (const [k, v] of Object.entries(o)) if (k.startsWith("blood_")) blood[k.slice(6)] = v;
    return { ...h, blood, beds: o.beds_available ?? h.beds };
  });
  const need = bloodNeeded(def.sev, def.injury, def.bleeding);
  const w = { ...W }; if (!need) { w.proximity += w.blood; w.blood = 0; }

  const demoBedsNow: Record<string, number> = {};
  if (def.forceCorridor) for (const h of hospitals) if (h.id !== "H20") { h.beds = 0; }
  let ranked = hospitals.map((h) => {
    const eta = etaMin(scene, [h.lat, h.lon]);
    const [bl, flag, units] = bloodScore(h, def.blood);
    const parts = { proximity: proximity(eta), trauma_match: traumaMatch(h.trauma, def.sev), beds: bedAvail(h.beds), specialty: specialtyMatch(def.injury, h.specialties), blood_match: bl, rf_qvalue: null };
    const total = w.proximity * parts.proximity + w.trauma * parts.trauma_match + w.beds * parts.beds + w.specialty * parts.specialty + w.blood * parts.blood_match;
    return { hospital_id: h.id, name: h.name, total_score: total, eta_min: eta, breakdown: parts, blood_flag: flag, row: h, bloodUnits: units };
  }).sort((a, b) => b.total_score - a.total_score).slice(0, 6);

  // live-state change AFTER ranking: the top hospital's last bed is taken (coordinator rule 2 demo)
  const top = ranked[0];
  if (def.forceCorridor) demoBedsNow[top.hospital_id] = top.row.beds > 0 ? 0 : 0;
  const bedsNow = { ...def.bedsNow, ...demoBedsNow };

  // ---- coordinator rules (mirrors choose_hospital) ----
  const rules: RuleEntry[] = [];
  const withBeds = ranked.filter((h) => (bedsNow[h.hospital_id] ?? (h.row.beds > 0 ? 1 : 0)) > 0);
  let chosen = withBeds[0] ?? top;
  const r2 = chosen.hospital_id !== top.hospital_id;
  rules.push({ rule: 2, id: "hospital_bed_shortfall", title: "Hospital bed shortfall", fired: r2, at: T.dispatch,
    detail: r2 ? `${top.hospital_id} ${top.name} has 0 beds. Switched to rank-${ranked.indexOf(chosen) + 1} ${chosen.hospital_id}; ${top.hospital_id} notified.` : `${top.hospital_id} has beds. No change.`, chose: chosen.hospital_id });
  let r6 = false, r6detail = `${chosen.hospital_id} blood flag: ${chosen.blood_flag}. ${need ? "No shortage." : "Transfusion risk not flagged."}`;
  if (need && chosen.blood_flag === "short") {
    const alt = withBeds.find((h) => h.blood_flag !== "short" && h.eta_min <= chosen.eta_min + 10);
    if (alt) { r6 = true; r6detail = `${chosen.hospital_id} short of blood. Chose ${alt.hospital_id} (+${(alt.eta_min - chosen.eta_min).toFixed(1)} min, blood ${alt.blood_flag}).`; chosen = alt; }
    else { r6 = true; r6detail = `${chosen.hospital_id} short of blood, no alternative within 10 min. Kept, blood transfer requested.`; }
  }
  rules.push({ rule: 6, id: "blood_shortage", title: "Blood shortage", fired: r6, detail: r6detail, chose: chosen.hospital_id, at: T.dispatch });

  // ambulance: nearest base
  const amb = [...AMBULANCE_BASES].sort((a, b) => haversineM([a.lat, a.lon], scene) - haversineM([b.lat, b.lon], scene))[0];
  const ambBase: LatLon = [amb.lat, amb.lon];

  // route (scene -> hospital). Corridor demo goes through the 4 verified junction coordinates.
  const dest: LatLon = [chosen.row.lat, chosen.row.lon];
  let skeleton: LatLon[];
  if (def.forceCorridor) skeleton = [scene, JUNCTIONS.bellandur, JUNCTIONS.marathahalli, JUNCTIONS.krpuram, dest];
  else {
    const mid: LatLon = [(scene[0] + dest[0]) / 2 + (dest[1] - scene[1]) * 0.08, (scene[1] + dest[1]) / 2 - (dest[0] - scene[0]) * 0.08];
    skeleton = [scene, mid, dest];
  }
  const route = densify(skeleton, 60);
  const cum = cumulative(route);
  const toScene = densify([ambBase, [(ambBase[0] + scene[0]) / 2 + (scene[1] - ambBase[1]) * 0.06, (ambBase[1] + scene[1]) / 2], scene], 60);
  const junctionAt: Partial<Record<JunctionId, number>> = {};
  for (const j of JUNCTION_ORDER) {
    const jp = JUNCTIONS[j];
    const d = projectOnPolyline(route, cum, jp);
    const idx = cum.findIndex((c) => c === d);
    if (haversineM(route[Math.max(idx, 0)], jp) < 400) junctionAt[j] = d;
  }

  // ---- other coordinator rules: honest "not triggered" (single incident in this replay) ----
  rules.unshift({ rule: 1, id: "cordon_on_route", title: "Cordon on route", fired: false, detail: "No police cordon crosses the route.", at: T.dispatch });
  rules.push(
    { rule: 3, id: "ambulance_contention", title: "Ambulance contention", fired: false, detail: "One active incident; no competing bid for the ambulance.", at: T.dispatch },
    { rule: 4, id: "police_units_short", title: "Police units short", fired: false, detail: "Requested units are within availability.", at: T.dispatch },
    { rule: 5, id: "junction_preemption", title: "Junction pre-emption", fired: false, detail: "No second ambulance within 30 s at any junction.", at: T.dispatch },
  );
  rules.sort((a, b) => a.rule - b.rule);

  // police sizing (mirror of size_alert)
  const base = [0, 1, 2, 3][def.sev];
  let units = base; if (units > 0) { units += Math.floor(Math.max(def.injured - 2, 0) / 2); if (def.injury === "road_accident") units += 1; units = Math.min(units, 6); }
  let perim = 0;
  if (units > 0 && !["cardiac_medical", "other"].includes(def.injury)) {
    perim = [0, 50, 100, 200][def.sev] + ({ road_accident: 50, burn: 50, fall: 0 } as Record<string, number>)[def.injury] + 20 * Math.max(def.injured - 1, 0);
    perim = Math.min(perim, 400);
  }
  const police: PoliceAlert = { incident_id: `MOCK-${def.id}`, gps: { lat: def.lat, lon: def.lon }, severity_class: def.sev, route_polyline: route, units_required: units, perimeter_m: perim, alert_status: "active", junction_posts: [] };

  const explanation = [
    `${chosen.hospital_id} ${chosen.name}: rank ${ranked.findIndex((h) => h.hospital_id === chosen.hospital_id) + 1}, score ${chosen.total_score.toFixed(2)}, ETA ${chosen.eta_min.toFixed(1)} min (straight-line, simulated)`,
    `trauma level ${chosen.row.trauma}, beds ${chosen.row.beds} (simulated), blood ${chosen.blood_flag}`,
  ].join("; ");
  const decision: DispatchDecision = {
    incident_id: police.incident_id, ambulance_id: amb.id, hospital_id: chosen.hospital_id, route_id: `route-${def.id}`, police_units: units,
    conflicts_resolved: rules.filter((r) => r.fired).map((r) => ({ type: r.id, resolution: r.detail })), explanation, timestamp: new Date(0).toISOString(), route_estimated: true,
  };
  const plan: Plan = { def, ranked, chosen, rules, ambulance: { id: amb.id, base: ambBase, name: amb.name }, route, cum, toScene, toSceneCum: cumulative(toScene), junctionAt, decision, police, transportM: cum[cum.length - 1] };
  planCache.set(def.id, plan);
  return plan;
}

/* ------------------------------------ vitals model ------------------------------------ */
const BASE = [
  { hr: 78, rr: 15, spo2: 98, sbp: 124, temp: 36.8 },
  { hr: 96, rr: 19, spo2: 96, sbp: 112, temp: 36.9 },
  { hr: 110, rr: 23, spo2: 94, sbp: 100, temp: 36.7 },
  { hr: 124, rr: 27, spo2: 91, sbp: 90, temp: 36.5 },
];
const DRIFT = [0, 0.25, 0.8, 1];
export function vitalsAt(def: ScenarioDef, k: number, readings: number): VitalsReading {
  const r = rng(`${def.id}:${k}`);
  const b = BASE[def.sev];
  const dr = DRIFT[def.sev] * smooth(k / Math.max(readings - 1, 1)) * (def.injury === "cardiac_medical" ? 1.15 : 1);
  const n = (a: number) => (r() - 0.5) * 2 * a;
  const hr = Math.round(b.hr + 24 * dr + n(2.5)), rr = Math.round(b.rr + 7 * dr + n(1)), spo2 = Math.round(clamp(b.spo2 - 6 * dr + n(0.7), 70, 100));
  const sbp = Math.round(b.sbp - 24 * dr + n(3)), temp = +(b.temp + n(0.08)).toFixed(1);
  const avpu: AVPU = def.sev >= 3 && dr > 0.75 ? "V" : def.sev >= 3 && dr > 0.95 ? "P" : "A";
  const dropout = def.sev === 3 && k === 7;           // one simulated sensor dropout
  return {
    incident_id: `MOCK-${def.id}`, patient_id: `MOCK-${def.id}-P1`, timestamp: new Date(1_790_000_000_000 + k * 30_000).toISOString(),
    hr, rr, spo2: dropout ? null : spo2, sbp, temp, avpu: dr > 0.6 && def.sev >= 2 ? (dr > 0.9 && def.sev === 3 ? "V" : "A") : avpu, supplemental_o2: dr > 0.45 && def.sev >= 2,
  };
}
export const readingCount = () => Math.floor((T.end - T.firstRead) / T.readEvery) + 1;
const FEATURES = ["hr", "rr", "spo2", "sbp", "temp", "avpu", "supplemental_o2"] as const;

function gaussProbs(mu: number, sd: number): number[] {
  const p = [0, 1, 2, 3].map((i) => Math.exp(-((i - mu) ** 2) / (2 * sd * sd)));
  const s = p.reduce((a, b) => a + b, 0);
  return p.map((x) => x / s);
}
function riskFrom(probs: number[], t: number, vit: VitalsReading[], def: ScenarioDef, vitalsUsed: boolean, trend: number | null): FusedRisk {
  const cls = probs.indexOf(Math.max(...probs)) as SeverityClass;
  const score = probs.reduce((a, p, i) => a + (p * i) / 3, 0);
  const last = vit[vit.length - 1];
  const d10 = vitalsUsed ? clamp((trend ?? 0) * 0.8 + 0.04, 0, 0.97) : null;
  return {
    incident_id: `MOCK-${def.id}`, patient_id: `MOCK-${def.id}-P1`, timestamp: new Date(1_790_000_000_000 + t * 1000).toISOString(),
    severity_probs: probs, severity_class: cls, risk_score: score, trend_prob: trend, trend_horizon_readings: 6,
    modalities_used: vitalsUsed ? ["text", "vitals"] : ["text"], news2_score: last ? news2(last) : null, model_version: "replay-mock",
    stale: false, deterioration_10min_prob: d10, deterioration_model: vitalsUsed ? "replay-mock" : null,
    deterioration_reasons: vitalsUsed && vit.length > 2 ? ["falling systolic BP", "rising heart rate"] : [],
  };
}

/* -------------------------------------- frame -------------------------------------- */
const CYCLE = 90; // s, base signal cycle for the scripted (non-pre-empted) phase pattern
function basePhase(j: JunctionId, t: number): Phase {
  const off = JUNCTION_ORDER.indexOf(j) * 23;
  const c = (t + off) % CYCLE;
  return c < 40 ? "main_green" : c < 44 ? "yellow" : c < 84 ? "cross_green" : "yellow";
}

export function frameAt(def: ScenarioDef, tIn: number): Frame {
  const plan = buildPlan(def);
  const t = clamp(tIn, 0, T.end);
  const phase: Phase4 = t < T.triage ? "intake" : t < T.dispatch ? "triage" : t < T.toScene ? "dispatch" : t < T.onScene ? "to_scene" : t < T.transport ? "on_scene" : t < T.end ? "transport" : "arrived";
  const progress = phase === "transport" ? (t - T.transport) / (T.end - T.transport) : phase === "arrived" ? 1 : 0;

  let position: LatLon, heading: number, trail: LatLon[] = [], speedKmh = 0, etaS: number;
  if (t < T.toScene) { position = plan.ambulance.base; heading = 0; etaS = 0; }
  else if (t < T.onScene) {
    const f = smooth((t - T.toScene) / (T.onScene - T.toScene)); const d = f * plan.toSceneCum[plan.toSceneCum.length - 1];
    const pa = pointAt(plan.toScene, plan.toSceneCum, d); position = pa.p; heading = pa.heading; speedKmh = 38;
    trail = plan.toScene.filter((_, i) => plan.toSceneCum[i] <= d).slice(-30); etaS = 0;
  } else if (t < T.transport) { position = [def.lat, def.lon]; heading = 0; etaS = 0; }
  else {
    const d = smooth(progress) * plan.transportM; const pa = pointAt(plan.route, plan.cum, d);
    position = pa.p; heading = pa.heading;
    trail = plan.route.filter((_, i) => plan.cum[i] <= d).slice(-45);
    const realMin = plan.chosen.eta_min; etaS = Math.max(0, (1 - progress) * realMin * 60);
    speedKmh = progress >= 1 ? 0 : 28 + 10 * Math.sin(progress * 9);
  }
  const dNow = phase === "transport" || phase === "arrived" ? smooth(progress) * plan.transportM : -1;

  // vitals so far
  const total = readingCount();
  const nRead = t < T.firstRead ? 0 : Math.min(total, Math.floor((t - T.firstRead) / T.readEvery) + 1);
  const all = Array.from({ length: nRead }, (_, k) => vitalsAt(def, k, total));
  const vit = all.slice(-6);

  // risk history (text-only before vitals)
  const textProbs = gaussProbs(def.sev + (def.sev === 3 ? -0.35 : def.sev === 0 ? 0.3 : 0), 0.75);
  const riskAt = (tt: number, k: number): FusedRisk | null => {
    if (tt < T.triage) return null;
    if (k === 0) return riskFrom(textProbs, tt, [], def, false, null);
    const win = Array.from({ length: Math.min(k, 6) }, (_, i) => vitalsAt(def, k - Math.min(k, 6) + i, total));
    const prog = smooth(k / Math.max(total - 1, 1));
    const mu = clamp(def.sev + (DRIFT[def.sev] * 0.9 * prog) - (def.sev === 3 ? 0.25 : 0), 0, 3);
    const trend = clamp(0.1 + DRIFT[def.sev] * 0.75 * prog + (def.sev >= 2 ? 0.05 : 0), 0, 0.97);
    return riskFrom(gaussProbs(mu, 0.55), tt, win, def, true, trend);
  };
  const risk = riskAt(t, nRead);
  const riskHistory: { t: number; score: number }[] = [];
  const first = riskAt(T.triage, 0); if (first && t >= T.triage) riskHistory.push({ t: T.triage, score: first.risk_score });
  for (let k = 1; k <= nRead; k++) { const tt = T.firstRead + (k - 1) * T.readEvery; const r = riskAt(tt, k); if (r) riskHistory.push({ t: tt, score: r.risk_score }); }

  // attributions (scripted): magnitude follows distance from the stable baseline
  let ig: Frame["ig"] = null;
  if (vit.length >= 2) {
    const last = vit[vit.length - 1]; const b = BASE[0];
    const z: Record<string, number> = { hr: ((last.hr ?? b.hr) - b.hr) / 25, rr: ((last.rr ?? b.rr) - b.rr) / 8, spo2: (b.spo2 - (last.spo2 ?? b.spo2)) / 6, sbp: (b.sbp - (last.sbp ?? b.sbp)) / 28, temp: Math.abs((last.temp ?? b.temp) - b.temp) / 2, avpu: last.avpu && last.avpu !== "A" ? 0.8 : 0.02, supplemental_o2: last.supplemental_o2 ? 0.45 : 0 };
    const norm = Math.max(...Object.values(z).map(Math.abs), 0.001);
    ig = FEATURES.map((f) => ({ feature: f, value: z[f] / norm * (risk?.risk_score ?? 0.5) })).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }

  // signals + scripted pre-emption
  const signals = {} as Frame["signals"]; const rlEvents: RlEvent[] = []; const passed: JunctionId[] = [];
  const speedMs = plan.transportM / (T.end - T.transport) * 1.0; // demo m per demo-second (avg)
  for (const j of JUNCTION_ORDER) {
    const at = plan.junctionAt[j]; const onRoute = at !== undefined;
    let ph = basePhase(j, t), pre = false, eta: number | null = null;
    if (onRoute && dNow >= 0) {
      const ahead = at! - dNow; eta = ahead > 0 ? ahead / (plan.transportM / (plan.chosen.eta_min * 60)) : null;
      const demoLead = ahead / Math.max(speedMs, 1);
      if (ahead < -80) passed.push(j);
      if (ahead < 1100 && ahead > -80) { pre = true; ph = "main_green"; void demoLead; }
      if (ahead < 1500 && ahead >= 1100) ph = "yellow";
      const startT = T.transport + (Math.max(at! - 1100, 0) / plan.transportM) * (T.end - T.transport); // approx first-green moment
      if (t >= startT - 1) rlEvents.push({ at: Math.round(startT), junction: j, action: "hold main green (corridor)" });
    }
    signals[j] = { phase: ph, preempted: pre, etaS: eta, onRoute };
  }
  rlEvents.sort((a, b) => a.at - b.at);

  const decision = t >= T.toScene ? plan.decision : null;
  const rules = t >= T.toScene ? plan.rules : [];
  const posts = t >= T.toScene ? JUNCTION_ORDER.filter((j) => plan.junctionAt[j] !== undefined).map((j) => {
    const etaOff = (plan.junctionAt[j]! / plan.transportM) * plan.chosen.eta_min * 60;
    return { junction_id: j, gps: { lat: JUNCTIONS[j][0], lon: JUNCTIONS[j][1] }, ambulance_eta: new Date(1_790_000_000_000 + etaOff * 1000).toISOString(), officer_by: new Date(1_790_000_000_000 + (etaOff - 120) * 1000).toISOString(), manual_clearing: false, status: passed.includes(j) ? "released" as const : "pending" as const };
  }) : [];
  const police = t >= T.toScene ? { ...plan.police, junction_posts: posts } : null;
  return { t, phase, progress, scenario: def, position, heading, trail, etaS, speedKmh, vitals: vit, vitalsAll: all, risk, riskHistory, textProbs, ig, decision, rules, signals, rlEvents, police, passed };
}
