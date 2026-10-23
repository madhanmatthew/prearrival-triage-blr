import type {
  DispatchDecision, FusedRisk, GPS, GuidanceResponse, Health, HospitalRecommendation, IncidentSummary, IncidentView, IntakeResponse, Language, PoliceAlert, Resources,
  VitalsReading,
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

/** Same-origin by default (FastAPI serves /app/). Override with VITE_API_BASE for a remote backend. */
const BASE: string = (import.meta.env?.VITE_API_BASE as string | undefined) ?? "";

export async function http<T>(path: string, init: RequestInit & { json?: unknown; signal?: AbortSignal } = {}): Promise<T> {
  const { json, ...rest } = init;
  const opts: RequestInit = { ...rest };
  if (json !== undefined) { opts.method = opts.method ?? "POST"; opts.headers = { "Content-Type": "application/json", ...(opts.headers ?? {}) }; opts.body = JSON.stringify(json); }
  const r = await fetch(BASE + path, opts);
  let data: unknown = null;
  try { data = await r.json(); } catch { /* empty body */ }
  if (!r.ok) {
    const d = (data as { detail?: unknown } | null)?.detail;
    throw new ApiError(r.status, typeof d === "string" ? d : d ? JSON.stringify(d) : r.statusText);
  }
  return data as T;
}

export const api = {
  health: (signal?: AbortSignal) => http<Health>("/health", { signal }),
  resources: () => http<Resources>("/resources"),
  incidents: () => http<IncidentSummary[]>("/incidents"),
  incident: (id: string) => http<IncidentView>(`/incident/${id}`),
  intakeText: (text: string, language: Language, gps: GPS) => http<IntakeResponse>("/incident/text", { json: { text, language, gps } }),
  intakeVoice: (audio: Blob, gps: GPS, language?: Language) => {
    const f = new FormData();
    f.append("audio", audio, "report.webm"); f.append("lat", String(gps.lat)); f.append("lon", String(gps.lon));
    if (language) f.append("language", language);
    return http<IntakeResponse>("/incident/voice", { method: "POST", body: f });
  },
  answer: (id: string, answer: string) => http<IntakeResponse>(`/incident/${id}/answer`, { json: { answer } }),
  hospitals: (id: string) => http<HospitalRecommendation>(`/incident/${id}/hospitals`),
  guidance: (id: string) => http<GuidanceResponse>(`/incident/${id}/guidance`),
  vitals: (id: string, r: VitalsReading) => http<FusedRisk>(`/incident/${id}/vitals`, { json: r }),
  dispatch: (id: string) => http<DispatchDecision>(`/incident/${id}/dispatch`, { method: "POST" }),
  position: (id: string, gps: GPS) => http<PoliceAlert>(`/incident/${id}/position`, { json: gps }),
  police: (id: string) => http<PoliceAlert>(`/police/${id}`),
  ack: (id: string, junction_id?: string) => http<PoliceAlert>(`/police/${id}/ack`, { json: junction_id ? { junction_id } : {} }),
  close: (id: string) => http<PoliceAlert>(`/police/${id}/close`, { method: "POST" }),
};
