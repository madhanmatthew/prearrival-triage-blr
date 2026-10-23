/**
 * TypeScript mirror of backend/schemas.py (docs/07 is the contract).
 * KEEP IN SYNC: when a Pydantic model changes, update docs/07 -> schemas.py -> this file.
 * Only fields that exist in schemas.py or in the FastAPI response dicts (backend/main.py) are declared.
 */
export type Language = "en" | "hi" | "kn";
export type SeverityClass = 0 | 1 | 2 | 3;
export type AVPU = "A" | "V" | "P" | "U";
export type AnswerValue = "yes" | "no" | "unknown";
export type QuestionId =
  | "Q_CONSCIOUS" | "Q_BREATHING" | "Q_BLEEDING" | "Q_SPEAK" | "Q_NUM_INJURED" | "Q_MECHANISM";
export type InjuryType = "road_accident" | "fall" | "burn" | "cardiac_medical" | "other";
export type JunctionId = "silkboard" | "bellandur" | "marathahalli" | "krpuram";
export type Phase = "main_green" | "cross_green" | "yellow";

export interface GPS { lat: number; lon: number }
export type LatLon = [number, number];

export interface DialogueTurn { q_id: QuestionId; question: string; answer_raw: string; answer_value: AnswerValue }

export interface IncidentReport {
  incident_id: string; timestamp: string; gps: GPS; language: Language; transcript: string;
  dialogue: DialogueTurn[]; text_features: number[]; text_severity_probs: number[];
  injury_type: InjuryType; num_injured: number;
}

export interface VitalsReading {
  incident_id: string; patient_id: string; timestamp: string;
  hr: number | null; rr: number | null; spo2: number | null; sbp: number | null; temp: number | null;
  avpu: AVPU | null; supplemental_o2: boolean | null;
}

export interface FusedRisk {
  incident_id: string; patient_id: string; timestamp: string;
  severity_probs: number[]; severity_class: SeverityClass; risk_score: number;
  trend_prob: number | null; trend_horizon_readings: number;
  modalities_used: ("text" | "vitals")[]; news2_score: number | null; model_version: string; stale: boolean;
  deterioration_10min_prob: number | null; deterioration_model: string | null; deterioration_reasons: string[];
}

export interface ScoreBreakdown {
  proximity: number; trauma_match: number; beds: number; specialty: number; blood_match: number; rf_qvalue: number | null;
}
export interface RankedHospital {
  hospital_id: string; name: string; total_score: number; eta_min: number; breakdown: ScoreBreakdown;
  blood_flag: "matched" | "unknown" | "short";
}
export interface HospitalRecommendation { incident_id: string; ranked: RankedHospital[] }

export interface JunctionState { id: JunctionId; phase: Phase; ambulance_eta_s: number | null; preempted: boolean }
export interface Conflict { type: string; detail: string }
export interface SignalState {
  incident_id: string; route_id: string; junctions: JunctionState[]; conflicts: Conflict[]; estimated_transit_s: number | null;
}

export interface JunctionPost {
  junction_id: string; gps: GPS; ambulance_eta: string; officer_by: string; manual_clearing: boolean;
  status: "pending" | "acknowledged" | "released";
}
export interface PoliceAlert {
  incident_id: string; gps: GPS; severity_class: SeverityClass; route_polyline: LatLon[]; units_required: number;
  perimeter_m: number; alert_status: "active" | "acknowledged" | "closed"; junction_posts: JunctionPost[];
}

export interface ConflictResolution { type: string; resolution: string }
export interface DispatchDecision {
  incident_id: string; ambulance_id: string; hospital_id: string; route_id: string; police_units: number;
  conflicts_resolved: ConflictResolution[]; explanation: string; timestamp: string; route_estimated: boolean;
}

export interface WSMessage<T = unknown> { type: "PoliceAlert" | "FusedRisk" | "SignalState" | "DispatchDecision"; data: T }

/* ---- FastAPI response dicts (backend/main.py) ---- */
export interface Question { q_id: QuestionId; text: string }
export interface IntakeResponse { incident_id: string; done: boolean; question: Question | null; report: IncidentReport | null; transcript?: string }
/** backend/rag/pipeline.py Guidance dataclass (asdict) + {problem} added by GET /incident/{id}/guidance. */
export interface RetrievedChunk { chunk_id: string; file?: string; publisher?: string; url?: string; score?: number }
export interface GuidanceItem {
  problem: string; query?: string; status: "ok" | "not_covered" | "llm_unavailable" | string; mode?: string; message: string;
  sentences: { text: string; chunk_ids: string[] }[]; retrieved: RetrievedChunk[]; groundedness?: number | null;
  refusal_reason?: string | null; topic_file?: string | null;
}
export interface GuidanceResponse { incident_id: string; status: string; message?: string; items: GuidanceItem[] }
export interface IncidentSummary {
  incident_id: string; created_at: string; gps: GPS; dialogue_done: boolean; injury_type: InjuryType | null;
  num_injured: number | null; severity_class: SeverityClass | null; risk_score: number | null; vitals_count: number;
  decision: DispatchDecision | null; dispatch_notes: string[];
}
export interface IncidentView {
  incident_id: string; created_at: string; gps: GPS; dialogue_done: boolean; next_question: Question | null;
  report: IncidentReport | null; vitals_count: number; risk: FusedRisk | null; decision: DispatchDecision | null; dispatch_notes: string[];
}
export interface Resources {
  hospitals: { hospital_id: string; name: string; lat: number; lon: number }[];
  ambulances: { ambulance_id: string; lat: number; lon: number; busy: boolean }[];
  junctions: { junction_id: string; lat: number; lon: number }[];
}
export interface Health { status: string; degraded: string[]; dispatch_ready: boolean }
