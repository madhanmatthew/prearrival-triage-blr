/** Typed access to REAL repo data imported at build time (scripts/sync-reports.mjs). */
import raw from "@/data/generated/repo.json";
import type { LatLon } from "@/api/types";

type Row = Record<string, string>;
interface Repo {
  repoFound: boolean; reports: Record<string, Row[] | null>; log: Row[]; hospitals: Row[]; ambulances: Row[]; scenarios: Row[];
  junctions: Record<string, { lat: number; lon: number }> | null; demoVitals: Record<string, unknown>[] | null; generatedAt: string;
}
export const repo = raw as unknown as Repo;

export interface HospitalRow {
  id: string; name: string; lat: number; lon: number; trauma: number; beds: number; icu: number; specialties: string[];
  blood: Record<string, number>;
}
const GROUPS = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"];
export const HOSPITALS: HospitalRow[] = repo.hospitals.map((r) => ({
  id: r.hospital_id, name: r.name, lat: +r.lat, lon: +r.lon, trauma: +r.trauma_level, beds: +r.beds_available || 0, icu: +r.icu_beds_available || 0,
  specialties: (r.specialties || "").split(";").map((s) => s.trim().toLowerCase()).filter(Boolean),
  blood: Object.fromEntries(GROUPS.map((g) => [g, +r[`blood_${g}`] || 0])),
}));
export const AMBULANCE_BASES = repo.ambulances.map((r) => ({ id: r.ambulance_id, name: r.base_name, lat: +r.lat, lon: +r.lon }));

export const JUNCTION_ORDER = ["silkboard", "bellandur", "marathahalli", "krpuram"] as const;
export const JUNCTION_LABEL: Record<string, string> = { silkboard: "Silk Board", bellandur: "Bellandur", marathahalli: "Marathahalli", krpuram: "KR Puram" };
export const JUNCTIONS: Record<string, LatLon> = Object.fromEntries(
  JUNCTION_ORDER.map((k) => [k, repo.junctions ? [repo.junctions[k].lat, repo.junctions[k].lon] : [0, 0]]),
) as Record<string, LatLon>;
