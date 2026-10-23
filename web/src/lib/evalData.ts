/** Parsing helpers for the Evaluation screen. Reads ONLY the build-time import of reports/*.csv and experiments/log.csv. */
import { repo } from "@/mocks/data";

export type Row = Record<string, string>;
export const report = (name: string): Row[] | null => { const r = repo.reports[name]; return r && r.length ? r : null; };
export const num = (s: string | undefined): number | null => (s === undefined || s === "" || Number.isNaN(Number(s)) ? null : Number(s));

/** "k=v k2=v2 ..." from the log's notes column. */
export function kv(notes: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const m of notes.matchAll(/([A-Za-z0-9_+.\-]+)=([^\s]+)/g)) out[m[1]] = m[2];
  return out;
}
/** Log rows with duplicates (re-runs with the same component/model/notes) collapsed to the most recent. */
export function logRows(filter: (r: Row) => boolean): Row[] {
  const latest = new Map<string, Row>();
  for (const r of repo.log) { if (!filter(r)) continue; const k = `${r.component}|${r.model}|${r.params_json}|${r.notes}|${r.data_version}`; const p = latest.get(k); if (!p || p.date <= r.date) latest.set(k, r); }
  return [...latest.values()];
}
export const meanStd = (m: string | undefined, s: string | undefined, d = 3) => { const a = num(m), b = num(s); return a == null ? "–" : b == null ? a.toFixed(d) : `${a.toFixed(d)} ± ${b.toFixed(d)}`; };
export const fmt = (x: string | number | undefined | null, d = 3) => { const n = typeof x === "number" ? x : num(x ?? undefined); return n == null ? "–" : n.toFixed(d); };
