export const pct = (x: number | null | undefined, d = 0) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
export const fix = (x: number | null | undefined, d = 1) => (x == null || Number.isNaN(x) ? "–" : x.toFixed(d));
export function mmss(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds));
  return `${Math.floor(s / 60).toString().padStart(2, "0")}:${(s % 60).toString().padStart(2, "0")}`;
}
export function hhmm(iso: string | null | undefined): string {
  if (!iso) return "–";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "–" : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
