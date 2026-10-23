import type { SeverityClass } from "@/api/types";

export const SEVERITY_NAMES = ["Stable", "Moderate", "Severe", "Critical"] as const;
/** Icon names are resolved in components/SeverityBadge (colour is never the only signal). */
export const SEVERITY_GLYPH = ["●", "▲", "◆", "■"] as const;

export function asSeverity(n: number | null | undefined): SeverityClass {
  const v = Math.round(n ?? 0);
  return (v < 0 ? 0 : v > 3 ? 3 : v) as SeverityClass;
}

export interface SeverityStyle {
  name: string; glyph: string; text: string; bg: string; ring: string; hex: string; var: string;
}
const T = (i: number): SeverityStyle => ({
  name: SEVERITY_NAMES[i], glyph: SEVERITY_GLYPH[i],
  text: `text-sev${i}`, bg: `bg-sev${i}/15`, ring: `ring-sev${i}/40`,
  hex: ["#34d399", "#fbbf24", "#fb7c3c", "#f43f5e"][i], var: `rgb(var(--c-sev${i}))`,
});
const STYLES = [T(0), T(1), T(2), T(3)];

/** Severity class (0-3) -> style tokens. Single source of truth for severity colour. */
export function severityStyle(c: number | null | undefined): SeverityStyle { return STYLES[asSeverity(c)]; }

/** Risk score in [0,1] -> class band used only for gauge colouring (class thresholds at the expected-severity grid i/3). */
export function riskBand(score: number): SeverityClass {
  if (score < 1 / 6) return 0;
  if (score < 0.5) return 1;
  if (score < 5 / 6) return 2;
  return 3;
}
