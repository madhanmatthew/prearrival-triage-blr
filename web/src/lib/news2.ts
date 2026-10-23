/** NEWS2 (SpO2 Scale 1). Mirrors backend/ml/news2.py (RCP bands, docs/02 §3). Used for the mock replay only. */
import type { VitalsReading } from "@/api/types";

export function news2(v: Pick<VitalsReading, "hr" | "rr" | "spo2" | "sbp" | "temp" | "avpu" | "supplemental_o2">): number | null {
  if ([v.hr, v.rr, v.spo2, v.sbp, v.temp].some((x) => x == null) || v.avpu == null || v.supplemental_o2 == null) return null;
  const rr = v.rr!, sp = v.spo2!, sbp = v.sbp!, hr = v.hr!, t = v.temp!;
  let s = 0;
  s += rr <= 8 ? 3 : rr <= 11 ? 1 : rr <= 20 ? 0 : rr <= 24 ? 2 : 3;
  s += sp <= 91 ? 3 : sp <= 93 ? 2 : sp <= 95 ? 1 : 0;
  s += v.supplemental_o2 ? 2 : 0;
  s += sbp <= 90 ? 3 : sbp <= 100 ? 2 : sbp <= 110 ? 1 : sbp <= 219 ? 0 : 3;
  s += hr <= 40 ? 3 : hr <= 50 ? 1 : hr <= 90 ? 0 : hr <= 110 ? 1 : hr <= 130 ? 2 : 3;
  s += v.avpu === "A" ? 0 : 3;
  s += t <= 35 ? 3 : t <= 36 ? 1 : t <= 38 ? 0 : t <= 39 ? 1 : 2;
  return s;
}
