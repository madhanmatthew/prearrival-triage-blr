import { describe, expect, it } from "vitest";
import { buildPlan, frameAt, listScenarios, T, CORRIDOR_DEMO_ID } from "./scenario";
import { news2 } from "@/lib/news2";

describe("replay engine", () => {
  const all = listScenarios();
  it("has 10 benchmark scenarios + the corridor demo", () => { expect(all.length).toBe(11); });
  it("is deterministic", () => {
    const d = all[3];
    expect(JSON.stringify(frameAt(d, 77))).toBe(JSON.stringify(frameAt(d, 77)));
  });
  it("risk is a float in [0,1], class in 0..3, probs sum to 1", () => {
    for (const d of all) for (const t of [8, 40, 80, 130]) {
      const r = frameAt(d, t).risk; if (!r) continue;
      expect(r.risk_score).toBeGreaterThanOrEqual(0); expect(r.risk_score).toBeLessThanOrEqual(1);
      expect([0, 1, 2, 3]).toContain(r.severity_class);
      expect(r.severity_probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    }
  });
  it("ambulance ends at the chosen hospital", () => {
    for (const d of all) { const p = buildPlan(d); const f = frameAt(d, T.end); expect(Math.abs(f.position[0] - p.chosen.row.lat)).toBeLessThan(0.002); }
  });
  it("corridor demo passes all four signals and fires bed-shortfall rule", () => {
    const d = all.find((s) => s.id === CORRIDOR_DEMO_ID)!; const p = buildPlan(d);
    expect(Object.keys(p.junctionAt).length).toBe(4);
    expect(p.rules.find((r) => r.id === "hospital_bed_shortfall")!.fired).toBe(true);
  });
  it("blood-match scenarios switch hospital via rule 6 or ranking", () => {
    for (const id of ["B1", "B2"]) { const p = buildPlan(all.find((s) => s.id === id)!); expect(p.chosen.blood_flag).not.toBe("short"); }
  });
  it("NEWS2 matches RCP bands", () => {
    expect(news2({ hr: 80, rr: 16, spo2: 98, sbp: 120, temp: 37, avpu: "A", supplemental_o2: false })).toBe(0);
    expect(news2({ hr: 135, rr: 26, spo2: 90, sbp: 85, temp: 39.5, avpu: "P", supplemental_o2: true })).toBe(3 + 3 + 2 + 3 + 3 + 3 + 2);
    expect(news2({ hr: null, rr: 16, spo2: 98, sbp: 120, temp: 37, avpu: "A", supplemental_o2: false })).toBeNull();
  });
});
