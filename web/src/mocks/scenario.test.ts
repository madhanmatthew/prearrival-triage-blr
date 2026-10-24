import { describe, expect, it } from "vitest";
import { buildPlan, frameAt, listScenarios, roadRoute, T, CORRIDOR_DEMO_ID } from "./scenario";
import { news2 } from "@/lib/news2";
import { JUNCTIONS } from "./data";

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

describe("replay routes follow real roads", () => {
  // ORS (OSRM for the 20 left after the ORS quota) geometry from scripts/build_demo_routes.py
  it("every scenario's transport and to-scene legs use road geometry", () => {
    let road = 0;
    for (const d of listScenarios()) {
      const p = buildPlan(d);
      const dest: [number, number] = [p.chosen.row.lat, p.chosen.row.lon];
      const leg: [number, number][] = d.forceCorridor
        ? [[d.lat, d.lon], JUNCTIONS.bellandur, JUNCTIONS.marathahalli, JUNCTIONS.krpuram, dest] : [[d.lat, d.lon], dest];
      const r = roadRoute(leg);
      expect(r, d.id).toBeTruthy();
      if (!r) continue;
      road++;
      expect(roadRoute([p.ambulance.base, [d.lat, d.lon]]), `${d.id} to scene`).toBeTruthy();
      expect(p.route[0], d.id).toEqual(r[0]);                         // plan starts on the road line
      expect(p.route.length, d.id).toBeGreaterThanOrEqual(r.length);
    }
    expect(road).toBe(listScenarios().length);
  });
});
