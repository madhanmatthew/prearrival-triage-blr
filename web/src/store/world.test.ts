import { describe, expect, it } from "vitest";
import { appendRiskHistory } from "./world";
import type { FusedRisk } from "@/api/types";

const risk = (id: string, ts: string, score: number) => ({ incident_id: id, timestamp: ts, risk_score: score }) as FusedRisk;

describe("live risk history", () => {
  it("appends per incident and ignores repeats of the same update (polling)", () => {
    let h = appendRiskHistory({}, risk("A", "2026-10-27T10:00:00Z", 0.3));
    h = appendRiskHistory(h, risk("A", "2026-10-27T10:00:00Z", 0.3));
    h = appendRiskHistory(h, risk("A", "2026-10-27T10:00:02Z", 0.5));
    h = appendRiskHistory(h, risk("B", "2026-10-27T10:00:01Z", 0.9));
    expect(h.A.map((p) => p.score)).toEqual([0.3, 0.5]);
    expect(h.B).toHaveLength(1);
  });
  it("keeps time order and skips bad timestamps", () => {
    let h = appendRiskHistory({}, risk("A", "2026-10-27T10:00:05Z", 0.6));
    h = appendRiskHistory(h, risk("A", "2026-10-27T10:00:01Z", 0.2));
    h = appendRiskHistory(h, risk("A", "not a time", 0.9));
    expect(h.A.map((p) => p.score)).toEqual([0.2, 0.6]);
  });
});
