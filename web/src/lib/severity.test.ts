import { describe, expect, it } from "vitest";
import { asSeverity, riskBand, severityStyle, SEVERITY_NAMES } from "./severity";

describe("severity mapping (exactly 4 classes, 0 to 3)", () => {
  it("names match the contract", () => { expect([...SEVERITY_NAMES]).toEqual(["Stable", "Moderate", "Severe", "Critical"]); });
  it("each class has a distinct colour token, glyph and text class", () => {
    const s = [0, 1, 2, 3].map((c) => severityStyle(c));
    expect(new Set(s.map((x) => x.hex)).size).toBe(4);
    expect(new Set(s.map((x) => x.glyph)).size).toBe(4);
    expect(s.map((x) => x.text)).toEqual(["text-sev0", "text-sev1", "text-sev2", "text-sev3"]);
  });
  it("clamps out-of-range classes instead of inventing a fifth", () => {
    expect(asSeverity(-2)).toBe(0); expect(asSeverity(7)).toBe(3); expect(asSeverity(null)).toBe(0); expect(asSeverity(2.4)).toBe(2);
  });
  it("risk bands follow the expected-severity grid", () => {
    expect([0, 0.1, 0.3, 0.49, 0.5, 0.7, 0.84, 1].map(riskBand)).toEqual([0, 0, 1, 1, 2, 2, 3, 3]);
  });
});
