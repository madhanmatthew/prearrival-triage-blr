import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "./client";

const ok = (body: unknown, status = 200) => ({ ok: status < 400, status, statusText: "x", json: async () => body }) as Response;
afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("POSTs JSON with the right path and body for a text report", async () => {
    const f = vi.fn().mockResolvedValue(ok({ incident_id: "INC-1", done: false, question: { q_id: "Q_CONSCIOUS", text: "?" }, report: null }));
    vi.stubGlobal("fetch", f);
    const r = await api.intakeText("fell off a bike", "en", { lat: 12.9, lon: 77.6 });
    expect(r.incident_id).toBe("INC-1");
    const [url, init] = f.mock.calls[0];
    expect(url).toBe("/incident/text"); expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ text: "fell off a bike", language: "en", gps: { lat: 12.9, lon: 77.6 } });
  });
  it("maps FastAPI error details to ApiError", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(ok({ detail: "finish the dialogue first" }, 409)));
    await expect(api.dispatch("INC-1")).rejects.toMatchObject({ status: 409, message: "finish the dialogue first" });
    await expect(api.dispatch("INC-1")).rejects.toBeInstanceOf(ApiError);
  });
  it("sends the answer and ack payloads the backend expects", async () => {
    const f = vi.fn().mockResolvedValue(ok({}));
    vi.stubGlobal("fetch", f);
    await api.answer("INC-9", "yes"); await api.ack("INC-9", "silkboard"); await api.ack("INC-9");
    expect(f.mock.calls[0][0]).toBe("/incident/INC-9/answer"); expect(JSON.parse(f.mock.calls[0][1].body)).toEqual({ answer: "yes" });
    expect(f.mock.calls[1][0]).toBe("/police/INC-9/ack"); expect(JSON.parse(f.mock.calls[1][1].body)).toEqual({ junction_id: "silkboard" });
    expect(JSON.parse(f.mock.calls[2][1].body)).toEqual({});
  });
  it("copes with an empty error body", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 503, statusText: "Service Unavailable", json: async () => { throw new Error("no body"); } }));
    await expect(api.resources()).rejects.toMatchObject({ status: 503, message: "Service Unavailable" });
  });
});
