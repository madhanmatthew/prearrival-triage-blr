import { api } from "./client";
import { openChannel, type Channel } from "./ws";
import { useSession } from "@/store/world";
import type { PoliceAlert, VitalsReading } from "./types";
import { densify } from "@/lib/geo";
import { repo } from "@/mocks/data";

/** Probe the backend. Resolves true if /health answers within 2.5 s. */
export async function probeBackend(): Promise<boolean> {
  const ac = new AbortController(); const to = setTimeout(() => ac.abort(), 2500);
  try { await api.health(ac.signal); return true; } catch { return false; } finally { clearTimeout(to); }
}

let channels: Channel[] = []; let poll: ReturnType<typeof setInterval> | undefined;

export function startLive() {
  stopLive();
  const { patchLive } = useSession.getState();
  const live = () => useSession.getState().live;
  let open = 0;
  const onState = (o: boolean) => { open += o ? 1 : -1; if (open < 0) open = 0; patchLive({ ws: open > 0 ? "open" : "reconnecting" }); };
  const refresh = async () => {
    try {
      const incidents = await api.incidents();
      let selectedId = live().selectedId;
      if (!selectedId || !incidents.some((i) => i.incident_id === selectedId)) selectedId = incidents[0]?.incident_id ?? null;
      const view = selectedId ? await api.incident(selectedId) : null;
      const police = { ...live().police };
      if (selectedId && view?.decision && !police[selectedId]) { try { police[selectedId] = await api.police(selectedId); } catch { /* not dispatched yet */ } }
      const ranking = { ...live().ranking };
      if (selectedId && view?.decision && !ranking[selectedId]) { try { ranking[selectedId] = await api.hospitals(selectedId); } catch { /* older backend */ } }
      patchLive({ incidents, selectedId, view, police, ranking, error: null, loaded: true });
    } catch (e) { patchLive({ error: (e as Error).message, loaded: true }); }
  };
  api.resources().then((resources) => patchLive({ resources })).catch(() => undefined);
  refresh();
  poll = setInterval(refresh, 3000);
  channels = [
    openChannel("/ws/map", () => { void refresh(); }, onState),
    openChannel("/ws/police", (m) => {
      if (m.type === "PoliceAlert") { const a = m.data as PoliceAlert; patchLive({ police: { ...live().police, [a.incident_id]: a } }); }
    }, onState),
  ];
}
export function stopLive() { channels.forEach((c) => c.close()); channels = []; clearInterval(poll); useSession.getState().patchLive({ ws: "idle" }); }
export function selectIncident(id: string) { useSession.getState().patchLive({ selectedId: id, view: null }); }

/* ---- demo helpers (same behaviour as the old dispatcher page) ---- */
export async function dispatchSelected() {
  const id = useSession.getState().live.selectedId; if (!id) return;
  await api.dispatch(id);
}
export async function replayVitals(onTick?: (k: number, n: number) => void): Promise<void> {
  const id = useSession.getState().live.selectedId; const rows = repo.demoVitals as Partial<VitalsReading>[] | null; if (!id || !rows) throw new Error("demo_vitals.json not found");
  for (let k = 0; k < rows.length; k++) {
    const reading = { incident_id: id, patient_id: `${id}-P1`, timestamp: new Date().toISOString().slice(0, 19) + "Z", hr: null, rr: null, spo2: null, sbp: null, temp: null, avpu: null, supplemental_o2: null, ...rows[k] } as VitalsReading;
    await api.vitals(id, reading);
    const s = useSession.getState(); s.patchLive({ localVitals: { ...s.live.localVitals, [id]: [...(s.live.localVitals[id] ?? []), reading] } });
    onTick?.(k + 1, rows.length);
    await new Promise((r) => setTimeout(r, 1800));
  }
}
export async function simulateDrive(): Promise<void> {
  const id = useSession.getState().live.selectedId; if (!id) return;
  const alert = await api.police(id); const pts = densify(alert.route_polyline, 120);
  const trail: [number, number][] = [];
  for (const p of pts) {
    trail.push(p); const prev = trail[Math.max(0, trail.length - 2)];
    const s = useSession.getState();
    s.patchLive({ ambulancePos: { ...s.live.ambulancePos, [id]: { pos: p, heading: Math.atan2(p[1] - prev[1], p[0] - prev[0]) * 180 / Math.PI, trail: trail.slice(-40) } } });
    try { await api.position(id, { lat: p[0], lon: p[1] }); } catch { /* ignore */ }
    await new Promise((r) => setTimeout(r, 400));
  }
}
