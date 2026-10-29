import { useState } from "react";
import { Car, HeartPulse, Send } from "lucide-react";
import type { World } from "@/store/world";
import { useSession } from "@/store/world";
import { dispatchSelected, replayVitals, selectIncident, simulateDrive } from "@/api/live";
import { SeverityBadge } from "@/components/ui";
import { repo } from "@/mocks/data";

/** Live-mode controls: incident picker + the same demo helpers as the old dispatcher page. */
export function LiveTools({ world }: { world: World }) {
  const { incidents, selectedId } = useSession((s) => s.live);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const sets = repo.demoVitalsSets ?? []; const [setId, setSetId] = useState(sets[0]?.id ?? "");
  const chosen = sets.find((s) => s.id === setId); const [tick, setTick] = useState(""); const [done, setDone] = useState<string | null>(null);
  const run = async (k: string, fn: () => Promise<unknown>) => { setBusy(k); setErr(null); try { await fn(); } catch (e) { setErr((e as Error).message); } finally { setBusy(null); } };
  const hasReport = !!world.incident?.injury, dispatched = !!world.decision;
  return (
    <div className="glass flex flex-wrap items-center gap-2 px-3 py-2" role="group" aria-label="Live incident controls">
      <select value={selectedId ?? ""} onChange={(e) => selectIncident(e.target.value)} className="h-8 max-w-[240px] rounded-lg border hairline bg-s3 px-2 text-[12.5px] outline-none focus-visible:ring-2 focus-visible:ring-brand" aria-label="Incident">
        {incidents.length === 0 && <option value="">No incidents yet</option>}
        {incidents.map((i) => <option key={i.incident_id} value={i.incident_id}>{i.incident_id}{i.injury_type ? ` · ${i.injury_type}` : ""}</option>)}
      </select>
      {world.risk && <SeverityBadge cls={world.risk.severity_class} size="sm" />}
      <button className="btn !h-8" disabled={!hasReport || dispatched || busy !== null} onClick={() => run("d", dispatchSelected)}><Send className="h-3.5 w-3.5" />{busy === "d" ? "Dispatching…" : "Dispatch"}</button>
      <select value={setId} onChange={(e) => { setSetId(e.target.value); setDone(null); }} disabled={busy !== null} aria-label="Demo vitals source" title={chosen?.source}
        className="h-8 max-w-[230px] rounded-lg border hairline bg-s3 px-2 text-[12.5px] outline-none focus-visible:ring-2 focus-visible:ring-brand">
        {sets.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
      </select>
      <button className="btn !h-8" disabled={!hasReport || !chosen || busy !== null} onClick={() => run("v", async () => { setDone(null); await replayVitals(setId, (k, n) => setTick(`${k}/${n}`)); setDone(chosen?.outcome ?? null); })}><HeartPulse className="h-3.5 w-3.5" />{busy === "v" ? `Replaying ${tick}` : "Demo vitals"}</button>
      <button className="btn !h-8" disabled={!dispatched || busy !== null} onClick={() => run("s", simulateDrive)}><Car className="h-3.5 w-3.5" />{busy === "s" ? "Driving…" : "Simulate drive"}</button>
      {done && <span className="text-[12px] text-mute">{done}</span>}
      {err && <span role="alert" className="text-[12px] text-sev3">{err}</span>}
    </div>
  );
}
