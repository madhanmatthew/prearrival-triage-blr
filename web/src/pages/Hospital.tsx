import { useMemo } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { Activity, BedDouble, Check, Clock, Droplets, HeartPulse, Stethoscope, X as XIcon, Siren } from "lucide-react";
import clsx from "clsx";
import { useWorld, useSession } from "@/store/world";
import { HOSPITALS } from "@/mocks/data";
import { ReplayBar } from "@/components/ReplayBar";
import { AnimatedNumber, EmptyState, Panel, RiskGauge, SeverityBadge, SimTag, Skeleton, Sparkline } from "@/components/ui";
import { VitalsStrip } from "@/components/command/Vitals";
import { mmss } from "@/lib/format";
import { bloodNeeded, listScenarios } from "@/mocks/scenario";
import type { InjuryType } from "@/api/types";

const SPECIALTY: Record<string, string[]> = { road_accident: ["trauma", "orthopaedics", "neurosurgery"], fall: ["trauma", "orthopaedics", "neurosurgery"], burn: ["burns", "plastic_surgery"], cardiac_medical: ["cardiology", "general_medicine"] };

export default function Hospital() {
  const world = useWorld(); const { id } = useParams(); const nav = useNavigate();
  const source = useSession((s) => s.source); const scenarioId = useSession((s) => s.mock.scenarioId);
  const hid = id ?? world.decision?.hospital_id ?? (world.ranking?.find((r) => r.chosen)?.hospital_id) ?? HOSPITALS[0]?.id;
  const h = HOSPITALS.find((x) => x.id === hid);
  const incoming = world.decision?.hospital_id === hid;
  const def = source === "mock" ? listScenarios().find((s) => s.id === scenarioId) : undefined;
  const injury = (world.incident?.injury ?? null) as InjuryType | null;
  const sev = world.risk?.severity_class ?? 0;
  const bloodGroup = def?.blood ?? null; const needBlood = def ? bloodNeeded(def.sev, def.injury, def.bleeding) : injury ? bloodNeeded(sev, injury, false) : false;
  const bg = bloodGroup ?? "O-"; const stock = h ? (def?.overrides[h.id]?.[`blood_${bg}`] ?? h.blood[bg] ?? 0) : 0;
  const eta = world.ambulance?.etaS ?? null;

  const checklist = useMemo(() => {
    if (!h || !incoming) return [];
    const want = injury ? SPECIALTY[injury] : undefined; const have = want ? want.filter((w) => h.specialties.includes(w)) : [];
    return [
      { ok: h.trauma >= sev || sev < 2, label: `Trauma team (level ${h.trauma}, case class ${sev})`, sim: false },
      { ok: h.beds > 0, label: `Bed ready (${h.beds} free)`, sim: true },
      { ok: h.icu > 0 || sev < 2, label: `ICU capacity (${h.icu} free)`, sim: true },
      { ok: !needBlood || stock >= 1, label: needBlood ? `Blood ${bloodGroup ?? "group unknown, O- proxy"}: ${stock} units` : "Blood: transfusion risk not flagged", sim: true },
      { ok: !want || have.length > 0, label: want ? `Specialty: ${have.join(", ") || "none listed for this injury"}` : "Specialty: no specific need", sim: false },
    ];
  }, [h, incoming, sev, injury, needBlood, stock, bloodGroup]);

  if (world.loading) return <div className="mx-auto max-w-6xl space-y-4 p-6"><Skeleton className="h-10 w-72" /><Skeleton className="h-72 w-full" /></div>;
  return (
    <div className="mx-auto w-full max-w-6xl space-y-4 px-4 py-5 lg:py-8">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div><div className="eyebrow">Hospital view</div><h1 className="text-[24px] font-semibold tracking-tight">{h?.name ?? "Select a hospital"}</h1>
          {h && <div className="mt-1 flex flex-wrap items-center gap-2 text-[12.5px] text-mute"><span className="num rounded-md bg-s3 px-1.5 py-0.5">{h.id}</span><span className="chip"><Stethoscope className="h-3 w-3" />Trauma level {h.trauma}</span>{h.specialties.slice(0, 4).map((s) => <span key={s} className="chip capitalize">{s.replace("_", " ")}</span>)}</div>}</div>
        <select value={hid} onChange={(e) => nav(`/hospital/${e.target.value}`)} aria-label="Hospital" className="h-9 max-w-[280px] rounded-xl border hairline bg-s2 px-3 text-[13px] outline-none focus-visible:ring-2 focus-visible:ring-brand">
          {HOSPITALS.map((x) => <option key={x.id} value={x.id}>{x.id} · {x.name}</option>)}
        </select>
      </div>
      <ReplayBar />

      <div className="grid gap-4 lg:grid-cols-[1.35fr_1fr]">
        <Panel title={<span className="flex items-center gap-2"><Siren className="h-3.5 w-3.5 text-sev3" />Incoming patient</span>} right={incoming ? <SeverityBadge cls={world.risk?.severity_class} pulse /> : undefined}>
          {!incoming ? <EmptyState icon={Activity} title="No patient inbound" body={world.decision ? <>The current incident is going to <Link className="text-brand hover:underline" to={`/hospital/${world.decision.hospital_id}`}>{world.decision.hospital_id}</Link>.</> : "When the coordinator dispatches an ambulance here, the patient card fills in live."} /> : (
            <div className="grid gap-5 md:grid-cols-[220px_1fr]">
              <div>
                <RiskGauge score={world.risk?.risk_score ?? null} cls={world.risk?.severity_class ?? null} size={220} />
                <div className="mt-3 rounded-xl border hairline bg-gradient-to-br from-brand/15 to-transparent p-3 text-center">
                  <div className="eyebrow !text-[10px] flex items-center justify-center gap-1"><Clock className="h-3 w-3" />Arrives in {world.source === "mock" && <SimTag>Sim</SimTag>}</div>
                  <motion.div key={eta == null ? "x" : Math.floor((eta ?? 0) / 10)} className="num text-[34px] font-semibold leading-tight">{eta == null ? "–:–" : mmss(eta)}</motion.div>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-2 text-center">
                  <div className="card px-2 py-2"><div className="eyebrow !text-[10px]">Trend ×6</div><AnimatedNumber value={world.risk?.trend_prob != null ? world.risk.trend_prob * 100 : null} suffix="%" className="text-[18px] font-semibold" /></div>
                  <div className="card px-2 py-2"><div className="eyebrow !text-[10px]">NEWS2</div><AnimatedNumber value={world.risk?.news2_score ?? null} className="text-[18px] font-semibold" /></div>
                </div>
              </div>
              <div className="space-y-4">
                <div><div className="mb-1.5 flex items-center justify-between"><span className="eyebrow flex items-center gap-1"><HeartPulse className="h-3 w-3" />Vitals trend</span>{world.source === "mock" && <SimTag />}</div><VitalsStrip vitals={world.vitals} compact /></div>
                <div>
                  <div className="eyebrow mb-1.5">Readiness checklist <SimTag>Sim data</SimTag></div>
                  <ul className="space-y-1.5">{checklist.map((c) => (
                    <li key={c.label} className="flex items-center gap-2.5 text-[13px]"><span className={clsx("grid h-5 w-5 shrink-0 place-items-center rounded-full", c.ok ? "bg-sev0/20 text-sev0" : "bg-sev3/20 text-sev3")}>{c.ok ? <Check className="h-3 w-3" /> : <XIcon className="h-3 w-3" />}</span><span className={clsx(!c.ok && "text-sev3")}>{c.label}</span></li>))}</ul>
                </div>
              </div>
            </div>
          )}
        </Panel>

        <div className="space-y-4">
          <Panel title={<span className="flex items-center gap-2"><BedDouble className="h-3.5 w-3.5" />Beds</span>} right={<SimTag />}>
            {h && <div className="grid grid-cols-2 gap-2">
              <div className="card px-3 py-2.5"><div className="eyebrow !text-[10px]">Available</div><div className={clsx("num text-[26px] font-semibold", h.beds === 0 && "text-sev3")}>{h.beds}</div></div>
              <div className="card px-3 py-2.5"><div className="eyebrow !text-[10px]">ICU available</div><div className={clsx("num text-[26px] font-semibold", h.icu === 0 && "text-sev3")}>{h.icu}</div></div>
            </div>}
          </Panel>
          <Panel title={<span className="flex items-center gap-2"><Droplets className="h-3.5 w-3.5" />Blood stock (units)</span>} right={<SimTag />}>
            {h && <ul className="grid grid-cols-2 gap-x-4 gap-y-2">{Object.entries(h.blood).map(([g, u]) => {
              const need = incoming && needBlood && g === bg; const units = def?.overrides[h.id]?.[`blood_${g}`] ?? u;
              return (<li key={g} className={clsx("grid grid-cols-[34px_1fr_26px] items-center gap-2 text-[12.5px]", need && "font-semibold")}>
                <span className={clsx("num", need && "text-brand")}>{g}</span>
                <div className="h-1.5 overflow-hidden rounded-full bg-s3"><motion.div className={clsx("h-full rounded-full", units < 1 ? "bg-sev3" : units < 4 ? "bg-sev1" : "bg-sev0")} initial={false} animate={{ width: `${Math.min(units / 30, 1) * 100}%` }} /></div>
                <span className="num text-right text-mute">{units}</span></li>); })}</ul>}
            {incoming && needBlood && <p className="mt-3 text-[12px] text-mute">Highlighted: required group {bloodGroup ?? "unknown (O− used as proxy)"}.</p>}
          </Panel>
          <Panel title="Ambulance vitals feed" pad>
            <div className="flex items-center justify-between text-[12.5px] text-mute"><span>Readings received</span><span className="num text-fg">{world.vitalsCount}</span></div>
            <div className="mt-2"><Sparkline values={world.riskHistory.map((r) => r.score)} width={300} height={48} minSpan={0.3} label="Risk score" /></div>
            <div className="mt-1 text-[11px] text-faint">Risk score history this incident</div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
