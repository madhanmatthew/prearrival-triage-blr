import { useState } from "react";
import { AnimatePresence, LayoutGroup, motion } from "framer-motion";
import { Activity, Ambulance, Check, ChevronRight, CircleDashed, Clock, Droplets, HeartPulse, MapPin, Sparkles, Users, Zap } from "lucide-react";
import { JUNCTION_LABEL, JUNCTION_ORDER } from "@/mocks/data";
import clsx from "clsx";
import type { World } from "@/store/world";
import { AnimatedNumber, EmptyState, RiskGauge, SeverityBadge, SimTag, Skeleton } from "@/components/ui";
import { VitalsStrip } from "./Vitals";
import { fix, mmss, pct } from "@/lib/format";
import { asSeverity, severityStyle } from "@/lib/severity";
import type { Phase4 } from "@/mocks/scenario";

/* ------------------------------------------- left: incident ------------------------------------------- */
const STEPS: { id: Phase4[]; label: string }[] = [
  { id: ["intake"], label: "Report" }, { id: ["triage"], label: "Triage" }, { id: ["dispatch"], label: "Dispatch" },
  { id: ["to_scene", "on_scene"], label: "To scene" }, { id: ["transport"], label: "Transport" }, { id: ["arrived"], label: "Arrived" },
];
export function PhaseStepper({ phase }: { phase: World["phase"] }) {
  const idx = phase === "live" ? -1 : STEPS.findIndex((s) => s.id.includes(phase));
  if (phase === "live") return null;
  return (
    <ol className="flex items-center gap-1" aria-label="Incident progress">
      {STEPS.map((s, i) => (
        <li key={s.label} className="flex flex-1 flex-col gap-1" aria-current={i === idx ? "step" : undefined}>
          <div className="h-1 overflow-hidden rounded-full bg-s3"><motion.div className="h-full rounded-full bg-brand" initial={false} animate={{ width: i < idx ? "100%" : i === idx ? "55%" : "0%" }} transition={{ duration: 0.4 }} /></div>
          <span className={clsx("truncate text-[9px] font-semibold uppercase", i <= idx ? "text-fg" : "text-faint")}>{s.label}</span>
        </li>
      ))}
    </ol>
  );
}

export function IncidentPanel({ world }: { world: World }) {
  const { incident, decision, ranking, ambulance, risk, textProbs } = world;
  const dest = ranking?.find((r) => r.chosen) ?? null;
  const textCls = textProbs ? asSeverity(textProbs.indexOf(Math.max(...textProbs))) : null;
  if (world.loading) return <div className="space-y-3 p-4"><Skeleton className="h-5 w-40" /><Skeleton className="h-16 w-full" /><Skeleton className="h-24 w-full" /></div>;
  if (!incident) return <EmptyState icon={MapPin} title="No incident selected" body="Reports appear here as soon as a bystander submits one. Use the Report screen to create one, or switch to the replay." />;
  const eta = ambulance?.etaS;
  return (
    <div className="space-y-3.5 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="eyebrow">Incident</div>
          <h2 className="num truncate text-[15px] font-semibold">{incident.id}</h2>
        </div>
        <SeverityBadge cls={risk?.severity_class ?? null} pulse />
      </div>
      <PhaseStepper phase={world.phase} />
      {incident.transcript && <p className="line-clamp-2 text-[12.5px] leading-relaxed text-mute">“{incident.transcript}”</p>}
      <div className="grid grid-cols-2 gap-2">
        <div className="card px-3 py-2"><div className="eyebrow !text-[10px]">Type</div><div className="text-[13px] font-medium capitalize">{incident.injury?.replace("_", " ") ?? "–"}</div></div>
        <div className="card px-3 py-2"><div className="eyebrow !text-[10px] flex items-center gap-1"><Users className="h-3 w-3" />Injured</div><div className="num text-[15px] font-semibold">{incident.injured ?? "–"}</div></div>
      </div>
      <div className="card px-3 py-2.5">
        <div className="flex items-center justify-between"><span className="eyebrow !text-[10px]">Text severity <span className="normal-case tracking-normal text-faint">(report only)</span></span>{world.source === "mock" && textProbs && <SimTag>Scripted</SimTag>}</div>
        <div className="mt-1.5 flex items-center gap-2">
          {textCls == null ? <span className="text-[12.5px] text-mute">Awaiting report</span> : <SeverityBadge cls={textCls} size="sm" />}
          {textProbs && <div className="flex h-1.5 flex-1 overflow-hidden rounded-full bg-s3" aria-hidden>{textProbs.map((p, i) => <div key={i} style={{ width: `${p * 100}%`, background: severityStyle(i).var, opacity: 0.85 }} />)}</div>}
        </div>
      </div>
      <div className="relative overflow-hidden rounded-xl border hairline bg-gradient-to-br from-brand/15 via-s2 to-s2 px-3.5 py-3">
        <div className="flex items-center justify-between"><span className="eyebrow !text-[10px] flex items-center gap-1"><Clock className="h-3 w-3" />ETA to hospital</span>{world.source === "mock" && <SimTag title="Straight-line ETA, simulated">Simulated</SimTag>}</div>
        <div className="num mt-1 text-[34px] font-semibold leading-none tracking-tight">{eta == null ? "–:–" : mmss(eta)}</div>
        <div className="mt-2 flex items-center gap-2 text-[12px] text-mute">
          <ChevronRight className="h-3.5 w-3.5 text-brand" /><span className="truncate">{decision ? (dest ? `${dest.hospital_id} · ${dest.name}` : decision.hospital_id) : "Awaiting dispatch"}</span>
        </div>
      </div>
      <SignalsCard world={world} />
      {ambulance && decision && (
        <div className="flex items-center justify-between rounded-xl border hairline px-3 py-2 text-[12px]">
          <span className="flex items-center gap-2"><Ambulance className="h-4 w-4 text-brand" /><span className="num font-medium">{ambulance.id}</span></span>
          <span className="num text-mute">{ambulance.speedKmh != null ? `${Math.round(ambulance.speedKmh)} km/h` : "–"}</span>
          <span className="chip">{decision.police_units} police {decision.police_units === 1 ? "unit" : "units"}</span>
        </div>
      )}
    </div>
  );
}


/* ------------------------------------------- ORR corridor signals ------------------------------------------- */
const PH = { main_green: { label: "Main green", cls: "text-sev0 bg-sev0/15" }, cross_green: { label: "Cross green", cls: "text-sev2 bg-sev2/10" }, yellow: { label: "Yellow", cls: "text-sev1 bg-sev1/15" } } as const;
export function SignalsCard({ world }: { world: World }) {
  const sig = world.signals;
  return (
    <div className="card px-3 py-2.5">
      <div className="mb-1.5 flex items-center justify-between"><span className="eyebrow !text-[10px] flex items-center gap-1"><Zap className="h-3 w-3" />ORR signals · RL corridor</span>{world.source === "mock" && <SimTag title="Scripted phases, not live RL output">Scripted</SimTag>}</div>
      {!sig ? <p className="text-[12px] text-mute">Signal state is not exposed by the live API yet.</p> : (
        <ul className="space-y-1">
          {JUNCTION_ORDER.map((j) => { const x = sig[j]; const ph = PH[x.phase]; return (
            <li key={j} className={clsx("flex items-center justify-between gap-2 text-[12px]", !x.onRoute && "opacity-45")}>
              <span className="flex min-w-0 items-center gap-1.5"><span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: { main_green: "#34d399", cross_green: "#fb7c3c", yellow: "#fbbf24" }[x.phase] }} /><span className="truncate font-medium">{JUNCTION_LABEL[j]}</span></span>
              <span className="flex items-center gap-1.5">
                {x.preempted && <span className="chip !h-[18px] !bg-sev0/20 text-sev0 !text-[10px]"><Zap className="h-2.5 w-2.5" />RL hold</span>}
                <span className={clsx("chip !h-[18px] !text-[10px]", ph.cls)}>{x.onRoute ? ph.label : "off route"}</span>
              </span>
            </li>); })}
        </ul>
      )}
    </div>
  );
}

/* ------------------------------------------- right: risk ------------------------------------------- */
export function RiskPanel({ world, onWhy }: { world: World; onWhy: () => void }) {
  const { risk, vitals } = world;
  const last = vitals[vitals.length - 1];
  if (world.loading) return <div className="space-y-3 p-4"><Skeleton className="mx-auto h-28 w-52" /><Skeleton className="h-16 w-full" /></div>;
  return (
    <div className="space-y-3.5 p-4">
      <div className="flex items-center justify-between">
        <div><div className="eyebrow">Emergency risk score</div><div className="text-[11px] text-faint">Fused · expected severity in [0, 1]</div></div>
        <button className="btn !h-8 !px-2.5" onClick={onWhy} disabled={!risk}><Sparkles className="h-3.5 w-3.5 text-brand" />Why this score</button>
      </div>
      <RiskGauge score={risk?.risk_score ?? null} cls={risk?.severity_class ?? null} size={236} />
      <div className="flex flex-wrap items-center justify-center gap-1.5">
        <SeverityBadge cls={risk?.severity_class ?? null} size="lg" pulse />
        {risk?.modalities_used.map((m) => <span key={m} className="chip capitalize">{m}</span>)}
        {risk?.stale && <span className="chip !bg-sev1/20 text-sev1">stale</span>}
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="card px-3 py-2" title="Probability of deterioration within the next 6 readings (BiGRU trend head)">
          <div className="eyebrow !text-[10px] flex items-center gap-1"><Activity className="h-3 w-3" />Trend ×6</div>
          <AnimatedNumber value={risk?.trend_prob != null ? risk.trend_prob * 100 : null} suffix="%" className="text-[20px] font-semibold" />
          <div className="mt-1 h-1 rounded-full bg-s3"><motion.div className="h-full rounded-full bg-sev2" animate={{ width: pct(risk?.trend_prob ?? 0) }} /></div>
        </div>
        <div className="card px-3 py-2" title="NEWS2 is a rule-based comparison baseline. It is never used to create labels.">
          <div className="eyebrow !text-[10px] flex items-center gap-1"><HeartPulse className="h-3 w-3" />NEWS2 baseline</div>
          <div className="flex items-baseline gap-1"><AnimatedNumber value={risk?.news2_score ?? null} className="text-[20px] font-semibold" /><span className="text-[10.5px] text-mute">/ 20</span></div>
          <div className="mt-1 text-[10px] text-faint">comparison only</div>
        </div>
      </div>
      <div className="card px-3 py-2" title="Short-horizon model trained on intra-operative data (VitalDB); shown as an additional signal.">
        <div className="eyebrow !text-[10px]">10-min drop in BP / SpO₂ <span className="normal-case tracking-normal text-faint">· intra-operative model</span></div>
        <div className="flex items-baseline justify-between"><AnimatedNumber value={risk?.deterioration_10min_prob != null ? risk.deterioration_10min_prob * 100 : null} suffix="%" className="text-[18px] font-semibold" /><span className="text-[10.5px] text-mute">{risk?.deterioration_model ?? "needs vitals"}</span></div>
      </div>
      <div>
        <div className="mb-1.5 flex items-center justify-between"><span className="eyebrow">Live vitals · 6-reading window</span>{world.source === "mock" && <SimTag />}</div>
        {last || world.source === "live" ? <VitalsStrip vitals={vitals} compact /> : <EmptyState icon={HeartPulse} title="Waiting for ambulance vitals" body="The strip starts when the first reading arrives after pickup." />}
      </div>
    </div>
  );
}

/* ------------------------------------------- bottom: hospitals + coordinator ------------------------------------------- */
const WEIGHTS = { proximity: 0.30, trauma_match: 0.25, beds: 0.20, specialty: 0.10, blood_match: 0.15 };
const LABEL = { proximity: "Proximity", trauma_match: "Trauma match", beds: "Beds", specialty: "Specialty", blood_match: "Blood match" } as const;

function Breakdown({ b }: { b: NonNullable<World["ranking"]>[number]["breakdown"] }) {
  return (
    <div className="glass w-[250px] space-y-1.5 p-3 text-[11.5px]" role="tooltip">
      <div className="eyebrow !text-[10px]">Score breakdown · factor × weight</div>
      {(Object.keys(LABEL) as (keyof typeof LABEL)[]).map((k) => (
        <div key={k} className="grid grid-cols-[84px_1fr_36px] items-center gap-2">
          <span className="text-mute">{LABEL[k]}</span>
          <div className="h-1.5 rounded-full bg-s3"><div className="h-full rounded-full bg-brand" style={{ width: `${b[k] * 100}%` }} /></div>
          <span className="num text-right">{(b[k] * 100).toFixed(0)}</span>
        </div>
      ))}
      <div className="border-t hairline pt-1.5 text-[10.5px] text-faint">Weights {Object.values(WEIGHTS).join(" / ")} (blood weight moves to proximity when no transfusion risk).</div>
    </div>
  );
}

export function HospitalTable({ world, limit = 5 }: { world: World; limit?: number }) {
  const [hover, setHover] = useState<string | null>(null);
  const rows = world.ranking?.slice(0, limit);
  if (!rows) return <EmptyState icon={Activity} title={world.source === "live" ? "Ranking not exposed by the API" : "Ranking appears after triage"} body={world.source === "live" ? "The backend returns only the chosen hospital (HospitalRecommendation has no endpoint yet). Switch to replay to see the full ranking." : "Hospitals are ranked as soon as the report has a severity."} />;
  return (
    <LayoutGroup>
      <div className="overflow-x-auto">
        <table className="w-full text-[12.5px]">
          <thead>
            <tr className="eyebrow !text-[10px] text-left [&>th]:whitespace-nowrap [&>th]:px-2.5 [&>th]:py-1.5 [&>th]:font-semibold">
              <th className="w-6">#</th><th>Hospital</th><th>Blood</th><th className="text-right">ETA</th><th className="w-[132px]">Score</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((h, i) => {
              const flag = { matched: ["text-sev0", "bg-sev0/15", "matched"], unknown: ["text-sev1", "bg-sev1/15", "unknown"], short: ["text-sev3", "bg-sev3/15", "short"] }[h.blood_flag];
              return (
                <motion.tr layout key={h.hospital_id} transition={{ type: "spring", stiffness: 260, damping: 28 }} onMouseEnter={() => setHover(h.hospital_id)} onMouseLeave={() => setHover(null)} onFocus={() => setHover(h.hospital_id)} onBlur={() => setHover(null)} tabIndex={0}
                  className={clsx("relative border-t hairline outline-none [&>td]:px-2.5 [&>td]:py-1.5", h.chosen && "bg-brand/10")}>
                  <td className="num text-mute">{i + 1}</td>
                  <td className="max-w-0 w-full">
                    <div className="flex items-center gap-1.5"><span className="truncate font-medium">{h.name}</span>{h.chosen && <span className="chip !h-5 shrink-0 !bg-brand/20 !text-[10px] text-brand"><Check className="h-3 w-3" />Destination</span>}</div>
                    <div className="num flex items-center gap-2 text-[10.5px] text-mute"><span>{h.hospital_id}</span>{h.trauma != null && <span>trauma L{h.trauma}</span>}{h.beds != null && <span className={clsx(h.beds === 0 && "text-sev3")}>beds {h.beds}</span>}{h.beds != null && <SimTag>Sim</SimTag>}</div>
                  </td>
                  <td><span className={clsx("chip whitespace-nowrap", flag[0], flag[1])}><Droplets className="h-3 w-3" />{flag[2]}</span></td>
                  <td className="num whitespace-nowrap text-right">{fix(h.eta_min, 1)} min</td>
                  <td>
                    <div className="flex items-center gap-2"><div className="h-1.5 flex-1 rounded-full bg-s3"><motion.div layout className="h-full rounded-full bg-gradient-to-r from-brand/60 to-brand" animate={{ width: `${h.total_score * 100}%` }} /></div><span className="num w-9 text-right font-semibold">{h.total_score.toFixed(2)}</span></div>
                    <AnimatePresence>{hover === h.hospital_id && <motion.div initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="absolute right-3 z-[1400] -translate-y-full" style={{ marginTop: -4 }}><Breakdown b={h.breakdown} /></motion.div>}</AnimatePresence>
                  </td>
                </motion.tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </LayoutGroup>
  );
}

export function DecisionLog({ world }: { world: World }) {
  const rules = world.rules;
  if (!rules || !rules.length) return <EmptyState icon={CircleDashed} title="No coordinator decision yet" body="The six conflict rules are evaluated when the dispatch decision is made." />;
  return (
    <div>
      <div className="mb-2 flex items-center justify-between px-1"><span className="chip !bg-brand/15 text-brand">Rule-based coordinator</span><span className="text-[11px] text-faint">fixed, inspectable rules · not RL · not a model</span></div>
      <ol className="relative space-y-1">
        <div className="absolute bottom-3 left-[15px] top-3 w-px bg-white/10" aria-hidden />
        <AnimatePresence initial={false}>
          {rules.map((r, i) => (
            <motion.li key={r.id} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.06 }} className="relative grid grid-cols-[32px_1fr] gap-2 rounded-xl px-1 py-1.5">
              <span className={clsx("z-10 grid h-[30px] w-[30px] place-items-center rounded-full text-[11px] font-semibold ring-4 ring-[rgb(var(--c-s1))]", r.fired ? "bg-sev1 text-black" : "bg-s3 text-mute")}>{r.fired ? <Check className="h-3.5 w-3.5" /> : r.rule}</span>
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-x-2"><span className="text-[12.5px] font-semibold">{r.rule}. {r.title}</span><span className={clsx("chip !h-5 text-[10px]", r.fired ? "!bg-sev1/20 text-sev1" : "text-faint")}>{r.fired ? "fired" : "not triggered"}</span></div>
                <p className="text-[12px] leading-relaxed text-mute">{r.detail}</p>
              </div>
            </motion.li>
          ))}
        </AnimatePresence>
      </ol>
    </div>
  );
}
