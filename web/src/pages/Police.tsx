import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { BadgeCheck, Cone, Radio, ShieldAlert, Timer, Users, Zap } from "lucide-react";
import clsx from "clsx";
import { useWorld, useSession, type World } from "@/store/world";
import { JUNCTION_LABEL, JUNCTION_ORDER } from "@/mocks/data";
import { ReplayBar } from "@/components/ReplayBar";
import { EmptyState, Panel, SeverityBadge, SimTag } from "@/components/ui";
import { cumulative, haversineM, projectOnPolyline } from "@/lib/geo";
import { hhmm } from "@/lib/format";
import { api } from "@/api/client";
import type { JunctionId, Phase } from "@/api/types";

const PHASE: Record<Phase, { label: string; hex: string }> = { main_green: { label: "Main green", hex: "#34d399" }, cross_green: { label: "Cross green", hex: "#fb7c3c" }, yellow: { label: "Yellow", hex: "#fbbf24" } };

/* ------------------------------------ corridor schematic ------------------------------------ */
function Corridor({ world }: { world: World }) {
  const W = 1000, H = 230, x0 = 70, x1 = 930;
  const legs = JUNCTION_ORDER.slice(1).map((j, i) => haversineM(world.junctions[JUNCTION_ORDER[i]], world.junctions[j]));
  const tot = legs.reduce((a, b) => a + b, 0) || 1;
  const xs = JUNCTION_ORDER.map((_, i) => x0 + ((legs.slice(0, i).reduce((a, b) => a + b, 0)) / tot) * (x1 - x0));
  const { route, ambulance, signals } = world;
  const cum = useMemo(() => (route.length > 1 ? cumulative(route) : []), [route]);
  const dj = JUNCTION_ORDER.map((j) => (cum.length ? projectOnPolyline(route, cum, world.junctions[j]) : 0));
  const onCorridor = signals ? JUNCTION_ORDER.filter((j) => signals[j].onRoute).length >= 2 : false;
  const dAmb = ambulance && cum.length ? projectOnPolyline(route, cum, ambulance.position) : 0;
  let ax = xs[0];
  if (onCorridor) {
    if (dAmb <= dj[0]) ax = xs[0] - ((dj[0] - dAmb) / 1500) * 40;
    else if (dAmb >= dj[3]) ax = xs[3] + Math.min((dAmb - dj[3]) / 1500, 1) * 36;
    else for (let i = 0; i < 3; i++) if (dAmb >= dj[i] && dAmb <= dj[i + 1]) ax = xs[i] + ((dAmb - dj[i]) / (dj[i + 1] - dj[i] || 1)) * (xs[i + 1] - xs[i]);
  }
  const greens = signals ? JUNCTION_ORDER.map((j, i) => signals[j].preempted ? [Math.max(ax, xs[i] - 150), xs[i] + 30] : null).filter(Boolean) as [number, number][] : [];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="ORR corridor schematic with four signalised junctions and the ambulance position">
      <rect x={x0 - 50} y={95} width={x1 - x0 + 100} height={40} rx={20} fill="rgb(var(--c-s3))" />
      <line x1={x0 - 40} x2={x1 + 40} y1={115} y2={115} stroke="rgb(var(--c-faint))" strokeOpacity=".5" strokeWidth="2" strokeDasharray="10 12" />
      {greens.map(([a, b], i) => <rect key={i} x={a} y={95} width={Math.max(b - a, 0)} height={40} rx={20} fill="#34d399" fillOpacity=".28" style={{ filter: "drop-shadow(0 0 10px #34d399)" }} />)}
      {JUNCTION_ORDER.map((j, i) => {
        const s = signals?.[j]; const ph = s ? PHASE[s.phase] : null; const x = xs[i];
        return (
          <g key={j} opacity={s && !s.onRoute ? 0.45 : 1}>
            <line x1={x} x2={x} y1={50} y2={180} stroke="rgb(var(--c-faint))" strokeOpacity=".4" strokeWidth="14" strokeLinecap="round" />
            <rect x={x - 17} y={14} width={34} height={64} rx={10} fill="rgb(var(--c-s1))" stroke="rgb(var(--c-line) / .25)" />
            {[0, 1, 2].map((k) => { const on = ph && ((k === 0 && s!.phase === "cross_green") || (k === 1 && s!.phase === "yellow") || (k === 2 && s!.phase === "main_green")); const col = ["#f43f5e", "#fbbf24", "#34d399"][k];
              return <circle key={k} cx={x} cy={29 + k * 17} r={6} fill={on ? col : "rgb(var(--c-line) / .12)"} style={on ? { filter: `drop-shadow(0 0 6px ${col})` } : undefined} />; })}
            <text x={x} y={205} textAnchor="middle" fontSize="15" fontWeight="600" fill="rgb(var(--c-fg))">{JUNCTION_LABEL[j]}</text>
            <text x={x} y={223} textAnchor="middle" fontSize="12" fill={ph ? ph.hex : "rgb(var(--c-faint))"}>{s ? (s.preempted ? "RL hold · " : "") + (s.onRoute ? ph!.label : "off route") : "–"}</text>
            {i < 3 && <text x={(xs[i] + xs[i + 1]) / 2} y={160} textAnchor="middle" fontSize="12" className="num" fill="rgb(var(--c-faint))">{(legs[i] / 1000).toFixed(1)} km</text>}
          </g>
        );
      })}
      {ambulance && world.decision && (
        <motion.g initial={false} animate={{ x: ax }} transition={{ type: "spring", stiffness: 60, damping: 20 }}>
          <circle cy={115} r={19} fill="rgb(var(--c-s1))" stroke="#fff" strokeWidth="2" />
          <path d="M-8 120 L8 120 L8 111 L2 106 L-8 106 Z" fill="rgb(var(--c-brand))" /><circle cx={-4} cy={121} r={2.4} fill="#fff" /><circle cx={4} cy={121} r={2.4} fill="#fff" />
          <circle cy={101} r={3} fill="#f43f5e"><animate attributeName="opacity" values="1;.2;1" dur="0.9s" repeatCount="indefinite" /></circle>
        </motion.g>
      )}
    </svg>
  );
}

/* ------------------------------------------ page ------------------------------------------ */
export default function Police() {
  const world = useWorld(); const source = useSession((s) => s.source);
  const { police, incident, risk, signals, rlEvents } = world;
  const [busy, setBusy] = useState<string | null>(null); const [err, setErr] = useState<string | null>(null);
  const live = async (k: string, fn: () => Promise<unknown>) => { setBusy(k); setErr(null); try { await fn(); } catch (e) { setErr((e as Error).message); } finally { setBusy(null); } };
  const injured = incident?.injured ?? 0, inj = incident?.injury ?? null, cls = police?.severity_class ?? risk?.severity_class ?? 0;
  const sizing = police && inj ? (() => {
    const base = [0, 1, 2, 3][cls]; const extra = base > 0 ? Math.floor(Math.max(injured - 2, 0) / 2) : 0; const road = base > 0 && inj === "road_accident" ? 1 : 0;
    return { base, extra, road, raw: base + extra + road };
  })() : null;
  return (
    <div className="mx-auto w-full max-w-6xl space-y-4 px-4 py-5 lg:py-8">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div><div className="eyebrow">Police / traffic view</div><h1 className="text-[24px] font-semibold tracking-tight">ORR corridor control</h1>
          <p className="mt-1 max-w-xl text-[13px] text-mute">Four signals on the Outer Ring Road, a reinforcement-learning controller (simulation) and rule-based police alert sizing.</p></div>
        {incident && <div className="flex items-center gap-2"><span className="num text-[13px] text-mute">{incident.id}</span><SeverityBadge cls={risk?.severity_class} pulse /></div>}
      </div>
      <ReplayBar />
      <Panel title={<span className="flex items-center gap-2"><Radio className="h-3.5 w-3.5" />Corridor schematic · eastbound</span>} right={source === "mock" ? <SimTag title="Scripted phases, not live RL output">Scripted signals</SimTag> : undefined}>
        {!signals && source === "live" ? <EmptyState icon={Radio} title="Signal state not exposed yet" body="SignalState has no endpoint or WebSocket message in the backend. Switch to the replay to see the corridor schematic animate." /> : <Corridor world={world} />}
      </Panel>

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title={<span className="flex items-center gap-2"><Zap className="h-3.5 w-3.5" />RL action timeline</span>} right={source === "mock" ? <SimTag>Scripted</SimTag> : undefined}>
          {rlEvents.length === 0 ? <EmptyState icon={Timer} title="No pre-emption yet" body="Actions appear as the ambulance approaches each signal." /> : (
            <ol className="space-y-2">{rlEvents.map((e, i) => (
              <motion.li key={`${e.junction}`} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }} className="flex items-center gap-3 text-[13px]">
                <span className="num rounded-md bg-s3 px-1.5 py-0.5 text-[11px] text-mute">t+{e.at}s</span><span className="font-medium">{JUNCTION_LABEL[e.junction as JunctionId]}</span><span className="text-mute">{e.action}</span></motion.li>))}</ol>)}
          <p className="mt-3 text-[11px] leading-relaxed text-faint">Evaluated in SUMO on an OSM corridor with speed-only calibration. Results live on the Evaluation screen.</p>
        </Panel>

        <Panel title={<span className="flex items-center gap-2"><Users className="h-3.5 w-3.5" />Alert sizing</span>} right={<span className="chip">Rule-based</span>}>
          {!police ? <EmptyState icon={ShieldAlert} title="No alert yet" body="A police alert is created when the incident is dispatched." /> : (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <div className="card px-3 py-2.5"><div className="eyebrow !text-[10px] flex items-center gap-1"><Users className="h-3 w-3" />Units required</div><div className="num text-[30px] font-semibold leading-tight">{police.units_required}</div></div>
                <div className="card px-3 py-2.5"><div className="eyebrow !text-[10px] flex items-center gap-1"><Cone className="h-3 w-3" />Perimeter</div><div className="num text-[30px] font-semibold leading-tight">{Math.round(police.perimeter_m)}<span className="text-[13px] text-mute"> m</span></div></div>
              </div>
              {sizing && <dl className="num space-y-1 text-[12px]">
                {([["class " + cls + " base", sizing.base], ["+ injured beyond 2 (per 2)", sizing.extra], ["+ road accident", sizing.road]] as [string, number][]).map(([k, v]) => <div key={k} className="flex justify-between"><dt className="text-mute">{k}</dt><dd>{v}</dd></div>)}
                <div className="flex justify-between border-t hairline pt-1 font-semibold"><dt>total (cap 6)</dt><dd>{Math.min(sizing.raw, 6)}</dd></div>
              </dl>}
              <p className="text-[11px] leading-relaxed text-faint">Constants are assumptions, to be verified with traffic-police practice. Decision support, not automation.</p>
            </div>)}
        </Panel>

        <Panel title={<span className="flex items-center gap-2"><BadgeCheck className="h-3.5 w-3.5" />Junction posts</span>} right={police && source === "live" ? <button className="btn !h-7 !px-2.5 !text-[11px]" disabled={busy !== null || police.alert_status === "closed"} onClick={() => live("close", () => api.close(police.incident_id))}>Close alert</button> : undefined}>
          {!police || police.junction_posts.length === 0 ? <EmptyState icon={Cone} title="No junction posts" body="Posts exist only when the route passes the four corridor junctions." /> : (
            <ul className="space-y-2">{police.junction_posts.map((p) => (
              <li key={p.junction_id} className={clsx("rounded-xl border-l-4 bg-s2 px-3 py-2", p.status === "released" ? "border-faint" : p.manual_clearing ? "border-sev3" : "border-brand")}>
                <div className="flex items-center justify-between"><span className="text-[13px] font-semibold">{JUNCTION_LABEL[p.junction_id] ?? p.junction_id}</span><span className={clsx("chip !h-5 !text-[10px] capitalize", p.status === "released" && "text-mute", p.status === "acknowledged" && "!bg-sev0/20 text-sev0", p.status === "pending" && "!bg-sev1/20 text-sev1")}>{p.status}</span></div>
                <div className="num mt-1 flex justify-between text-[11.5px] text-mute"><span>Officer by {hhmm(p.officer_by)}</span><span>ETA {hhmm(p.ambulance_eta)}</span></div>
                {source === "live" && p.status === "pending" && <button className="btn mt-2 !h-7 !px-2.5 !text-[11px]" disabled={busy !== null} onClick={() => live(p.junction_id, () => api.ack(police.incident_id, p.junction_id))}>Acknowledge</button>}
              </li>))}</ul>)}
          {err && <p role="alert" className="mt-2 text-[12px] text-sev3">{err}</p>}
        </Panel>
      </div>
    </div>
  );
}
