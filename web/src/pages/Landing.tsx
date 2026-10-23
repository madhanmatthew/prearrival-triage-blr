import { Link } from "react-router-dom";
import { motion, useReducedMotion } from "framer-motion";
import { Activity, ArrowRight, Hospital, Mic, Radio, ShieldAlert, TriangleAlert } from "lucide-react";
import { useSession } from "@/store/world";
import { repo } from "@/mocks/data";

const NODES: { id: string; x: number; y: number; t: string; c: string; sim?: boolean }[] = [
  { id: "report", x: 30, y: 110, t: "Bystander report", c: "voice or text · EN HI KN" },
  { id: "dialogue", x: 170, y: 110, t: "Guided dialogue", c: "6 fixed questions" },
  { id: "text", x: 310, y: 110, t: "Text severity", c: "TF-IDF + LR" },
  { id: "fusion", x: 450, y: 110, t: "Fused risk score", c: "BiGRU + text · 0 to 1" },
  { id: "hospital", x: 590, y: 110, t: "Hospital agent", c: "trauma · beds · blood" },
  { id: "coord", x: 730, y: 110, t: "Coordinator", c: "6 rules · rule-based" },
  { id: "dispatch", x: 870, y: 110, t: "Dispatch", c: "ambulance + police" },
  { id: "rag", x: 170, y: 250, t: "First-aid retrieval", c: "IFRC text, with source" },
  { id: "guide", x: 310, y: 250, t: "Bystander guidance", c: "or: wait for help" },
  { id: "vitals", x: 450, y: 250, t: "Ambulance vitals", c: "6-reading window", sim: true },
  { id: "rl", x: 730, y: 250, t: "Signal control", c: "RL in SUMO · 4 junctions", sim: true },
];
const W = 110, H = 64;
const EDGES: [string, string, string][] = [
  ["report", "dialogue", "m140 142 h30"], ["dialogue", "text", "m280 142 h30"], ["text", "fusion", "m420 142 h30"], ["fusion", "hospital", "m560 142 h30"], ["hospital", "coord", "m700 142 h30"], ["coord", "dispatch", "m840 142 h30"],
  ["dialogue", "rag", "m225 174 v76"], ["rag", "guide", "m280 282 h30"], ["vitals", "fusion", "m505 250 v-44"], ["rl", "coord", "m785 250 v-44"],
];

function Pipeline() {
  const reduce = useReducedMotion();
  return (
    <svg viewBox="0 92 1000 250" className="w-full" role="img" aria-label="System pipeline: a bystander report goes through a guided dialogue and text severity, is fused with ambulance vitals into a risk score, the hospital agent ranks hospitals, and a rule-based coordinator dispatches with signal control and police alerts.">
      <defs><linearGradient id="edge" x1="0" x2="1"><stop offset="0" stopColor="rgb(var(--c-brand))" stopOpacity=".2" /><stop offset="1" stopColor="rgb(var(--c-brand))" stopOpacity=".7" /></linearGradient></defs>
      {EDGES.map(([, , d], i) => (
        <g key={i}><path id={`e${i}`} d={d.replace(/^m/, "M")} fill="none" stroke="url(#edge)" strokeWidth="2" strokeDasharray="3 5" />
          {!reduce && <circle r="3.5" fill="rgb(var(--c-brand))" style={{ filter: "drop-shadow(0 0 5px rgb(var(--c-brand)))" }}><animateMotion dur={`${2.2 + (i % 3) * 0.5}s`} repeatCount="indefinite" begin={`${i * 0.25}s`}><mpath href={`#e${i}`} /></animateMotion></circle>}</g>
      ))}
      {NODES.map((n) => (
        <g key={n.id} transform={`translate(${n.x} ${n.y + 0})`}>
          <rect width={W} height={H} rx="14" fill="rgb(var(--c-s2))" stroke="rgb(var(--c-line) / .16)" />
          <rect width={W} height={H} rx="14" fill="none" stroke={n.sim ? "rgb(var(--c-sim) / .5)" : "rgb(var(--c-brand) / .35)"} strokeDasharray={n.sim ? "4 4" : undefined} />
          <text x={W / 2} y={26} textAnchor="middle" fontSize="12.5" fontWeight="600" fill="rgb(var(--c-fg))">{n.t}</text>
          <text x={W / 2} y={44} textAnchor="middle" fontSize="9.5" fill="rgb(var(--c-mute))">{n.c}</text>
        </g>
      ))}
      <text x="30" y="330" fontSize="11" fill="rgb(var(--c-faint))">Dashed purple outline = simulated input (vitals stream, SUMO traffic).</text>
    </svg>
  );
}

const CARDS = [
  { to: "/report", icon: Mic, title: "Report", body: "Mobile-first intake in three languages with source-cited first-aid steps." },
  { to: "/command", icon: Radio, title: "Command Center", body: "Map, fused risk score, hospital ranking and the coordinator's decision log." },
  { to: "/hospital", icon: Hospital, title: "Hospital", body: "Incoming-patient card, readiness checklist, bed and blood status." },
  { to: "/police", icon: ShieldAlert, title: "Police / traffic", body: "ORR corridor signals, RL actions and rule-based alert sizing." },
  { to: "/evaluation", icon: Activity, title: "Evaluation", body: "Every result traced to the experiment log, including the ones that disappoint." },
];
const LIMITS = [
  "Fusion training pairs are synthetic: MIMIC has no bystander text, so the ablation measures the pair generator's assumptions as well as the model.",
  "MIMIC is ICU data and VitalDB is intra-operative. Neither is pre-hospital: the domain gap is a stated limitation.",
  "MIMIC-IV Demo has 100 patients. Expect high variance and wide confidence intervals.",
  "RL results are from simulation on an approximated corridor, calibrated on speeds only.",
  "Blood stock and bed occupancy are simulated. ETAs are straight-line estimates.",
  "Hindi and Kannada speech recognition is clearly weaker than English.",
];

export default function Landing() {
  const play = useSession((s) => s.play); const source = useSession((s) => s.source);
  return (
    <div className="relative">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[520px] overflow-hidden" aria-hidden>
        <div className="absolute inset-0 opacity-[.5]" style={{ backgroundImage: "linear-gradient(rgb(var(--c-line) / .06) 1px, transparent 1px), linear-gradient(90deg, rgb(var(--c-line) / .06) 1px, transparent 1px)", backgroundSize: "44px 44px", maskImage: "radial-gradient(70% 80% at 50% 0%, #000 30%, transparent 75%)", WebkitMaskImage: "radial-gradient(70% 80% at 50% 0%, #000 30%, transparent 75%)" }} />
        <div className="absolute left-1/2 top-[-160px] h-[420px] w-[820px] -translate-x-1/2 rounded-full bg-brand/20 blur-[110px]" />
      </div>
      <div className="relative mx-auto max-w-6xl px-4 pb-16 pt-12 lg:pt-20">
        <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="max-w-3xl">
          <span className="chip !h-7 gap-2 !px-3 text-[12px]"><span className="h-1.5 w-1.5 rounded-full bg-sev3" />Research prototype · Bengaluru · ORR corridor</span>
          <h1 className="mt-5 text-[40px] font-semibold leading-[1.05] tracking-tight sm:text-[56px]">From the first call to the <span className="bg-gradient-to-r from-brand to-sev0 bg-clip-text text-transparent">right hospital</span>.</h1>
          <p className="mt-5 max-w-2xl text-[16px] leading-relaxed text-mute">A pre-arrival triage and coordination system. A bystander reports by voice or text, vitals from the ambulance are fused into a single risk score, signals on the Outer Ring Road are cleared in simulation, and a rule-based coordinator picks the receiving hospital. Decision support for a human dispatcher.</p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link to="/command" onClick={() => source === "mock" && play(true)} className="btn btn-brand !h-11 !px-5 !text-[14px]">Open the Command Center<ArrowRight className="h-4 w-4" /></Link>
            <Link to="/report" className="btn !h-11 !px-5 !text-[14px]"><Mic className="h-4 w-4" />Report an emergency</Link>
          </div>
        </motion.div>

        <div className="mt-12 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[[repo.hospitals.length || "–", "hospitals, matched to OSM"], [repo.scenarios.length || "–", "benchmark scenarios"], ["4", "ORR signals in the RL corridor"], ["3", "languages: EN · हिन्दी · ಕನ್ನಡ"]].map(([n, l], i) => (
            <motion.div key={l as string} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 + i * 0.07 }} className="surface px-4 py-3"><div className="num text-[28px] font-semibold">{n}</div><div className="text-[12px] text-mute">{l}</div></motion.div>))}
        </div>

        <section className="mt-14" aria-label="Pipeline">
          <h2 className="eyebrow mb-3">How a case flows</h2>
          <div className="surface hidden p-5 md:block"><Pipeline /></div>
          <ol className="surface space-y-2 p-4 md:hidden text-[13px]">{["Bystander report (voice or text)", "Guided dialogue, 6 fixed questions", "Text severity + first-aid retrieval", "Fusion with ambulance vitals → risk score", "Hospital agent ranks hospitals", "Rule-based coordinator resolves 6 conflicts", "Dispatch: ambulance, police, signals"].map((s, i) => <li key={s} className="flex gap-3"><span className="num grid h-5 w-5 shrink-0 place-items-center rounded-md bg-s3 text-[11px]">{i + 1}</span>{s}</li>)}</ol>
        </section>

        <section className="mt-14 grid gap-3 sm:grid-cols-2 lg:grid-cols-3" aria-label="Screens">
          {CARDS.map(({ to, icon: Icon, title, body }, i) => (
            <motion.div key={to} initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.05 }}>
              <Link to={to} className="surface group block h-full p-5 transition hover:-translate-y-0.5 hover:border-brand/40">
                <div className="grid h-10 w-10 place-items-center rounded-xl bg-brand/15 text-brand ring-1 ring-inset ring-brand/25"><Icon className="h-5 w-5" /></div>
                <h3 className="mt-3 flex items-center gap-1.5 text-[15px] font-semibold">{title}<ArrowRight className="h-3.5 w-3.5 -translate-x-1 opacity-0 transition group-hover:translate-x-0 group-hover:opacity-100" /></h3>
                <p className="mt-1 text-[13px] leading-relaxed text-mute">{body}</p>
              </Link>
            </motion.div>))}
        </section>

        <section className="mt-14 rounded-2xl border border-sev1/30 bg-sev1/[.06] p-6" aria-label="Honest limits">
          <h2 className="flex items-center gap-2 text-[16px] font-semibold"><TriangleAlert className="h-4 w-4 text-sev1" />Honest limits</h2>
          <ul className="mt-3 grid gap-x-8 gap-y-2 text-[13px] leading-relaxed text-mute md:grid-cols-2">{LIMITS.map((l) => <li key={l} className="flex gap-2"><span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-sev1" />{l}</li>)}</ul>
          <p className="mt-4 text-[12.5px] text-fg/80">The contribution is context and integration (pre-arrival, multilingual, low-resource, Bengaluru), not a new architecture. Not a medical device.</p>
        </section>
      </div>
    </div>
  );
}
