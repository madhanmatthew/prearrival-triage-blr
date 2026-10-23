import { animate, motion, useReducedMotion } from "framer-motion";
import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { AlertTriangle, CircleAlert, FlaskConical, Inbox, ShieldCheck, Siren, WifiOff } from "lucide-react";
import clsx from "clsx";
import { severityStyle, riskBand } from "@/lib/severity";

/* ------------------------------- severity badge (icon + label + colour) ------------------------------- */
const SEV_ICON = [ShieldCheck, CircleAlert, AlertTriangle, Siren] as const;
export function SeverityBadge({ cls, size = "md", pulse = false }: { cls: number | null | undefined; size?: "sm" | "md" | "lg"; pulse?: boolean }) {
  if (cls == null) return <span className="chip text-mute">No class yet</span>;
  const s = severityStyle(cls); const Icon = SEV_ICON[Math.round(cls)];
  const sz = size === "lg" ? "h-8 px-3 text-[13px]" : size === "sm" ? "h-5 px-1.5 text-[10px]" : "h-6 px-2.5 text-[12px]";
  return (
    <span className={clsx("relative inline-flex items-center gap-1.5 rounded-full font-semibold ring-1 ring-inset", sz, s.text, s.bg, s.ring)} aria-label={`Severity class ${cls}: ${s.name}`}>
      {pulse && cls === 3 && <span className="absolute inset-0 rounded-full bg-sev3/25 animate-pulseRing" aria-hidden />}
      <Icon className={size === "sm" ? "h-3 w-3" : "h-3.5 w-3.5"} aria-hidden />
      <span className="num">{cls}</span><span>{s.name}</span>
    </span>
  );
}

export function SimTag({ children = "Simulated", title }: { children?: ReactNode; title?: string }) {
  return (
    <span title={title ?? "Simulated or scripted value, not real data"} className="inline-flex items-center gap-1 rounded-md bg-sim/15 px-1.5 h-[18px] text-[10px] font-semibold uppercase tracking-wider text-sim ring-1 ring-inset ring-sim/30">
      <FlaskConical className="h-2.5 w-2.5" aria-hidden />{children}
    </span>
  );
}

export function Skeleton({ className }: { className?: string }) { return <div className={clsx("skeleton", className)} aria-hidden />; }

export function EmptyState({ icon: Icon = Inbox, title, body, action }: { icon?: typeof Inbox; title: string; body?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      <div className="grid h-11 w-11 place-items-center rounded-2xl bg-s3 ring-1 ring-inset ring-white/5"><Icon className="h-5 w-5 text-mute" aria-hidden /></div>
      <h3 className="text-[14px] font-semibold">{title}</h3>
      {body && <p className="max-w-sm text-[12.5px] leading-relaxed text-mute">{body}</p>}
      {action}
    </div>
  );
}
export function ErrorState({ title, body, onRetry }: { title: string; body?: ReactNode; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-col items-center gap-2 px-6 py-8 text-center">
      <div className="grid h-11 w-11 place-items-center rounded-2xl bg-sev3/15 ring-1 ring-inset ring-sev3/30"><WifiOff className="h-5 w-5 text-sev3" aria-hidden /></div>
      <h3 className="text-[14px] font-semibold">{title}</h3>
      {body && <p className="max-w-sm text-[12.5px] text-mute">{body}</p>}
      {onRetry && <button className="btn mt-1" onClick={onRetry}>Try again</button>}
    </div>
  );
}
/** "Pending: not yet run" for results that have no row in reports/ or experiments/log.csv. */
export function Pending({ what }: { what: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-dashed hairline bg-s2/50 px-4 py-5">
      <div className="grid h-9 w-9 place-items-center rounded-xl bg-s3"><Inbox className="h-4 w-4 text-faint" /></div>
      <div><div className="text-[13px] font-medium">Pending: not yet run</div><div className="text-[12px] text-mute">{what} has no row in <span className="num">reports/</span> or <span className="num">experiments/log.csv</span>.</div></div>
    </div>
  );
}

export function Panel({ title, right, children, className, pad = true }: { title?: ReactNode; right?: ReactNode; children: ReactNode; className?: string; pad?: boolean }) {
  return (
    <section className={clsx("surface", className)}>
      {(title || right) && (
        <header className="flex items-center justify-between gap-3 px-4 pt-3.5">
          <h2 className="eyebrow">{title}</h2>{right}
        </header>
      )}
      <div className={clsx(pad && "p-4", title && pad && "pt-3")}>{children}</div>
    </section>
  );
}

/* ----------------------------------- tweened number ----------------------------------- */
export function AnimatedNumber({ value, decimals = 0, suffix = "", className, fallback = "–" }: { value: number | null | undefined; decimals?: number; suffix?: string; className?: string; fallback?: string }) {
  const reduce = useReducedMotion();
  const [shown, setShown] = useState<number | null>(value ?? null);
  const prev = useRef<number | null>(value ?? null);
  useEffect(() => {
    if (value == null) { setShown(null); prev.current = null; return; }
    if (reduce || prev.current == null) { setShown(value); prev.current = value; return; }
    const c = animate(prev.current, value, { duration: 0.45, ease: "easeOut", onUpdate: (v) => setShown(v) });
    prev.current = value; return () => c.stop();
  }, [value, reduce]);
  return <span className={clsx("num tnum", className)}>{shown == null ? fallback : `${shown.toFixed(decimals)}${suffix}`}</span>;
}

/* ---------------------------------------- sparkline ---------------------------------------- */
export function Sparkline({ values, width = 120, height = 34, color = "rgb(var(--c-brand))", minSpan = 0, fill = true, label }: {
  values: (number | null)[]; width?: number; height?: number; color?: string; minSpan?: number; fill?: boolean; label?: string;
}) {
  const id = useId();
  const nums = values.filter((v): v is number => v != null);
  if (nums.length < 2) return <svg width={width} height={height} role="img" aria-label={label ? `${label}: waiting for readings` : "waiting"}><line x1="0" x2={width} y1={height / 2} y2={height / 2} stroke="currentColor" strokeOpacity=".15" strokeDasharray="3 4" /></svg>;
  const dMin = Math.min(...nums), dMax = Math.max(...nums), mid = (dMin + dMax) / 2, half = Math.max((dMax - dMin) / 2, minSpan / 2, 0.5), lo = mid - half * 1.15, hi = mid + half * 1.15, span = hi - lo || 1, pad = 3;
  const x = (i: number) => pad + (i / Math.max(values.length - 1, 1)) * (width - pad * 2);
  const y = (v: number) => height - pad - ((v - lo) / span) * (height - pad * 2);
  let d = "", open = false;
  values.forEach((v, i) => { if (v == null) { open = false; return; } d += `${open ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)} `; open = true; });
  const lastI = values.length - 1 - [...values].reverse().findIndex((v) => v != null); const lastV = values[lastI] as number;
  return (
    <svg width={width} height={height} role="img" aria-label={label ? `${label} trend, latest ${lastV}` : "trend"} className="overflow-visible">
      <defs><linearGradient id={id} x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={color} stopOpacity=".28" /><stop offset="1" stopColor={color} stopOpacity="0" /></linearGradient></defs>
      {fill && <path d={`${d} L${x(lastI)} ${height} L${x(0)} ${height} Z`} fill={`url(#${id})`} stroke="none" />}
      <path d={d} fill="none" stroke={color} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
      {values.map((v, i) => v == null && <line key={i} x1={x(i)} x2={x(i)} y1={4} y2={height - 4} stroke="currentColor" strokeOpacity=".3" strokeDasharray="2 3" />)}
      <circle cx={x(lastI)} cy={y(lastV)} r="3" fill={color}><title>{`${lastV}`}</title></circle>
    </svg>
  );
}

/* ----------------------------------------- risk gauge ----------------------------------------- */
export function RiskGauge({ score, cls, size = 220 }: { score: number | null; cls: number | null; size?: number }) {
  const reduce = useReducedMotion();
  const R = 84, C = Math.PI * R; // semicircle length
  const s = score ?? 0; const band = severityStyle(cls ?? riskBand(s));
  const bandsArc = [0, 1 / 6, 0.5, 5 / 6, 1];
  const pt = (t: number, r = R): [number, number] => [100 - r * Math.cos(Math.PI * t), 100 - r * Math.sin(Math.PI * t)];
  const arc = (a: number, b: number) => { const [x1, y1] = pt(a), [x2, y2] = pt(b); return `M${x1} ${y1} A${R} ${R} 0 0 1 ${x2} ${y2}`; };
  const [dx, dy] = pt(Math.min(Math.max(s, 0.004), 0.996));
  const spring = reduce ? { duration: 0 } : { type: "spring" as const, stiffness: 70, damping: 18 };
  return (
    <div className="relative mx-auto" style={{ width: size, height: size * 0.6 }}>
      <svg viewBox="0 -9 200 127" width="100%" height="100%" role="img" aria-label={score == null ? "Risk score not available yet" : `Emergency risk score ${score.toFixed(2)} of 1, class ${cls}`}>
        {[0, 1, 2, 3].map((i) => <path key={i} d={arc(bandsArc[i] + 0.008, bandsArc[i + 1] - 0.008)} stroke={`rgb(var(--c-sev${i}))`} strokeOpacity={score == null ? 0.15 : 0.22} strokeWidth="12" fill="none" strokeLinecap="round" />)}
        <motion.path d={arc(0.003, 0.997)} stroke={band.var} strokeWidth="12" fill="none" strokeLinecap="round" strokeDasharray={C} initial={false}
          animate={{ strokeDashoffset: C * (1 - Math.max(s, 0.004)) }} transition={spring} style={{ filter: `drop-shadow(0 0 6px ${band.var})`, opacity: score == null ? 0 : 1 }} />
        <motion.circle r="7.5" fill="rgb(var(--c-s1))" stroke="rgb(var(--c-fg))" strokeWidth="3" initial={false} animate={{ cx: dx, cy: dy, opacity: score == null ? 0 : 1 }} transition={spring} />
        {[[0, "0"], [0.5, "0.5"], [1, "1"]].map(([t, l]) => { const [x, y] = pt(t as number, 99); return <text key={l} x={x} y={y + (t === 0.5 ? 3 : 4)} fontSize="7.5" textAnchor={t === 0 ? "start" : t === 1 ? "end" : "middle"} fill="rgb(var(--c-faint))" className="num" transform={t === 0 ? "translate(-4 12)" : t === 1 ? "translate(4 12)" : ""}>{l as string}</text>; })}
      </svg>
      <div className="pointer-events-none absolute inset-x-0 bottom-[6%] flex flex-col items-center">
        <AnimatedNumber value={score} decimals={2} className="text-[38px] font-semibold leading-none" fallback="0.00" />
      </div>
    </div>
  );
}

/* --------------------------------------- status & misc --------------------------------------- */
export function Dot({ tone = "ok", pulse }: { tone?: "ok" | "warn" | "bad" | "sim"; pulse?: boolean }) {
  const c = { ok: "bg-sev0", warn: "bg-sev1", bad: "bg-sev3", sim: "bg-sim" }[tone];
  return <span className="relative inline-flex h-2 w-2"><span className={clsx("absolute inset-0 rounded-full", c, pulse && "animate-pulseRing")} /><span className={clsx("relative h-2 w-2 rounded-full", c)} /></span>;
}
export function Kbd({ children }: { children: ReactNode }) { return <kbd className="num rounded-md border hairline bg-s3 px-1.5 py-0.5 text-[10px] text-mute">{children}</kbd>; }
