import { motion } from "framer-motion";
import type { VitalsReading } from "@/api/types";
import { AnimatedNumber, Sparkline } from "@/components/ui";
import clsx from "clsx";

interface Def { key: "hr" | "rr" | "spo2" | "sbp" | "temp"; label: string; unit: string; dec: number; lo: number; hi: number; warn: (v: number) => boolean }
const DEFS: Def[] = [
  { key: "hr", label: "Heart rate", unit: "bpm", dec: 0, lo: 40, hi: 160, warn: (v) => v > 110 || v < 50 },
  { key: "rr", label: "Resp. rate", unit: "/min", dec: 0, lo: 8, hi: 36, warn: (v) => v > 24 || v < 9 },
  { key: "spo2", label: "SpO₂", unit: "%", dec: 0, lo: 80, hi: 100, warn: (v) => v < 94 },
  { key: "sbp", label: "Systolic BP", unit: "mmHg", dec: 0, lo: 60, hi: 160, warn: (v) => v < 101 },
  { key: "temp", label: "Temp", unit: "°C", dec: 1, lo: 34, hi: 40, warn: (v) => v < 36 || v > 38 },
];

export function VitalsStrip({ vitals, compact }: { vitals: VitalsReading[]; compact?: boolean }) {
  const last = vitals[vitals.length - 1];
  return (
    <div className={clsx("grid gap-1.5", compact ? "grid-cols-1" : "grid-cols-1 sm:grid-cols-2")}>
      {DEFS.map((d) => {
        const series = vitals.map((v) => v[d.key]); const cur = last?.[d.key] ?? null; const bad = cur != null && d.warn(cur);
        return (
          <div key={d.key} className="card grid grid-cols-[1fr_auto_88px] items-center gap-2 px-3 py-1.5">
            <div className="min-w-0">
              <div className="eyebrow !text-[10px] whitespace-nowrap">{d.label}</div>
              <div className="text-[10.5px] text-mute">{cur == null && last ? "no signal" : d.unit}</div>
            </div>
            <AnimatedNumber value={cur} decimals={d.dec} className={clsx("text-[22px] font-semibold", bad ? "text-sev2" : "text-fg")} />
            <Sparkline values={series} width={88} height={28} minSpan={(d.hi - d.lo) * 0.3} color={bad ? "rgb(var(--c-sev2))" : "rgb(var(--c-brand))"} label={d.label} />
          </div>
        );
      })}
      <div className="card flex items-center gap-2 px-3 py-2">
        <span className="eyebrow !text-[10px]">AVPU</span>
        <motion.span key={last?.avpu ?? "-"} initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className={clsx("num chip", last?.avpu && last.avpu !== "A" && "!bg-sev2/20 text-sev2")}>{last?.avpu ?? "–"}</motion.span>
        <span className="eyebrow !text-[10px] ml-2">O₂</span>
        <span className="num chip">{last?.supplemental_o2 == null ? "–" : last.supplemental_o2 ? "supplemental" : "room air"}</span>
      </div>
    </div>
  );
}
