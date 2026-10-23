import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import type { World } from "@/store/world";
import { SimTag } from "@/components/ui";
import clsx from "clsx";

const NAMES: Record<string, string> = { hr: "Heart rate", rr: "Resp. rate", spo2: "SpO₂", sbp: "Systolic BP", temp: "Temperature", avpu: "AVPU", supplemental_o2: "Supplemental O₂" };

export function XaiDrawer({ world, open, onClose }: { world: World; open: boolean; onClose: () => void }) {
  const ig = world.ig; const max = Math.max(...(ig ?? []).map((x) => Math.abs(x.value)), 0.001);
  const reasons = world.risk?.deterioration_reasons ?? [];
  return (
    <AnimatePresence>
      {open && (
        <motion.aside initial={{ x: 40, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 40, opacity: 0 }} transition={{ duration: 0.22 }}
          className="glass absolute inset-y-0 right-0 z-[1300] flex w-full max-w-[400px] flex-col overflow-hidden !rounded-none sm:inset-y-4 sm:right-4 sm:!rounded-2xl" role="dialog" aria-label="Why this score">
          <header className="flex items-center justify-between border-b hairline px-4 py-3">
            <div><div className="eyebrow">Explainability</div><h2 className="text-[15px] font-semibold">Why this score</h2></div>
            <button className="btn !h-8 !w-8 !px-0" onClick={onClose} aria-label="Close explanation"><X className="h-4 w-4" /></button>
          </header>
          <div className="flex-1 space-y-5 overflow-y-auto p-4">
            <section>
              <div className="mb-2 flex items-center justify-between"><h3 className="text-[13px] font-semibold">Integrated Gradients <span className="font-normal text-mute">· BiGRU severity model</span></h3>{world.source === "mock" && <SimTag>Scripted values</SimTag>}</div>
              {ig ? (
                <ul className="space-y-1.5">
                  {ig.map((x) => (
                    <li key={x.feature} className="grid grid-cols-[104px_1fr_44px] items-center gap-2 text-[12px]">
                      <span className="truncate text-mute">{NAMES[x.feature] ?? x.feature}</span>
                      <div className="relative h-2.5 rounded-full bg-s3">
                        <div className="absolute left-1/2 top-0 h-full w-px bg-white/20" />
                        <motion.div className={clsx("absolute top-0 h-full rounded-full", x.value >= 0 ? "bg-sev2" : "bg-sev0")} initial={false}
                          animate={{ width: `${(Math.abs(x.value) / max) * 50}%`, left: x.value >= 0 ? "50%" : `${50 - (Math.abs(x.value) / max) * 50}%` }} transition={{ type: "spring", stiffness: 120, damping: 20 }} />
                      </div>
                      <span className="num text-right text-mute">{x.value >= 0 ? "+" : ""}{x.value.toFixed(2)}</span>
                    </li>
                  ))}
                </ul>
              ) : <p className="text-[12.5px] text-mute">Needs at least two vitals readings. Attributions are not exposed by the live API yet.</p>}
              <p className="mt-2 text-[11px] leading-relaxed text-faint">Bars right of centre push the score towards higher severity. Integrated Gradients (Captum) is used for the neural model; SHAP TreeExplainer is never applied to the BiGRU.</p>
            </section>
            <section>
              <div className="mb-2 flex items-center justify-between"><h3 className="text-[13px] font-semibold">SHAP <span className="font-normal text-mute">· gradient-boosted trees (10-min model)</span></h3></div>
              {reasons.length ? (
                <ul className="space-y-1.5">{reasons.map((r) => <li key={r} className="card px-3 py-2 text-[12.5px]">{r}</li>)}</ul>
              ) : <p className="text-[12.5px] text-mute">No tree-model reasons yet (needs vitals).</p>}
              <p className="mt-2 text-[11px] leading-relaxed text-faint">Top reasons as text from the tree model; the model was trained on intra-operative data (VitalDB), not pre-hospital data.</p>
            </section>
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}
