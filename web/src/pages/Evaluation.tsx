import { Fragment, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import clsx from "clsx";
import { Database, Search } from "lucide-react";
import { Panel, Pending, SimTag } from "@/components/ui";
import { useSession } from "@/store/world";
import { repo } from "@/mocks/data";
import { fmt, kv, logRows, meanStd, num, report, type Row } from "@/lib/evalData";

/* ------------------------------------------- small building blocks ------------------------------------------- */
function useChartTheme() {
  const dark = useSession((s) => s.theme) === "dark";
  return { axis: dark ? "#94a0b8" : "#525c70", grid: dark ? "rgba(255,255,255,.08)" : "rgba(15,23,42,.1)", tip: dark ? "#151a27" : "#fff", palette: dark ? ["#7c9cff", "#fbbf24", "#34d399", "#c4b5fd", "#fb7c3c"] : ["#3456dc", "#b47000", "#047857", "#6d46dc", "#cd500a"] };
}
function Table({ cols, rows, empty, hi }: { cols: { k: string; h: string; align?: "right"; f?: (r: Row) => React.ReactNode }[]; rows: Row[]; empty?: string; hi?: (r: Row) => boolean }) {
  if (!rows.length) return <Pending what={empty ?? "This table"} />;
  return (
    <div className="overflow-x-auto"><table className="w-full text-[12.5px]">
      <thead><tr className="eyebrow !text-[10px] text-left">{cols.map((c) => <th key={c.k} className={clsx("whitespace-nowrap px-3 py-2 font-semibold", c.align === "right" && "text-right")}>{c.h}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i} className={clsx("border-t hairline transition hover:bg-s2", hi?.(r) && "bg-brand/10")}>{cols.map((c) => <td key={c.k} className={clsx("px-3 py-1.5", c.align === "right" && "num whitespace-nowrap text-right")}>{c.f ? c.f(r) : r[c.k]}</td>)}</tr>)}</tbody>
    </table></div>
  );
}
function Section({ id, title, sub, children, sim }: { id: string; title: string; sub?: string; children: React.ReactNode; sim?: string }) {
  return (
    <section id={id} className="scroll-mt-20">
      <div className="mb-3 flex flex-wrap items-center gap-2"><h2 className="text-[19px] font-semibold tracking-tight">{title}</h2>{sim && <SimTag>{sim}</SimTag>}</div>
      {sub && <p className="mb-3 max-w-3xl text-[13px] leading-relaxed text-mute">{sub}</p>}
      <div className="space-y-4">{children}</div>
    </section>
  );
}
function Confusion({ rows, title }: { rows: Row[]; title: string }) {
  const [norm, setNorm] = useState(true); const dark = useSession((s) => s.theme) === "dark";
  const cols = ["pred_0", "pred_1", "pred_2", "pred_3"];
  const data = rows.map((r) => cols.map((c) => num(r[c]) ?? 0)); const max = Math.max(...data.flat(), 1);
  return (
    <div className="card p-3">
      <div className="mb-2 flex items-center justify-between"><h4 className="text-[13px] font-semibold">{title}</h4><button className="chip hover:bg-s3" onClick={() => setNorm(!norm)}>{norm ? "row %" : "counts"}</button></div>
      <div className="grid grid-cols-[auto_repeat(4,1fr)] gap-1 text-center text-[11px]">
        <div />{cols.map((c, i) => <div key={c} className="eyebrow !text-[9.5px]">pred {i}</div>)}
        {data.map((row, i) => { const sum = row.reduce((a, b) => a + b, 0) || 1; return (<Fragment key={i}>
          <div key={`h${i}`} className="eyebrow !text-[9.5px] self-center pr-1 text-right">true {i}</div>
          {row.map((v, j) => { const f = norm ? v / sum : v / max; return <div key={`${i}-${j}`} className={clsx("num grid h-11 place-items-center rounded-lg text-[12px]", i === j && "ring-1 ring-inset ring-fg/40")} style={{ background: `rgb(var(--c-brand) / ${0.08 + f * 0.8})`, color: dark ? (f > 0.3 ? "#fff" : undefined) : (f > 0.78 ? "#fff" : "rgb(15 23 42)") }}>{norm ? `${(f * 100).toFixed(0)}%` : v}</div>; })}
        </Fragment>); })}
      </div>
    </div>
  );
}

/* ----------------------------------------------------- page ----------------------------------------------------- */
const TOC = [["vitals", "Vitals models"], ["external", "External checks"], ["fusion", "Fusion ablation"], ["rl", "Traffic RL"], ["benchmark", "Hospital benchmark"], ["speech", "Speech & dialogue"], ["text", "Text & RAG"], ["log", "Experiment log"]] as const;

export default function Evaluation() {
  const ct = useChartTheme();
  const [model, setModel] = useState("gb");
  const [q, setQ] = useState(""); const [comp, setComp] = useState("all"); const [page, setPage] = useState(0);

  const vit = useMemo(() => logRows((r) => r.component === "vitals" && r.data_version.startsWith("mimic-iv-demo")), []);
  const sev = vit.filter((r) => kv(r.notes).task === "severity"), trend = vit.filter((r) => kv(r.notes).task === "trend");
  const order = ["majority", "news2", "logreg", "rf", "gb", "bigru"];
  const merged = order.map((m) => { const s = sev.find((r) => r.model === m), t = trend.find((r) => r.model === m); return s || t ? { model: m, f1: meanStd(s?.cv_mean, s?.cv_std), pr: meanStd(t?.cv_mean, t?.cv_std), auroc: t ? kv(t.notes).auroc?.replace("+/-", " ± ") ?? "–" : "–", n: kv((s ?? t)!.notes).n } : null; }).filter(Boolean) as Record<string, string>[];
  const confBase = report("baselines_confusion"); const confModels = confBase ? [...new Set(confBase.map((r) => r.model))] : [];
  const ext = logRows((r) => r.component === "vitals" && r.notes.startsWith("external check")).map((r) => ({ ...r, ...kv(r.notes) })).sort((a, b) => (a.task + a.subset + a.model).localeCompare(b.task + b.subset + b.model));
  const vdb = logRows((r) => r.component === "vitals" && r.model.startsWith("vitaldb_")).sort((a, b) => (num(b.cv_mean) ?? 0) - (num(a.cv_mean) ?? 0)).map((r) => ({ ...r, auroc: kv(r.notes).auroc?.replace("+/-", " ± ") ?? "–" }));
  const fus = report("fusion_sensitivity");
  const rl = report("rl_comparison"); const rlChart = useMemo(() => {
    if (!rl) return []; return ["low", "medium", "peak"].map((d) => { const o: Record<string, number | string> = { demand: d }; for (const r of rl.filter((x) => x.demand === d)) { o[r.policy] = +(num(r.amb_transit_s_mean) ?? 0).toFixed(0); o[`${r.policy}_sd`] = +(num(r.amb_transit_s_std) ?? 0).toFixed(0); } return o; });
  }, [rl]);
  const rlPolicies = rl ? [...new Set(rl.map((r) => r.policy))] : [];
  const osm = logRows((r) => r.component === "traffic_rl" && r.model.startsWith("osm_")).map((r) => ({ ...r, ...kv(r.notes), slot: JSON.parse(r.params_json || "{}").slot as string } as Row)).sort((a, b) => (a.slot + a.model).localeCompare(b.slot + b.model));
  const bsum = report("benchmark_summary"), bres = report("benchmark_results");
  const bScen = bres ? [...new Set(bres.map((r) => r.scenario_id))].map((id) => ({ id, name: bres.find((r) => r.scenario_id === id)!.name, naive: bres.find((r) => r.scenario_id === id && r.method === "naive"), ai: bres.find((r) => r.scenario_id === id && r.method === "ai") })) : [];
  const wer = (["asr_wer_heldout_en_turbo_novad", "asr_wer_heldout_hi_ft_novad", "asr_wer_heldout_kn_ft_novad"] as const).flatMap((k) => report(k) ?? []);
  const snr = report("asr_wer_snr_large-v3-turbo_rp1.1_cuda"); const snrData = useMemo(() => snr ? [...new Set(snr.map((r) => +r.snr_db))].sort((a, b) => b - a).map((s) => { const o: Record<string, number> = { snr: s }; for (const r of snr.filter((x) => +x.snr_db === s && x.condition === "vad_off" && x.noise_source === "recorded")) o[r.language] = +(+r.wer * 100).toFixed(1); return o; }) : [], [snr]);
  const tri = report("asr_triage_summary_heldout_finetuned");
  const textLog = logRows((r) => r.component === "reporting"), ragLog = logRows((r) => r.component === "reporting_rag");
  const comps = ["all", ...new Set(repo.log.map((r) => r.component))];
  const logAll = useMemo(() => [...repo.log].reverse().filter((r) => (comp === "all" || r.component === comp) && (!q || (r.model + r.notes + r.data_version).toLowerCase().includes(q.toLowerCase()))), [comp, q]);

  const tip = { contentStyle: { background: ct.tip, border: `1px solid ${ct.grid}`, borderRadius: 10, fontSize: 12 }, labelStyle: { color: ct.axis } };
  return (
    <div className="mx-auto grid w-full max-w-6xl gap-8 px-4 py-5 lg:grid-cols-[180px_1fr] lg:py-8">
      <nav aria-label="Sections" className="sticky top-4 hidden self-start lg:block"><div className="eyebrow mb-2">On this page</div><ul className="space-y-0.5">{TOC.map(([id, l]) => <li key={id}><a href={`#${id}`} className="block rounded-lg px-2.5 py-1.5 text-[13px] text-mute transition hover:bg-s2 hover:text-fg">{l}</a></li>)}</ul></nav>
      <div className="min-w-0 space-y-10">
        <header>
          <div className="eyebrow">Evaluation</div><h1 className="text-[28px] font-semibold tracking-tight">What the experiments say</h1>
          <p className="mt-2 max-w-3xl text-[14px] leading-relaxed text-mute">Every number on this page is read from <span className="num">reports/*.csv</span> and <span className="num">experiments/log.csv</span> at build time. Nothing is typed in by hand. If a result is not in the log, it shows as pending.</p>
          <div className="mt-3 flex flex-wrap gap-2 text-[12px]"><span className="chip"><Database className="h-3 w-3" />{repo.log.length} logged runs</span><span className="chip">{Object.values(repo.reports).filter(Boolean).length} report files</span><span className="chip">synced {new Date(repo.generatedAt).toLocaleDateString()}</span></div>
          {!repo.repoFound && <div className="mt-4"><Pending what="The repo reports folder" /></div>}
        </header>

        <Section id="vitals" title="Vitals models" sub="Nested grouped cross-validation (outer 5-fold, inner 3-fold, split by patient) on MIMIC-IV Demo, an ICU dataset of 100 patients: expect wide variance, and an in-hospital to pre-hospital domain gap. NEWS2 is a rule-based comparison baseline, never a label source.">
          <Panel title="Severity (macro-F1) and 6-reading trend (PR-AUC)" pad={false}><Table rows={merged} cols={[{ k: "model", h: "Model", f: (r) => <span className="font-medium uppercase">{r.model}</span> }, { k: "f1", h: "Severity F1 (mean ± std)", align: "right" }, { k: "pr", h: "Trend PR-AUC", align: "right" }, { k: "auroc", h: "Trend AUROC", align: "right" }, { k: "n", h: "Windows", align: "right" }]} hi={(r) => r.model === "news2" || r.model === "bigru"} empty="Vitals nested-CV results" /></Panel>
          <div className="grid gap-4 md:grid-cols-2">
            {report("bigru_confusion") ? <Confusion rows={report("bigru_confusion")!} title="BiGRU · severity confusion (pooled outer folds)" /> : <Pending what="BiGRU confusion matrix" />}
            {confBase ? <div className="space-y-2"><select value={model} onChange={(e) => setModel(e.target.value)} aria-label="Baseline model" className="h-8 rounded-lg border hairline bg-s2 px-2 text-[12.5px]">{confModels.map((m) => <option key={m}>{m}</option>)}</select><Confusion rows={confBase.filter((r) => r.model === model)} title={`${model} · severity confusion`} /></div> : <Pending what="Baseline confusion matrices" />}
          </div>
          <Panel title="Intra-operative deterioration (VitalDB, 10-minute label)" right={<SimTag title="Intra-operative data, not pre-hospital">Intra-operative</SimTag>} pad={false}><Table rows={vdb} cols={[{ k: "model", h: "Model", f: (r) => r.model.replace("vitaldb_", "") }, { k: "cv_mean", h: "PR-AUC (mean ± std)", align: "right", f: (r) => meanStd(r.cv_mean, r.cv_std, 3) }, { k: "auroc", h: "AUROC", align: "right" }]} empty="VitalDB deterioration results" /></Panel>
        </Section>

        <Section id="external" title="External check: MIMIC-IV-ED Demo" sub="The frozen BiGRU compared to NEWS2 on an emergency-department sample it was not trained on (95% CI from bootstrap). The ICU model does not transfer cleanly; confidence intervals are wide." >
          <Panel pad={false}><Table rows={ext} cols={[{ k: "task", h: "Outcome" }, { k: "subset", h: "Subset" }, { k: "model", h: "Scorer", f: (r) => r.model === "bigru_frozen" ? "BiGRU (frozen)" : "NEWS2" }, { k: "test_score", h: "AUROC", align: "right", f: (r) => fmt(r.test_score, 3) }, { k: "ci95", h: "95% CI", align: "right" }, { k: "n", h: "n (pos)", align: "right", f: (r) => `${r.n} (${r.pos})` }]} hi={(r) => r.model === "news2"} empty="ED external check" /></Panel>
        </Section>

        <Section id="fusion" title="Fusion ablation" sim="Synthetic text" sub="MIMIC has no bystander text, so the text side of each pair is generated. This ablation measures the pairing generator's assumptions as much as the model. At the agreement levels tried, the fused model did not beat text-only.">
          {fus ? <Panel title="Severity macro-F1 by text-vitals agreement"><div className="h-[260px]"><ResponsiveContainer><BarChart data={fus.map((r) => ({ agreement: r.agreement, "Text only": +(+r.text_only_f1).toFixed(3), "Vitals only": +(+r.vitals_only_f1).toFixed(3), Fused: +(+r.fused_f1).toFixed(3) }))}>
            <CartesianGrid stroke={ct.grid} vertical={false} /><XAxis dataKey="agreement" stroke={ct.axis} fontSize={12} label={{ value: "Synthetic agreement", position: "insideBottom", offset: -2, fill: ct.axis, fontSize: 11 }} /><YAxis stroke={ct.axis} fontSize={12} domain={[0, 1]} /><Tooltip {...tip} /><Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="Text only" fill={ct.palette[1]} radius={[4, 4, 0, 0]} /><Bar dataKey="Vitals only" fill={ct.palette[3]} radius={[4, 4, 0, 0]} /><Bar dataKey="Fused" fill={ct.palette[0]} radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></div>
            <Table rows={fus} cols={[{ k: "agreement", h: "Agreement" }, { k: "gain_vs_vitals_mean", h: "Gain vs vitals", align: "right", f: (r) => `${fmt(r.gain_vs_vitals_mean)} ± ${fmt(r.gain_vs_vitals_std)}` }, { k: "folds_fused_beat_vitals", h: "Folds fused > vitals", align: "right", f: (r) => `${r.folds_fused_beat_vitals}/5` }, { k: "gain_vs_text_mean", h: "Gain vs text", align: "right", f: (r) => `${fmt(r.gain_vs_text_mean)} ± ${fmt(r.gain_vs_text_std)}` }, { k: "folds_fused_beat_text", h: "Folds fused > text", align: "right", f: (r) => `${r.folds_fused_beat_text}/5` }]} /></Panel> : <Pending what="Fusion ablation" />}
        </Section>

        <Section id="rl" title="Traffic signal control (SUMO)" sub="Reinforcement learning (DQN) against fixed-time and always-green baselines. Results are from simulation of an approximated corridor. The OSM corridor is speed-calibrated, not demand-calibrated, and some DQN seeds were unstable.">
          {rl ? <Panel title="Ambulance transit time, schematic corridor (mean ± std over episodes)" right={<span className="chip">lower is better</span>}>
            <div className="h-[280px]"><ResponsiveContainer><BarChart data={rlChart}><CartesianGrid stroke={ct.grid} vertical={false} /><XAxis dataKey="demand" stroke={ct.axis} fontSize={12} /><YAxis stroke={ct.axis} fontSize={12} unit=" s" /><Tooltip {...tip} /><Legend wrapperStyle={{ fontSize: 12 }} />
              {rlPolicies.map((p, i) => <Bar key={p} dataKey={p} fill={ct.palette[i % 5]} radius={[4, 4, 0, 0]}><ErrorBar dataKey={`${p}_sd`} stroke={ct.axis} width={4} strokeWidth={1.2} /></Bar>)}</BarChart></ResponsiveContainer></div></Panel> : <Pending what="RL comparison" />}
          <Panel title="OSM corridor runs (ambulance on the 7.7 km controlled stretch)" pad={false}><Table rows={osm} cols={[{ k: "slot", h: "Slot" }, { k: "model", h: "Policy", f: (r) => r.model.replace("osm_", "") }, { k: "completed", h: "Completed", align: "right", f: (r) => `${r.completed}/${r.episodes}` }, { k: "transit_s", h: "Transit (s)", align: "right", f: (r) => fmt(r.transit_s, 0) }, { k: "gen_wait_s", h: "General wait (s)", align: "right", f: (r) => fmt(r.gen_wait_s, 1) }]} empty="OSM RL runs" /></Panel>
        </Section>

        <Section id="benchmark" title="Hospital recommendation benchmark" sim="Simulated beds & blood" sub="Eight scenarios plus two blood-match scenarios, run for the hospital agent and a fair naive baseline on identical inputs. Straight-line ETAs. This tests decision logic, not patient outcomes.">
          {bsum ? <div className="grid gap-3 sm:grid-cols-2">{bsum.map((r) => <div key={r.method} className={clsx("card p-4", r.method === "ai" && "ring-1 ring-brand/40")}><div className="eyebrow">{r.method === "ai" ? "Hospital agent" : "Naive baseline"}</div><div className="num mt-1 text-[34px] font-semibold">{r.appropriate}<span className="text-[16px] text-mute">/{r.scenarios}</span></div><div className="text-[12px] text-mute">appropriate hospital · blood ok {r.blood_ok}/{r.scenarios} · mean total {fmt(r.mean_total_min, 1)} min</div></div>)}</div> : <Pending what="Benchmark summary" />}
          <Panel pad={false}><Table rows={bScen.map((s) => ({ id: s.id, name: s.name, naive: s.naive ? `${s.naive.hospital_id} ${s.naive.appropriate === "True" ? "✓" : "✗"}` : "–", ai: s.ai ? `${s.ai.hospital_id} ${s.ai.appropriate === "True" ? "✓" : "✗"}` : "–", total: s.ai ? `${fmt(s.naive?.total_min, 1)} / ${fmt(s.ai.total_min, 1)}` : "–" }))} cols={[{ k: "id", h: "ID" }, { k: "name", h: "Scenario" }, { k: "naive", h: "Naive", align: "right" }, { k: "ai", h: "Agent", align: "right" }, { k: "total", h: "Total min (naive / agent)", align: "right" }]} empty="Benchmark scenarios" /></Panel>
        </Section>

        <Section id="speech" title="Speech recognition and dialogue cues" sub="Word error rate on held-out team recordings (10 clips per language, small sample). English is best; Kannada is far behind. Language-specific fine-tuned Whisper models are used for Hindi and Kannada.">
          <Panel title="Held-out WER" pad={false}><Table rows={wer} cols={[{ k: "language", h: "Language", f: (r) => ({ en: "English", hi: "Hindi", kn: "Kannada" } as Record<string, string>)[r.language] }, { k: "utterances", h: "Clips", align: "right" }, { k: "wer", h: "WER", align: "right", f: (r) => `${(+r.wer * 100).toFixed(1)}%` }, { k: "cer", h: "CER", align: "right", f: (r) => `${(+r.cer * 100).toFixed(1)}%` }]} empty="Held-out WER" /></Panel>
          {snrData.length > 0 && <Panel title="WER vs signal-to-noise ratio (recorded noise)"><div className="h-[240px]"><ResponsiveContainer><LineChart data={snrData}><CartesianGrid stroke={ct.grid} vertical={false} /><XAxis dataKey="snr" reversed stroke={ct.axis} fontSize={12} unit=" dB" /><YAxis stroke={ct.axis} fontSize={12} unit="%" /><Tooltip {...tip} /><Legend wrapperStyle={{ fontSize: 12 }} />{["en", "hi", "kn"].map((l, i) => <Line key={l} type="monotone" dataKey={l} stroke={ct.palette[i]} strokeWidth={2} dot={{ r: 3 }} />)}</LineChart></ResponsiveContainer></div></Panel>}
          {tri && <Panel title="Dialogue-cue agreement from ASR output, held-out" pad={false}><Table rows={tri} cols={[{ k: "language", h: "Lang" }, { k: "field", h: "Field" }, { k: "clips", h: "Clips", align: "right" }, { k: "asr", h: "Agreement", align: "right", f: (r) => fmt(r.asr, 2) }, { k: "ref_vs_en", h: "Reference vs English", align: "right", f: (r) => fmt(r.ref_vs_en, 2) }]} /></Panel>}
        </Section>

        <Section id="text" title="Text severity and first-aid retrieval" sub="Both are evaluated on small, partly synthetic sets. The text classifier mostly recovers its own templates; on the real held-out recordings its predictions are inconsistent.">
          <div className="grid gap-3 sm:grid-cols-2">
            {textLog.map((r, i) => <div key={i} className="card p-4"><div className="flex items-center justify-between"><span className="eyebrow">Text severity · TF-IDF + LR</span><SimTag>Template data</SimTag></div><div className="num mt-1 text-[30px] font-semibold">{fmt(r.test_score, 3)}</div><div className="text-[12px] text-mute">test macro-F1 · {kv(r.notes).per_lang?.replaceAll("_", " ")}</div></div>)}
            {ragLog.slice(-1).map((r, i) => { const k = kv(r.notes); return <div key={i} className="card p-4"><span className="eyebrow">First-aid retrieval (extractive)</span><div className="num mt-1 text-[30px] font-semibold">{fmt(k.hit_at_k, 2)}</div><div className="text-[12px] text-mute">hit@k · answer {fmt(k.answer_rate, 2)} · refusal {fmt(k.refusal_rate, 2)} · fully grounded {fmt(k.pct_fully_grounded, 2)} · {k.n_in_scope} in-scope / {k.n_out_of_scope} out-of-scope queries</div></div>; })}
            {!textLog.length && <Pending what="Text severity classifier" />}{!ragLog.length && <Pending what="RAG groundedness" />}
          </div>
          <Pending what="Hospital RF surrogate (not trained: scores are rule-only in the benchmark)" />
        </Section>

        <Section id="log" title="Experiment log" sub="The full experiments/log.csv, newest first. Every number above traces to a row here.">
          <div className="flex flex-wrap items-center gap-2">
            <label className="relative"><Search className="pointer-events-none absolute left-2.5 top-2.5 h-3.5 w-3.5 text-faint" /><input value={q} onChange={(e) => { setQ(e.target.value); setPage(0); }} placeholder="Filter model, notes, dataset…" className="h-9 w-64 rounded-xl border hairline bg-s2 pl-8 pr-3 text-[13px] outline-none focus-visible:ring-2 focus-visible:ring-brand" aria-label="Filter log" /></label>
            <select value={comp} onChange={(e) => { setComp(e.target.value); setPage(0); }} aria-label="Component" className="h-9 rounded-xl border hairline bg-s2 px-2 text-[13px]">{comps.map((c) => <option key={c}>{c}</option>)}</select>
            <span className="num ml-auto text-[12px] text-mute">{logAll.length} rows</span>
          </div>
          <Panel pad={false}><Table rows={logAll.slice(page * 20, page * 20 + 20)} cols={[{ k: "date", h: "Date" }, { k: "component", h: "Component" }, { k: "model", h: "Model" }, { k: "cv_mean", h: "cv", align: "right", f: (r) => meanStd(r.cv_mean, r.cv_std) }, { k: "test_score", h: "test", align: "right", f: (r) => fmt(r.test_score) }, { k: "notes", h: "Notes", f: (r) => <span className="line-clamp-2 max-w-[360px] text-[11.5px] text-mute" title={r.notes}>{r.notes}</span> }]} /></Panel>
          <div className="flex items-center justify-between"><button className="btn" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button><span className="num text-[12px] text-mute">{page + 1} / {Math.max(1, Math.ceil(logAll.length / 20))}</span><button className="btn" disabled={(page + 1) * 20 >= logAll.length} onClick={() => setPage(page + 1)}>Next</button></div>
        </Section>
      </div>
    </div>
  );
}
