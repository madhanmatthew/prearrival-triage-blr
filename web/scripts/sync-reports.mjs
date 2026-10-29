// Build-time import of REAL repo results into src/data/generated/repo.json.
// The Evaluation screen renders only what is in here. Nothing is hard-coded or invented.
// Sources: reports/*.csv, experiments/log.csv, data/{hospitals,ambulance_bases,benchmark_scenarios}.csv,
//          scripts/traffic_points.json, frontend/demo_vitals.json
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");
const outDir = path.resolve(here, "..", "src", "data", "generated");

function parseCsv(text) {
  const rows = []; let row = [], cur = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"') { if (text[i + 1] === '"') { cur += '"'; i++; } else q = false; } else cur += c;
    } else if (c === '"') q = true;
    else if (c === ",") { row.push(cur); cur = ""; }
    else if (c === "\n") { row.push(cur); rows.push(row); row = []; cur = ""; }
    else if (c !== "\r") cur += c;
  }
  if (cur.length || row.length) { row.push(cur); rows.push(row); }
  const [head, ...body] = rows.filter((r) => r.length > 1 || r[0] !== "");
  return body.map((r) => Object.fromEntries(head.map((h, i) => [h === "" ? "_row" : h, r[i] ?? ""])));
}
const readCsv = (rel) => {
  const p = path.join(root, rel);
  return fs.existsSync(p) ? parseCsv(fs.readFileSync(p, "utf8")) : null;
};
const readJson = (rel) => {
  const p = path.join(root, rel);
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null;
};

const reportNames = [
  "baselines_results", "baselines_confusion", "bigru_confusion", "rl_comparison", "rl_baselines_summary",
  "rl_dqn_summary_s1", "rl_dqn_summary_s2", "rl_dqn_summary_s3", "rl_calibration", "benchmark_summary",
  "benchmark_results", "fusion_sensitivity", "fusion_confusion_vitals_only", "fusion_confusion_fused_agree0.65",
  "fusion_confusion_text_only_agree0.65", "ed_demo_auroc", "asr_wer", "asr_wer_heldout", "asr_wer_heldout_en_turbo_novad", "asr_wer_heldout_hi_ft_novad", "asr_wer_heldout_kn_ft_novad", "asr_wer_snr_large-v3-turbo_rp1.1_cuda", "asr_triage_summary", "asr_triage_summary_heldout_finetuned",
];
const reports = {};
for (const n of reportNames) reports[n] = readCsv(`reports/${n}.csv`);

// First-aid corpus files (IFRC 2020) used by the Report screen's replay so guidance is real corpus text with its source.
const corpus = {};
const cdir = path.join(root, "data", "first_aid_corpus");
if (fs.existsSync(cdir)) for (const f of fs.readdirSync(cdir).filter((x) => x.endsWith("_ifrc.md"))) {
  const txt = fs.readFileSync(path.join(cdir, f), "utf8").replace(/\r\n/g, "\n"); const m = txt.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!m) continue;
  const meta = Object.fromEntries(m[1].split("\n").map((l) => { const i = l.indexOf(":"); return [l.slice(0, i).trim(), l.slice(i + 1).trim()]; }));
  corpus[f] = { ...meta, body: m[2].trim() };
}

const out = {
  generatedAt: new Date().toISOString(),
  repoFound: fs.existsSync(path.join(root, "reports")),
  reports,
  log: readCsv("experiments/log.csv") ?? [],
  hospitals: readCsv("data/hospitals.csv") ?? [],
  ambulances: readCsv("data/ambulance_bases.csv") ?? [],
  scenarios: readCsv("data/benchmark_scenarios.csv") ?? [],
  corpus,
  junctions: readJson("scripts/traffic_points.json")?.junctions ?? null,
  // Demo vitals replays: real VitalDB windows (scripts/build_demo_vitals.py) first, then the hand-written one.
  demoVitalsSets: [
    ...(readJson("frontend/demo_vitals_vitaldb.json")?.sets ?? []).map((s) => ({ id: s.id, name: s.name, source: "VitalDB, real intra-operative", outcome: s.outcome, readings: s.readings })),
    ...((h) => (h ? [{ id: "handwritten", name: "Hand-written example", source: "hand-written, not patient data", outcome: null, readings: h }] : []))(readJson("frontend/demo_vitals.json")?.readings),
  ],
};
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, "repo.json"), JSON.stringify(out));
console.log(`sync-reports: ${Object.values(reports).filter(Boolean).length}/${reportNames.length} reports, ${out.log.length} log rows, ${out.hospitals.length} hospitals`);
