import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BookOpen, Check, ChevronDown, Keyboard, LocateFixed, Mic, RefreshCw, Send, Square, Volume2, TriangleAlert, Ambulance } from "lucide-react";
import clsx from "clsx";
import { Link } from "react-router-dom";
import { api } from "@/api/client";
import type { GPS, GuidanceItem, Language, Question } from "@/api/types";
import { useSession } from "@/store/world";
import { ANSWER_TOKEN, LANGS, QKEY, QUESTIONS, QUESTION_ORDER, STATUS, UI } from "@/i18n/strings";
import { repo } from "@/mocks/data";
import { Waveform } from "@/components/Waveform";
import { Panel, SimTag } from "@/components/ui";

type Stage = "idle" | "recording" | "processing" | "dialogue" | "guidance";
const DEMO_GPS: GPS = { lat: 12.9177, lon: 77.6238 };

/* ----------------------------- replay-mode helpers: real corpus text + real source ----------------------------- */
const TOPICS: [RegExp, string][] = [
  [/bleed|blood|खून|ರಕ್ತ|cut|wound/i, "severe_bleeding_ifrc.md"], [/not breathing|unconscious|collapse|heart|cpr|बेहोश|ಪ್ರಜ್ಞೆ/i, "cpr_adult_ifrc.md"],
  [/burn|fire|जल|ಸುಟ್ಟ/i, "burns_ifrc.md"], [/head|fall|fell|गिर|ಬಿದ್/i, "head_injury_ifrc.md"],
];
function mockGuidance(text: string): GuidanceItem {
  const hit = TOPICS.find(([re]) => re.test(text));
  const doc = hit ? (repo as unknown as { corpus: Record<string, Record<string, string>> }).corpus?.[hit[1]] : undefined;
  if (!doc) return { problem: "report", status: "not_covered", message: "", sentences: [], retrieved: [], refusal_reason: "below_threshold" };
  const steps = doc.body.split("\n").map((l) => l.match(/^\d+\.\s+(.*)$/)?.[1]).filter(Boolean) as string[];
  const lines = [doc.body.split("\n")[0], ...steps].slice(0, 5);
  return {
    problem: doc.topic, status: "ok", mode: "replay", message: "", groundedness: 1, topic_file: hit![1],
    sentences: lines.map((t, i) => ({ text: t, chunk_ids: [`${doc.topic}#${i}`] })),
    retrieved: lines.map((_, i) => ({ chunk_id: `${doc.topic}#${i}`, file: hit![1], publisher: doc.publisher, url: doc.url, score: undefined })),
  };
}

/* ---------------------------------------------- recorder ---------------------------------------------- */
function useRecorder() {
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const rec = useRef<MediaRecorder | null>(null); const chunks = useRef<Blob[]>([]); const stream = useRef<MediaStream | null>(null); const ctx = useRef<AudioContext | null>(null);
  const start = useCallback(async () => {
    const s = await navigator.mediaDevices.getUserMedia({ audio: true }); stream.current = s;
    const ac = new AudioContext(); ctx.current = ac; const an = ac.createAnalyser(); an.fftSize = 128; ac.createMediaStreamSource(s).connect(an); setAnalyser(an);
    const r = new MediaRecorder(s); chunks.current = []; r.ondataavailable = (e) => chunks.current.push(e.data); r.start(); rec.current = r;
  }, []);
  const stop = useCallback((): Promise<Blob> => new Promise((res) => {
    const r = rec.current!; r.onstop = () => { stream.current?.getTracks().forEach((t) => t.stop()); void ctx.current?.close(); setAnalyser(null); res(new Blob(chunks.current, { type: r.mimeType || "audio/webm" })); }; r.stop();
  }), []);
  return { analyser, start, stop };
}

/* ------------------------------- speech: items play in order; a new say() stops the previous one -------------------------------
   {key} = native recording /prompts/<key>_<lang>.mp3 (browser voice if missing), {text, l} = browser voice only */
type SayItem = { key?: string; text: string; l?: Language };
const VOICE_LANG: Record<Language, string> = { en: "en-IN", hi: "hi-IN", kn: "kn-IN" };
function useSpeech() {
  const gen = useRef(0); const playing = useRef<HTMLAudioElement | null>(null);
  const stopSpeech = useCallback(() => {
    gen.current++; playing.current?.pause(); playing.current = null;
    if ("speechSynthesis" in window) speechSynthesis.cancel();
  }, []);
  const voice = (text: string, l: Language) => new Promise<void>((done) => {
    if (!("speechSynthesis" in window) || !text) return done();
    const v = speechSynthesis.getVoices().find((x) => x.lang === VOICE_LANG[l] || x.lang.startsWith(l));
    if (!v && l !== "en") return done();   // no voice for this language on the device: text only
    const u = new SpeechSynthesisUtterance(text); u.lang = VOICE_LANG[l]; if (v) u.voice = v;
    u.onend = () => done(); u.onerror = () => done(); speechSynthesis.speak(u);
  });
  const one = (it: SayItem, lang: Language) => {
    if (!it.key) return voice(it.text, it.l ?? lang);
    return new Promise<void>((done) => {
      const a = new Audio(`/prompts/${it.key}_${lang}.mp3`); let fell = false;
      const fallback = () => { if (!fell) { fell = true; void voice(it.text, lang).then(done); } };
      a.onended = () => done(); a.onerror = fallback; playing.current = a; a.play().catch(fallback);
    });
  };
  const say = useCallback(async (items: SayItem[], lang: Language) => {
    stopSpeech(); const g = gen.current;
    for (const it of items) { if (g !== gen.current) return; await one(it, lang); }
  }, [stopSpeech]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => stopSpeech, [stopSpeech]);
  return { say, stopSpeech };
}

export default function Report() {
  const source = useSession((s) => s.source);
  const [lang, setLang] = useState<Language>("en");
  const [stage, setStage] = useState<Stage>("idle");
  const [typing, setTyping] = useState(false);
  const [text, setText] = useState("");
  const [gps, setGps] = useState<GPS | null>(null);
  const [gpsInfo, setGpsInfo] = useState<{ kind: "locating" | "gps" | "approx" | "fail" | "demo"; acc?: number }>({ kind: "locating" });
  const [incident, setIncident] = useState<string | null>(null);
  const [q, setQ] = useState<Question | null>(null);
  const [asked, setAsked] = useState(0);
  const [guid, setGuid] = useState<GuidanceItem[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [transcript, setTranscript] = useState("");
  const { analyser, start, stop } = useRecorder();
  const t = UI[lang]; const font = LANGS.find((l) => l.code === lang)!.font;
  const { say, stopSpeech } = useSpeech();

  const locate = useCallback(() => {
    setGpsInfo({ kind: "locating" });
    if (!navigator.geolocation) { setGpsInfo({ kind: "fail" }); return; }
    navigator.geolocation.getCurrentPosition(
      (p) => { setGps({ lat: p.coords.latitude, lon: p.coords.longitude }); setGpsInfo({ kind: p.coords.accuracy > 200 ? "approx" : "gps", acc: Math.round(p.coords.accuracy) }); },
      () => setGpsInfo({ kind: "fail" }), { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 });
  }, []);
  useEffect(() => { locate(); }, [locate]);
  const useDemo = () => { setGps(DEMO_GPS); setGpsInfo({ kind: "demo" }); };

  /* ---- dialogue plumbing: live API, or a local replay of the same 6-question state machine ---- */
  const mockQ = (i: number): Question | null => (i >= QUESTION_ORDER.length ? null : { q_id: QUESTION_ORDER[i], text: QUESTIONS[QUESTION_ORDER[i]][lang] });
  const finish = async (id: string | null, transcriptText: string) => {
    setStage("guidance");
    if (source === "mock" || !id) { setGuid([mockGuidance(transcriptText)]); return; }
    try { const g = await api.guidance(id); setGuid(g.items.length ? g.items : [{ problem: "report", status: g.status, message: g.message ?? "", sentences: [], retrieved: [] }]); }
    catch (e) { setErr((e as Error).message); }
  };
  const submit = async (kind: "text" | "voice", blob?: Blob) => {
    const where = gps ?? DEMO_GPS; setErr(null); setStage("processing");
    try {
      if (source === "mock") { await new Promise((r) => setTimeout(r, 700)); setTranscript(text); setAsked(0); setQ(mockQ(0)); setStage("dialogue"); return; }
      const r = kind === "voice" ? await api.intakeVoice(blob!, where, lang) : await api.intakeText(text, lang, where);
      setIncident(r.incident_id); setTranscript(r.transcript ?? text); setAsked(0);
      if (r.done) await finish(r.incident_id, r.transcript ?? text); else { setQ(r.question); setStage("dialogue"); }
    } catch (e) { setErr((e as Error).message); setStage("idle"); setTyping(true); }
  };
  const answer = async (a: string) => {
    setErr(null);
    if (source === "mock") { const n = asked + 1; setAsked(n); const nq = mockQ(n); if (nq) setQ(nq); else { setQ(null); await finish(null, transcript); } return; }
    try { const r = await api.answer(incident!, a); setAsked((n) => n + 1); if (r.done) { setQ(null); await finish(incident, transcript); } else setQ(r.question); }
    catch (e) { setErr((e as Error).message); }
  };
  const toggleMic = async () => {
    setErr(null);
    if (stage === "recording") { const b = await stop(); void submit("voice", b); return; }
    try { await start(); setStage("recording"); } catch { setErr(t.micDenied); setTyping(true); }
  };
  const playQuestion = (intro = false) => {
    if (!q) return;
    void say([...(intro ? [{ key: "s3", text: STATUS[lang].s3 }] : []), { key: QKEY[q.q_id], text: q.text }], lang);
  };
  // guidance steps are English corpus text (hi/kn reviewed translations pending), so they use the English voice
  const playGuidance = () => {
    const items: SayItem[] = [{ key: "s2", text: STATUS[lang].s2 }, { key: "s1", text: STATUS[lang].s1 }];
    for (const it of guid) {
      if (it.status !== "ok") { items.push({ text: it.refusal_reason === "medication" ? t.noMeds : t.notCovered }, { text: t.wait }); continue; }
      items.push({ key: "s4", text: STATUS[lang].s4 }, ...it.sentences.map((s) => ({ text: s.text, l: "en" as Language })));
    }
    void say(items, lang);
  };
  // speak each question, and the guidance, automatically as it appears
  useEffect(() => { if (stage === "dialogue" && q) playQuestion(asked === 0); }, [q, stage]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (stage === "guidance" && guid.length) playGuidance(); }, [guid, stage]); // eslint-disable-line react-hooks/exhaustive-deps
  const reset = () => { stopSpeech(); setStage("idle"); setTranscript(""); setText(""); setQ(null); setIncident(null); setGuid([]); setAsked(0); setTyping(false); setErr(null); };

  const gpsChip = {
    locating: { icon: <RefreshCw className="h-3 w-3 animate-spin" />, label: t.locating, tone: "" }, gps: { icon: <LocateFixed className="h-3 w-3 text-sev0" />, label: `${t.gps} · ±${gpsInfo.acc} m`, tone: "" },
    approx: { icon: <LocateFixed className="h-3 w-3 text-sev1" />, label: `${t.gpsApprox} · ±${gpsInfo.acc && gpsInfo.acc > 999 ? (gpsInfo.acc / 1000).toFixed(1) + " km" : gpsInfo.acc + " m"}`, tone: "" },
    fail: { icon: <TriangleAlert className="h-3 w-3 text-sev1" />, label: t.gpsFail, tone: "" }, demo: { icon: <LocateFixed className="h-3 w-3 text-sim" />, label: "Demo location · Silk Board", tone: "" },
  }[gpsInfo.kind];

  return (
    <div className={clsx("mx-auto grid w-full max-w-5xl gap-6 px-4 pb-10 pt-5 lg:grid-cols-[minmax(0,460px)_1fr] lg:pt-8", font)}>
      <div className="mx-auto w-full max-w-[460px]">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-1 rounded-xl border hairline bg-s2 p-1" role="radiogroup" aria-label="Language">
            {LANGS.map((l) => <button key={l.code} role="radio" aria-checked={lang === l.code} onClick={() => setLang(l.code)} disabled={stage !== "idle"} className={clsx("h-8 rounded-lg px-3 text-[13px] font-medium transition disabled:opacity-60", l.font, lang === l.code ? "bg-brand text-onbrand shadow" : "text-mute hover:text-fg")}>{l.label}</button>)}
          </div>
          {source === "mock" && <SimTag>Replay</SimTag>}
        </div>

        <h1 className="mt-5 text-[26px] font-semibold leading-tight tracking-tight">{t.title}</h1>
        <p className="mt-1 text-[14px] text-mute">{t.sub}</p>

        <div className="mt-4 flex items-center gap-2">
          <span className="chip h-7 gap-2 !px-3 text-[12px]" aria-live="polite">{gpsChip.icon}{gpsChip.label}</span>
          {(gpsInfo.kind === "fail" || gpsInfo.kind === "approx") && <button className="text-[12px] font-medium text-brand underline-offset-2 hover:underline" onClick={useDemo}>{t.demoLoc}</button>}
          {gpsInfo.kind === "fail" && <button className="text-[12px] text-mute hover:text-fg" onClick={locate}>↻</button>}
        </div>

        {(stage === "dialogue" || stage === "guidance") && transcript.trim() && (
          <div className="mt-5 rounded-2xl border hairline bg-s2 p-4">
            <div className="eyebrow">{t.reported}</div>
            <p className="mt-1 text-[15px] leading-snug">“{transcript}”</p>
          </div>
        )}

        <AnimatePresence mode="wait">
          {(stage === "idle" || stage === "recording" || stage === "processing") && (
            <motion.div key="intake" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} className="surface mt-5 overflow-hidden p-5">
              <div className="flex flex-col items-center gap-4">
                <div className="relative grid h-[132px] w-[132px] place-items-center">
                  {stage === "recording" && <><span className="absolute inset-0 rounded-full bg-sev3/25 animate-pulseRing" /><span className="absolute inset-3 rounded-full bg-sev3/20 animate-pulseRing [animation-delay:.5s]" /></>}
                  <button onClick={toggleMic} disabled={stage === "processing"} aria-label={stage === "recording" ? t.stop : t.tap} aria-pressed={stage === "recording"}
                    className={clsx("relative grid h-[104px] w-[104px] place-items-center rounded-full text-onbrand shadow-float transition active:scale-95 disabled:opacity-60", stage === "recording" ? "bg-sev3" : "bg-gradient-to-br from-brand to-brand/70 shadow-glow hover:brightness-110")}>
                    {stage === "recording" ? <Square className="h-9 w-9 fill-current" /> : stage === "processing" ? <RefreshCw className="h-9 w-9 animate-spin" /> : <Mic className="h-10 w-10" />}
                  </button>
                </div>
                <div className="text-[15px] font-medium" aria-live="polite">{stage === "recording" ? t.listening : stage === "processing" ? t.processing : t.tap}</div>
                <Waveform analyser={analyser} active={stage === "recording"} />
                {stage === "recording" && <div className="text-[12px] text-mute">{t.stop}</div>}
              </div>
              {stage === "idle" && (
                <div className="mt-4 border-t hairline pt-4">
                  {!typing ? <button className="mx-auto flex items-center gap-2 text-[13px] font-medium text-mute hover:text-fg" onClick={() => setTyping(true)}><Keyboard className="h-4 w-4" />{t.type}</button> : (
                    <div className="space-y-2.5">
                      <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder={t.placeholder} rows={3} className="w-full resize-none rounded-xl border hairline bg-s2 p-3 text-[15px] outline-none placeholder:text-faint focus-visible:ring-2 focus-visible:ring-brand" />
                      <button className="btn btn-brand !h-11 w-full !text-[14px]" disabled={!text.trim()} onClick={() => submit("text")}><Send className="h-4 w-4" />{t.send}</button>
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          )}

          {stage === "dialogue" && q && (
            <motion.div key={`q-${q.q_id}`} initial={{ opacity: 0, x: 24 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -24 }} className="surface mt-5 p-5">
              <div className="flex items-center gap-1.5" role="progressbar" aria-valuemin={1} aria-valuemax={6} aria-valuenow={Math.min(asked + 1, 6)} aria-label={t.qOf.replace("{n}", String(Math.min(asked + 1, 6)))}>
                {Array.from({ length: 6 }, (_, i) => <span key={i} className={clsx("h-1.5 flex-1 rounded-full transition-colors", i < asked ? "bg-brand" : i === asked ? "bg-brand/60" : "bg-s3")} />)}
              </div>
              <div className="mt-2 flex items-center justify-between"><span className="eyebrow">{t.qOf.replace("{n}", String(Math.min(asked + 1, 6)))}</span><button className="btn !h-8 !px-2.5 !text-[12px]" onClick={() => playQuestion()}><Volume2 className="h-3.5 w-3.5" />{t.playQ}</button></div>
              <p className="mt-3 text-[22px] font-medium leading-snug">{q.text}</p>
              {q.q_id === "Q_NUM_INJURED" ? (
                <div className="mt-5"><div className="mb-2 text-[12px] text-mute">{t.number}</div><div className="grid grid-cols-5 gap-2">{["1", "2", "3", "4", "5+"].map((n) => <button key={n} className="btn !h-14 !text-[20px] num" onClick={() => answer(n === "5+" ? "5" : n)}>{n}</button>)}</div></div>
              ) : (
                <div className="mt-5 grid grid-cols-3 gap-2">
                  <button className="btn !h-14 !text-[16px] !border-sev0/40 !bg-sev0/15 text-sev0" onClick={() => answer(ANSWER_TOKEN[lang].yes)}>{t.yes}</button>
                  <button className="btn !h-14 !text-[16px] !border-sev3/40 !bg-sev3/15 text-sev3" onClick={() => answer(ANSWER_TOKEN[lang].no)}>{t.no}</button>
                  <button className="btn !h-14 !text-[16px]" onClick={() => answer(ANSWER_TOKEN[lang].unknown)}>{t.notSure}</button>
                </div>
              )}
              <p className="mt-4 text-[12px] text-faint">{STATUS[lang].s3}</p>
            </motion.div>
          )}

          {stage === "guidance" && (
            <motion.div key="guidance" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="mt-5 space-y-3">
              <div className="rounded-2xl border border-sev0/30 bg-sev0/10 p-4"><div className="flex items-center gap-2 text-[14px] font-semibold text-sev0"><Ambulance className="h-4 w-4" />{STATUS[lang].s2}</div><p className="mt-1 text-[14px]">{STATUS[lang].s1}</p>
                {source === "mock" && <Link to="/command" className="mt-2 inline-block text-[12px] font-medium text-brand hover:underline">See this on the Command Center →</Link>}</div>
              {guid.length > 0 && <button className="btn w-full" onClick={playGuidance}><Volume2 className="h-3.5 w-3.5" />{t.listen}</button>}
              <GuidanceCards items={guid} lang={lang} mock={source === "mock"} />
              <button className="btn w-full" onClick={reset}><RefreshCw className="h-3.5 w-3.5" />{t.again}</button>
            </motion.div>
          )}
        </AnimatePresence>
        {err && <p role="alert" className="mt-3 rounded-xl border border-sev3/30 bg-sev3/10 px-3 py-2 text-[13px] text-sev3">{err}</p>}
      </div>

      <aside className="hidden space-y-4 lg:block" aria-label="How this screen works">
        <Panel title="How the report is handled">
          <ol className="space-y-3 text-[13px] leading-relaxed text-mute">
            <li className="flex gap-3"><span className="num grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-s3 text-fg">1</span><span><strong className="text-fg">Voice or text</strong>, in English, Hindi or Kannada. Speech is transcribed locally.</span></li>
            <li className="flex gap-3"><span className="num grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-s3 text-fg">2</span><span><strong className="text-fg">Six fixed questions.</strong> A bounded state machine decides what to ask; a language model never drives the dialogue.</span></li>
            <li className="flex gap-3"><span className="num grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-s3 text-fg">3</span><span><strong className="text-fg">Location is device GPS only.</strong> It is never read from what you say.</span></li>
            <li className="flex gap-3"><span className="num grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-s3 text-fg">4</span><span><strong className="text-fg">First-aid text is retrieved, not generated.</strong> Each step shows its source. If the guidelines do not cover the situation, we say so. No medicine or dose advice.</span></li>
          </ol>
        </Panel>
        <Panel title="Honest limits"><p className="text-[12.5px] leading-relaxed text-mute">Hindi and Kannada speech recognition is weaker than English (see Evaluation). Always call your local emergency number if you can. This prototype does not replace it.</p></Panel>
      </aside>
    </div>
  );
}

/* -------------------------------------- guidance cards with expandable source -------------------------------------- */
function GuidanceCards({ items, lang, mock }: { items: GuidanceItem[]; lang: Language; mock: boolean }) {
  const t = UI[lang];
  return (
    <div className="space-y-3">
      {items.map((it, k) => {
        if (it.status !== "ok") {
          const meds = it.refusal_reason === "medication";
          return (
            <div key={k} className="rounded-2xl border border-dashed hairline bg-s2/60 p-5 text-center">
              <BookOpen className="mx-auto h-6 w-6 text-mute" aria-hidden /><h3 className="mt-2 text-[15px] font-semibold">{meds ? t.noMeds : t.notCovered}</h3>
              <p className="mt-1 text-[14px] text-mute">{t.wait}</p>
            </div>
          );
        }
        return (
          <div key={k} className="surface p-4">
            <div className="mb-2 flex items-center justify-between"><h3 className="eyebrow">{t.steps} · {it.problem.replace(/_/g, " ")}</h3>{mock && <SimTag title="Real IFRC guideline text, replayed without the live retrieval backend">Replay</SimTag>}</div>
            <p className="mb-3 text-[13px] text-mute">{STATUS[lang].s4}</p>
            <ol className="space-y-2">{it.sentences.map((s, i) => <Step key={i} n={i + 1} text={s.text} chunkIds={s.chunk_ids} retrieved={it.retrieved} lang={lang} />)}</ol>
          </div>
        );
      })}
    </div>
  );
}
function Step({ n, text, chunkIds, retrieved, lang }: { n: number; text: string; chunkIds: string[]; retrieved: GuidanceItem["retrieved"]; lang: Language }) {
  const [open, setOpen] = useState(false); const t = UI[lang];
  const chunks = retrieved.filter((r) => chunkIds.includes(r.chunk_id)); const c = chunks[0];
  return (
    <li className="rounded-xl bg-s2 p-3">
      <div className="flex gap-3"><span className="num grid h-6 w-6 shrink-0 place-items-center rounded-lg bg-brand/20 text-[12px] font-semibold text-brand">{n}</span><p className="text-[16px] leading-snug">{text}</p></div>
      <button onClick={() => setOpen(!open)} aria-expanded={open} className="chip mt-2 ml-9 cursor-pointer !bg-s3 text-[11px] text-mute transition hover:text-fg"><BookOpen className="h-3 w-3" />{open ? t.hideSource : t.source}{c?.publisher ? ` · ${c.publisher}` : ""}<ChevronDown className={clsx("h-3 w-3 transition", open && "rotate-180")} /></button>
      <AnimatePresence initial={false}>{open && (
        <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
          <div className="ml-9 mt-2 space-y-1 rounded-lg border-l-2 border-brand/50 bg-s1 px-3 py-2 text-[12px]">
            <div className="eyebrow !text-[10px] flex items-center gap-1"><Check className="h-3 w-3 text-sev0" />{t.verbatim}</div>
            <p className="italic text-fg/90">“{text}”</p>
            <dl className="num grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 pt-1 text-[11px] text-mute">
              <dt>chunk</dt><dd className="break-all">{chunkIds.join(", ") || "–"}</dd>
              {c?.file && <><dt>file</dt><dd className="break-all">{c.file}</dd></>}
              {c?.score != null && <><dt>score</dt><dd>{c.score.toFixed(2)}</dd></>}
              {c?.url && <><dt>url</dt><dd className="break-all"><a className="text-brand hover:underline" href={c.url} target="_blank" rel="noreferrer">{c.url}</a></dd></>}
            </dl>
          </div>
        </motion.div>
      )}</AnimatePresence>
    </li>
  );
}
