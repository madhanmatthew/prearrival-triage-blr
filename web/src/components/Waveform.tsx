import { useEffect, useRef } from "react";

/** Live microphone waveform (Web Audio AnalyserNode). Draws rounded bars; flat idle line when no analyser. */
export function Waveform({ analyser, active, height = 56 }: { analyser: AnalyserNode | null; active: boolean; height?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current; if (!cv) return;
    const ctx = cv.getContext("2d")!; const dpr = window.devicePixelRatio || 1;
    const resize = () => { cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr; };
    resize(); const ro = new ResizeObserver(resize); ro.observe(cv);
    const buf = new Uint8Array(analyser?.fftSize ?? 64); let raf = 0, phase = 0;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const style = getComputedStyle(document.documentElement);
    const draw = () => {
      const { width: W, height: H } = cv; ctx.clearRect(0, 0, W, H);
      const bars = Math.floor(W / (6 * dpr)); const brand = `rgb(${style.getPropertyValue("--c-brand")})`;
      if (analyser && active) analyser.getByteTimeDomainData(buf);
      for (let i = 0; i < bars; i++) {
        let v = 0.04;
        if (analyser && active) { const a = Math.abs(buf[Math.floor((i / bars) * buf.length)] - 128) / 128; v = Math.min(1, 0.06 + a * 2.6); }
        else if (!reduce) v = 0.05 + 0.03 * Math.sin(phase + i * 0.5);
        const h = Math.max(3 * dpr, v * H); const x = i * 6 * dpr + 1;
        ctx.fillStyle = brand; ctx.globalAlpha = active ? 0.35 + 0.65 * v : 0.25;
        ctx.beginPath(); ctx.roundRect(x, (H - h) / 2, 3.5 * dpr, h, 2 * dpr); ctx.fill();
      }
      phase += 0.06; raf = requestAnimationFrame(draw);
    };
    draw(); return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, [analyser, active]);
  return <canvas ref={ref} style={{ height }} className="w-full" aria-hidden />;
}
