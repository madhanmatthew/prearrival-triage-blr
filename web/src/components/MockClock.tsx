import { useEffect } from "react";
import { useSession } from "@/store/world";
import { T } from "@/mocks/scenario";

/** Advances the replay clock. Mounted once. Runs only while playing in mock mode and the tab is visible. */
export function MockClock() {
  useEffect(() => {
    let raf = 0, last = performance.now(), acc = 0;
    const loop = (now: number) => {
      const dt = Math.min((now - last) / 1000, 0.25); last = now;
      const s = useSession.getState();
      if (s.source === "mock" && s.mock.playing && !document.hidden) {
        acc += dt * s.mock.speed;
        if (acc >= 0.1) { const t = Math.min(T.end, s.mock.t + acc); acc = 0; s.setT(t); if (t >= T.end) s.play(false); }
      } else acc = 0;
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);
  return null;
}
