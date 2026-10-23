import React from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./styles/index.css";
import { App, routes } from "./App";
import { useSession } from "./store/world";
import { probeBackend, startLive } from "./api/live";

const router = createBrowserRouter(routes, { basename: import.meta.env.BASE_URL.replace(/\/$/, "") });

// Mode: ?mock=1 forces replay, ?mock=0 forces live; otherwise live if /health answers, else replay (demo resilience).
async function boot() {
  const q = new URLSearchParams(location.search).get("mock");
  const envMock = import.meta.env.VITE_MOCK === "1";
  const s = useSession.getState();
  const tq = new URLSearchParams(location.search).get("t"), sq = new URLSearchParams(location.search).get("s");
  if (sq) useSession.setState({ mock: { ...useSession.getState().mock, scenarioId: sq } });
  if (q === "1" || envMock) { s.setSource("mock"); s.setBackend("down"); if (tq) { s.setT(+tq); s.play(false); } else s.play(true); return; }
  const up = await probeBackend();
  s.setBackend(up ? "up" : "down");
  if (up || q === "0") { s.setSource("live"); startLive(); } else { s.setSource("mock"); s.play(true); }
}
void boot();

createRoot(document.getElementById("root")!).render(<React.StrictMode><App><RouterProvider router={router} /></App></React.StrictMode>);
