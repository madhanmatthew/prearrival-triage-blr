import type { WSMessage } from "./types";

export interface Channel { close: () => void }
/** WebSocket with exponential back-off (1s -> 15s) and jitter. onState is called on every open/close. */
export function openChannel(path: string, onMsg: (m: WSMessage) => void, onState: (open: boolean) => void): Channel {
  let ws: WebSocket | null = null, stopped = false, attempt = 0, timer: ReturnType<typeof setTimeout> | undefined;
  const connect = () => {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const base = (import.meta.env?.VITE_WS_BASE as string | undefined) ?? `${proto}//${location.host}`;
    try { ws = new WebSocket(base + path); } catch { schedule(); return; }
    ws.onopen = () => { attempt = 0; onState(true); };
    ws.onmessage = (e) => { try { onMsg(JSON.parse(e.data) as WSMessage); } catch { /* ignore malformed */ } };
    ws.onclose = () => { onState(false); if (!stopped) schedule(); };
    ws.onerror = () => ws?.close();
  };
  const schedule = () => { const d = Math.min(15000, 1000 * 2 ** attempt++) * (0.8 + Math.random() * 0.4); timer = setTimeout(connect, d); };
  connect();
  return { close: () => { stopped = true; clearTimeout(timer); ws?.close(); } };
}
