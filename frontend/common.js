// Shared helpers for the demo pages (served by FastAPI at /ui/). Research prototype, not a medical device.
const API = "";   // same origin as the FastAPI app

async function api(path, opts = {}) {
  const init = { ...opts };
  if (init.json !== undefined) {
    init.method = init.method || "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(init.json);
    delete init.json;
  }
  const r = await fetch(API + path, init);
  let data = null;
  try { data = await r.json(); } catch (_) { /* empty body */ }
  if (!r.ok) {
    const msg = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : r.statusText;
    throw new Error(`${r.status}: ${msg}`);
  }
  return data;
}

// WebSocket with auto-reconnect. onMsg receives {type, data}.
function openChannel(path, onMsg, onState) {
  let ws, stopped = false;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const connect = () => {
    ws = new WebSocket(`${proto}//${location.host}${path}`);
    ws.onopen = () => onState && onState(true);
    ws.onmessage = (e) => { try { onMsg(JSON.parse(e.data)); } catch (_) {} };
    ws.onclose = () => { onState && onState(false); if (!stopped) setTimeout(connect, 2000); };
  };
  connect();
  return { close() { stopped = true; ws && ws.close(); } };
}

const SEVERITY = ["Stable", "Moderate", "Severe", "Critical"];
const SEV_CLASS = ["sev0", "sev1", "sev2", "sev3"];

function sevBadge(cls) {
  if (cls === null || cls === undefined) return `<span class="badge">–</span>`;
  return `<span class="badge ${SEV_CLASS[cls]}">${cls} · ${SEVERITY[cls]}</span>`;
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function hhmm(iso) {
  if (!iso) return "–";
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function pct(x) { return x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`; }

// Leaflet base map centred on Bengaluru ORR.
function baseMap(id) {
  const map = L.map(id).setView([12.94, 77.66], 12);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  return map;
}

function dot(color, size = 14) {
  return L.divIcon({ className: "", html: `<span class="dot" style="background:${color};width:${size}px;height:${size}px"></span>`, iconSize: [size, size] });
}
