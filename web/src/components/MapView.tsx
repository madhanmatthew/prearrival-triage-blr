import { useEffect, useMemo, useRef } from "react";
import { MapContainer, Marker, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";
import L from "leaflet";
import { LocateFixed, Minus, Plus } from "lucide-react";
import type { LatLon } from "@/api/types";
import type { World } from "@/store/world";
import { useSession } from "@/store/world";
import { JUNCTION_LABEL, JUNCTION_ORDER } from "@/mocks/data";
import { cumulative, pointAt, projectOnPolyline } from "@/lib/geo";
import { severityStyle } from "@/lib/severity";

const CENTER: LatLon = [12.96, 77.65];
const PHASE_COLOR = { main_green: "#34d399", cross_green: "#fb7c3c", yellow: "#fbbf24" } as const;
const OSM_TILES = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";   // same free tiles as the old /ui/ pages

/* --------------------------------- marker icons (DOM, styled with tokens) --------------------------------- */
const incidentIcon = (hex: string, critical: boolean) => L.divIcon({
  className: "", iconSize: [40, 40], iconAnchor: [20, 20],
  html: `<div style="position:relative;width:40px;height:40px;color:${hex}">
    ${critical ? '<span class="marker-pulse"></span>' : ""}
    <span style="position:absolute;inset:11px;border-radius:999px;background:${hex};box-shadow:0 0 0 4px rgb(0 0 0/.35),0 0 22px ${hex};border:2px solid #fff"></span></div>`,
});
const ambulanceIcon = (heading: number) => L.divIcon({
  className: "amb-glide", iconSize: [34, 34], iconAnchor: [17, 17],
  html: `<div style="width:34px;height:34px;transform:rotate(${heading}deg);transition:transform .25s linear;display:grid;place-items:center">
    <svg width="34" height="34" viewBox="0 0 34 34"><circle cx="17" cy="17" r="15" fill="rgb(15 19 29 / .92)" stroke="#fff" stroke-width="1.5"/>
    <path d="M17 6 L24 24 L17 20 L10 24 Z" fill="#7c9cff" stroke="#fff" stroke-width="1.2" stroke-linejoin="round"/></svg></div>`,
});
const hospitalIcon = (rank: number | null, chosen: boolean) => L.divIcon({
  className: "", iconSize: [26, 26], iconAnchor: [13, 13],
  html: `<div style="width:26px;height:26px;border-radius:9px;display:grid;place-items:center;font:600 11px 'JetBrains Mono',monospace;
    background:${chosen ? "rgb(124 156 255)" : "rgb(var(--glass) / .92)"};color:${chosen ? "#fff" : "rgb(var(--c-fg))"};
    border:1.5px solid ${chosen ? "#fff" : "rgb(var(--c-line) / .35)"};box-shadow:0 4px 14px rgb(0 0 0 / .4)${chosen ? ",0 0 18px rgb(124 156 255 / .7)" : ""}">${rank ? rank : "H"}</div>`,
});
const smallHospital = L.divIcon({ className: "", iconSize: [10, 10], iconAnchor: [5, 5], html: `<div style="width:10px;height:10px;border-radius:3px;background:rgb(var(--c-faint));border:1px solid rgb(var(--c-base))"></div>` });
const junctionIcon = (label: string, phase: keyof typeof PHASE_COLOR, preempted: boolean, onRoute: boolean) => {
  const c = PHASE_COLOR[phase];
  return L.divIcon({
    className: "", iconSize: [96, 48], iconAnchor: [48, 14],
    html: `<div style="display:flex;flex-direction:column;align-items:center;gap:3px;opacity:${onRoute ? 1 : 0.55}">
      <div style="position:relative;width:26px;height:26px;border-radius:999px;background:rgb(var(--glass)/.9);border:2px solid ${c};box-shadow:0 0 ${preempted ? 20 : 8}px ${c};display:grid;place-items:center">
        <span style="width:10px;height:10px;border-radius:999px;background:${c}"></span></div>
      <span style="font:600 10px Inter,sans-serif;color:rgb(var(--c-fg));background:rgb(var(--glass)/.85);padding:1px 6px;border-radius:999px;border:1px solid rgb(var(--c-line)/.18);white-space:nowrap">${label}${preempted ? " · RL" : ""}</span></div>`,
  });
};

/* ------------------------------------------- helpers ------------------------------------------- */
function FitOnce({ pts, keyId, padding }: { pts: LatLon[]; keyId: string; padding: [number, number, number, number] }) {
  const map = useMap(); const done = useRef<string>("");
  useEffect(() => {
    if (pts.length < 2 || done.current === keyId) return;
    done.current = keyId;
    map.invalidateSize();
    map.fitBounds(L.latLngBounds(pts), { paddingTopLeft: [padding[3], padding[0]], paddingBottomRight: [padding[1], padding[2]], animate: true, duration: 0.8, maxZoom: 14 });
  }, [pts, keyId, map, padding]);
  return null;
}
function Controls({ pts, padding }: { pts: LatLon[]; padding: [number, number, number, number] }) {
  const map = useMap(); const ref = useRef<HTMLDivElement>(null);
  useEffect(() => { if (ref.current) { L.DomEvent.disableClickPropagation(ref.current); L.DomEvent.disableScrollPropagation(ref.current); } }, []);
  const btn = "grid h-9 w-9 place-items-center text-fg transition hover:bg-white/10 active:scale-95";
  return (
    <div ref={ref} className="glass absolute bottom-4 left-4 z-[900] hidden flex-col overflow-hidden !rounded-xl md:flex xl:bottom-auto xl:left-[368px] xl:top-[76px]">
      <button className={btn} aria-label="Zoom in" onClick={() => map.zoomIn()}><Plus className="h-4 w-4" /></button>
      <div className="h-px bg-white/10" />
      <button className={btn} aria-label="Zoom out" onClick={() => map.zoomOut()}><Minus className="h-4 w-4" /></button>
      <div className="h-px bg-white/10" />
      <button className={btn} aria-label="Recenter on incident and route" onClick={() => pts.length > 1 && map.fitBounds(L.latLngBounds(pts), { paddingTopLeft: [padding[3], padding[0]], paddingBottomRight: [padding[1], padding[2]] })}><LocateFixed className="h-4 w-4" /></button>
    </div>
  );
}

export function MapView({ world, padding = [90, 380, 230, 360], className }: { world: World; padding?: [number, number, number, number]; className?: string }) {
  const theme = useSession((s) => s.theme);
  const { incident, route, ambulance, signals, ranking, hospitals, junctions, risk, decision } = world;
  const cum = useMemo(() => (route.length > 1 ? cumulative(route) : []), [route]);

  // distance along route of ambulance, and of each junction (for the glowing RL green corridor)
  const amb = ambulance?.position;
  const dAmb = useMemo(() => (cum.length && amb ? projectOnPolyline(route, cum, amb) : 0), [route, cum, amb]);
  const traveled = useMemo(() => (cum.length ? route.filter((_, i) => cum[i] <= dAmb).concat(amb ? [amb] : []) : []), [route, cum, dAmb, amb]);
  const remaining = useMemo(() => (cum.length ? [...(amb ? [amb] : []), ...route.filter((_, i) => cum[i] > dAmb)] : []), [route, cum, dAmb, amb]);
  const greenSegs = useMemo(() => {
    if (!signals || !cum.length) return [] as LatLon[][];
    const segs: LatLon[][] = [];
    for (const j of JUNCTION_ORDER) {
      const s = signals[j]; const jp = junctions[j]; if (!s?.preempted || !jp) continue;
      const dj = projectOnPolyline(route, cum, jp);
      const a = Math.max(dAmb, dj - 1100), b = dj + 120;
      if (b <= a) continue;
      const seg: LatLon[] = [pointAt(route, cum, a).p];
      route.forEach((p, i) => { if (cum[i] > a && cum[i] < b) seg.push(p); });
      seg.push(pointAt(route, cum, b).p); segs.push(seg);
    }
    return segs;
  }, [signals, junctions, route, cum, dAmb]);

  const fitPts = useMemo<LatLon[]>(() => {
    const pts: LatLon[] = [...route];
    if (incident) pts.push([incident.gps.lat, incident.gps.lon]);
    return pts;
  }, [route, incident]);
  const sev = severityStyle(risk?.severity_class ?? 1);
  const rankOf = new Map((ranking ?? []).map((r, i) => [r.hospital_id, i + 1]));
  const rankedIds = new Set((ranking ?? []).map((r) => r.hospital_id));
  const corridor = JUNCTION_ORDER.map((j) => junctions[j]).filter(Boolean);

  return (
    <MapContainer center={CENTER} zoom={12} zoomControl={false} attributionControl preferCanvas className={className ?? "h-full w-full"} worldCopyJump={false} minZoom={9}>
      <TileLayer key={theme} url={OSM_TILES} className={theme === "dark" ? "osm-dark" : undefined} attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' subdomains="abc" maxZoom={19} />
      <FitOnce pts={fitPts} keyId={`${world.source}:${incident?.id ?? "none"}:${route.length}`} padding={padding} />
      <Controls pts={fitPts} padding={padding} />

      {/* ORR corridor spine (verified junction coordinates, straight segments) */}
      {corridor.length === 4 && <Polyline positions={corridor} pathOptions={{ color: "#7c9cff", weight: 3, opacity: 0.3, dashArray: "2 8", lineCap: "round" }} />}

      {/* route: remaining (bright), traveled (dim) */}
      {route.length > 1 && <Polyline positions={route} pathOptions={{ color: "#000", weight: 9, opacity: 0.35, lineCap: "round", lineJoin: "round" }} />}
      {remaining.length > 1 && <Polyline positions={remaining} pathOptions={{ color: "#7c9cff", weight: 5, opacity: 0.95, lineCap: "round", lineJoin: "round" }} />}
      {traveled.length > 1 && <Polyline positions={traveled} pathOptions={{ color: "#7c9cff", weight: 5, opacity: 0.28, lineCap: "round" }} />}
      {greenSegs.map((s, i) => <Polyline key={i} positions={s} pathOptions={{ color: "#34d399", weight: 8, opacity: 0.95, lineCap: "round", className: "corridor-glow" }} />)}

      {/* hospitals */}
      {hospitals.map((h) => {
        const rank = rankOf.get(h.id) ?? null; const ranked = rankedIds.has(h.id);
        return (
          <Marker key={h.id} keyboard={false} position={[h.lat, h.lon]} icon={ranked || h.chosen ? hospitalIcon(rank, h.chosen) : smallHospital} zIndexOffset={h.chosen ? 800 : ranked ? 400 : 0}>
            <Tooltip direction="top" offset={[0, -10]}>{h.id} · {h.name}{rank ? ` · rank ${rank}` : ""}{h.chosen ? " · destination" : ""}</Tooltip>
          </Marker>
        );
      })}

      {/* signals */}
      {signals && JUNCTION_ORDER.map((j) => junctions[j] && (
        <Marker key={j} position={junctions[j]} icon={junctionIcon(JUNCTION_LABEL[j], signals[j].phase, signals[j].preempted, signals[j].onRoute)} zIndexOffset={600} interactive={false} />
      ))}

      {/* incident */}
      {incident && <Marker keyboard={false} position={[incident.gps.lat, incident.gps.lon]} icon={incidentIcon(sev.hex, (risk?.severity_class ?? 0) >= 3)} zIndexOffset={900}>
        <Tooltip direction="top" offset={[0, -16]}>{incident.id}{risk ? ` · class ${risk.severity_class} ${sev.name}` : ""}</Tooltip>
      </Marker>}

      {/* ambulance trail + marker */}
      {ambulance && ambulance.trail.length > 1 && <Polyline positions={ambulance.trail} pathOptions={{ color: "#fff", weight: 2, opacity: 0.6, dashArray: "1 6", lineCap: "round" }} />}
      {ambulance && decision && <Marker keyboard={false} position={ambulance.position} icon={ambulanceIcon(ambulance.heading)} zIndexOffset={1000}><Tooltip direction="top" offset={[0, -18]}>{ambulance.id}</Tooltip></Marker>}
    </MapContainer>
  );
}
