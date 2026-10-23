import type { LatLon } from "@/api/types";
import { lerp } from "./format";

const R = 6371000;
export function haversineM(a: LatLon, b: LatLon): number {
  const p = Math.PI / 180;
  const dLat = (b[0] - a[0]) * p, dLon = (b[1] - a[1]) * p;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a[0] * p) * Math.cos(b[0] * p) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}
export function bearing(a: LatLon, b: LatLon): number {
  const p = Math.PI / 180;
  const y = Math.sin((b[1] - a[1]) * p) * Math.cos(b[0] * p);
  const x = Math.cos(a[0] * p) * Math.sin(b[0] * p) - Math.sin(a[0] * p) * Math.cos(b[0] * p) * Math.cos((b[1] - a[1]) * p);
  return (Math.atan2(y, x) / p + 360) % 360;
}
export function cumulative(poly: LatLon[]): number[] {
  const out = [0];
  for (let i = 1; i < poly.length; i++) out.push(out[i - 1] + haversineM(poly[i - 1], poly[i]));
  return out;
}
/** Point + heading at distance d (metres) along a polyline. */
export function pointAt(poly: LatLon[], cum: number[], d: number): { p: LatLon; heading: number; seg: number } {
  const total = cum[cum.length - 1];
  const x = Math.min(Math.max(d, 0), total);
  let i = 1;
  while (i < cum.length - 1 && cum[i] < x) i++;
  const t = cum[i] === cum[i - 1] ? 0 : (x - cum[i - 1]) / (cum[i] - cum[i - 1]);
  const a = poly[i - 1], b = poly[i];
  return { p: [lerp(a[0], b[0], t), lerp(a[1], b[1], t)], heading: bearing(a, b), seg: i - 1 };
}
/** Distance along polyline of the vertex nearest to `pt` (used to place junctions on the route). */
export function projectOnPolyline(poly: LatLon[], cum: number[], pt: LatLon): number {
  let best = 0, bestD = Infinity;
  for (let i = 0; i < poly.length; i++) {
    const d = haversineM(poly[i], pt);
    if (d < bestD) { bestD = d; best = cum[i]; }
  }
  return best;
}
/** Densify a polyline so that no segment is longer than `stepM`. */
export function densify(poly: LatLon[], stepM: number): LatLon[] {
  const out: LatLon[] = [poly[0]];
  for (let i = 1; i < poly.length; i++) {
    const d = haversineM(poly[i - 1], poly[i]);
    const n = Math.max(1, Math.ceil(d / stepM));
    for (let k = 1; k <= n; k++) out.push([lerp(poly[i - 1][0], poly[i][0], k / n), lerp(poly[i - 1][1], poly[i][1], k / n)]);
  }
  return out;
}
