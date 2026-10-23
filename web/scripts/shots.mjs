// Visual QA: screenshots of every screen at desktop (1440x900) and mobile (390x844) into frontend/screenshots.
// Usage: node scripts/shots.mjs [baseUrl] [outDir]. Map tiles are replaced by a neutral placeholder when offline.
import { chromium } from "playwright";
import fs from "node:fs";
const base = process.argv[2] ?? "http://127.0.0.1:5173/app";
const out = process.argv[3] ?? "screenshots";
fs.mkdirSync(out, { recursive: true });
const exe = ["/opt/pw-browsers/chromium", "/opt/pw-browsers/chromium-1243/chrome-linux64/chrome"].find((p) => fs.existsSync(p) && fs.statSync(p).isFile());
const tile = (dark) => `<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256"><rect width="256" height="256" fill="${dark ? "#0f131c" : "#e9edf3"}"/><path d="M0 128H256M128 0V256M0 40L256 90M30 256L120 0M200 256L256 150" stroke="${dark ? "#1a2030" : "#d5dbe6"}" stroke-width="${dark ? 3 : 4}" fill="none"/><path d="M0 200L256 210" stroke="${dark ? "#222a3d" : "#cfd6e3"}" stroke-width="7"/></svg>`;
const shots = [
  ["landing", "/?mock=1", 1], ["command", "/command?mock=1&t=95&s=C0", 1], ["command-early", "/command?mock=1&t=20&s=C0", 1], ["command-blood", "/command?mock=1&t=70&s=B1", 1],
  ["hospital", "/hospital?mock=1&t=95&s=C0", 1], ["police", "/police?mock=1&t=95&s=C0", 1], ["report", "/report?mock=1", 1], ["evaluation", "/evaluation?mock=1", 1],
];
const browser = await chromium.launch({ executablePath: exe, args: ["--no-sandbox"] });
for (const theme of ["dark", "light"]) for (const [name, url, _] of shots) for (const [vw, vh, tag] of [[1440, 900, "desktop"], [390, 844, "mobile"]]) {
  if (theme === "light" && !["command", "landing"].includes(name)) continue;
  const ctx = await browser.newContext({ viewport: { width: vw, height: vh }, deviceScaleFactor: 1, colorScheme: theme, reducedMotion: "reduce", geolocation: { latitude: 12.9177, longitude: 77.6238 }, permissions: ["geolocation"] });
  await ctx.addInitScript((t) => { try { localStorage.setItem("theme", t); } catch {} }, theme);
  const page = await ctx.newPage();
  await page.route(/basemaps\.cartocdn\.com/, (r) => r.fulfill({ contentType: "image/svg+xml", body: tile(theme === "dark") }));
  await page.route(/\/(health|incidents|resources)/, (r) => r.abort());
  const errors = []; page.on("pageerror", (e) => errors.push(e.message)); page.on("console", (m) => m.type() === "error" && !/Failed to load resource|net::ERR/.test(m.text()) && errors.push(m.text()));
  await page.goto(base + url, { waitUntil: "networkidle" });
  await page.waitForTimeout(1600);
  await page.screenshot({ path: `${out}/${name}-${tag}${theme === "light" ? "-light" : ""}.png`, fullPage: tag === "mobile" && name !== "command" });
  if (errors.length) console.log(`[${name}/${tag}/${theme}] errors:`, errors.slice(0, 3));
  await ctx.close();
}
await browser.close();
console.log("shots done");
