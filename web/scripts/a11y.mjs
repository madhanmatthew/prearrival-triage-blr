// Automated accessibility audit (axe-core, WCAG 2.1 A/AA) of every screen in dark and light themes.
import { chromium } from "playwright";
import AxeBuilder from "@axe-core/playwright";
import fs from "node:fs";
const base = process.argv[2] ?? "http://127.0.0.1:5173/app";
const exe = ["/opt/pw-browsers/chromium", "/opt/pw-browsers/chromium-1243/chrome-linux64/chrome"].find((p) => fs.existsSync(p) && fs.statSync(p).isFile());
const pages = ["/?mock=1", "/report?mock=1", "/command?mock=1&t=95&s=C0", "/hospital?mock=1&t=95&s=C0", "/police?mock=1&t=95&s=C0", "/evaluation?mock=1"];
const browser = await chromium.launch({ executablePath: exe, args: ["--no-sandbox"] });
let total = 0;
for (const theme of ["dark", "light"]) for (const url of pages) {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: theme, reducedMotion: "reduce" });
  await ctx.addInitScript((t) => { try { localStorage.setItem("theme", t); } catch {} }, theme);
  const page = await ctx.newPage();
  await page.route(/basemaps\.cartocdn\.com/, (r) => r.abort());
  await page.goto(base + url, { waitUntil: "networkidle" }); await page.waitForTimeout(800);
  const res = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  for (const v of res.violations) { total++; console.log(`[${theme}] ${url} :: ${v.id} (${v.impact}) x${v.nodes.length} :: ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`); }
  await ctx.close();
}
await browser.close(); console.log(total ? `${total} violation groups` : "axe: no violations");
