import { NavLink, Outlet, useLocation } from "react-router-dom";
import { Activity, Hospital, LayoutDashboard, Moon, Radio, ShieldAlert, Sun, FlaskConical, Siren, Mic } from "lucide-react";
import clsx from "clsx";
import { useSession } from "@/store/world";
import { Dot } from "./ui";
import { startLive, stopLive, probeBackend } from "@/api/live";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/report", label: "Report", icon: Mic },
  { to: "/command", label: "Command", icon: Radio },
  { to: "/hospital", label: "Hospital", icon: Hospital },
  { to: "/police", label: "Police", icon: ShieldAlert },
  { to: "/evaluation", label: "Evaluation", icon: Activity },
];

export function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="relative grid h-8 w-8 place-items-center rounded-[10px] bg-gradient-to-br from-brand to-brand/60 shadow-glow">
        <Siren className="h-[17px] w-[17px] text-onbrand" aria-hidden />
      </div>
      <div className="min-w-0 leading-tight">
        <div className="text-[13.5px] font-semibold tracking-tight">Pre-Arrival Triage</div>
        <div className="eyebrow hidden !text-[9.5px] !tracking-[.14em] sm:block">Bengaluru · ORR corridor</div>
      </div>
    </div>
  );
}

export function StatusPill() {
  const source = useSession((s) => s.source);
  const ws = useSession((s) => s.live.ws);
  const backend = useSession((s) => s.backend);
  const setSource = useSession((s) => s.setSource);
  const tone = source === "mock" ? "sim" : ws === "open" ? "ok" : "warn";
  const label = source === "mock" ? "Replay (mock)" : ws === "open" ? "Live" : ws === "reconnecting" ? "Reconnecting" : "Connecting";
  const toggle = async () => {
    if (source === "mock") { if (await probeBackend()) { setSource("live"); startLive(); } else useSession.getState().setBackend("down"); }
    else { stopLive(); setSource("mock"); }
  };
  return (
    <button onClick={toggle} className="group flex h-8 shrink-0 items-center gap-2 whitespace-nowrap rounded-full border hairline bg-s2 pl-2.5 pr-3 text-[12px] font-medium transition hover:bg-s3"
      title={source === "mock" ? (backend === "down" ? "Backend not reachable. Click to retry." : "Scripted replay. Click to connect to the live backend.") : "Connected to backend. Click to switch to replay."}
      aria-live="polite">
      <Dot tone={tone} pulse={tone !== "sim"} />
      <span>{label}</span>
      {source === "mock" && <FlaskConical className="h-3 w-3 text-sim" aria-hidden />}
    </button>
  );
}

function ThemeToggle() {
  const theme = useSession((s) => s.theme); const toggle = useSession((s) => s.toggleTheme);
  return (
    <button className="grid h-8 w-8 place-items-center rounded-full border hairline bg-s2 text-mute transition hover:bg-s3 hover:text-fg" onClick={toggle} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
      {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  );
}

export function HonestyBar({ compact }: { compact?: boolean }) {
  return (
    <div role="note" className={clsx("flex items-center justify-center gap-2 border-t hairline bg-s1/95 px-3 text-[11px] text-mute backdrop-blur", compact ? "h-6" : "h-7")}>
      <span className="inline-block h-1.5 w-1.5 rounded-full bg-sim" aria-hidden />
      <span><strong className="font-semibold text-fg/80">Research prototype.</strong> Not a medical device. Simulated data.</span>
    </div>
  );
}

export function Shell() {
  const { pathname } = useLocation();
  const full = pathname.startsWith("/command");
  return (
    <div className="flex h-dvh flex-col bg-base">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-[2000] focus:rounded-lg focus:bg-brand focus:px-3 focus:py-2 focus:text-onbrand">Skip to content</a>
      <header className="z-[1100] flex h-14 shrink-0 items-center gap-3 border-b hairline bg-s1/90 px-3 backdrop-blur-xl sm:px-5">
        <NavLink to="/" aria-label="Overview"><Logo /></NavLink>
        <nav aria-label="Primary" className="ml-2 hidden items-center gap-0.5 lg:flex">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} className={({ isActive }) => clsx("relative flex h-8 items-center gap-1.5 rounded-lg px-3 text-[13px] font-medium transition", isActive ? "bg-s3 text-fg" : "text-mute hover:bg-s2 hover:text-fg")}>
              {({ isActive }) => (<><Icon className={clsx("h-3.5 w-3.5", isActive && "text-brand")} aria-hidden />{label}</>)}
            </NavLink>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2"><StatusPill /><ThemeToggle /></div>
      </header>
      <main id="main" className={clsx("min-h-0 flex-1", full ? "relative overflow-hidden" : "overflow-y-auto pb-16 lg:pb-0")}><Outlet /></main>
      <HonestyBar compact={full} />
      {/* mobile tab bar */}
      <nav aria-label="Primary mobile" className="fixed inset-x-0 bottom-7 z-[1100] grid grid-cols-6 border-t hairline bg-s1/95 backdrop-blur-xl lg:hidden" style={{ paddingBottom: "env(safe-area-inset-bottom)" }}>
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} className={({ isActive }) => clsx("flex h-12 flex-col items-center justify-center gap-0.5 text-[9.5px] font-medium", isActive ? "text-brand" : "text-mute")}>
            <Icon className="h-[17px] w-[17px]" aria-hidden />{label}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
