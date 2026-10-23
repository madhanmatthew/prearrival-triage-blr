import { useState } from "react";
import { Activity, GitMerge } from "lucide-react";
import clsx from "clsx";
import { MapView } from "@/components/MapView";
import { ReplayBar } from "@/components/ReplayBar";
import { IncidentPanel, RiskPanel, HospitalTable, DecisionLog } from "@/components/command/Panels";
import { XaiDrawer } from "@/components/command/XaiDrawer";
import { LiveTools } from "@/components/command/LiveTools";
import { useWorld, useSession } from "@/store/world";
import { ErrorState } from "@/components/ui";

export default function Command() {
  const world = useWorld();
  const source = useSession((s) => s.source);
  const err = useSession((s) => s.live.error);
  const [tab, setTab] = useState<"rank" | "log">("rank");
  const [why, setWhy] = useState(false);
  const wide = typeof window !== "undefined" && window.matchMedia("(min-width: 1280px) and (min-height: 800px)").matches;
  const pad: [number, number, number, number] = wide ? [84, 376, 322, 356] : [30, 30, 30, 30];
  return (
    <div className="flex h-full flex-col overflow-y-auto fl:block fl:overflow-hidden">
      <div className="relative h-[62vh] min-h-[360px] shrink-0 fl:absolute fl:inset-0 fl:h-auto">
        <MapView world={world} padding={pad} />
        {source === "live" && err && <div className="glass absolute left-1/2 top-4 z-[1000] -translate-x-1/2"><ErrorState title="Backend not responding" body={err} /></div>}
      </div>

      <div className="z-[1000] p-3 fl:pointer-events-none fl:absolute fl:inset-x-0 fl:top-0 fl:p-0">
        <div className="fl:pointer-events-auto fl:absolute fl:left-[356px] fl:right-[376px] fl:top-4">
          {source === "mock" ? <ReplayBar floating /> : <LiveTools world={world} />}
        </div>
      </div>

      <aside aria-label="Incident" className="glass z-[1000] m-3 fl:absolute fl:left-4 fl:top-4 fl:m-0 fl:max-h-[calc(100%-2rem)] fl:w-[328px] fl:overflow-y-auto">
        <IncidentPanel world={world} />
      </aside>

      <section aria-label="Hospital ranking and coordinator" className="glass z-[1000] m-3 fl:absolute fl:bottom-4 fl:left-[356px] fl:right-[376px] fl:m-0 fl:max-h-[286px] fl:overflow-y-auto">
        <div role="tablist" className="sticky top-0 z-10 flex items-center gap-1 border-b hairline bg-[rgb(var(--glass)/.6)] px-3 py-2 backdrop-blur-xl">
          {([["rank", "Hospital ranking", Activity], ["log", "Coordinator decision log", GitMerge]] as const).map(([k, label, Icon]) => (
            <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)} className={clsx("flex h-8 items-center gap-1.5 rounded-lg px-3 text-[12.5px] font-medium transition", tab === k ? "bg-s3 text-fg" : "text-mute hover:text-fg")}>
              <Icon className={clsx("h-3.5 w-3.5", tab === k && "text-brand")} />{label}
              {k === "log" && world.rules?.some((r) => r.fired) && <span className="num grid h-4 min-w-4 place-items-center rounded-full bg-sev1 px-1 text-[9.5px] font-bold text-black">{world.rules.filter((r) => r.fired).length}</span>}
            </button>
          ))}
        </div>
        <div className="p-2" role="tabpanel">{tab === "rank" ? <HospitalTable world={world} /> : <div className="p-2"><DecisionLog world={world} /></div>}</div>
      </section>

      <aside aria-label="Risk and vitals" className="glass z-[1000] m-3 fl:absolute fl:right-4 fl:top-4 fl:m-0 fl:max-h-[calc(100%-2rem)] fl:w-[348px] fl:overflow-y-auto">
        <RiskPanel world={world} onWhy={() => setWhy(true)} />
      </aside>
      <XaiDrawer world={world} open={why} onClose={() => setWhy(false)} />
      {world.gaps.length > 0 && source === "live" && <p className="px-4 pb-4 text-[11px] text-faint fl:hidden">{world.gaps.join(" ")}</p>}
    </div>
  );
}
