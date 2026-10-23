import { Pause, Play, RotateCcw } from "lucide-react";
import { useSession } from "@/store/world";
import { listScenarios, T } from "@/mocks/scenario";
import { mmss } from "@/lib/format";
import { SimTag } from "./ui";
import clsx from "clsx";

const SPEEDS = [0.5, 1, 2, 4];
export function ReplayBar({ className, floating }: { className?: string; floating?: boolean }) {
  const { mock, setScenario, play, setSpeed, setT, restart } = useSession();
  const source = useSession((s) => s.source);
  if (source !== "mock") return null;
  const list = listScenarios();
  return (
    <div className={clsx(floating ? "glass" : "surface", "flex flex-wrap items-center gap-x-2.5 gap-y-2 px-3 py-2", className)} role="group" aria-label="Replay controls">
      <label className="flex min-w-0 items-center gap-2">
                <select aria-label="Scenario" value={mock.scenarioId} onChange={(e) => setScenario(e.target.value)} className="h-8 max-w-[210px] truncate rounded-lg border hairline bg-s3 px-2 text-[12.5px] font-medium text-fg outline-none focus-visible:ring-2 focus-visible:ring-brand">
          {list.map((s) => <option key={s.id} value={s.id}>{s.id} · {s.name}</option>)}
        </select>
      </label>
      <div className="flex items-center gap-1">
        <button className="btn !h-8 !w-8 !px-0" onClick={() => play()} aria-label={mock.playing ? "Pause replay" : "Play replay"}>{mock.playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}</button>
        <button className="btn !h-8 !w-8 !px-0" onClick={restart} aria-label="Restart replay"><RotateCcw className="h-3.5 w-3.5" /></button>
      </div>
      <div className="flex items-center rounded-lg border hairline bg-s3 p-0.5" role="radiogroup" aria-label="Replay speed">
        {SPEEDS.map((s) => (
          <button key={s} role="radio" aria-checked={mock.speed === s} onClick={() => setSpeed(s)} className={clsx("num h-7 rounded-md px-2 text-[11px] font-medium transition", mock.speed === s ? "bg-brand text-onbrand" : "text-mute hover:text-fg")}>{s}×</button>
        ))}
      </div>
      <div className="flex min-w-[110px] flex-1 items-center gap-2">
        <input type="range" min={0} max={T.end} step={0.5} value={mock.t} onChange={(e) => setT(+e.target.value)} className="h-1 w-full accent-[rgb(var(--c-brand))]" aria-label="Replay position" />
        <span className="num hidden shrink-0 whitespace-nowrap text-right text-[11px] text-mute min-[1500px]:inline">{mmss(mock.t)} / {mmss(T.end)}</span>
      </div>
      {!floating && <SimTag>Scripted replay</SimTag>}
    </div>
  );
}
