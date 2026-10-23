import { lazy, Suspense, type ReactNode } from "react";
import { Navigate, type RouteObject } from "react-router-dom";
import { Shell } from "./components/Shell";
import { MockClock } from "./components/MockClock";
import { Skeleton } from "./components/ui";

const Landing = lazy(() => import("./pages/Landing"));
const Report = lazy(() => import("./pages/Report"));
const Command = lazy(() => import("./pages/Command"));
const Hospital = lazy(() => import("./pages/Hospital"));
const Police = lazy(() => import("./pages/Police"));
const Evaluation = lazy(() => import("./pages/Evaluation"));

const Fallback = () => <div className="space-y-4 p-6"><Skeleton className="h-8 w-64" /><Skeleton className="h-64 w-full" /></div>;
const wrap = (el: ReactNode) => <Suspense fallback={<Fallback />}>{el}</Suspense>;

export const routes: RouteObject[] = [
  {
    path: "/", element: <Shell />,
    children: [
      { index: true, element: wrap(<Landing />) },
      { path: "report", element: wrap(<Report />) },
      { path: "command", element: wrap(<Command />) },
      { path: "hospital/:id?", element: wrap(<Hospital />) },
      { path: "police", element: wrap(<Police />) },
      { path: "evaluation", element: wrap(<Evaluation />) },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
];
export function App({ children }: { children: ReactNode }) { return <><MockClock />{children}</>; }
