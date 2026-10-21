"""Download a subset of the open VitalDB dataset (vitaldb.net) as 1-minute vitals per case.

    python scripts/download_vitaldb.py [--n-cases 6388 (all usable)] [--workers 8]

Writes data/raw/vitaldb/cases.csv (clinical info, outcomes) and case_<id>.csv.gz
(columns: minute, hr, spo2, rr, temp, sbp_nibp, sbp_art; median of raw samples per minute).
Resumable: finished cases are skipped. Cases that died in hospital are always included,
the rest are a seeded random sample of cases having HR, SpO2, RR and temperature tracks.
Raw values are NOT cleaned here (ART_SBP etc. contain artefacts); clean at load time.
Open data, ~5 KB per case. Record licence/terms and access date in data/PROVENANCE.md.
"""
from __future__ import annotations

import argparse
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

API = "https://api.vitaldb.net"
TRACKS = {"Solar8000/HR": "hr", "Solar8000/PLETH_SPO2": "spo2", "Solar8000/RR_CO2": "rr",
          "Solar8000/BT": "temp", "Solar8000/NIBP_SBP": "sbp_nibp", "Solar8000/ART_SBP": "sbp_art"}
REQUIRED = ["hr", "spo2", "rr", "temp"]
OUT = Path("data/raw/vitaldb")


def fetch_track(tid: str, name: str) -> pd.Series:
    for attempt in range(6):
        try:
            r = requests.get(f"{API}/{tid}", timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            minute = (df.iloc[:, 0] // 60).astype(int)
            return df.iloc[:, 1].groupby(minute).median().rename(name)
        except Exception:  # noqa: BLE001 - retry any network/parse error
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def download_case(caseid: int, trks: pd.DataFrame) -> str:
    path = OUT / f"case_{caseid}.csv.gz"
    if path.exists():
        return "skip"
    try:
        series = [fetch_track(row.tid, TRACKS[row.tname]) for row in trks.itertuples()]
    except Exception:  # noqa: BLE001 - skip this case; a re-run retries it
        return "fail"
    df = pd.concat(series, axis=1).sort_index()
    df.index.name = "minute"
    df.to_csv(path.with_suffix(".tmp"), compression="gzip")
    path.with_suffix(".tmp").replace(path)
    return "ok"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-cases", type=int, default=6388)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cases = pd.read_csv(f"{API}/cases")
    trks = pd.read_csv(f"{API}/trks")
    trks = trks[trks.tname.isin(TRACKS)]
    have = trks.groupby("caseid").tname.apply(lambda s: {TRACKS[t] for t in s})
    ok = have[have.apply(lambda s: set(REQUIRED) <= s and bool({"sbp_nibp", "sbp_art"} & s))].index
    cases = cases[cases.caseid.isin(ok)]
    died = cases[cases.death_inhosp == 1]
    rest = cases[cases.death_inhosp != 1].sample(n=min(len(cases) - len(died), max(0, a.n_cases - len(died))), random_state=a.seed)
    chosen = pd.concat([died, rest]).sort_values("caseid")
    chosen.to_csv(OUT / "cases.csv", index=False)
    print(f"{len(chosen)} cases ({len(died)} in-hospital deaths) -> {OUT}", flush=True)

    groups = {cid: g for cid, g in trks[trks.caseid.isin(chosen.caseid)].groupby("caseid")}
    done, failed = 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        for cid, res in zip(groups, ex.map(lambda c: download_case(c, groups[c]), groups)):
            done += 1
            failed += res == "fail"
            if done % 25 == 0 or done == len(groups):
                print(f"{done}/{len(groups)} cases ({failed} failed)", flush=True)
    print("done" if not failed else f"done with {failed} failed cases: re-run to retry them")


if __name__ == "__main__":
    main()
