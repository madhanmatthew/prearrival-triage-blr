"""Build data/hospitals.csv and data/ambulance_bases.csv (docs/08 §7.1, docs/15 D11).

    python scripts/build_hospitals.py

Sources (data/raw/hospitals_src/, provenance in data/PROVENANCE.md):
  bengaluru_hospitals.geojson      OpenStreetMap hospitals (overpass-turbo, 2026-05-09, ODbL)
                                   -> name, lat, lon (REAL)
  curated_emergency_hospitals.csv  capability flags from the 6th-sem project
                                   (emergency, polytrauma, accident_victims, neurosurgery,
                                   orthopaedics) [TODO-VERIFY source of each flag]
  ambulance_bases_mvp.csv          16 base locations from the 6th-sem project; APPROXIMATED zone
                                   points, not official 108 base locations

Each curated hospital is joined to one OSM feature by an explicit, hand-checked OSM id (MATCH).
Hospitals with no confident OSM match are dropped, never given guessed coordinates.

Derived columns ([ASSUMPTION], rule-based, documented):
  trauma_level   = polytrauma + neurosurgery + (emergency or accident_victims), 0..3
  specialties    = trauma (polytrauma or accident_victims), neurosurgery, orthopaedics, emergency
  open_24x7      = emergency flag
SIMULATED columns (no public source; labelled SIM everywhere): beds_available,
icu_beds_available, blood_* units. Seeded, so every run gives the same table.
beds_total is left empty: no public source was found.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/raw/hospitals_src")
OUT_HOSP = Path("data/hospitals.csv")
OUT_AMB = Path("data/ambulance_bases.csv")
SEED = 42
BLOOD_GROUPS = ("A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-")

# curated id -> (OSM @id, note). None = dropped (no confident match in the OSM extract).
MATCH: dict[int, tuple[str | None, str]] = {
    1: ("relation/2187206", ""),
    2: ("relation/4497967", "OSM: Vydehi Institute Of Medical Sciences And Research Centre"),
    3: ("way/148032600", "OSM: Ramaiah Memorial Hospital (separate from id 1, ~200 m away)"),
    4: (None, "Rajarajeswari Medical College: not in OSM extract"),
    5: ("relation/5397008", "OSM: Sapthagiri Hospital, Chikkasandra, Hesaraghatta Main Road"),
    6: ("node/11982389397", ""),
    7: (None, "PES University Institute of Medical Sciences: not in OSM extract"),
    8: ("node/1228960966", "OSM: BGS Gleneagles Hospitals, Kengeri (same campus)"),
    9: ("node/5983331355", "OSM: CyteCare"),
    10: ("way/332071569", "OSM spelling: Sanjeevani Hospital [TODO-VERIFY same facility]"),
    11: ("node/1474875685", "OSM: Dr. Ambedkar Medical College Hospital"),
    12: ("node/6944276882", ""),
    13: ("way/44091443", ""),
    14: ("way/44091445", "OSM: Apollo Hospital, opp. IIM, Bannerghatta Road"),
    15: ("way/160906231", "now Manipal Hospital Hebbal (ex Columbia Asia), Kirloskar Business Park"),
    16: ("way/133061664", "OSM: Manipal Hospital, 98 Rusthom Bagh, Old Airport Road"),
    17: ("way/631580848", "OSM: Sakra World"),
    18: ("relation/17069662", ""),
    19: (None, "Narayana Health City (Bommasandra): outside the OSM extract"),
    20: ("way/1040899005", ""),
}


def load_osm(path: Path) -> dict[str, dict]:
    g = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for f in g["features"]:
        lon, lat = f["geometry"]["coordinates"]
        out[f["properties"]["@id"]] = {"osm_name": f["properties"].get("name", ""), "lat": lat, "lon": lon}
    return out


def _flag(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def trauma_level(row) -> int:
    return int(_flag(row.polytrauma)) + int(_flag(row.neurosurgery)) + int(
        _flag(row.emergency) or _flag(row.accident_victims))


def specialties(row) -> str:
    s = []
    if _flag(row.polytrauma) or _flag(row.accident_victims):
        s.append("trauma")
    if _flag(row.neurosurgery):
        s.append("neurosurgery")
    if _flag(row.orthopaedics):
        s.append("orthopaedics")
    if _flag(row.emergency):
        s.append("emergency")
    return ";".join(s)


def build_hospitals(curated: pd.DataFrame, osm: dict[str, dict], seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for r in curated.itertuples():
        osm_id, note = MATCH.get(int(r.id), (None, "no match entry"))
        if osm_id is None:
            continue
        if osm_id not in osm:
            raise KeyError(f"OSM id {osm_id} for curated id {r.id} not in the geojson")
        o = osm[osm_id]
        rows.append({
            "hospital_id": f"H{int(r.id):02d}", "name": r.name, "lat": round(o["lat"], 6),
            "lon": round(o["lon"], 6), "trauma_level": trauma_level(r), "beds_total": "",
            "beds_available": int(rng.integers(2, 26)),                       # SIM
            "icu_beds_available": int(rng.integers(0, 7)),                    # SIM
            "specialties": specialties(r), "open_24x7": _flag(r.emergency),
            **{f"blood_{g}": int(rng.integers(0, 31)) for g in BLOOD_GROUPS},  # SIM
            "osm_id": osm_id, "osm_name": o["osm_name"], "match_note": note,
        })
    return pd.DataFrame(rows)


def build_ambulances(mvp: pd.DataFrame) -> pd.DataFrame:
    out = mvp.rename(columns={"id": "ambulance_id", "name": "base_name"})
    out["location_quality"] = "approximated"
    return out[["ambulance_id", "base_name", "lat", "lon", "zone", "location_quality"]]


def main() -> None:
    p = argparse.ArgumentParser(description="Build hospitals + ambulance bases tables")
    p.add_argument("--src", type=Path, default=SRC)
    a = p.parse_args()
    hosp = build_hospitals(pd.read_csv(a.src / "curated_emergency_hospitals.csv"),
                           load_osm(a.src / "bengaluru_hospitals.geojson"))
    hosp.to_csv(OUT_HOSP, index=False)
    amb = build_ambulances(pd.read_csv(a.src / "ambulance_bases_mvp.csv"))
    amb.to_csv(OUT_AMB, index=False)
    dropped = [f"{k}: {n}" for k, (o, n) in MATCH.items() if o is None]
    print(f"wrote {OUT_HOSP}: {len(hosp)} hospitals (dropped {len(dropped)}: {'; '.join(dropped)})")
    print(f"wrote {OUT_AMB}: {len(amb)} ambulance bases (approximated locations)")
    print("trauma levels:", hosp.trauma_level.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
