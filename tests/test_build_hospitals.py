import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from backend.agents import hospital_agent as ha

spec = importlib.util.spec_from_file_location("bh", Path("scripts/build_hospitals.py"))
bh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bh)


def curated(**over):
    row = {"id": 1, "name": "X", "emergency": "True", "polytrauma": "True", "accident_victims": "False",
           "neurosurgery": "True", "orthopaedics": "False"}
    row.update(over)
    return pd.DataFrame([row])


def test_trauma_level_and_specialties():
    r = next(curated().itertuples())
    assert bh.trauma_level(r) == 3 and bh.specialties(r) == "trauma;neurosurgery;emergency"
    r0 = next(curated(emergency="False", polytrauma="False", neurosurgery="False").itertuples())
    assert bh.trauma_level(r0) == 0 and bh.specialties(r0) == ""
    r1 = next(curated(emergency="False", polytrauma="False", neurosurgery="False",
                      accident_victims="True").itertuples())
    assert bh.trauma_level(r1) == 1 and bh.specialties(r1) == "trauma"


def test_dropped_and_unknown_osm_ids():
    osm = {"relation/2187206": {"osm_name": "A", "lat": 13.0, "lon": 77.5}}
    assert len(bh.build_hospitals(curated(id=4), osm)) == 0                 # mapped to None
    with pytest.raises(KeyError):
        bh.build_hospitals(curated(id=2), osm)                              # id not in geojson


def test_seeded_simulation_is_reproducible():
    osm = {"relation/2187206": {"osm_name": "A", "lat": 13.0, "lon": 77.5}}
    a, b = bh.build_hospitals(curated(), osm), bh.build_hospitals(curated(), osm)
    pd.testing.assert_frame_equal(a, b)


def test_real_tables_load_into_agents():
    src = Path("data/raw/hospitals_src")
    hosp = bh.build_hospitals(pd.read_csv(src / "curated_emergency_hospitals.csv"),
                              bh.load_osm(src / "bengaluru_hospitals.geojson"))
    assert len(hosp) == 17 and hosp.hospital_id.is_unique
    assert hosp.lat.between(12.7, 13.3).all() and hosp.lon.between(77.3, 77.9).all()
    assert hosp.trauma_level.between(0, 3).all()
    agent = ha.HospitalAgent(hosp)                                           # schema accepted
    amb = bh.build_ambulances(pd.read_csv(src / "ambulance_bases_mvp.csv"))
    assert list(amb.columns[:4]) == ["ambulance_id", "base_name", "lat", "lon"]
    assert (amb.location_quality == "approximated").all() and amb.ambulance_id.is_unique
    assert agent is not None
