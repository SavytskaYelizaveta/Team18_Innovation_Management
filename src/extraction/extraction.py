from pathlib import Path
import pandas as pd
import yaml

from epo.tipdata.patstat import PatstatClient
from epo.tipdata.patstat.database.models import TLS901_TECHN_FIELD_IPC
from src.extraction.helpers import (
    Scope,
    application_scope,
    count_by_office,
    count_per_year,
    family_scope,
    fractional_by_country,
    fractional_by_cpc_subclass,
    fractional_by_techn_field,
    sector_family_ids_with_sec,
)

# Setup
with open("config/01.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

sec = cfg["sector"]
tf = cfg["technology_filter"]

ENV = "PROD"
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

patstat = PatstatClient(env=ENV)
db = patstat.orm()
print(f"Connecting to PATSTAT {ENV}... extracting {sec['name']}")

sector_ids = sector_family_ids_with_sec(db, tf["core_cpc"], tf["secondary_cpc"], tf["keywords"])

y_min, y_max = sec["time_window"]["start_year"], sec["time_window"]["end_year"]
fam_col, app_col = sec["time_columns"]["family"], sec["time_columns"]["application"]
ipr_type = sec["ipr_type"]

SCOPES = [
    Scope("all patents", "family", family_scope(db, None, y_min, y_max, fam_col, ipr_type)),
    Scope(sec["name"], "family", family_scope(db, sector_ids, y_min, y_max, fam_col, ipr_type)),
    Scope("all patents", "application", application_scope(db, None, y_min, y_max, app_col, ipr_type)),
    Scope(sec["name"], "application", application_scope(db, sector_ids, y_min, y_max, app_col, ipr_type)),
]
SECTOR_SCOPES = [s for s in SCOPES if s.name == sec["name"]]


# Extract and Save
print("1/6 Extracting volume...")
volume = pd.concat(
    [count_per_year(patstat, db, s).assign(series="all") for s in SCOPES]
    + [count_per_year(patstat, db, s, only_pct=True).assign(series="via PCT") for s in SCOPES],
    ignore_index=True,
)
volume.to_csv(OUTPUT_DIR / "volume.csv", index=False)

print("2/6 Extracting CPC subclass mix...")
cpc_mix = pd.concat([fractional_by_cpc_subclass(patstat, db, s) for s in SECTOR_SCOPES], ignore_index=True)
cpc_mix.to_csv(OUTPUT_DIR / "cpc_subclasses.csv", index=False)

print("3/6 Extracting fields...")
field_names = patstat.df(
    db.query(TLS901_TECHN_FIELD_IPC.techn_field_nr, TLS901_TECHN_FIELD_IPC.techn_field,
             TLS901_TECHN_FIELD_IPC.techn_sector).distinct()
).drop_duplicates("techn_field_nr")
fields = pd.concat([fractional_by_techn_field(patstat, db, s) for s in SECTOR_SCOPES], ignore_index=True)
fields = fields.merge(field_names, on="techn_field_nr", how="left")
fields.to_csv(OUTPUT_DIR / "techn_fields.csv", index=False)

print("4/6 Extracting IP5 office filings...")
offices = pd.concat([count_by_office(patstat, db, s) for s in SECTOR_SCOPES], ignore_index=True)
offices["office"] = offices["office"].where(offices["office"].isin(sec["ip5_offices"] + ["WO"]), "Other")
offices = offices.groupby(["level", "office", "year"], as_index=False)["n"].sum()
offices.to_csv(OUTPUT_DIR / "offices.csv", index=False)

print("5/6 Extracting applicant/inventor countries...")
countries = pd.concat(
    [fractional_by_country(patstat, db, s, role) for s in SECTOR_SCOPES for role in ("applicant", "inventor")],
    ignore_index=True,
)
countries.to_csv(OUTPUT_DIR / "countries.csv", index=False)

print("6/6 Extracting list of sector family IDs...")
sector_families_df = patstat.df(sector_ids)
sector_families_df.to_csv(OUTPUT_DIR / "sector_family_ids.csv", index=False)

print(f"\n All files saved to {OUTPUT_DIR.resolve()} ({len(sector_families_df):,} families found).")