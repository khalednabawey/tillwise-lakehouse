"""Download the public sources into data/raw/ as CSV, ready to upload to the Databricks Volume.

Sources (see data/README.md for licences and attribution):
  1. UCI Online Retail II (dataset 502), CC BY 4.0: two XLSX sheets -> two CSVs
  2. Open-Meteo historical weather for London, daily, CC BY 4.0
  3. UK (England) public holidays from the `holidays` package (offline, MIT)

Usage:  uv run python -m tillwise.ingest.download_sources [--skip-weather]
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
LONDON = {"latitude": 51.5072, "longitude": -0.1276}
START, END = "2009-12-01", "2011-12-31"

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"


def download_retail(out_dir: Path) -> list[Path]:
    print(f"Downloading {UCI_URL} ...")
    resp = requests.get(UCI_URL, timeout=300)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xlsx_name = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        xlsx_bytes = zf.read(xlsx_name)
    print(f"Reading {xlsx_name} (two sheets, this takes a minute) ...")
    sheets = pd.read_excel(io.BytesIO(xlsx_bytes), sheet_name=None, dtype=str, engine="openpyxl")
    written = []
    for sheet, df in sheets.items():
        # "Year 2009-2010" -> online_retail_ii_2009_2010.csv
        suffix = sheet.lower().replace("year", "").strip().replace("-", "_")
        path = out_dir / f"online_retail_ii_{suffix}.csv"
        df.to_csv(path, index=False)
        print(f"  {path.name}: {len(df):,} rows")
        written.append(path)
    return written


def download_weather(out_dir: Path) -> Path:
    params = {**LONDON, "start_date": START, "end_date": END, "timezone": "Europe/London",
              "daily": "temperature_2m_mean,precipitation_sum"}
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=120)
    resp.raise_for_status()
    daily = resp.json()["daily"]
    df = pd.DataFrame({"date": daily["time"], "temp_mean_c": daily["temperature_2m_mean"],
                       "precip_mm": daily["precipitation_sum"]})
    path = out_dir / "weather_london_daily.csv"
    df.to_csv(path, index=False)
    print(f"  {path.name}: {len(df):,} days")
    return path


def write_holidays(out_dir: Path) -> Path:
    import holidays

    uk = holidays.country_holidays("GB", subdiv="ENG", years=[2009, 2010, 2011, 2012])
    df = pd.DataFrame(sorted(uk.items()), columns=["date", "holiday_name"])
    path = out_dir / "uk_holidays.csv"
    df.to_csv(path, index=False)
    print(f"  {path.name}: {len(df)} holidays")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-weather", action="store_true")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    download_retail(RAW)
    if not args.skip_weather:
        download_weather(RAW)
    write_holidays(RAW)
    print(f"Done. Files are in {RAW}")


if __name__ == "__main__":
    main()
