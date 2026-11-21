# Databricks notebook source
# MAGIC %md
# MAGIC # 00b · Load the source files into the landing Volume
# MAGIC Downloads UCI Online Retail II (CC BY 4.0) and writes the CSVs to
# MAGIC `/Volumes/<catalog>/raw/landing/retail/`, plus England bank holidays (and London weather, if the
# MAGIC workspace can reach Open-Meteo) to `.../reference/`.
# MAGIC
# MAGIC **If the download is blocked** (Free Edition limits outbound internet), upload the files from your
# MAGIC local `data/raw/` folder instead: Catalog → `<catalog>` → `raw` → `landing` → Upload to this volume.

# COMMAND ----------

# MAGIC %pip install openpyxl holidays --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from pathlib import Path

from tillwise.ingest.download_sources import download_retail, download_weather, write_holidays

retail = Path(f"{T.landing_path}/retail")
reference = Path(f"{T.landing_path}/reference")
retail.mkdir(parents=True, exist_ok=True)
reference.mkdir(parents=True, exist_ok=True)

write_holidays(reference)
try:
    download_weather(reference)
except Exception as e:  # noqa: BLE001
    print(f"Weather download skipped ({type(e).__name__}); it is optional.")

if sorted(p.name for p in retail.glob("online_retail_ii_*.csv")):
    print("Retail CSVs already in the volume; skipping download.")
else:
    download_retail(retail)

display(dbutils.fs.ls(f"{T.landing_path}/retail"))
