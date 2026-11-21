# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Unity Catalog setup
# MAGIC Creates the catalog, the medallion schemas and the landing Volume.
# MAGIC
# MAGIC **Free Edition note:** if `CREATE CATALOG` is not allowed in your workspace, set the
# MAGIC `catalog` widget to `workspace` (the default catalog) and run again; the schemas and
# MAGIC Volume are then created inside it.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from tillwise.config import LANDING_VOLUME, SCHEMAS

try:
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")  # noqa: F821
except Exception as e:  # noqa: BLE001
    print(f"Could not create catalog {CATALOG!r} ({type(e).__name__}). "
          "Set the widget to 'workspace' and re-run.")
    raise

for schema in SCHEMAS:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")  # noqa: F821

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.raw.{LANDING_VOLUME}")  # noqa: F821
spark.sql(f"COMMENT ON SCHEMA {CATALOG}.bronze IS 'Raw source rows as strings, with lineage'")  # noqa: F821
spark.sql(f"COMMENT ON SCHEMA {CATALOG}.silver IS 'Typed, de-duplicated, rule-checked data'")  # noqa: F821
spark.sql(f"COMMENT ON SCHEMA {CATALOG}.gold IS 'Star schema for BI, forecasting and the agent'")  # noqa: F821

display(spark.sql(f"SHOW SCHEMAS IN {CATALOG}"))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC **Next:** upload the CSVs from your local `data/raw/` folder to
# MAGIC `/Volumes/<catalog>/raw/landing/retail/` (Catalog Explorer → Volume → Upload),
# MAGIC and `uk_holidays.csv` / `weather_london_daily.csv` to `/Volumes/<catalog>/raw/landing/reference/`.

# COMMAND ----------

dbutils.fs.mkdirs(f"{T.landing_path}/retail")  # noqa: F821
dbutils.fs.mkdirs(f"{T.landing_path}/reference")  # noqa: F821
display(dbutils.fs.ls(T.landing_path))  # noqa: F821
