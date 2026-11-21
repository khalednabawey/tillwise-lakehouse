# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze with Auto Loader
# MAGIC Incrementally ingests every CSV that lands in `/Volumes/<catalog>/raw/landing/retail/`.
# MAGIC Re-running only picks up new files (the checkpoint remembers what was loaded).
# MAGIC All columns stay strings; `standardize_raw` normalises names and adds lineage.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from pyspark.sql import functions as F

from tillwise.transforms.bronze import standardize_raw

source = f"{T.landing_path}/retail/"
checkpoint = f"{T.landing_path}/_checkpoints/bronze_transactions"

stream = (
    spark.readStream.format("cloudFiles")  # noqa: F821
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", checkpoint)
    .option("cloudFiles.inferColumnTypes", "false")  # keep everything as strings in bronze
    .option("header", "true")
    .option("multiLine", "true")
    .option("escape", '"')
    .load(source)
)

bronze = standardize_raw(stream, source_file=F.col("_metadata.file_path"))

query = (
    bronze.writeStream.option("checkpointLocation", checkpoint)
    .trigger(availableNow=True)  # batch-style run of an incremental stream
    .toTable(T.bronze_transactions)
)
query.awaitTermination()

# COMMAND ----------

display(spark.sql(f"""
  SELECT _source_file, count(*) AS rows, max(_ingest_ts) AS loaded_at
  FROM {T.bronze_transactions} GROUP BY _source_file ORDER BY _source_file
"""))  # noqa: F821
