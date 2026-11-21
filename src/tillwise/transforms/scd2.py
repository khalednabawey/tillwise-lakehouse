"""SCD Type 2 product dimension, rebuilt from transaction history.

The source has no product master: a stock code's description is whatever staff typed
on the invoice, and it changes over time (renames, typo fixes). To get one stable
description per day, take the most frequent description per stock code per day, then
start a new version whenever that daily description changes.

Each version gets:
  product_sk  deterministic surrogate key (hash of stock_code + valid_from)
  valid_from  start of the first day the description was seen
  valid_to    valid_from of the next version (exclusive), or NULL for the current version
              (same convention as Lakeflow AUTO CDC's __END_AT)
  is_current  true for the latest version

Week 2 moves this into Lakeflow's AUTO CDC (formerly APPLY CHANGES) with the same
output columns, so gold does not change.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def daily_descriptions(clean: DataFrame) -> DataFrame:
    """Most frequent description per stock code per day (ties: alphabetical, for determinism)."""
    counts = (
        clean.where(F.col("description").isNotNull())
        .withColumn("sale_date", F.to_date("invoice_ts"))
        .groupBy("stock_code", "sale_date", "description")
        .agg(F.count("*").alias("n"))
    )
    w = Window.partitionBy("stock_code", "sale_date").orderBy(F.col("n").desc(), F.col("description"))
    return (
        counts.withColumn("rk", F.row_number().over(w))
        .where("rk = 1")
        .select("stock_code", "sale_date", "description")
    )


def build_product_scd2(clean: DataFrame) -> DataFrame:
    daily = daily_descriptions(clean)
    by_day = Window.partitionBy("stock_code").orderBy("sale_date")
    versions = (
        daily.withColumn("prev_description", F.lag("description").over(by_day))
        .withColumn(
            "is_change",
            F.when(F.col("prev_description").isNull(), 1)
            .when(F.col("prev_description") != F.col("description"), 1)
            .otherwise(0),
        )
        .withColumn("version", F.sum("is_change").over(by_day.rowsBetween(Window.unboundedPreceding, 0)))
        .groupBy("stock_code", "version")
        .agg(
            F.first("description").alias("description"),
            F.min("sale_date").alias("first_day"),
            F.max("sale_date").alias("last_seen_day"),
        )
    )
    by_version = Window.partitionBy("stock_code").orderBy("version")
    return (
        versions.withColumn("valid_from", F.col("first_day").cast("timestamp_ntz"))
        .withColumn("valid_to", F.lead("valid_from").over(by_version))
        .withColumn("is_current", F.col("valid_to").isNull())
        .withColumn("product_sk", F.xxhash64("stock_code", "valid_from"))
        .select(
            "product_sk",
            "stock_code",
            "description",
            F.col("version").cast("int").alias("version"),
            "valid_from",
            "valid_to",
            "is_current",
            "last_seen_day",
        )
    )
