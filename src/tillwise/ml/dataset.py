"""Direct multi-horizon training data: one model predicts every horizon 1..H.

Each row is (series, origin week o, horizon h): features are what is known at the end
of week o, plus calendar facts about the target week o+h, and the label is units in
week o+h. No feature ever looks past week o, so there is no leakage, and the same frame
builder serves training, backtesting and future forecasts.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from tillwise.ml.features import SERIES

FEATURES = [
    "h", "last_units", "ma4", "ma13", "ma52", "units_same_week_last_year", "has_last_year",
    "last_price", "target_week_of_year", "target_holiday", "stock_idx", "country_idx",
]


def origin_features(weekly: DataFrame) -> DataFrame:
    w = Window.partitionBy(*SERIES).orderBy("week_idx")
    return weekly.select(
        *SERIES,
        F.col("week_idx").alias("origin_idx"),
        F.col("week_start").alias("origin_week"),
        F.col("units").alias("last_units"),
        F.avg("units").over(w.rowsBetween(-3, 0)).alias("ma4"),
        F.avg("units").over(w.rowsBetween(-12, 0)).alias("ma13"),
        F.avg("units").over(w.rowsBetween(-51, 0)).alias("ma52"),
        F.coalesce(F.col("avg_price"), F.lit(0.0)).alias("last_price"),
    )


def target_frame(weekly: DataFrame) -> DataFrame:
    w = Window.partitionBy(*SERIES).orderBy("week_idx")
    return weekly.select(
        *SERIES,
        F.col("week_idx").alias("target_idx"),
        F.col("week_start").alias("target_week"),
        F.col("units").alias("label"),
        F.col("week_of_year").alias("target_week_of_year"),
        F.col("holiday_in_week").alias("target_holiday"),
        F.lag("units", 52).over(w).alias("units_same_week_last_year"),
    )


def build_direct_frame(origins: DataFrame, targets: DataFrame, horizons: int) -> DataFrame:
    spark = origins.sparkSession
    hz = spark.range(1, horizons + 1).select(F.col("id").cast("int").alias("h"))
    frame = (
        origins.crossJoin(hz)
        .withColumn("target_idx", F.col("origin_idx") + F.col("h"))
        .join(targets, [*SERIES, "target_idx"])
    )
    return (
        frame.withColumn("has_last_year", F.col("units_same_week_last_year").isNotNull().cast("int"))
        # no history a year back yet: fall back to the 13-week average, and say so
        .withColumn("units_same_week_last_year", F.coalesce("units_same_week_last_year", "ma13"))
        .withColumn("target_week_of_year", F.col("target_week_of_year").cast("double"))
        .withColumn("target_holiday", F.col("target_holiday").cast("double"))
        .withColumn("h", F.col("h").cast("double"))
    )


def training_frame(weekly: DataFrame, horizons: int = 12, min_history: int = 13) -> DataFrame:
    """All (origin, horizon) pairs whose target is observed. Origins need 13 weeks of history."""
    origins = origin_features(weekly).where(F.col("origin_idx") >= min_history - 1)
    return build_direct_frame(origins, target_frame(weekly), horizons)


def future_frame(weekly: DataFrame, dim_date: DataFrame, horizons: int = 8) -> DataFrame:
    """Rows to forecast the next H weeks from the last observed week."""
    origins = origin_features(weekly)
    last_idx = weekly.agg(F.max("week_idx")).first()[0]
    origins = origins.where(F.col("origin_idx") == last_idx)
    hz = weekly.sparkSession.range(1, horizons + 1).select(F.col("id").cast("int").alias("h"))
    hol = dim_date.groupBy("week_start").agg(F.max(F.col("is_holiday").cast("int")).alias("target_holiday"))
    ly = weekly.select(*SERIES, F.col("week_idx").alias("ly_idx"),
                       F.col("units").alias("units_same_week_last_year"))
    base = (
        origins.crossJoin(hz)
        .withColumn("target_idx", F.col("origin_idx") + F.col("h"))
        .withColumn("target_week", F.date_add(F.col("origin_week"), (F.col("h") * 7).cast("int")))
        .join(hol, F.col("target_week") == F.col("week_start"), "left").drop("week_start")
        .fillna({"target_holiday": 0})
        .withColumn("target_week_of_year", F.weekofyear("target_week"))
    )
    frame = (
        base.alias("b").join(
            ly.alias("l"),
            (F.col("b.stock_code") == F.col("l.stock_code")) & (F.col("b.country") == F.col("l.country"))
            & (F.col("l.ly_idx") == F.col("b.target_idx") - 52),
            "left",
        )
        .select("b.*", "l.units_same_week_last_year")
        .withColumn("label", F.lit(None).cast("double"))
    )
    return (
        frame.withColumn("has_last_year", F.col("units_same_week_last_year").isNotNull().cast("int"))
        .withColumn("units_same_week_last_year", F.coalesce("units_same_week_last_year", "ma13"))
        .withColumn("target_week_of_year", F.col("target_week_of_year").cast("double"))
        .withColumn("target_holiday", F.col("target_holiday").cast("double"))
        .withColumn("h", F.col("h").cast("double"))
    )
