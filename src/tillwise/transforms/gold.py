"""Gold: star schema for analysis, dashboards, the forecast and the agent.

fact_sales  one row per clean invoice line, keyed to the product version valid at sale time
dim_product SCD Type 2 product versions
dim_customer, dim_country, dim_date
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


def date_key(col: F.Column) -> F.Column:
    """yyyymmdd as an int, the usual warehouse date key."""
    return F.date_format(col, "yyyyMMdd").cast("int")


def build_dim_date(
    spark: SparkSession, start: date, end: date, holidays: Iterable[tuple[date, str]] = ()
) -> DataFrame:
    days = spark.sql(
        f"SELECT explode(sequence(DATE'{start.isoformat()}', DATE'{end.isoformat()}', INTERVAL 1 DAY)) AS d"
    )
    hol = spark.createDataFrame(list(holidays) or [(date(1900, 1, 1), "none")], "d date, holiday_name string")
    return dim_date_from_days(days, hol)


def build_dim_date_for(clean: DataFrame, holidays: DataFrame | None = None) -> DataFrame:
    """Lazy version for Lakeflow pipelines (no actions): 1 Jan of the first sale year to 31 Dec of the
    year after the last sale. `holidays` has columns date (string or date) and holiday_name."""
    days = clean.agg(F.min("invoice_ts").alias("lo"), F.max("invoice_ts").alias("hi")).select(
        F.explode(F.sequence(F.make_date(F.year("lo"), F.lit(1), F.lit(1)),
                             F.make_date(F.year("hi") + 1, F.lit(12), F.lit(31)),
                             F.expr("INTERVAL 1 DAY"))).alias("d"))
    if holidays is None:
        hol = clean.sparkSession.createDataFrame([(date(1900, 1, 1), "none")], "d date, holiday_name string")
    else:
        hol = holidays.select(F.col("date").cast("date").alias("d"), "holiday_name")
    return dim_date_from_days(days, hol)


def dim_date_from_days(days: DataFrame, hol: DataFrame) -> DataFrame:
    return (
        days.join(hol, "d", "left")
        .select(
            date_key(F.col("d")).alias("date_key"),
            F.col("d").alias("date"),
            F.year("d").alias("year"),
            F.quarter("d").alias("quarter"),
            F.month("d").alias("month"),
            F.date_format("d", "MMMM").alias("month_name"),
            F.weekofyear("d").alias("iso_week"),
            F.date_trunc("week", "d").cast("date").alias("week_start"),
            F.dayofweek("d").alias("day_of_week"),  # 1 = Sunday ... 7 = Saturday
            F.date_format("d", "EEEE").alias("day_name"),
            F.dayofweek("d").isin(1, 7).alias("is_weekend"),
            F.col("holiday_name").isNotNull().alias("is_holiday"),
            "holiday_name",
        )
        .orderBy("date_key")
    )


def build_dim_country(clean: DataFrame) -> DataFrame:
    return (
        clean.where(F.col("country").isNotNull())
        .select("country")
        .distinct()
        .withColumn("country_key", F.xxhash64("country"))
        .select("country_key", "country")
    )


def build_dim_customer(clean: DataFrame) -> DataFrame:
    known = clean.where(F.col("customer_id").isNotNull())
    by_country = Window.partitionBy("customer_id").orderBy(F.col("lines").desc(), F.col("country"))
    primary_country = (
        known.groupBy("customer_id", "country")
        .agg(F.count("*").alias("lines"))
        .withColumn("rk", F.row_number().over(by_country))
        .where("rk = 1")
        .select("customer_id", F.col("country").alias("primary_country"))
    )
    activity = known.groupBy("customer_id").agg(
        F.min("invoice_ts").alias("first_purchase_ts"),
        F.max("invoice_ts").alias("last_purchase_ts"),
        F.countDistinct(F.when(~F.col("is_return"), F.col("invoice_no"))).alias("orders"),
        F.countDistinct(F.when(F.col("is_return"), F.col("invoice_no"))).alias("return_invoices"),
    )
    return activity.join(primary_country, "customer_id").select(
        "customer_id", "primary_country", "first_purchase_ts", "last_purchase_ts", "orders", "return_invoices"
    )


def build_fact_sales(clean: DataFrame, dim_product: DataFrame) -> DataFrame:
    """Attach the product version that was valid when the line was sold.

    The join is on stock_code plus valid_from <= invoice_ts < valid_to (open-ended when
    valid_to is NULL, i.e. the current version). Lines whose
    stock code has no description at all (no version) keep a null product_sk, and the
    tests check that every line with a described product gets exactly one version.
    """
    p = dim_product.select("product_sk", "stock_code", "valid_from", "valid_to")
    joined = clean.alias("s").join(
        p.alias("p"),
        (F.col("s.stock_code") == F.col("p.stock_code"))
        & (F.col("s.invoice_ts") >= F.col("p.valid_from"))
        & (F.col("p.valid_to").isNull() | (F.col("s.invoice_ts") < F.col("p.valid_to"))),
        "left",
    )
    return joined.select(
        F.col("s.invoice_no").alias("invoice_no"),
        date_key(F.col("s.invoice_ts")).alias("date_key"),
        F.col("s.invoice_ts").alias("invoice_ts"),
        F.col("p.product_sk").alias("product_sk"),
        F.col("s.stock_code").alias("stock_code"),
        F.col("s.customer_id").alias("customer_id"),
        F.xxhash64(F.col("s.country")).alias("country_key"),
        F.col("s.quantity").alias("quantity"),
        F.col("s.unit_price").alias("unit_price"),
        F.col("s.line_revenue").alias("line_revenue"),
        F.col("s.is_return").alias("is_return"),
    )
