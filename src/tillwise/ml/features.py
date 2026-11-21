"""Weekly demand table: product x country x week, on a complete grid of full weeks.

Target is gross units sold (returns are tracked separately), which is what a planner
orders against. Weeks with no sales are explicit zeros, so lags and moving averages
mean the same thing for every series. The first and last weeks of the data are partial
(the source starts on a Tuesday and ends on a Friday), so they are dropped.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

SERIES = ["stock_code", "country"]


def top_series(fact: DataFrame, n_products: int = 50, n_countries: int = 5) -> DataFrame:
    """Top countries by net revenue, then the top products by net revenue inside them."""
    countries = (
        fact.groupBy("country_key").agg(F.sum("line_revenue").alias("rev"))
        .orderBy(F.col("rev").desc()).limit(n_countries).select("country_key")
    )
    in_scope = fact.join(countries, "country_key")
    products = (
        in_scope.groupBy("stock_code").agg(F.sum("line_revenue").alias("rev"))
        .orderBy(F.col("rev").desc(), "stock_code").limit(n_products).select("stock_code")
    )
    return in_scope.join(products, "stock_code").select("stock_code", "country_key").distinct()


def full_weeks(fact: DataFrame) -> DataFrame:
    """Monday-starting weeks fully covered by the data. Lazy (no actions), so it works in Lakeflow."""
    bounds = fact.agg(F.min("invoice_ts").cast("date").alias("lo"), F.max("invoice_ts").cast("date").alias("hi"))
    return bounds.select(F.explode(F.expr(
        "sequence(CAST(date_trunc('week', date_add(lo, 6)) AS DATE),"            # first Monday on/after lo
        " date_add(CAST(date_trunc('week', date_add(hi, 1)) AS DATE), -7),"     # last week ending on/before hi
        " INTERVAL 7 DAY)")).alias("week_start"))


def weekly_demand(fact: DataFrame, dim_country: DataFrame, dim_date: DataFrame,
                  n_products: int = 50, n_countries: int = 5) -> DataFrame:
    series = top_series(fact, n_products, n_countries).join(dim_country, "country_key")
    weeks = full_weeks(fact)

    lines = fact.join(series.select("stock_code", "country_key"), ["stock_code", "country_key"]).withColumn(
        "week_start", F.date_trunc("week", "invoice_ts").cast("date")
    )
    agg = lines.groupBy("stock_code", "country_key", "week_start").agg(
        F.sum(F.when(~F.col("is_return"), F.col("quantity")).otherwise(0)).alias("units"),
        F.sum(F.when(F.col("is_return"), -F.col("quantity")).otherwise(0)).alias("return_units"),
        F.sum(F.when(~F.col("is_return"), F.col("line_revenue")).otherwise(0)).alias("gross_revenue"),
    )
    holidays = (
        dim_date.groupBy("week_start").agg(F.max(F.col("is_holiday").cast("int")).alias("holiday_in_week"))
    )
    grid = series.crossJoin(weeks)
    w = Window.partitionBy(*SERIES).orderBy("week_start")
    out = (
        grid.join(agg, ["stock_code", "country_key", "week_start"], "left")
        .join(holidays, "week_start", "left")
        .fillna({"units": 0, "return_units": 0, "gross_revenue": 0, "holiday_in_week": 0})
        .withColumn("avg_price", F.when(F.col("units") > 0,
                                        F.col("gross_revenue") / F.col("units")).cast("double"))
        # carry the last known price forward through zero-sale weeks
        .withColumn("avg_price", F.last("avg_price", ignorenulls=True).over(
            w.rowsBetween(Window.unboundedPreceding, 0)))
        .withColumn("week_of_year", F.weekofyear("week_start"))
        .withColumn("week_idx", F.row_number().over(w) - 1)
    )
    return out.select("stock_code", "country", "week_start", "week_idx", "week_of_year", "holiday_in_week",
                      F.col("units").cast("double").alias("units"),
                      F.col("return_units").cast("double").alias("return_units"),
                      "avg_price")
