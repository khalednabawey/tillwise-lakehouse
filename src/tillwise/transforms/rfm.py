"""Customer RFM segments (recency, frequency, monetary) for gold.customer_rfm."""

from __future__ import annotations

from datetime import date

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# Rules are checked top to bottom on the 1-5 scores (5 = best). First match wins.
SEGMENT_RULES: list[tuple[str, str]] = [
    ("Champions", "r >= 4 AND f >= 4 AND m >= 4"),
    ("Loyal", "f >= 4 AND r >= 3"),
    ("New", "r >= 4 AND f <= 2"),
    ("At risk", "r <= 2 AND f >= 3"),
    ("Hibernating", "r <= 2 AND f <= 2"),
    ("Needs attention", "TRUE"),
]


def build_customer_rfm(fact: DataFrame, as_of: date) -> DataFrame:
    """One row per known customer, scored 1-5 on each dimension with quintiles."""
    known = fact.where(F.col("customer_id").isNotNull())
    base = known.groupBy("customer_id").agg(
        F.datediff(F.lit(as_of), F.max(F.when(~F.col("is_return"), F.col("invoice_ts")))).alias("recency_days"),
        F.countDistinct(F.when(~F.col("is_return"), F.col("invoice_no"))).alias("frequency"),
        F.sum("line_revenue").cast("decimal(14,2)").alias("monetary"),
    ).where(F.col("frequency") > 0)  # customers with only returns have no purchase to score

    def quintile(col: str, ascending: bool) -> F.Column:
        # Order worst -> best and score ceil(cume_dist * 5). Unlike ntile, tied values always
        # get the same score, so results don't depend on row order. For recency, fewer days is
        # better, so it is ordered descending.
        order = F.col(col).asc() if ascending else F.col(col).desc()
        return F.ceil(F.cume_dist().over(Window.orderBy(order)) * 5).cast("int")

    scored = (
        base.withColumn("r", quintile("recency_days", ascending=False))
        .withColumn("f", quintile("frequency", ascending=True))
        .withColumn("m", quintile("monetary", ascending=True))
    )
    segment = None
    for name, rule in SEGMENT_RULES:
        cond = F.expr(rule)
        segment = F.when(cond, F.lit(name)) if segment is None else segment.when(cond, F.lit(name))
    return scored.withColumn("segment", segment).select(
        "customer_id", "recency_days", "frequency", "monetary",
        F.col("r").alias("r_score"), F.col("f").alias("f_score"), F.col("m").alias("m_score"),
        F.concat_ws("", "r", "f", "m").alias("rfm_cell"), "segment",
    )
