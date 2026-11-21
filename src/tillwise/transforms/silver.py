"""Silver: typed, de-duplicated transactions, split into clean / quarantine / non-product.

Nothing is silently dropped. Each row ends up in exactly one of:
  - clean        : product sales and returns that pass every rule
  - quarantine   : rows that break a rule, with the rule name in `dq_reason`
  - non_product  : postage, fees and manual adjustments (valid rows, not products)
Exact duplicates are removed and counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from tillwise.config import NON_PRODUCT_CODE_PATTERN, QUARANTINE_RULES, SOURCE_TIMESTAMP_FORMAT

BUSINESS_COLUMNS = [
    "invoice_no",
    "stock_code",
    "description",
    "quantity",
    "invoice_ts",
    "unit_price",
    "customer_id",
    "country",
]


def type_transactions(bronze: DataFrame) -> DataFrame:
    """Cast bronze strings to proper types without failing on bad values (ANSI-safe)."""
    return (
        bronze.select(
            F.upper(F.col("invoice_no")).alias("invoice_no"),
            F.upper(F.col("stock_code")).alias("stock_code"),
            # Collapse repeated spaces; descriptions are free text typed by staff.
            F.regexp_replace(F.col("description"), r"\s+", " ").alias("description"),
            F.expr("try_cast(quantity AS INT)").alias("quantity"),
            # Shop-local wall-clock times with no zone in the source: TIMESTAMP_NTZ, not UTC.
            F.expr(f"CAST(try_to_timestamp(invoice_ts, '{SOURCE_TIMESTAMP_FORMAT}') AS TIMESTAMP_NTZ)")
            .alias("invoice_ts"),
            F.expr("try_cast(unit_price AS DECIMAL(12, 2))").alias("unit_price"),
            # Spreadsheet exports turn IDs into floats ("13085.0"); keep the integer part.
            F.regexp_replace(F.col("customer_id"), r"\.0+$", "").alias("customer_id"),
            F.col("country"),
            F.col("_source_file"),
            F.col("_ingest_ts"),
        )
        .withColumn("is_return", F.coalesce(F.col("invoice_no").startswith("C"), F.lit(False)))
    )


def _first_matching_rule() -> F.Column:
    """CASE WHEN rule_1 THEN 'rule_1' WHEN rule_2 ... END, in QUARANTINE_RULES order."""
    expr = None
    for name, condition in QUARANTINE_RULES.items():
        cond = F.coalesce(F.expr(condition), F.lit(False))
        expr = F.when(cond, F.lit(name)) if expr is None else expr.when(cond, F.lit(name))
    return expr  # type: ignore[return-value]


@dataclass
class SilverResult:
    clean: DataFrame
    quarantine: DataFrame
    non_product: DataFrame
    duplicates_removed: DataFrame = field(repr=False)


def split_transactions(typed: DataFrame) -> SilverResult:
    """De-duplicate, then route every row to clean, quarantine or non_product."""
    deduped = typed.dropDuplicates(BUSINESS_COLUMNS)
    duplicates = typed.exceptAll(deduped)

    flagged = deduped.withColumn("dq_reason", _first_matching_rule()).withColumn(
        "is_non_product", F.coalesce(F.col("stock_code").rlike(NON_PRODUCT_CODE_PATTERN), F.lit(False))
    )

    quarantine = flagged.where(F.col("dq_reason").isNotNull()).drop("is_non_product")
    passed = flagged.where(F.col("dq_reason").isNull()).drop("dq_reason")
    non_product = passed.where(F.col("is_non_product")).drop("is_non_product")
    clean = (
        passed.where(~F.col("is_non_product"))
        .drop("is_non_product")
        .withColumn("line_revenue", (F.col("quantity") * F.col("unit_price")).cast("decimal(14, 2)"))
    )
    return SilverResult(clean=clean, quarantine=quarantine, non_product=non_product,
                        duplicates_removed=duplicates)


def dq_metrics(typed: DataFrame, result: SilverResult) -> DataFrame:
    """One row per outcome, for the README table and the dq_metrics Delta table."""
    spark = typed.sparkSession
    total = typed.count()
    rows = [("input_rows", total), ("duplicates_removed", result.duplicates_removed.count())]
    reasons = {r["dq_reason"]: r["n"] for r in
               result.quarantine.groupBy("dq_reason").agg(F.count("*").alias("n")).collect()}
    rows += [(f"quarantine:{name}", int(reasons.get(name, 0))) for name in QUARANTINE_RULES]
    rows += [("non_product_lines", result.non_product.count()), ("clean_rows", result.clean.count())]
    df = spark.createDataFrame(rows, "metric string, row_count long")
    return df.withColumn("pct_of_input", F.round(F.col("row_count") * 100.0 / F.lit(max(total, 1)), 2))
