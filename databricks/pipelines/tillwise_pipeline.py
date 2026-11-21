# Lakeflow Declarative Pipeline (formerly Delta Live Tables) for Tillwise.
#
# One pipeline, bronze -> silver -> gold, using the same tested functions as the
# notebooks and the local runner. Configuration (set in resources/pipeline.yml):
#   tillwise.catalog       Unity Catalog catalog (e.g. tillwise or workspace)
#   tillwise.src_path      workspace path of the repo's src/ folder
#   tillwise.landing_path  /Volumes/<catalog>/raw/landing
#
# Data quality: every silver rule is a named expectation, so the pipeline UI shows a
# pass/fail count per rule on each update. Rows that break a rule are kept in the
# quarantine table instead of being dropped.
#
# No actions (count/collect/first) inside table functions: upstream tables don't exist yet when
# the pipeline graph is planned, so every function here is a pure transformation.

import sys

import dlt
from pyspark.sql import functions as F

sys.path.insert(0, spark.conf.get("tillwise.src_path"))

from tillwise.config import NON_PRODUCT_CODE_PATTERN, QUARANTINE_RULES  # noqa: E402
from tillwise.ml.features import weekly_demand  # noqa: E402
from tillwise.transforms import gold, rfm, scd2, silver  # noqa: E402
from tillwise.transforms.bronze import standardize_raw  # noqa: E402

CATALOG = spark.conf.get("tillwise.catalog")
LANDING = spark.conf.get("tillwise.landing_path")


def fq(schema: str, table: str) -> str:
    return f"{CATALOG}.{schema}.{table}"


# Expectation per rule: a row passes when it does NOT match the rule's condition.
RULE_EXPECTATIONS = {name: f"coalesce(NOT ({cond}), true)" for name, cond in QUARANTINE_RULES.items()}


# ---------------------------------------------------------------- bronze
@dlt.table(name=fq("bronze", "transactions_raw"),
           comment="Raw Online Retail II rows as strings, with source file and ingest time")
def bronze_transactions_raw():
    stream = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.inferColumnTypes", "false")
        .option("header", "true").option("multiLine", "true").option("escape", '"')
        .load(f"{LANDING}/retail/")
    )
    return standardize_raw(stream, source_file=F.col("_metadata.file_path"))


# ---------------------------------------------------------------- silver
@dlt.view(comment="Typed, de-duplicated rows with the first failing rule (if any)")
@dlt.expect_all(RULE_EXPECTATIONS)  # warn-only: records a metric per rule, keeps every row
def silver_flagged():
    typed = silver.type_transactions(spark.read.table(fq("bronze", "transactions_raw")))
    deduped = typed.dropDuplicates(silver.BUSINESS_COLUMNS)
    return (deduped.withColumn("dq_reason", silver.first_matching_rule())
            .withColumn("is_non_product",
                        F.coalesce(F.col("stock_code").rlike(NON_PRODUCT_CODE_PATTERN), F.lit(False))))


@dlt.table(name=fq("silver", "transactions"), comment="Clean product sales and returns")
@dlt.expect_or_fail("no_quarantined_rows", "dq_reason IS NULL")
def silver_transactions():
    return (spark.read.table("silver_flagged")
            .where("dq_reason IS NULL AND NOT is_non_product")
            .withColumn("line_revenue", (F.col("quantity") * F.col("unit_price")).cast("decimal(14, 2)"))
            .drop("is_non_product"))


@dlt.table(name=fq("silver", "transactions_quarantine"), comment="Rows that broke a rule, with dq_reason")
def silver_quarantine():
    return spark.read.table("silver_flagged").where("dq_reason IS NOT NULL").drop("is_non_product")


@dlt.table(name=fq("silver", "non_product_lines"), comment="Postage, fees and manual adjustments")
def silver_non_product():
    return (spark.read.table("silver_flagged").where("dq_reason IS NULL AND is_non_product")
            .drop("is_non_product", "dq_reason"))


@dlt.table(name=fq("silver", "dq_metrics"), comment="Row counts per data-quality outcome, latest update")
def silver_dq_metrics():
    typed = silver.type_transactions(spark.read.table(fq("bronze", "transactions_raw")))
    return silver.dq_metrics(typed, silver.split_transactions(typed)).withColumn("run_ts", F.current_timestamp())


@dlt.table(name=fq("silver", "products_scd2"), comment="SCD Type 2 product versions (daily modal description)")
@dlt.expect_or_fail("valid_range", "valid_to IS NULL OR valid_to > valid_from")
def silver_products_scd2():
    return scd2.build_product_scd2(spark.read.table(fq("silver", "transactions")))


# ---------------------------------------------------------------- gold
@dlt.table(name=fq("gold", "dim_product"), comment="Product versions; join on valid_from/valid_to")
def gold_dim_product():
    return spark.read.table(fq("silver", "products_scd2"))


@dlt.table(name=fq("gold", "dim_date"), comment="Calendar with England bank holidays")
def gold_dim_date():
    holidays = spark.read.option("header", True).csv(f"{LANDING}/reference/uk_holidays.csv")
    return gold.build_dim_date_for(spark.read.table(fq("silver", "transactions")), holidays)


@dlt.table(name=fq("gold", "dim_country"))
def gold_dim_country():
    return gold.build_dim_country(spark.read.table(fq("silver", "transactions")))


@dlt.table(name=fq("gold", "dim_customer"))
def gold_dim_customer():
    return gold.build_dim_customer(spark.read.table(fq("silver", "transactions")))


@dlt.table(name=fq("gold", "fact_sales"), comment="One row per clean invoice line")
@dlt.expect("has_product_version", "product_sk IS NOT NULL")
def gold_fact_sales():
    return gold.build_fact_sales(spark.read.table(fq("silver", "transactions")),
                                 spark.read.table(fq("gold", "dim_product")))


@dlt.table(name=fq("gold", "customer_rfm"), comment="RFM scores and segments as of the last sale date")
def gold_customer_rfm():
    return rfm.build_customer_rfm(spark.read.table(fq("gold", "fact_sales")))  # as of the last sale date


@dlt.table(name=fq("gold", "weekly_product_demand"),
           comment="Top 50 products x top 5 countries x full weeks, zero-filled")
def gold_weekly_demand():
    return weekly_demand(spark.read.table(fq("gold", "fact_sales")), spark.read.table(fq("gold", "dim_country")),
                         spark.read.table(fq("gold", "dim_date")))
