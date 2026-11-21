from __future__ import annotations

from datetime import date, datetime

from chispa import assert_df_equality
from pyspark.sql import functions as F

from tillwise.transforms import gold, scd2

CLEAN_SCHEMA = ("invoice_no string, stock_code string, description string, quantity int, "
                "invoice_ts timestamp_ntz, customer_id string, country string, is_return boolean, "
                "line_revenue decimal(14,2)")


def clean_df(spark, rows):
    return spark.createDataFrame(rows, CLEAN_SCHEMA)


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


def test_scd2_opens_new_version_when_daily_description_changes(spark):
    rows = [
        # day 1: typo wins 2 vs 1 -> "LUNCH BAG RED SPOTY"
        ("1", "20725", "LUNCH BAG RED SPOTY", 1, ts("2010-01-01 09:00:00"), "1", "UK", False, None),
        ("2", "20725", "LUNCH BAG RED SPOTY", 1, ts("2010-01-01 10:00:00"), "1", "UK", False, None),
        ("3", "20725", "LUNCH BAG RED SPOTTY", 1, ts("2010-01-01 11:00:00"), "1", "UK", False, None),
        # day 2: same typo, no new version
        ("4", "20725", "LUNCH BAG RED SPOTY", 1, ts("2010-01-02 09:00:00"), "1", "UK", False, None),
        # day 5: fixed spelling -> version 2
        ("5", "20725", "LUNCH BAG RED SPOTTY", 1, ts("2010-01-05 09:00:00"), "1", "UK", False, None),
        # other product, single version
        ("6", "22423", "REGENCY CAKESTAND 3 TIER", 1, ts("2010-01-03 09:00:00"), "1", "UK", False, None),
    ]
    out = scd2.build_product_scd2(clean_df(spark, rows)).orderBy("stock_code", "version")
    got = [(r["stock_code"], r["version"], r["description"], r["valid_from"], r["valid_to"], r["is_current"])
           for r in out.collect()]
    assert got == [
        ("20725", 1, "LUNCH BAG RED SPOTY", ts("2010-01-01 00:00:00"), ts("2010-01-05 00:00:00"), False),
        ("20725", 2, "LUNCH BAG RED SPOTTY", ts("2010-01-05 00:00:00"), None, True),
        ("22423", 1, "REGENCY CAKESTAND 3 TIER", ts("2010-01-03 00:00:00"), None, True),
    ]
    assert out.select("product_sk").distinct().count() == 3  # surrogate keys are unique


def test_scd2_handles_a_b_a_flip(spark):
    rows = [
        ("1", "X1", "A", 1, ts("2010-01-01 09:00:00"), "1", "UK", False, None),
        ("2", "X1", "B", 1, ts("2010-01-02 09:00:00"), "1", "UK", False, None),
        ("3", "X1", "A", 1, ts("2010-01-03 09:00:00"), "1", "UK", False, None),
    ]
    out = scd2.build_product_scd2(clean_df(spark, rows)).orderBy("version")
    assert [(r["version"], r["description"]) for r in out.collect()] == [(1, "A"), (2, "B"), (3, "A")]


def test_fact_joins_the_version_valid_at_sale_time(spark):
    rows = [
        ("1", "20725", "OLD NAME", 2, ts("2010-01-01 09:00:00"), "13085", "UK", False, None),
        ("2", "20725", "NEW NAME", 3, ts("2010-02-01 09:00:00"), "13085", "UK", False, None),
        ("3", "20725", None, 1, ts("2010-02-02 09:00:00"), None, "UK", False, None),  # no description
    ]
    clean = clean_df(spark, rows).withColumn("unit_price", F.lit(1).cast("decimal(12,2)"))
    products = scd2.build_product_scd2(clean)
    fact = gold.build_fact_sales(clean, products)
    sk = {r["description"]: r["product_sk"] for r in products.collect()}
    got = {r["invoice_no"]: r["product_sk"] for r in fact.collect()}
    assert got == {"1": sk["OLD NAME"], "2": sk["NEW NAME"], "3": sk["NEW NAME"]}
    assert fact.count() == 3  # the range join never duplicates lines
    assert fact.where("invoice_no = '1'").first()["date_key"] == 20100101


def test_dim_date_flags_weekends_and_holidays(spark):
    dim = gold.build_dim_date(spark, date(2010, 12, 24), date(2010, 12, 27),
                              holidays=[(date(2010, 12, 25), "Christmas Day")])
    got = [(r["date_key"], r["day_name"], r["is_weekend"], r["holiday_name"]) for r in dim.collect()]
    assert got == [
        (20101224, "Friday", False, None),
        (20101225, "Saturday", True, "Christmas Day"),
        (20101226, "Sunday", True, None),
        (20101227, "Monday", False, None),
    ]


def test_dim_customer_primary_country_and_order_counts(spark):
    rows = [
        ("1", "A", "x", 1, ts("2010-01-01 09:00:00"), "100", "UK", False, None),
        ("1", "B", "y", 1, ts("2010-01-01 09:00:00"), "100", "UK", False, None),
        ("2", "A", "x", 1, ts("2010-03-01 09:00:00"), "100", "France", False, None),
        ("C3", "A", "x", -1, ts("2010-03-02 09:00:00"), "100", "UK", True, None),
        ("4", "A", "x", 1, ts("2010-03-02 09:00:00"), None, "UK", False, None),  # unknown customer
    ]
    dim = gold.build_dim_customer(clean_df(spark, rows))
    expected = spark.createDataFrame(
        [("100", "UK", ts("2010-01-01 09:00:00"), ts("2010-03-02 09:00:00"), 2, 1)],
        "customer_id string, primary_country string, first_purchase_ts timestamp_ntz, "
        "last_purchase_ts timestamp_ntz, orders long, return_invoices long",
    )
    assert_df_equality(dim, expected, ignore_nullable=True)


def test_lazy_dim_date_for_matches_eager_version(spark):
    rows = [("1", "A", "x", 1, ts("2010-03-01 09:00:00"), "1", "UK", False, None),
            ("2", "A", "x", 1, ts("2011-02-01 09:00:00"), "1", "UK", False, None)]
    hol = spark.createDataFrame([("2010-12-25", "Christmas Day")], "date string, holiday_name string")
    lazy = gold.build_dim_date_for(clean_df(spark, rows), hol)
    eager = gold.build_dim_date(spark, date(2010, 1, 1), date(2012, 12, 31), [(date(2010, 12, 25), "Christmas Day")])
    assert_df_equality(lazy, eager, ignore_row_order=True, ignore_nullable=True)
