from __future__ import annotations

from decimal import Decimal

import pytest
from pyspark.sql import functions as F

from tillwise.transforms import silver
from tillwise.transforms.bronze import standardize_raw

ROWS = [
    # invoice, stock, description, qty, ts, price, customer, country
    ("489434", "85048", "15CM CHRISTMAS GLASS BALL 20 LIGHTS", "12", "2009-12-01 07:45:00", "6.95", "13085.0",
     "United Kingdom"),  # clean sale
    ("489434", "85048", "15CM CHRISTMAS GLASS BALL 20 LIGHTS", "12", "2009-12-01 07:45:00", "6.95", "13085.0",
     "United Kingdom"),  # exact duplicate
    ("C489449", "22087", "PAPER BUNTING WHITE LACE", "-12", "2009-12-01 10:33:00", "2.95", "16321",
     "Australia"),  # clean return
    ("489436", "21523", "DOORMAT  FANCY FONT HOME SWEET HOME", "6", "2009-12-01 09:06:00", "5.95", None,
     "United Kingdom"),  # clean, no customer, double space in description
    ("489437", "POST", "POSTAGE", "1", "2009-12-01 09:08:00", "18.00", "12682", "France"),  # non-product
    ("A506401", "B", "Adjust bad debt", "1", "2010-04-29 13:36:00", "-53594.36", None,
     "United Kingdom"),  # adjustment
    ("489440", "22355", None, "-20", "2009-12-01 10:00:00", "0.00", None, "United Kingdom"),  # price 0
    ("489441", "22356", "CHARLOTTE BAG PINK", "-3", "2009-12-01 10:00:00", "0.85", "13085",
     "United Kingdom"),  # negative qty, not a return
    ("C489442", "22357", "PINK BAG", "3", "2009-12-01 10:00:00", "0.85", "13085",
     "United Kingdom"),  # return with positive qty
    ("489443", "22358", "BLUE BAG", "2", "not a date", "0.85", "13085", "United Kingdom"),  # bad timestamp
]


@pytest.fixture
def typed(raw_df):
    bronze = standardize_raw(raw_df(ROWS), source_file=F.lit("test.csv"))
    return silver.type_transactions(bronze)


def test_bronze_normalises_names_and_keeps_strings(raw_df):
    bronze = standardize_raw(raw_df(ROWS[:1]), source_file=F.lit("test.csv"))
    assert bronze.columns == [
        "invoice_no", "stock_code", "description", "quantity", "invoice_ts", "unit_price",
        "customer_id", "country", "_source_file", "_ingest_ts",
    ]
    assert {f.dataType.simpleString() for f in bronze.schema.fields[:8]} == {"string"}
    assert bronze.first()["_source_file"] == "test.csv"


def test_bronze_rejects_unexpected_source(spark):
    with pytest.raises(ValueError, match="missing columns"):
        standardize_raw(spark.createDataFrame([("x",)], "Invoice string"), source_file=F.lit("x"))


def test_typing_is_ansi_safe_and_cleans_values(typed):
    rows = {r["invoice_no"]: r for r in typed.collect()}
    assert rows["489434"]["customer_id"] == "13085"  # float-looking ID fixed
    assert rows["489434"]["unit_price"] == Decimal("6.95")
    assert rows["489436"]["description"] == "DOORMAT FANCY FONT HOME SWEET HOME"
    assert rows["489443"]["invoice_ts"] is None  # bad date becomes null, no exception
    assert rows["C489449"]["is_return"] is True
    assert rows["489434"]["is_return"] is False


def test_every_row_lands_in_exactly_one_place(typed):
    result = silver.split_transactions(typed)
    total = (result.clean.count() + result.quarantine.count() + result.non_product.count()
             + result.duplicates_removed.count())
    assert total == typed.count() == len(ROWS)
    assert result.duplicates_removed.count() == 1


def test_quarantine_reasons(typed):
    result = silver.split_transactions(typed)
    reasons = {r["invoice_no"]: r["dq_reason"] for r in result.quarantine.collect()}
    assert reasons == {
        "A506401": "adjustment_invoice",
        "489440": "non_positive_price",
        "489441": "non_positive_quantity_not_return",
        "C489442": "return_with_positive_quantity",
        "489443": "unparseable_timestamp",
    }


def test_clean_and_non_product_split(typed):
    result = silver.split_transactions(typed)
    assert sorted(r["invoice_no"] for r in result.clean.collect()) == ["489434", "489436", "C489449"]
    assert [r["stock_code"] for r in result.non_product.collect()] == ["POST"]
    ret = result.clean.where("invoice_no = 'C489449'").first()
    assert ret["line_revenue"] == Decimal("-35.40")  # returns reduce revenue


def test_dq_metrics_add_up(typed):
    result = silver.split_transactions(typed)
    m = {r["metric"]: r["row_count"] for r in silver.dq_metrics(typed, result).collect()}
    accounted = (m["duplicates_removed"] + m["non_product_lines"] + m["clean_rows"]
                 + sum(v for k, v in m.items() if k.startswith("quarantine:")))
    assert accounted == m["input_rows"] == len(ROWS)
