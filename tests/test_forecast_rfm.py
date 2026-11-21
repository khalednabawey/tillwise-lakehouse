from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import pytest
from pyspark.sql import functions as F

from tillwise.ml.dataset import FEATURES, future_frame, training_frame
from tillwise.ml.features import full_weeks, weekly_demand
from tillwise.ml.model import add_baselines, backtest, wape_and_bias
from tillwise.transforms import gold, rfm

FACT_SCHEMA = ("invoice_no string, date_key int, invoice_ts timestamp_ntz, product_sk long, stock_code string, "
               "customer_id string, country_key long, quantity int, unit_price decimal(12,2), "
               "line_revenue decimal(14,2), is_return boolean")


def fact_rows(spark, rows):
    """rows: (invoice, ts, stock, country, qty, price, customer, is_return)"""
    out = []
    for inv, ts, stock, country, qty, price, cust, ret in rows:
        out.append((inv, int(ts.strftime("%Y%m%d")), ts, 1, stock, cust, hash(country) % 1000, qty,
                    price, round(qty * price, 2), ret))
    df = spark.createDataFrame(out, FACT_SCHEMA.replace("unit_price decimal(12,2)", "unit_price double")
                               .replace("line_revenue decimal(14,2)", "line_revenue double"))
    countries = spark.createDataFrame([(hash(c) % 1000, c) for c in {r[3] for r in rows}],
                                      "country_key long, country string")
    return df, countries


def test_full_weeks_drop_partial_first_and_last_week(spark):
    # Tue 2009-12-01 .. Fri 2009-12-25 -> full weeks start Mon 7th, 14th (21st week ends Sun 27th > Fri 25th)
    df = spark.createDataFrame([(datetime(2009, 12, 1, 8),), (datetime(2009, 12, 25, 18),)],
                               "invoice_ts timestamp_ntz")
    weeks = [r["week_start"] for r in full_weeks(df).orderBy("week_start").collect()]
    assert weeks == [date(2009, 12, 7), date(2009, 12, 14)]


def test_weekly_demand_zero_fills_and_separates_returns(spark):
    mon = datetime(2010, 1, 4, 10)  # Monday
    rows = [
        ("1", mon - timedelta(days=1), "A", "UK", 1, 1.0, "c1", False),            # partial week, dropped
        ("2", mon, "A", "UK", 10, 2.0, "c1", False),
        ("C3", mon + timedelta(days=2), "A", "UK", -3, 2.0, "c1", True),          # return in week 1
        ("4", mon + timedelta(days=14), "A", "UK", 5, 2.5, "c1", False),          # week 3
        ("5", mon + timedelta(days=27), "A", "UK", 1, 2.5, "c1", False),          # Sun of week 4 (end)
    ]
    fact, countries = fact_rows(spark, rows)
    dim_date = gold.build_dim_date(spark, date(2010, 1, 1), date(2010, 2, 28))
    weekly = weekly_demand(fact, countries, dim_date, n_products=5, n_countries=5).orderBy("week_idx")
    got = [(r["week_start"], r["units"], r["return_units"], r["avg_price"]) for r in weekly.collect()]
    assert got == [
        (date(2010, 1, 4), 10.0, 3.0, 2.0),
        (date(2010, 1, 11), 0.0, 0.0, 2.0),   # zero week, price carried forward
        (date(2010, 1, 18), 5.0, 0.0, 2.5),
        (date(2010, 1, 25), 1.0, 0.0, 2.5),
    ]


@pytest.fixture
def synthetic_weekly(spark):
    """Two series, 80 weeks, with yearly seasonality, so lag-52 features exist."""
    rows = []
    start = date(2010, 1, 4)
    for code, base in (("A", 20.0), ("B", 5.0)):
        for i in range(80):
            wk = start + timedelta(weeks=i)
            units = base + 10 * math.sin(2 * math.pi * i / 52) + (i % 3)
            rows.append((code, "UK", wk, i, int(wk.strftime("%V")), 0, max(units, 0.0), 0.0, 2.0))
    return spark.createDataFrame(rows, "stock_code string, country string, week_start date, week_idx int, "
                                       "week_of_year int, holiday_in_week int, units double, "
                                       "return_units double, avg_price double")


def test_training_frame_has_no_leakage(synthetic_weekly):
    frame = training_frame(synthetic_weekly, horizons=4)
    assert set(FEATURES) - {"stock_idx", "country_idx"} <= set(frame.columns)
    bad = frame.where((F.col("target_idx") != F.col("origin_idx") + F.col("h")) | (F.col("h") < 1)
                      | (F.col("h") > 4))
    assert bad.count() == 0
    # last_units is the value AT the origin, never the label
    row = frame.where("stock_code = 'A' AND origin_idx = 60 AND h = 1").first()
    w60 = synthetic_weekly.where("stock_code = 'A' AND week_idx = 60").first()["units"]
    w61 = synthetic_weekly.where("stock_code = 'A' AND week_idx = 61").first()["units"]
    assert row["last_units"] == w60 and row["label"] == w61
    # lag-52 feature for the target week 61 is week 9
    w9 = synthetic_weekly.where("stock_code = 'A' AND week_idx = 9").first()["units"]
    assert row["units_same_week_last_year"] == w9 and row["has_last_year"] == 1


def test_wape_and_bias(spark):
    df = spark.createDataFrame([(10.0, 12.0), (10.0, 6.0)], "label double, p double")
    wape, bias = wape_and_bias(df, "p")
    assert wape == pytest.approx(6 / 20) and bias == pytest.approx(-2 / 20)


def test_backtest_and_future_frame(spark, synthetic_weekly):
    frame = training_frame(synthetic_weekly, horizons=4)
    result = backtest(frame, test_weeks=4, params={"maxIter": 20, "maxDepth": 3})
    assert result.cutoff_idx == 75
    assert result.predictions.count() == 2 * 4  # 2 series x 4 horizons from one origin
    assert set(result.metrics) == {"gbt", "seasonal_naive", "ma4", "last_value"}
    assert 0 <= result.metrics["gbt"]["wape"] < 0.5  # sane on smooth data; real accuracy is in the README
    assert set(result.residual_quantiles) == {1, 2, 3, 4}

    dim_date = gold.build_dim_date(spark, date(2010, 1, 1), date(2012, 12, 31))
    fut = future_frame(synthetic_weekly, dim_date, horizons=3)
    assert fut.count() == 2 * 3
    assert fut.agg(F.min("target_idx")).first()[0] == 80
    preds = add_baselines(fut)
    assert "pred_seasonal_naive" in preds.columns


def test_rfm_segments(spark):
    as_of = date(2011, 12, 9)
    recent = datetime(2011, 12, 1)
    rows = []
    # 10 customers: c0 buys most often, most recently and spends most; c9 the opposite
    for i in range(10):
        n_orders = 10 - i
        ts = recent - timedelta(days=60 * i)
        for k in range(n_orders):
            rows.append((f"{i}-{k}", ts, "A", "UK", 1, 10.0 * (10 - i), f"c{i}", False))
    rows.append(("C99", recent, "A", "UK", -1, 5.0, "c_returns_only", True))
    fact, _ = fact_rows(spark, rows)
    out = {r["customer_id"]: r for r in rfm.build_customer_rfm(fact, as_of).collect()}
    assert "c_returns_only" not in out  # nothing bought, nothing to score
    assert out["c0"]["segment"] == "Champions"
    assert out["c9"]["segment"] == "Hibernating"
    assert out["c0"]["recency_days"] == 8
    assert {r["segment"] for r in out.values()} <= {name for name, _ in rfm.SEGMENT_RULES}


def test_rfm_default_as_of_is_last_sale_date(spark):
    rows = [("1", datetime(2011, 12, 1), "A", "UK", 1, 10.0, "c1", False),
            ("2", datetime(2011, 12, 9, 12), "A", "UK", 1, 10.0, "c2", False)]
    fact, _ = fact_rows(spark, rows)
    out = {r["customer_id"]: r["recency_days"] for r in rfm.build_customer_rfm(fact).collect()}
    assert out == {"c1": 8, "c2": 0}
