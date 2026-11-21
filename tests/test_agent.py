"""Agent tests: tools on a tiny gold layer (DuckDB), grounding, the graph with a fake model, and the API.

No API keys, no network, no Spark: these run in CI on every push.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from itertools import count

import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from tillwise.agent.graph import ask, build_graph
from tillwise.agent.grounding import extract_numbers, ungrounded
from tillwise.agent.store import DuckDBStore
from tillwise.agent.tools import Toolbox, ToolError

UK, FR = 1, 2


@pytest.fixture(scope="module")
def gold_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("gold")
    fact = pd.DataFrame([
        # invoice, ts, stock, customer, country, qty, price, return
        ("1", datetime(2010, 3, 1, 10), "85123A", "100", UK, 10, 2.5, False),
        ("2", datetime(2010, 3, 2, 10), "22423", "101", UK, 2, 10.0, False),
        ("C3", datetime(2010, 3, 3, 10), "22423", "101", UK, -1, 10.0, True),
        ("4", datetime(2011, 5, 1, 10), "85123A", "200", FR, 4, 2.5, False),
        ("5", datetime(2011, 11, 20, 10), "85123A", None, UK, 6, 2.5, False),
    ], columns=["invoice_no", "invoice_ts", "stock_code", "customer_id", "country_key", "quantity", "unit_price",
                "is_return"])
    fact["line_revenue"] = fact["quantity"] * fact["unit_price"]
    fact["date_key"] = fact["invoice_ts"].dt.strftime("%Y%m%d").astype(int)
    fact["product_sk"] = 1
    fact.to_parquet(d / "fact_sales.parquet")
    pd.DataFrame({"country_key": [UK, FR], "country": ["United Kingdom", "France"]}).to_parquet(
        d / "dim_country.parquet")
    pd.DataFrame([(1, "85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 1, datetime(2009, 12, 1), None, True),
                  (2, "22423", "REGENCY CAKESTAND 3 TIER", 1, datetime(2009, 12, 1), None, True)],
                 columns=["product_sk", "stock_code", "description", "version", "valid_from", "valid_to",
                          "is_current"]).to_parquet(d / "dim_product.parquet")
    pd.DataFrame({"customer_id": ["100"]}).to_parquet(d / "dim_customer.parquet")
    pd.DataFrame({"date_key": [20100301], "date": [date(2010, 3, 1)]}).to_parquet(d / "dim_date.parquet")
    pd.DataFrame([("100", 10, 5, 1000.0, "Champions"), ("101", 300, 1, 10.0, "Hibernating")],
                 columns=["customer_id", "recency_days", "frequency", "monetary", "segment"]).to_parquet(
        d / "customer_rfm.parquet")
    pd.DataFrame({"stock_code": ["85123A"]}).to_parquet(d / "weekly_product_demand.parquet")
    pd.DataFrame([("85123A", "United Kingdom", date(2011, 11, 28), date(2011, 12, 5), 1, 120.0, 80.0, 200.0),
                  ("85123A", "United Kingdom", date(2011, 11, 28), date(2011, 12, 12), 2, 90.0, 50.0, 150.0)],
                 columns=["stock_code", "country", "origin_week", "target_week", "h", "forecast_units", "p10",
                          "p90"]).to_parquet(d / "forecast_weekly.parquet")
    pd.DataFrame({"stock_code": ["85123A"]}).to_parquet(d / "forecast_backtest.parquet")
    return d


@pytest.fixture(scope="module")
def box(gold_dir):
    return Toolbox(DuckDBStore(gold_dir))


# ---------------------------------------------------------------- store and tools
def test_store_is_read_only(box):
    with pytest.raises(ValueError):
        box.store.query("DELETE FROM gold.fact_sales")
    with pytest.raises(ValueError):
        box.store.query("SELECT 1; DROP VIEW gold.fact_sales")


def test_sales_summary_definitions(box):
    s = box.sales_summary("2010-01-01", "2010-12-31")
    assert s["net_revenue"] == 35.0          # 25 + 20 - 10
    assert s["gross_revenue"] == 45.0
    assert s["returns_value"] == 10.0
    assert s["units_sold"] == 12             # returns not subtracted
    assert s["orders"] == 2 and s["customers"] == 2


def test_filters_aliases_and_validation(box):
    assert box.sales_summary(country="uk")["country"] == "United Kingdom"
    assert box.sales_summary(country="France")["net_revenue"] == 10.0
    with pytest.raises(ToolError, match="Unknown country"):
        box.sales_summary(country="Atlantis")
    with pytest.raises(ToolError, match="YYYY-MM-DD"):
        box.sales_summary(start_date="March 2010")
    assert box.sales_summary("2015-01-01", "2015-12-31")["no_data"] is True
    with pytest.raises(ToolError):
        box.top_products(n=500)


def test_top_products_and_lookup(box):
    top = box.top_products(metric="units")
    assert top["products"][0]["stock_code"] == "85123A" and top["products"][0]["units_sold"] == 20
    found = box.find_products("heart")
    assert found["matches"][0]["stock_code"] == "85123A"


def test_returns_trend_segments_forecast(box):
    r = box.returns_report(by="product")["rows"][0]
    assert r["stock_code"] == "22423" and r["returns_value"] == 10.0 and r["returns_pct_of_sales"] == 50.0
    months = {m["month"]: m["net_revenue"] for m in box.monthly_trend()["months"]}
    assert months["2010-03"] == 35.0 and months["2011-11"] == 15.0
    seg = {s["segment"]: s for s in box.customer_segments()["segments"]}
    assert seg["Champions"]["revenue_share_pct"] == 99.0
    fc = box.demand_forecast("85123a", "UK", weeks=1)
    assert fc["forecast"] == [{"target_week": "2011-12-05", "weeks_ahead": 1, "forecast_units": 120.0,
                               "p10": 80.0, "p90": 200.0}]
    assert box.demand_forecast("22423", "France")["no_data"] is True


# ---------------------------------------------------------------- grounding
def test_extract_numbers_with_suffixes():
    got = [v for v, _ in extract_numbers("£9.14M, 1,234.50 units, 3 million, 12.5% and 2.5k")]
    assert got == [9_140_000.0, 1234.5, 3_000_000.0, 12.5, 2500.0]


def test_ungrounded_allows_rounding_sums_and_small_numbers():
    tools = [json.dumps({"a": 9136724.93, "b": 9013836.88})]
    assert ungrounded("Revenue was £9.14M in 2010 and £9,013,836.88 in 2011.", tools) == []
    assert ungrounded("Together that is £18,150,561.81 over 2 years.", tools) == []   # a sum is fine
    assert ungrounded("Top 3 products, 12% growth.", tools) == []                      # small/pct skipped
    assert ungrounded("Revenue was £7,500,000.", tools) == [7_500_000.0]


# ---------------------------------------------------------------- graph and API
def scripted_model(replies):
    it = iter(replies)
    return RunnableLambda(lambda _msgs: next(it))


def test_graph_calls_tool_then_answers(box):
    ids = count()
    model = scripted_model([
        AIMessage("", tool_calls=[{"name": "sales_summary", "id": f"c{next(ids)}",
                                   "args": {"start_date": "2010-01-01", "end_date": "2010-12-31"}}]),
        AIMessage("Net revenue in 2010 was £35.00 from 2 orders."),
    ])
    out = ask(build_graph(box, model=model), "net revenue 2010?")
    assert out["answer"].startswith("Net revenue in 2010")
    assert out["tool_calls"] == [{"tool": "sales_summary", "args": {"start_date": "2010-01-01",
                                                                     "end_date": "2010-12-31"}, "ok": True}]


def test_graph_returns_tool_errors_to_the_model(box):
    model = scripted_model([
        AIMessage("", tool_calls=[{"name": "sales_summary", "id": "c1", "args": {"country": "Atlantis"}}]),
        AIMessage("There is no country called Atlantis in the data."),
    ])
    out = ask(build_graph(box, model=model), "sales in Atlantis")
    assert out["tool_calls"][0]["ok"] is False
    assert "Atlantis" in out["answer"]


def test_graph_retries_once_when_answer_has_made_up_numbers(box):
    model = scripted_model([
        AIMessage("", tool_calls=[{"name": "sales_summary", "id": "c1", "args": {}}]),
        AIMessage("Net revenue was £123,456.00."),        # not in any tool result
        AIMessage("Net revenue was £60.00."),             # grounded: 25+20-10+10+15
    ])
    out = ask(build_graph(box, model=model), "total net revenue?")
    assert out["answer"] == "Net revenue was £60.00."
    assert out["ungrounded_numbers"] == []


def test_api_ask_and_health(box):
    from fastapi.testclient import TestClient

    from tillwise.agent import app as app_module

    model = scripted_model([AIMessage("", tool_calls=[{"name": "customer_segments", "id": "c1", "args": {}}]),
                            AIMessage("Champions bring 99.0% of revenue.")])
    app_module.app.dependency_overrides[app_module.get_toolbox] = lambda: box
    app_module.app.dependency_overrides[app_module.get_graph] = lambda: build_graph(box, model=model)
    try:
        client = TestClient(app_module.app)
        assert client.get("/health").json()["data"] == {"first_day": "2010-03-01", "last_day": "2011-11-20"}
        r = client.post("/ask", json={"question": "Who are our best customers?"})
        assert r.status_code == 200
        assert r.json()["answer"] == "Champions bring 99.0% of revenue."
        assert r.json()["tool_calls"][0]["tool"] == "customer_segments"
        assert client.post("/ask", json={"question": "x"}).status_code == 422
    finally:
        app_module.app.dependency_overrides.clear()
