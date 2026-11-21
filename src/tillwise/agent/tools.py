"""The agent's fixed tools. Each one validates its inputs and runs one parameterised query on gold.

There is no free-form SQL: the model can only pick a tool and fill in its arguments. Definitions
used everywhere:
  net_revenue    sales minus returns (GBP)
  gross_revenue  sales only
  returns_value  value of returned goods (positive number)
  units_sold     units on sales invoices (returns not subtracted)
  orders         distinct sales invoices
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from tillwise.agent.store import GoldStore

DATA_START = date(2009, 12, 1)
DATA_END = date(2011, 12, 9)
MAX_ROWS = 25


class ToolError(ValueError):
    """Bad arguments. The message goes back to the model so it can correct itself."""


def _clean(value: Any) -> Any:
    if isinstance(value, Decimal):
        return round(float(value), 2)
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return value


def _rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: _clean(v) for k, v in r.items()} for r in rows]


def _parse_date(value: str | None, default: date, name: str) -> date:
    if value in (None, ""):
        return default
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ToolError(f"{name} must be YYYY-MM-DD, got {value!r}") from exc


def _period(start_date: str | None, end_date: str | None) -> tuple[date, date, dict[str, Any] | None]:
    start = _parse_date(start_date, DATA_START, "start_date")
    end = _parse_date(end_date, DATA_END, "end_date")
    if end < start:
        raise ToolError("end_date is before start_date")
    if end < DATA_START or start > DATA_END:
        return start, end, {"no_data": True,
                            "message": f"No data for {start} to {end}. Data covers {DATA_START} to {DATA_END}."}
    return max(start, DATA_START), min(end, DATA_END), None


def _limit(n: int) -> int:
    try:
        n = int(n)
    except (TypeError, ValueError) as exc:
        raise ToolError("n must be an integer") from exc
    if not 1 <= n <= MAX_ROWS:
        raise ToolError(f"n must be between 1 and {MAX_ROWS}")
    return n


class Toolbox:
    def __init__(self, store: GoldStore):
        self.store = store
        self._countries: list[str] | None = None

    # ------------------------------------------------------------- helpers
    def _q(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        return _rows(self.store.query(sql, params))

    def countries(self) -> list[str]:
        if self._countries is None:
            self._countries = [r["country"] for r in self._q("SELECT country FROM gold.dim_country ORDER BY country")]
        return self._countries

    def _country(self, country: str | None) -> str | None:
        if not country:
            return None
        aliases = {"uk": "United Kingdom", "england": "United Kingdom", "britain": "United Kingdom",
                   "ireland": "EIRE", "usa": "USA", "united states": "USA", "holland": "Netherlands"}
        wanted = aliases.get(country.strip().lower(), country.strip())
        for c in self.countries():
            if c.lower() == wanted.lower():
                return c
        raise ToolError(f"Unknown country {country!r}. Valid countries: {', '.join(self.countries())}")

    # ------------------------------------------------------------- tools
    def data_coverage(self) -> dict[str, Any]:
        """What the data covers: date range, countries, product and customer counts."""
        r = self._q("""
            SELECT min(sale_date) AS first_day, max(sale_date) AS last_day,
                   count(DISTINCT stock_code) AS products, count(DISTINCT customer_id) AS customers,
                   count(DISTINCT CASE WHEN NOT is_return THEN invoice_no END) AS orders
            FROM gold.v_sales""")[0]
        return {**r, "countries": self.countries(), "currency": "GBP",
                "source": "UK online gift retailer (UCI Online Retail II)"}

    def sales_summary(self, start_date: str | None = None, end_date: str | None = None,
                      country: str | None = None, stock_code: str | None = None) -> dict[str, Any]:
        start, end, empty = _period(start_date, end_date)
        if empty:
            return empty
        c = self._country(country)
        code = stock_code.strip().upper() if stock_code else None
        r = self._q("""
            SELECT round(sum(line_revenue), 2) AS net_revenue,
                   round(sum(CASE WHEN NOT is_return THEN line_revenue ELSE 0 END), 2) AS gross_revenue,
                   round(-sum(CASE WHEN is_return THEN line_revenue ELSE 0 END), 2) AS returns_value,
                   sum(CASE WHEN NOT is_return THEN quantity ELSE 0 END) AS units_sold,
                   count(DISTINCT CASE WHEN NOT is_return THEN invoice_no END) AS orders,
                   count(DISTINCT CASE WHEN NOT is_return THEN customer_id END) AS customers
            FROM gold.v_sales
            WHERE sale_date BETWEEN CAST(:start AS DATE) AND CAST(:end AS DATE)
              AND (:country IS NULL OR country = :country)
              AND (:code IS NULL OR stock_code = :code)""",
                    start=start.isoformat(), end=end.isoformat(), country=c, code=code)[0]
        if r["orders"] in (0, None) and r["net_revenue"] in (0, None):
            return {"no_data": True, "message": "No sales match these filters.",
                    "filters": {"start": str(start), "end": str(end), "country": c, "stock_code": code}}
        return {"period": {"start": str(start), "end": str(end)}, "country": c or "all", "stock_code": code or "all",
                **r}

    def top_products(self, start_date: str | None = None, end_date: str | None = None,
                     metric: Literal["revenue", "units"] = "revenue", n: int = 10,
                     country: str | None = None) -> dict[str, Any]:
        start, end, empty = _period(start_date, end_date)
        if empty:
            return empty
        if metric not in ("revenue", "units"):
            raise ToolError("metric must be 'revenue' or 'units'")
        n, c = _limit(n), self._country(country)
        order = "net_revenue" if metric == "revenue" else "units_sold"
        rows = self._q(f"""
            SELECT stock_code, any_value(description) AS description,
                   round(sum(line_revenue), 2) AS net_revenue,
                   sum(CASE WHEN NOT is_return THEN quantity ELSE 0 END) AS units_sold
            FROM gold.v_sales
            WHERE sale_date BETWEEN CAST(:start AS DATE) AND CAST(:end AS DATE)
              AND (:country IS NULL OR country = :country)
            GROUP BY stock_code
            ORDER BY {order} DESC, stock_code
            LIMIT {n}""", start=start.isoformat(), end=end.isoformat(), country=c)
        return {"period": {"start": str(start), "end": str(end)}, "country": c or "all", "ranked_by": order,
                "products": rows}

    def country_breakdown(self, start_date: str | None = None, end_date: str | None = None,
                          n: int = 10) -> dict[str, Any]:
        start, end, empty = _period(start_date, end_date)
        if empty:
            return empty
        n = _limit(n)
        rows = self._q(f"""
            SELECT country, round(sum(line_revenue), 2) AS net_revenue,
                   count(DISTINCT CASE WHEN NOT is_return THEN invoice_no END) AS orders,
                   count(DISTINCT CASE WHEN NOT is_return THEN customer_id END) AS customers
            FROM gold.v_sales
            WHERE sale_date BETWEEN CAST(:start AS DATE) AND CAST(:end AS DATE)
            GROUP BY country ORDER BY net_revenue DESC, country LIMIT {n}""",
                       start=start.isoformat(), end=end.isoformat())
        return {"period": {"start": str(start), "end": str(end)}, "countries": rows}

    def monthly_trend(self, start_date: str | None = None, end_date: str | None = None,
                      country: str | None = None) -> dict[str, Any]:
        start, end, empty = _period(start_date, end_date)
        if empty:
            return empty
        c = self._country(country)
        rows = self._q("""
            SELECT CAST(year(sale_date) AS STRING) || '-' || lpad(CAST(month(sale_date) AS STRING), 2, '0') AS month,
                   round(sum(line_revenue), 2) AS net_revenue,
                   count(DISTINCT CASE WHEN NOT is_return THEN invoice_no END) AS orders
            FROM gold.v_sales
            WHERE sale_date BETWEEN CAST(:start AS DATE) AND CAST(:end AS DATE)
              AND (:country IS NULL OR country = :country)
            GROUP BY 1 ORDER BY 1""", start=start.isoformat(), end=end.isoformat(), country=c)
        return {"period": {"start": str(start), "end": str(end)}, "country": c or "all", "months": rows,
                "note": "Months at the edges of the period may be partial."}

    def returns_report(self, start_date: str | None = None, end_date: str | None = None,
                       by: Literal["country", "product"] = "country", n: int = 10) -> dict[str, Any]:
        start, end, empty = _period(start_date, end_date)
        if empty:
            return empty
        if by not in ("country", "product"):
            raise ToolError("by must be 'country' or 'product'")
        n = _limit(n)
        key = "country" if by == "country" else "stock_code, any_value(description) AS description"
        group = "country" if by == "country" else "stock_code"
        rows = self._q(f"""
            SELECT {key},
                   round(-sum(CASE WHEN is_return THEN line_revenue ELSE 0 END), 2) AS returns_value,
                   round(sum(CASE WHEN NOT is_return THEN line_revenue ELSE 0 END), 2) AS gross_revenue,
                   round(100.0 * -sum(CASE WHEN is_return THEN line_revenue ELSE 0 END)
                         / nullif(sum(CASE WHEN NOT is_return THEN line_revenue ELSE 0 END), 0), 2)
                     AS returns_pct_of_sales
            FROM gold.v_sales
            WHERE sale_date BETWEEN CAST(:start AS DATE) AND CAST(:end AS DATE)
            GROUP BY {group} ORDER BY returns_value DESC LIMIT {n}""",
                       start=start.isoformat(), end=end.isoformat())
        return {"period": {"start": str(start), "end": str(end)}, "by": by, "rows": rows}

    def customer_segments(self) -> dict[str, Any]:
        rows = self._q("""
            SELECT segment, count(*) AS customers, round(sum(monetary), 2) AS net_revenue,
                   round(avg(recency_days), 1) AS avg_days_since_last_purchase,
                   round(avg(frequency), 1) AS avg_orders
            FROM gold.customer_rfm GROUP BY segment ORDER BY net_revenue DESC""")
        total = sum(r["net_revenue"] for r in rows) or 1
        for r in rows:
            r["revenue_share_pct"] = round(100 * r["net_revenue"] / total, 1)
        return {"as_of": str(DATA_END), "segments": rows,
                "method": "RFM quintile scores (1-5) on recency, frequency and net spend; rule-based segments"}

    def find_products(self, text: str, n: int = 10) -> dict[str, Any]:
        if not text or len(text.strip()) < 2:
            raise ToolError("text must be at least 2 characters")
        n = _limit(n)
        # Search every SCD2 version, so products are found by names they had in the past too
        # (e.g. 85123A was "WHITE HANGING HEART..." before it became "CREAM HANGING HEART...").
        rows = self._q(f"""
            WITH hits AS (
              SELECT stock_code, min(description) AS matched_name FROM gold.dim_product
              WHERE description ILIKE :pat OR stock_code ILIKE :pat GROUP BY stock_code)
            SELECT p.stock_code, p.description AS current_description, h.matched_name,
                   round(coalesce(sum(f.line_revenue), 0), 2) AS net_revenue
            FROM hits h
            JOIN gold.dim_product p ON p.stock_code = h.stock_code AND p.is_current
            LEFT JOIN gold.fact_sales f ON f.stock_code = p.stock_code
            GROUP BY p.stock_code, p.description, h.matched_name ORDER BY net_revenue DESC LIMIT {n}""",
                       pat=f"%{text.strip()}%")
        return {"query": text, "matches": rows}

    def demand_forecast(self, stock_code: str, country: str = "United Kingdom", weeks: int = 8) -> dict[str, Any]:
        code, c = stock_code.strip().upper(), self._country(country)
        if not 1 <= int(weeks) <= 8:
            raise ToolError("weeks must be between 1 and 8")
        rows = self._q("""
            SELECT target_week, h AS weeks_ahead, forecast_units, p10, p90
            FROM gold.forecast_weekly
            WHERE stock_code = :code AND country = :country AND h <= :weeks
            ORDER BY h""", code=code, country=c, weeks=int(weeks))
        if not rows:
            covered = self._q("SELECT DISTINCT country FROM gold.forecast_weekly ORDER BY country")
            return {"no_data": True,
                    "message": "Forecasts exist only for the top 50 products in the top 5 countries "
                               f"({', '.join(r['country'] for r in covered)}). {code} in {c} is not one of them."}
        return {"stock_code": code, "country": c, "origin_week": "2011-11-28", "unit": "units sold per week",
                "forecast": rows,
                "method": "Gradient-boosted trees; p10/p90 from backtest errors. Backtest WAPE 0.669."}


TOOL_NAMES = ["data_coverage", "sales_summary", "top_products", "country_breakdown", "monthly_trend",
              "returns_report", "customer_segments", "find_products", "demand_forecast"]
