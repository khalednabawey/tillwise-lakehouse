"""Read-only access to the gold layer.

Two backends with the same interface:
  DuckDBStore      local Parquet export (data/gold/), for development, tests and evals
  DatabricksStore  Databricks SQL warehouse via databricks-sql-connector, with a read-only token

Tools write SQL once, with named parameters (`:name`), against the schema `gold`. Values are
always passed as parameters, never formatted into the SQL text, except integers that the tools
have already validated (LIMIT n).
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLD_DIR = ROOT / "data" / "gold"

GOLD_TABLES = ("fact_sales", "dim_product", "dim_country", "dim_customer", "dim_date", "customer_rfm",
               "weekly_product_demand", "forecast_weekly", "forecast_backtest")

# Agent-facing view: every sales line with its country name and the product's current description.
SALES_VIEW = """
CREATE OR REPLACE VIEW {schema}.v_sales AS
SELECT f.invoice_no, f.invoice_ts, CAST(f.invoice_ts AS DATE) AS sale_date, f.stock_code,
       p.description, c.country, f.customer_id, f.quantity, f.unit_price, f.line_revenue, f.is_return
FROM {schema}.fact_sales f
JOIN {schema}.dim_country c ON c.country_key = f.country_key
LEFT JOIN {schema}.dim_product p ON p.stock_code = f.stock_code AND p.is_current
"""


class GoldStore(Protocol):
    schema: str

    def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...


_READ_ONLY = re.compile(r"^\s*(WITH|SELECT)\b", re.IGNORECASE)


def _check_read_only(sql: str) -> None:
    if not _READ_ONLY.match(sql) or ";" in sql.strip().rstrip(";"):
        raise ValueError("Only single SELECT statements are allowed")


class DuckDBStore:
    """In-memory DuckDB with views over the Parquet files. Thread-safe for FastAPI (one cursor per call)."""

    schema = "gold"

    def __init__(self, gold_dir: Path | str = DEFAULT_GOLD_DIR):
        import duckdb

        gold_dir = Path(gold_dir)
        missing = [t for t in GOLD_TABLES if not (gold_dir / f"{t}.parquet").exists()]
        if missing:
            raise FileNotFoundError(f"Missing gold files in {gold_dir}: {missing}. "
                                    "Run: uv run python scripts/export_gold_local.py")
        self._con = duckdb.connect(":memory:")
        self._lock = threading.Lock()
        self._con.execute("CREATE SCHEMA gold")
        for t in GOLD_TABLES:
            path = (gold_dir / f"{t}.parquet").as_posix().replace("'", "''")
            self._con.execute(f"CREATE VIEW gold.{t} AS SELECT * FROM read_parquet('{path}')")
        self._con.execute(SALES_VIEW.format(schema="gold"))

    def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        _check_read_only(sql)
        duck_sql = re.sub(r"(?<!:):([a-zA-Z_]\w*)", r"$\1", sql)  # :name -> $name (skip ::casts)
        with self._lock:
            cur = self._con.cursor()
            try:
                rel = cur.execute(duck_sql, params or {})
                cols = [d[0] for d in rel.description]
                return [dict(zip(cols, row, strict=True)) for row in rel.fetchall()]
            finally:
                cur.close()


class DatabricksStore:
    """Databricks SQL warehouse. Use a token that only has SELECT on the gold schema."""

    def __init__(self, catalog: str | None = None):
        from databricks import sql as dbsql

        self.schema = f"{catalog or os.environ.get('TILLWISE_CATALOG', 'tillwise')}.gold"
        self._connect = lambda: dbsql.connect(
            server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"],
            http_path=os.environ["DATABRICKS_HTTP_PATH"],
            access_token=os.environ["DATABRICKS_TOKEN"],
        )

    def query(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        _check_read_only(sql)
        sql = sql.replace("gold.", f"{self.schema}.")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params or {})
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def store_from_env() -> GoldStore:
    if os.environ.get("TILLWISE_BACKEND", "duckdb").lower() == "databricks":
        return DatabricksStore()
    return DuckDBStore(os.environ.get("TILLWISE_GOLD_DIR", DEFAULT_GOLD_DIR))
