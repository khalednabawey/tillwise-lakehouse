"""Names and rules shared by local runs, tests and Databricks notebooks."""

from __future__ import annotations

import os
from dataclasses import dataclass

# Free Edition may not allow CREATE CATALOG; in that case set TILLWISE_CATALOG=workspace.
DEFAULT_CATALOG = os.getenv("TILLWISE_CATALOG", "tillwise")

SCHEMAS = ("raw", "bronze", "silver", "gold", "ml")
LANDING_VOLUME = "landing"  # lives in the raw schema: /Volumes/<catalog>/raw/landing/


@dataclass(frozen=True)
class Tables:
    catalog: str = DEFAULT_CATALOG

    def fq(self, schema: str, table: str) -> str:
        return f"{self.catalog}.{schema}.{table}"

    @property
    def landing_path(self) -> str:
        return f"/Volumes/{self.catalog}/raw/{LANDING_VOLUME}"

    # bronze
    @property
    def bronze_transactions(self) -> str:
        return self.fq("bronze", "transactions_raw")

    # silver
    @property
    def silver_transactions(self) -> str:
        return self.fq("silver", "transactions")

    @property
    def silver_quarantine(self) -> str:
        return self.fq("silver", "transactions_quarantine")

    @property
    def silver_non_product(self) -> str:
        return self.fq("silver", "non_product_lines")

    @property
    def silver_dq_metrics(self) -> str:
        return self.fq("silver", "dq_metrics")

    @property
    def silver_products_scd2(self) -> str:
        return self.fq("silver", "products_scd2")

    # gold
    @property
    def gold_fact_sales(self) -> str:
        return self.fq("gold", "fact_sales")

    @property
    def gold_dim_product(self) -> str:
        return self.fq("gold", "dim_product")

    @property
    def gold_dim_customer(self) -> str:
        return self.fq("gold", "dim_customer")

    @property
    def gold_dim_country(self) -> str:
        return self.fq("gold", "dim_country")

    @property
    def gold_dim_date(self) -> str:
        return self.fq("gold", "dim_date")


# Source column -> bronze column. The source file has a space in "Customer ID",
# which Delta only accepts with column mapping, so every name is normalised here.
SOURCE_COLUMNS: dict[str, str] = {
    "Invoice": "invoice_no",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_ts",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}

SOURCE_TIMESTAMP_FORMAT = "yyyy-MM-dd HH:mm:ss"

# Stock codes that are fees, postage or manual adjustments rather than products.
NON_PRODUCT_CODE_PATTERN = (
    r"^(POST|DOT|M|C2|D|CRUK|PADS|B|S|BANK CHARGES|AMAZONFEE|ADJUST\d*|TEST\d*|gift_\d+_\d+)$"
)

# Data-quality rules for silver, in the order they are checked. The first matching
# rule wins, so each quarantined row carries exactly one reason. The same names are
# reused as Lakeflow expectations in week 2.
QUARANTINE_RULES: dict[str, str] = {
    "unparseable_timestamp": "invoice_ts IS NULL",
    "missing_stock_code": "stock_code IS NULL",
    "adjustment_invoice": "invoice_no LIKE 'A%'",
    "non_positive_price": "unit_price IS NULL OR unit_price <= 0",
    "non_positive_quantity_not_return": "quantity <= 0 AND NOT is_return",
    "return_with_positive_quantity": "quantity > 0 AND is_return",
}
