from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tillwise.spark import local_spark  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    session = local_spark("tillwise-tests", partitions=1)
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


RAW_COLUMNS = [
    "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country",
]


@pytest.fixture
def raw_df(spark):
    """A tiny slice shaped like the UCI file, covering every silver rule."""

    def make(rows):
        return spark.createDataFrame(rows, ", ".join(f"`{c}` string" for c in RAW_COLUMNS))

    return make
