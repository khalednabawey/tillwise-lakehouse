"""Local SparkSession for tests and the local runner. On Databricks, use the provided `spark`."""

from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession


def local_spark(app_name: str = "tillwise-local", partitions: int = 4) -> SparkSession:
    # Make sure Spark workers use the same interpreter as the driver (uv venv on Windows).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.master(f"local[{partitions}]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", str(partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.ansi.enabled", "true")  # same default as Databricks serverless
        .getOrCreate()
    )
