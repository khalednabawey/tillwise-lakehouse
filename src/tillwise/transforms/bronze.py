"""Bronze: land the source as-is, with normalised names and lineage columns.

Every business column stays a string. Typing and validation happen in silver, so a
bad value never stops ingestion and the raw value is always kept for audit.
Works on batch and streaming DataFrames (Auto Loader), since it is a pure projection.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from tillwise.config import SOURCE_COLUMNS


def standardize_raw(df: DataFrame, source_file: Column | None = None) -> DataFrame:
    """Rename source columns to snake_case, cast to trimmed strings, add lineage.

    `source_file` defaults to Spark's `_metadata.file_path` hidden column, which Auto
    Loader and file sources provide. Pass a literal in tests or for in-memory data.
    """
    missing = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Source is missing columns: {missing}")

    if source_file is None:
        source_file = F.col("_metadata.file_path")

    cleaned = [
        F.nullif(F.trim(F.col(f"`{src}`").cast("string")), F.lit("")).alias(dst)
        for src, dst in SOURCE_COLUMNS.items()
    ]
    return df.select(
        *cleaned,
        source_file.cast("string").alias("_source_file"),
        F.current_timestamp().alias("_ingest_ts"),
    )
