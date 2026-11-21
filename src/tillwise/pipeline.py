"""Week 1 batch pipeline: bronze -> silver -> gold, shared by the local runner and notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from tillwise.transforms import gold, scd2, silver


@dataclass
class PipelineOutputs:
    silver_transactions: DataFrame
    silver_quarantine: DataFrame
    silver_non_product: DataFrame
    dq_metrics: DataFrame
    products_scd2: DataFrame
    dim_date: DataFrame
    dim_country: DataFrame
    dim_customer: DataFrame
    fact_sales: DataFrame


def run_silver_and_gold(bronze: DataFrame, holidays: list[tuple[date, str]] | None = None) -> PipelineOutputs:
    typed = silver.type_transactions(bronze)
    result = silver.split_transactions(typed)
    metrics = silver.dq_metrics(typed, result)

    products = scd2.build_product_scd2(result.clean)
    bounds = result.clean.agg(F.min("invoice_ts").alias("lo"), F.max("invoice_ts").alias("hi")).first()
    if bounds is None or bounds["lo"] is None:
        raise ValueError("No clean rows: check the bronze input")
    start, end = bounds["lo"].date(), bounds["hi"].date()
    dim_date = gold.build_dim_date(bronze.sparkSession, date(start.year, 1, 1), date(end.year + 1, 12, 31),
                                   holidays or [])

    return PipelineOutputs(
        silver_transactions=result.clean,
        silver_quarantine=result.quarantine,
        silver_non_product=result.non_product,
        dq_metrics=metrics,
        products_scd2=products,
        dim_date=dim_date,
        dim_country=gold.build_dim_country(result.clean),
        dim_customer=gold.build_dim_customer(result.clean),
        fact_sales=gold.build_fact_sales(result.clean, products),
    )
