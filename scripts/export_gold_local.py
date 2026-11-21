"""Build the gold layer locally and save it as Parquet in data/gold/, for the agent and its evals.

Runs the same code as the pipeline and notebooks (bronze -> gold, RFM, backtest, 8-week forecast),
then writes each table with pandas/pyarrow (no Hadoop tools needed on Windows).

Usage:  uv run python scripts/export_gold_local.py
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from tillwise.ml.dataset import future_frame, training_frame  # noqa: E402
from tillwise.ml.features import weekly_demand  # noqa: E402
from tillwise.ml.model import backtest, fit, predict, with_intervals  # noqa: E402
from tillwise.pipeline import run_silver_and_gold  # noqa: E402
from tillwise.spark import local_spark  # noqa: E402
from tillwise.transforms import rfm  # noqa: E402
from tillwise.transforms.bronze import standardize_raw  # noqa: E402

RAW = ROOT / "data" / "raw"
GOLD = ROOT / "data" / "gold"


def save(df, name: str) -> None:
    pdf = df.toPandas()
    for col in pdf.columns:  # decimals -> float for DuckDB/pyarrow portability
        if pdf[col].dtype == object and len(pdf) and type(pdf[col].dropna().iloc[0]).__name__ == "Decimal":
            pdf[col] = pdf[col].astype(float)
    pdf.to_parquet(GOLD / f"{name}.parquet", index=False)
    print(f"  {name}: {len(pdf):,} rows")


def main() -> None:
    GOLD.mkdir(parents=True, exist_ok=True)
    spark = local_spark("tillwise-export", partitions=8)
    spark.sparkContext.setLogLevel("ERROR")
    t0 = time.time()

    raw = spark.read.option("header", True).option("multiLine", True).option("escape", '"').csv(
        [str(f) for f in sorted(RAW.glob("online_retail_ii_*.csv"))])
    h = pd.read_csv(RAW / "uk_holidays.csv")
    holidays = [(date.fromisoformat(d), n) for d, n in zip(h["date"], h["holiday_name"], strict=True)]
    out = run_silver_and_gold(standardize_raw(raw, source_file=F.input_file_name()), holidays)
    fact = out.fact_sales.cache()

    weekly = weekly_demand(fact, out.dim_country, out.dim_date).cache()
    result = backtest(training_frame(weekly, horizons=12), test_weeks=12)
    model = fit(training_frame(weekly, horizons=8))
    forecast = with_intervals(predict(model, future_frame(weekly, out.dim_date, horizons=8)),
                              result.residual_quantiles)
    as_of = fact.agg(F.max("invoice_ts")).first()[0].date()

    print("Writing gold tables to data/gold/:")
    save(fact, "fact_sales")
    save(out.products_scd2, "dim_product")
    save(out.dim_country, "dim_country")
    save(out.dim_customer, "dim_customer")
    save(out.dim_date, "dim_date")
    save(rfm.build_customer_rfm(fact, as_of), "customer_rfm")
    save(weekly, "weekly_product_demand")
    save(forecast.select("stock_code", "country", "origin_week", "target_week", F.col("h").cast("int").alias("h"),
                         F.round("prediction", 1).alias("forecast_units"), F.round("p10", 1).alias("p10"),
                         F.round("p90", 1).alias("p90")), "forecast_weekly")
    save(result.predictions.select("stock_code", "country", "origin_week", "target_week", "h", "label",
                                   "prediction", "pred_seasonal_naive", "pred_ma4", "pred_last_value"),
         "forecast_backtest")
    print(f"Finished in {time.time() - t0:.0f}s")
    spark.stop()


if __name__ == "__main__":
    main()
