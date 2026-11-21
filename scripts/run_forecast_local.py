"""Week 2 locally: weekly demand, 12-week backtest vs baselines, RFM segments, next-8-week forecast.

Usage:  uv run python scripts/run_forecast_local.py [--no-mlflow]
MLflow runs go to ./mlruns (view with: uv run mlflow ui).
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    spark = local_spark("tillwise-forecast", partitions=8)
    spark.sparkContext.setLogLevel("ERROR")
    t0 = time.time()

    raw = spark.read.option("header", True).option("multiLine", True).option("escape", '"').csv(
        [str(f) for f in sorted(RAW.glob("online_retail_ii_*.csv"))])
    h = pd.read_csv(RAW / "uk_holidays.csv")
    holidays = [(date.fromisoformat(d), n) for d, n in zip(h["date"], h["holiday_name"], strict=True)]
    out = run_silver_and_gold(standardize_raw(raw, source_file=F.input_file_name()), holidays)
    fact = out.fact_sales.cache()

    weekly = weekly_demand(fact, out.dim_country, out.dim_date).cache()
    n_series = weekly.select("stock_code", "country").distinct().count()
    weeks = weekly.agg(F.min("week_start"), F.max("week_start"), F.countDistinct("week_start")).first()
    print(f"\nWeekly demand: {n_series} series x {weeks[2]} full weeks ({weeks[0]} to {weeks[1]})")
    print("Countries:", [r[0] for r in weekly.select("country").distinct().orderBy("country").collect()])

    frame = training_frame(weekly, horizons=12).cache()
    print(f"Training frame: {frame.count():,} (series, origin, horizon) rows")

    result = backtest(frame, test_weeks=12)
    print("\nBacktest, last 12 weeks, one forecast origin (lower WAPE is better):")
    for name, m in sorted(result.metrics.items(), key=lambda kv: kv[1]["wape"]):
        print(f"  {name:15s} WAPE {m['wape']:.3f}   bias {m['bias']:+.3f}")
    print("\nWAPE by horizon (weeks ahead):")
    result.by_horizon.select("h", F.round("wape_gbt", 3).alias("gbt"),
                             F.round("wape_seasonal", 3).alias("seasonal_naive")).show(12)

    if not args.no_mlflow:
        import os

        import mlflow

        from tillwise.ml.tracking import log_backtest

        # Local only: the plain-folder store is fine for one person; Databricks uses its own tracking server.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
        print("MLflow run:", log_backtest(result, experiment="tillwise-demand-forecast"))

    # Final model on all history -> next 8 weeks with p10/p90 from backtest residuals
    model = fit(training_frame(weekly, horizons=8))
    forecast = with_intervals(predict(model, future_frame(weekly, out.dim_date, horizons=8)),
                              result.residual_quantiles)
    top = weekly.groupBy("stock_code", "country").agg(F.sum("units").alias("u")).orderBy(F.desc("u")).first()
    print(f"\nNext 8 weeks for the biggest series ({top['stock_code']}, {top['country']}):")
    (forecast.where((F.col("stock_code") == top["stock_code"]) & (F.col("country") == top["country"]))
     .select("target_week", F.round("prediction", 0).alias("forecast"), F.round("p10", 0).alias("p10"),
             F.round("p90", 0).alias("p90")).orderBy("target_week").show())

    as_of = fact.agg(F.max("invoice_ts")).first()[0].date()
    segments = rfm.build_customer_rfm(fact, as_of)
    print(f"RFM segments as of {as_of}:")
    total = segments.agg(F.sum("monetary")).first()[0]
    (segments.groupBy("segment").agg(F.count("*").alias("customers"), F.round(F.sum("monetary"), 0).alias("revenue"))
     .withColumn("revenue_share_pct", F.round(F.col("revenue") * 100 / F.lit(float(total)), 1))
     .orderBy(F.desc("revenue")).show(truncate=False))

    print(f"Finished in {time.time() - t0:.0f}s")
    spark.stop()


if __name__ == "__main__":
    main()
