# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Forecast backtest (MLflow)
# MAGIC Direct multi-horizon gradient-boosted model on `gold.weekly_product_demand`.
# MAGIC Trains on everything up to the cutoff (last week − 12), forecasts the 12 held-out weeks
# MAGIC from the cutoff, and compares with seasonal naive, 4-week mean and last value.
# MAGIC Writes `gold.forecast_backtest` and `ml.residual_quantiles` (used for intervals in 05).

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from pyspark.sql import functions as F

from tillwise.ml.dataset import training_frame
from tillwise.ml.model import backtest
from tillwise.ml.tracking import log_backtest
from tillwise.spark import try_cache

weekly = spark.table(T.fq("gold", "weekly_product_demand"))
frame = try_cache(training_frame(weekly, horizons=12))
result = backtest(frame, test_weeks=12)

user = spark.sql("SELECT current_user()").first()[0]
run_id = log_backtest(result, experiment=f"/Users/{user}/tillwise-demand-forecast")
print("MLflow run:", run_id)
display(spark.createDataFrame([(k, v["wape"], v["bias"]) for k, v in result.metrics.items()],
                              "model string, wape double, bias double"))

# COMMAND ----------

(result.predictions
 .select("stock_code", "country", "origin_week", "target_week", "h", "label", "prediction",
         "pred_seasonal_naive", "pred_ma4", "pred_last_value")
 .withColumn("run_ts", F.current_timestamp())
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T.fq("gold", "forecast_backtest")))

spark.createDataFrame([(float(h), lo, hi) for h, (lo, hi) in result.residual_quantiles.items()],
                      "h double, q10 double, q90 double") \
    .write.mode("overwrite").saveAsTable(T.fq("ml", "residual_quantiles"))

display(result.by_horizon)
