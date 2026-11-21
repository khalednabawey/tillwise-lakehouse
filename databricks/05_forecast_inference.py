# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Final model and next-8-week forecast
# MAGIC Refits the model on all observed weeks, registers it in Unity Catalog as
# MAGIC `<catalog>.ml.demand_gbt` with alias `@champion`, and writes `gold.forecast_weekly`
# MAGIC with p10/p90 bounds taken from the backtest residuals.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from pyspark.sql import functions as F

from tillwise.ml.dataset import future_frame, training_frame
from tillwise.ml.model import fit, predict, with_intervals
from tillwise.ml.tracking import log_and_register

weekly = spark.table(T.fq("gold", "weekly_product_demand"))
dim_date = spark.table(T.gold_dim_date)

model = fit(training_frame(weekly, horizons=8))
future = predict(model, future_frame(weekly, dim_date, horizons=8))

quantiles = {int(r["h"]): (r["q10"], r["q90"]) for r in spark.table(T.fq("ml", "residual_quantiles")).collect()}
forecast = with_intervals(future, quantiles)

user = spark.sql("SELECT current_user()").first()[0]
log_and_register(model, forecast, experiment=f"/Users/{user}/tillwise-demand-forecast",
                 registered_model_name=T.fq("ml", "demand_gbt"))

(forecast.select("stock_code", "country", "origin_week", "target_week", F.col("h").cast("int").alias("h"),
                 F.round("prediction", 1).alias("forecast_units"),
                 F.round("p10", 1).alias("p10"), F.round("p90", 1).alias("p90"))
 .withColumn("run_ts", F.current_timestamp())
 .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T.fq("gold", "forecast_weekly")))

display(spark.table(T.fq("gold", "forecast_weekly")).orderBy("stock_code", "country", "h"))
