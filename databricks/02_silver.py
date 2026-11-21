# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver: types, de-duplication and data-quality rules
# MAGIC Writes three tables so nothing is silently dropped:
# MAGIC `silver.transactions` (clean), `silver.transactions_quarantine` (with `dq_reason`),
# MAGIC `silver.non_product_lines` (postage, fees, adjustments), plus a run log in `silver.dq_metrics`.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from pyspark.sql import functions as F

from tillwise.transforms import silver

bronze = spark.table(T.bronze_transactions)  # noqa: F821
typed = silver.type_transactions(bronze)
result = silver.split_transactions(typed)

result.clean.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T.silver_transactions)
result.quarantine.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T.silver_quarantine)
result.non_product.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T.silver_non_product)

metrics = silver.dq_metrics(typed, result).withColumn("run_ts", F.current_timestamp())
metrics.write.mode("append").saveAsTable(T.silver_dq_metrics)
display(metrics)  # noqa: F821

# COMMAND ----------

# MAGIC %md Quarantine reasons with examples

# COMMAND ----------

display(spark.sql(f"""
  SELECT dq_reason, count(*) AS rows, any_value(invoice_no) AS example_invoice,
         any_value(stock_code) AS example_stock_code
  FROM {T.silver_quarantine} GROUP BY dq_reason ORDER BY rows DESC
"""))  # noqa: F821
