# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold: SCD2 products and the star schema
# MAGIC `silver.products_scd2` / `gold.dim_product`, `gold.dim_customer`, `gold.dim_country`,
# MAGIC `gold.dim_date` (with UK holidays) and `gold.fact_sales`.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from datetime import date

from pyspark.sql import functions as F

from tillwise.transforms import gold, scd2

clean = spark.table(T.silver_transactions)  # noqa: F821

products = scd2.build_product_scd2(clean)
products.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T.silver_products_scd2)
spark.table(T.silver_products_scd2).write.mode("overwrite").option(  # noqa: F821
    "overwriteSchema", "true").saveAsTable(T.gold_dim_product)

holidays_path = f"{T.landing_path}/reference/uk_holidays.csv"
try:
    hol_rows = spark.read.option("header", True).csv(holidays_path).collect()  # noqa: F821
    holidays = [(date.fromisoformat(r["date"]), r["holiday_name"]) for r in hol_rows]
except Exception:  # noqa: BLE001
    print("No holidays file found; dim_date.is_holiday will be false everywhere.")
    holidays = []

bounds = clean.agg(F.min("invoice_ts").alias("lo"), F.max("invoice_ts").alias("hi")).first()
dim_date = gold.build_dim_date(spark, date(bounds["lo"].year, 1, 1), date(bounds["hi"].year, 12, 31),  # noqa: F821
                               holidays)

dims = {
    T.gold_dim_date: dim_date,
    T.gold_dim_country: gold.build_dim_country(clean),
    T.gold_dim_customer: gold.build_dim_customer(clean),
    T.gold_fact_sales: gold.build_fact_sales(clean, spark.table(T.gold_dim_product)),  # noqa: F821
}
for name, df in dims.items():
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(name)
    print(f"{name}: {spark.table(name).count():,} rows")  # noqa: F821

# COMMAND ----------

# MAGIC %md Check: every described line maps to exactly one product version

# COMMAND ----------

display(spark.sql(f"""
  SELECT count(*) AS lines,
         count_if(product_sk IS NULL) AS lines_without_version,
         round(sum(line_revenue), 2) AS net_revenue
  FROM {T.gold_fact_sales}
"""))  # noqa: F821
