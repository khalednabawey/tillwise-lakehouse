# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Agent views and read-only access
# MAGIC Creates `gold.v_sales` (the view the agent's tools query) and shows how to give the agent a
# MAGIC principal that can only read `gold`. Then point the API at Databricks:
# MAGIC ```
# MAGIC TILLWISE_BACKEND=databricks TILLWISE_CATALOG=<catalog>
# MAGIC DATABRICKS_SERVER_HOSTNAME=...  DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/...  DATABRICKS_TOKEN=...
# MAGIC ```

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from tillwise.agent.store import SALES_VIEW

spark.sql(SALES_VIEW.format(schema=f"{CATALOG}.gold"))
display(spark.sql(f"SELECT * FROM {CATALOG}.gold.v_sales LIMIT 5"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read-only principal
# MAGIC Create a service principal (or use a separate user), generate its token, and grant only these.
# MAGIC Replace `agent-sp` with its application ID. On Free Edition, a personal access token for a
# MAGIC second user works the same way.

# COMMAND ----------

print(f"""
GRANT USE CATALOG ON CATALOG {CATALOG} TO `agent-sp`;
GRANT USE SCHEMA ON SCHEMA {CATALOG}.gold TO `agent-sp`;
GRANT SELECT ON SCHEMA {CATALOG}.gold TO `agent-sp`;
-- and CAN USE on the SQL warehouse (Warehouse → Permissions)
""")
