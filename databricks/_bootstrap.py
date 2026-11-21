# Databricks notebook source
# MAGIC %md
# MAGIC ### Bootstrap (run with `%run ./_bootstrap` from the other notebooks)
# MAGIC Puts the repo's `src/` folder on the Python path so notebooks import the same tested
# MAGIC code that runs locally and in CI, and reads the `catalog` widget.

# COMMAND ----------

import os
import sys

_src = os.path.abspath(os.path.join(os.getcwd(), "..", "src"))
if _src not in sys.path:
    sys.path.insert(0, _src)

dbutils.widgets.text("catalog", "tillwise", "Unity Catalog catalog")  # noqa: F821
CATALOG = dbutils.widgets.get("catalog")  # noqa: F821

from tillwise.config import Tables  # noqa: E402

T = Tables(catalog=CATALOG)
print(f"Using catalog: {CATALOG}; source code from {_src}")
