# LinkedIn post draft

The Databricks run is done (21 Nov 2026). Post after recording the demo video, and replace the bracketed parts.

I built a retail sales lakehouse on Databricks to learn the stack properly: PySpark, Delta, Unity Catalog and
Lakeflow.

Tillwise takes about 1 million real transactions from a UK online retailer (UCI Online Retail II) and runs them
through bronze, silver and gold:

- Data quality without silent drops: every row ends up clean, quarantined (with a reason) or flagged as a
  fee/postage line, and the counts add up. De-duplication caught a week in Dec 2010 that appears in both
  source sheets, about 22,000 rows that would have been counted twice.
- SCD Type 2 products, so old sales keep the product name they had at the time.
- One Lakeflow pipeline with a named expectation per rule, run by a Databricks Job, all defined as code in an
  Asset Bundle.
- A weekly demand forecast (gradient-boosted trees, one model for 1–12 weeks ahead, tracked in MLflow). In a
  12-week backtest it beat three baselines. The honest result: WAPE improved only slightly, but bias dropped
  from about −40% to −3% going into Christmas.
- A small agent (FastAPI + LangGraph, Groq with an OpenRouter fallback) that answers questions from the gold
  layer through fixed, read-only tools. Every number it states must come from a tool result. It scored
  30/30 on an eval set whose answers are computed straight from the data.

Code: [GitHub link]   Demo: [video link]

#Databricks #PySpark #DataEngineering #Lakehouse #MLflow #AIAgents
