# Tillwise

**A retail sales lakehouse on Databricks.** About 1 million real transactions from a UK online retailer,
landed with Auto Loader, cleaned with explicit data-quality rules, and modelled into a star schema with an
SCD Type 2 product dimension. Coming next: one Lakeflow pipeline with expectations, weekly demand
forecasting tracked in MLflow, a dashboard, and an agent that answers business questions from the gold layer.

> Status: **week 1 of 3 done** (bronze → silver → gold, tested). See [Roadmap](#roadmap).

## Architecture

```text
UCI Online Retail II CSVs ──► UC Volume /raw/landing/retail/
                                   │  Auto Loader (incremental, all strings + lineage)
                                   ▼
BRONZE  bronze.transactions_raw
                                   │  types (ANSI-safe), de-duplication, rule checks
                                   ▼
SILVER  silver.transactions            clean sales and returns
        silver.transactions_quarantine rows that break a rule, with dq_reason
        silver.non_product_lines       postage, fees, adjustments
        silver.dq_metrics              row counts per outcome, per run
        silver.products_scd2           product versions (SCD Type 2)
                                   ▼
GOLD    fact_sales ── dim_product (SCD2) · dim_customer · dim_country · dim_date (UK holidays)
```

The same Python functions (`src/tillwise/transforms/`) run in three places: unit tests on a local
Spark session, the local runner, and the Databricks notebooks. Notebooks contain no business logic.

## Data-quality rules (silver)

Nothing is dropped silently. Each row ends in exactly one place, and the counts are logged.

| Rule | Outcome |
|---|---|
| Exact duplicate line | removed, counted |
| Timestamp can't be parsed | quarantine `unparseable_timestamp` |
| Missing stock code | quarantine `missing_stock_code` |
| Invoice starts with `A` (bad-debt adjustment) | quarantine `adjustment_invoice` |
| Price missing or ≤ 0 | quarantine `non_positive_price` |
| Quantity ≤ 0 on a normal invoice | quarantine `non_positive_quantity_not_return` |
| Positive quantity on a `C` (return) invoice | quarantine `return_with_positive_quantity` |
| Postage, fees, manual codes (`POST`, `M`, `BANK CHARGES`, …) | `non_product_lines` |
| Everything else | `silver.transactions` (returns kept, with `is_return`) |

<!-- RESULTS:START -->
### Results on the full dataset (local run, 21 Nov 2026)

| Outcome | Rows | % of input |
|---|---:|---:|
| Input (bronze) | 1,067,371 | 100.00 |
| Duplicates removed | 34,335 | 3.22 |
| Quarantine: `non_positive_price` | 6,032 | 0.57 |
| Quarantine: `adjustment_invoice` | 6 | 0.00 |
| Quarantine: `return_with_positive_quantity` | 1 | 0.00 |
| Quarantine: other rules | 0 | 0.00 |
| Non-product lines (postage, fees, manual) | 5,665 | 0.53 |
| **Clean (`silver.transactions`)** | **1,021,332** | **95.69** |

Every input row is accounted for (the outcomes add up to the input).

**Why de-duplication matters here:** the source's two sheets overlap. 1–9 December 2010 appears in both
(22,523 rows), on top of about 12,000 repeated lines inside the sheets. Without the de-duplication step,
that week's revenue would be counted twice.

| Gold | Rows |
|---|---:|
| `fact_sales` | 1,021,332 (all matched to a product version) |
| `dim_product` (SCD2) | 6,695 versions for 4,745 stock codes; 561 codes changed description |
| `dim_customer` | 5,875 |
| `dim_country` | 43 |
| `dim_date` | 1,095 (2009–2011, with England bank holidays) |

Net revenue (sales minus returns): £9.14M in 2010 and £9.01M in 2011.
<!-- RESULTS:END -->

## SCD Type 2 products

The source has no product master: the description is whatever was typed on each invoice, and it
changes over time. Tillwise takes the most frequent description per stock code per day and opens a new
version whenever it changes (so A → B → A gives three versions). `fact_sales` joins each line to the
version valid at sale time (`valid_from <= invoice_ts < valid_to`, with `valid_to` NULL for the current version), so historical reports keep the
names that were used then.

## Run it

**Local (tests and a full run on your laptop)** — needs Java 17+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups
uv run pytest                                        # unit tests on local Spark
uv run python -m tillwise.ingest.download_sources    # ~45 MB download into data/raw/
uv run python scripts/run_local.py                   # bronze → gold, prints the results
```

**Databricks Free Edition**

1. Create a Git folder from this repository (Workspace → Create → Git folder).
2. Run `databricks/00_setup_unity_catalog.py` (set the `catalog` widget to `workspace` if creating a
   catalog isn't allowed).
3. Upload the CSVs from `data/raw/` to the landing Volume, as described in `data/README.md`.
4. Run `01_bronze_autoloader`, `02_silver`, `03_gold` in order.

## Roadmap

- [x] **Week 1** — Unity Catalog setup, Auto Loader bronze, silver rules with quarantine and metrics,
      SCD2 products, gold star schema, unit tests, CI
- [ ] **Week 2** — one Lakeflow Declarative Pipeline (expectations, AUTO CDC for SCD2), Databricks Job,
      demand features, baseline + gradient-boosted forecast with a time-based backtest, MLflow, AI/BI dashboard
- [ ] **Week 3** — FastAPI agent with fixed tools on gold views (Groq, OpenRouter fallback),
      30-question evaluation set, demo video

## Data and licences

See [`data/README.md`](data/README.md). The retail data is *Online Retail II* (UCI, CC BY 4.0).
