# Tillwise

**A retail sales lakehouse on Databricks.** About 1 million real transactions from a UK online retailer,
landed with Auto Loader, cleaned with explicit data-quality rules, and modelled into a star schema with an
SCD Type 2 product dimension. One Lakeflow pipeline runs it all with a named expectation per rule. On top:
customer RFM segments and a weekly demand forecast (gradient-boosted trees, tracked in MLflow, registered in
Unity Catalog) that is backtested against three baselines. A small data agent answers business questions
from the gold layer through fixed, read-only tools. It scored 30/30 on an evaluation set whose answers are
computed straight from the data.

> Status: **all 3 weeks built, tested in CI, and running on Databricks Free Edition** (21 Nov 2026): the
> Lakeflow pipeline, the daily job (pipeline → backtest → forecast) and the Unity Catalog model registration.
> See [Running on Databricks](#running-on-databricks).

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
        customer_rfm                   recency / frequency / monetary scores and segments
        weekly_product_demand          top 50 products × top 5 countries × full weeks, zero-filled
                                   ▼
ML      04 backtest (MLflow) ──► gold.forecast_backtest, ml.residual_quantiles
        05 final model ──► UC model ml.demand_gbt@champion ──► gold.forecast_weekly (p10 / p90)
```

**Orchestration:** bronze → gold is one **Lakeflow Declarative Pipeline**
(`databricks/pipelines/tillwise_pipeline.py`). A **Databricks Job** (`resources/job.yml`) runs the pipeline,
then the backtest, then the forecast. Both are defined as code in a **Databricks Asset Bundle**
(`databricks.yml`). The week 1 notebooks (`01`–`03`) do the same steps by hand, which is useful for learning
and debugging.

The same Python functions (`src/tillwise/`) run in four places: unit tests on a local Spark session, the
local runners, the notebooks, and the Lakeflow pipeline. None of them contain business logic of their own.

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
| `dim_date` | 1,461 (2009–2012, with England bank holidays) |

Net revenue (sales minus returns): £9.14M in 2010 and £9.01M in 2011.
<!-- RESULTS:END -->

## SCD Type 2 products

The source has no product master: the description is whatever was typed on each invoice, and it
changes over time. Tillwise takes the most frequent description per stock code per day and opens a new
version whenever it changes (so A → B → A gives three versions). `fact_sales` joins each line to the
version valid at sale time (`valid_from <= invoice_ts < valid_to`, with `valid_to` NULL for the current version), so historical reports keep the
names that were used then.

In the Lakeflow pipeline, SCD2 is a materialized view rebuilt from history by the same tested function,
not `AUTO CDC`. AUTO CDC needs a streaming change feed, such as a product master's change log. Here the
"changes" can only be worked out after de-duplicating and aggregating a whole day, which is a batch step.

## Demand forecast

- **Data:** `gold.weekly_product_demand`. 227 series (the top 50 products in the top 5 countries:
  UK, EIRE, Germany, France, Netherlands) × 104 full weeks. Weeks with no sales are explicit zeros, and the
  partial first and last weeks are dropped.
- **Target:** gross units sold per week (returns are tracked separately).
- **Method:** one gradient-boosted tree model (Spark ML `GBTRegressor`, log1p target) covers every horizon
  from 1 to 12 weeks ("direct multi-horizon"). Each row has only what was known at the forecast origin
  (last week, 4/13/52-week means, last price) plus facts about the target week (week of year, bank
  holiday, same week last year). The tests check there is no leakage.
- **Backtest:** train on everything up to 12 weeks before the end, then forecast the last 12 weeks from that
  single origin. The model never sees those weeks.
- **Intervals:** p10/p90 come from the backtest's *relative* errors at each horizon, so bands scale with the
  series. Absolute errors pooled across all series would give big sellers bands that are far too narrow. For
  the biggest series (85099B, UK), the first week's forecast is 884 units with a p10–p90 band of 276–3,074.
  The band is wide and skewed upward, as it should be for lumpy wholesale demand.

| Backtest (last 12 weeks) | WAPE ↓ | Bias |
|---|---:|---:|
| **Gradient-boosted model** | **0.669** | **−0.034** |
| 4-week moving average | 0.677 | −0.408 |
| Last value | 0.756 | −0.390 |
| Seasonal naive (same week last year) | 0.833 | +0.163 |

**Reading it honestly:**
- Weekly demand for one product in one country is very lumpy: many zero weeks, then wholesale-sized orders. So
  every method has a high WAPE here.
- The model's WAPE is only about 1% better than the 4-week average, which is a hard baseline.
- The bigger gain is **bias**. The test weeks are September to November 2011, the run-up to Christmas. The
  recent-average baselines under-forecast by about 40% and seasonal naive over-forecasts by 16%; the model
  is within 3.4%. For ordering stock, that matters more than a small WAPE gain.
- The model beats seasonal naive at 11 of 12 horizons.

## Customer segments (RFM)

`gold.customer_rfm` scores every known customer from 1 to 5 on recency, frequency and monetary value, as of
the last sale date (9 Dec 2011). Scoring uses `ceil(cume_dist × 5)`, so customers with tied values always get
the same score (unlike `ntile`, which splits ties by row order). Rule-based segments:

| Segment | Customers | Net revenue | Share |
|---|---:|---:|---:|
| Champions | 1,335 | £11.45M | 70.0% |
| Loyal | 838 | £1.93M | 11.8% |
| At risk | 1,206 | £1.67M | 10.2% |
| Needs attention | 1,101 | £0.86M | 5.3% |
| Hibernating | 1,133 | £0.37M | 2.2% |
| New | 239 | £0.08M | 0.5% |

About 23% of customers (the Champions) bring in 70% of revenue. The 1,206 "At risk" customers used to buy
often but have gone quiet: £1.67M of past revenue to win back.

## Data agent ("ask the lakehouse")

`POST /ask {"question": "..."}` → an answer, the tools it called, and the model that answered.

- **Fixed tools, no free SQL:** nine read-only tools (`sales_summary`, `top_products`, `country_breakdown`,
  `monthly_trend`, `returns_report`, `customer_segments`, `find_products`, `demand_forecast`, `data_coverage`).
  Each validates its arguments (dates, countries, limits) and runs one parameterised query on the gold layer.
  The store refuses anything that isn't a single `SELECT`.
- **Grounded numbers:** before an answer leaves, every number of 100 or more must match a tool result
  (within 1%, or a sum or difference of two results). If not, the model gets one retry to fix it.
- **Uses the SCD2 history:** `find_products` searches past names too. "White hanging heart t-light holder"
  finds 85123A, whose current name is "CREAM HANGING HEART…".
- **Models:** Groq (`openai/gpt-oss-120b`) first, OpenRouter (free model) on any Groq error. Every answer
  records which provider produced it.
- **Backends:** DuckDB over a local Parquet export of gold (development, tests, evals), or a Databricks SQL
  warehouse with a read-only token (`TILLWISE_BACKEND=databricks`; see `databricks/06_agent_views.py`).

```text
START → agent ──tool calls──► tools ──► agent … ──answer──► ground ──ok──► END
                                                              └─ made-up number → one retry → agent
```

### Evaluation: 30 questions

[`evals/cases.yaml`](evals/cases.yaml) has 30 questions, including 4 the agent should decline (a forecast it
doesn't have, a future year, a delete request, a customer's personal details). Ground truth comes from SQL run
directly on the gold tables, not through the agent's tools, so a bug in a tool shows up as a failure.
Numbers must match within 1%.

| Run (21 Nov 2026) | Score |
|---|---|
| Full run, as first scored | 29/30 |
| Same answers, after fixing a scorer false positive | **30/30** |

The one first-pass failure was the scorer's fault: the agent correctly declined the forecast question but
repeated the stock code from the question (22423), and the scorer counted it as an invented number. Details
are in [`evals/reports/latest.md`](evals/reports/latest.md). Groq's free tier hit its rate limit during the
run, so 20 of the 30 answers came from the OpenRouter fallback. That is a useful test of the fallback, and the
score held.

## Running on Databricks

Run on Databricks Free Edition (serverless) on 21 November 2026, using this repo as a Git folder:

1. `00_setup_unity_catalog` created the `tillwise` catalog, the five schemas and the landing volume (Free
   Edition allowed `CREATE CATALOG`).
2. `00b_load_sources` downloaded the UCI data straight into the volume.
3. `07_deploy_pipeline_and_job` created the Lakeflow pipeline and the `tillwise-daily` job with the Databricks
   SDK, then started the job.

| Check (SQL on the Databricks tables) | Databricks | Local run |
|---|---:|---:|
| Bronze rows | 1,067,371 | 1,067,371 |
| `fact_sales` rows | 1,021,332 | 1,021,332 |
| Duplicates removed | 34,335 | 34,335 |
| Net revenue 2010 | £9,136,724.93 | £9,136,724.93 |
| Champions customers | 1,335 | 1,335 |
| Forecast rows (227 series × 8 weeks) | 1,816 | 1,816 |
| Backtest WAPE, gradient-boosted model | 0.646 | 0.669 |
| Backtest WAPE, 4-week average | 0.677 | 0.677 |

The data numbers match exactly. The model's WAPE differs slightly because gradient boosting samples rows
randomly, and Spark splits the data differently on the two machines; the deterministic baseline is identical.
The job took about 16 minutes: pipeline 8.5 min (including waiting for serverless compute), backtest 2.5 min,
forecast 2 min.

**What the first runs taught (each is fixed in the code):**
- **No actions in pipeline functions.** `count()`, `first()` and `collect()` inside a Lakeflow table function
  fail, because upstream tables don't exist yet when the graph is planned. The calendar bounds, RFM as-of date,
  full-week sequence and data-quality metrics are now pure transformations.
- **`TIMESTAMP_NTZ` needs a Delta table feature.** Every pipeline table sets
  `delta.feature.timestampNtz = supported`.
- **No caching on serverless.** `try_cache()` skips it there.
- **Spark ML models on serverless need a UC volume.** `mlflow.spark.log_model` saves through `dfs_tmpdir`, here
  `/Volumes/tillwise/ml/mlflow_tmp`. The forecast table is written before registration, so a registry problem
  can't block the forecast.

## Run it

**Local (tests and a full run on your laptop)** — needs Java 17+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups
uv run pytest                                        # unit tests on local Spark
uv run python -m tillwise.ingest.download_sources    # ~45 MB download into data/raw/
uv run python scripts/run_local.py                   # bronze → gold, prints the results
uv run python scripts/run_forecast_local.py          # demand, backtest, RFM, 8-week forecast (MLflow in ./mlruns)

# the agent
uv sync --all-groups --extra agent
uv run python scripts/export_gold_local.py           # gold tables → data/gold/*.parquet (~8 min)
cp .env.example .env                                 # add GROQ_API_KEY and OPENROUTER_API_KEY
uv run uvicorn tillwise.agent.app:app --reload       # http://localhost:8000/docs
uv run python evals/run_eval.py                      # 30-question eval → evals/reports/latest.md
```

**Databricks Free Edition**

1. Create a Git folder from this repository (Workspace → Create → Git folder).
2. Run `databricks/00_setup_unity_catalog.py` (set the `catalog` widget to `workspace` if creating a
   catalog isn't allowed).
3. Upload the CSVs from `data/raw/` to the landing Volume, as described in `data/README.md`.
4. Either run the notebooks `01_bronze_autoloader`, `02_silver`, `03_gold` by hand, **or** deploy the
   pipeline and job:
   `databricks bundle deploy -t dev` then `databricks bundle run tillwise_daily -t dev`. If bundles
   aren't available, create the pipeline and job in the UI with the settings in `resources/`.
   Use one route or the other: a Lakeflow pipeline can't take over tables the notebooks already
   created, so drop those first if you switch.
5. Build the dashboard from `dashboards/retail_overview.sql` (seven queries), then export it to
   `dashboards/retail_overview.lvdash.json`.

## Roadmap

- [x] **Week 1** — Unity Catalog setup, Auto Loader bronze, silver rules with quarantine and metrics,
      SCD2 products, gold star schema, unit tests, CI
- [x] **Week 2** — Lakeflow pipeline with an expectation per rule, Databricks Job and Asset Bundle, RFM
      segments, weekly demand features, direct multi-horizon GBT with backtest vs three baselines, MLflow
      tracking, UC model registration, p10/p90 intervals, dashboard queries
- [x] **Run in Databricks Free Edition** — pipeline, job, forecast and UC model registration all succeeded
- [ ] Take screenshots (pipeline graph, expectation metrics, MLflow run, job run) and build the AI/BI dashboard
- [x] **Week 3** — FastAPI + LangGraph agent with nine read-only tools, number grounding, Groq → OpenRouter
      fallback, DuckDB and Databricks SQL backends, 30-question eval (30/30), demo script
      ([docs/demo_script.md](docs/demo_script.md))
- [ ] **Record the 3-minute demo video** and post it ([docs/linkedin_post.md](docs/linkedin_post.md))

## Data and licences

See [`data/README.md`](data/README.md). The retail data is *Online Retail II* (UCI, CC BY 4.0).
