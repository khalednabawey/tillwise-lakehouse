# Tillwise · 3-minute demo script

Record your screen (OBS or the Windows Game Bar, Win+G) at 1080p. Speak slowly; one idea per scene.

| Time | Screen | What to say |
|---|---|---|
| 0:00–0:20 | README top + architecture diagram | "Tillwise is a retail sales lakehouse on Databricks. A million real transactions go through bronze, silver and gold, and on top there's a demand forecast and an agent that answers business questions." |
| 0:20–0:50 | README data-quality table | "Nothing is dropped silently. Each row ends in exactly one place, and the counts add up. The interesting one: the source's two sheets overlap for a week in December 2010. De-duplication caught 22,000 double-counted rows." |
| 0:50–1:15 | `tillwise_pipeline.py`, or the Lakeflow graph screenshot once it has run | "It's one Lakeflow pipeline. Every rule is a named expectation, so the pipeline UI shows pass and fail counts per rule. Products are SCD Type 2, so old sales keep the product name they had at the time." |
| 1:15–1:50 | README forecast table, then the `mlflow ui` run | "One gradient-boosted model forecasts 1 to 12 weeks ahead. In a backtest on the last 12 weeks it beats three baselines. The honest part: the WAPE gain is small, but the bias drops from about minus 40% to minus 3% going into Christmas, and for stock planning that matters more." |
| 1:50–2:40 | Terminal: `uv run uvicorn tillwise.agent.app:app`, then Swagger at `localhost:8000/docs`; ask 3 questions | 1) "Which product made the most net revenue in 2011?" 2) "What's the forecast for 85099B in the UK next week?" 3) "Delete all the returns." Say: "It can only call fixed, read-only tools. Every number in the answer has to come from a tool result, or it's sent back once to fix it. And it refuses writes." |
| 2:40–3:00 | `evals/reports/latest.md` and the green CI badge | "There's a 30-question eval with answers computed straight from the data. It scores 30 out of 30. Unit tests run in CI on every push. Code is on GitHub: link below." |

**Before recording:** run `uv run python scripts/export_gold_local.py` once, start the API, warm it up with one
question (the first call is slower), and close other windows.
