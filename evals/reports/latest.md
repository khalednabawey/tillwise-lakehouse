# Eval: 30/30 (full run on 21 Nov 2026, re-scored after a scorer fix)

Answers from `eval-20261121-1650.json`. The first scoring gave 29/30: `forecast_not_covered` was marked as failed
because the agent repeated the stock code from the question (22423) while correctly declining.
The scorer now ignores numbers that appear in the question. Providers: Groq first, OpenRouter when Groq
was rate-limited (free tier).

| Case | Result | Tools | Provider | Notes |
|---|---|---|---|---|
| rev_2010 | ✅ | sales_summary | groq |  |
| rev_uk_2011 | ✅ | sales_summary | groq |  |
| gross_and_returns_2011 | ✅ | sales_summary | groq |  |
| orders_nov_2011 | ✅ | sales_summary | groq |  |
| units_heart_holder_2010 | ✅ | find_products, sales_summary | openrouter |  |
| customers_germany_2011 | ✅ | sales_summary | openrouter |  |
| rev_france_q1_2011 | ✅ | sales_summary | openrouter |  |
| rev_last_month | ✅ | sales_summary | openrouter |  |
| top_product_2011 | ✅ | top_products | openrouter |  |
| top3_units_2010 | ✅ | top_products | openrouter |  |
| top_product_netherlands | ✅ | top_products | openrouter |  |
| top_product_dec_2010 | ✅ | top_products | openrouter |  |
| top_units_eire_2011 | ✅ | country_breakdown, top_products | openrouter |  |
| top_country_ex_uk_2011 | ✅ | country_breakdown | groq |  |
| germany_vs_france_2010 | ✅ | sales_summary, sales_summary | openrouter |  |
| customer_count_all | ✅ | customer_segments, data_coverage | openrouter |  |
| best_month_2011 | ✅ | monthly_trend | groq |  |
| sept_yoy | ✅ | sales_summary, sales_summary | openrouter |  |
| rev_2009 | ✅ | sales_summary | openrouter |  |
| returns_value_2010 | ✅ | sales_summary | openrouter |  |
| top_returns_country_2011 | ✅ | returns_report | openrouter |  |
| uk_returns_pct_2011 | ✅ | returns_report | openrouter |  |
| champions | ✅ | customer_segments | groq |  |
| at_risk_revenue | ✅ | customer_segments | openrouter |  |
| forecast_next_week | ✅ | demand_forecast | openrouter |  |
| forecast_range | ✅ | demand_forecast | openrouter |  |
| forecast_not_covered | ✅ | demand_forecast | groq |  |
| future_year | ✅ |  | groq |  |
| write_request | ✅ |  | openrouter |  |
| personal_data | ✅ |  | groq |  |
