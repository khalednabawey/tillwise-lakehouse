-- Tillwise · Retail overview dashboard (Databricks AI/BI)
-- Create a dashboard, add one dataset per query below (replace `tillwise` with your catalog),
-- then export it to dashboards/retail_overview.lvdash.json and commit it.

-- 1. Weekly net revenue and order count (line chart)
SELECT d.week_start,
       round(sum(f.line_revenue), 2)                                   AS net_revenue,
       count(DISTINCT CASE WHEN NOT f.is_return THEN f.invoice_no END) AS orders
FROM tillwise.gold.fact_sales f
JOIN tillwise.gold.dim_date d USING (date_key)
GROUP BY d.week_start
ORDER BY d.week_start;

-- 2. Top 15 products by net revenue, with the current description (bar chart)
SELECT p.description, f.stock_code, round(sum(f.line_revenue), 2) AS net_revenue, sum(f.quantity) AS net_units
FROM tillwise.gold.fact_sales f
JOIN tillwise.gold.dim_product p ON p.stock_code = f.stock_code AND p.is_current
GROUP BY p.description, f.stock_code
ORDER BY net_revenue DESC
LIMIT 15;

-- 3. Return rate by country, countries with 1,000+ lines (table)
SELECT c.country,
       count(*)                                                        AS lines,
       round(100 * count_if(f.is_return) / count(*), 2)                AS return_line_pct,
       round(-sum(CASE WHEN f.is_return THEN f.line_revenue END), 2)   AS returned_value
FROM tillwise.gold.fact_sales f
JOIN tillwise.gold.dim_country c USING (country_key)
GROUP BY c.country
HAVING count(*) >= 1000
ORDER BY return_line_pct DESC;

-- 4. Customer segments (bar chart: customers and revenue share per segment)
SELECT segment, count(*) AS customers, round(sum(monetary), 2) AS revenue,
       round(100 * sum(monetary) / sum(sum(monetary)) OVER (), 1) AS revenue_share_pct
FROM tillwise.gold.customer_rfm
GROUP BY segment
ORDER BY revenue DESC;

-- 5. Backtest: forecast vs actual, all top series summed (line chart)
SELECT target_week,
       sum(label)               AS actual_units,
       round(sum(prediction))   AS gbt_forecast,
       round(sum(pred_seasonal_naive)) AS seasonal_naive
FROM tillwise.gold.forecast_backtest
GROUP BY target_week
ORDER BY target_week;

-- 6. Next 8 weeks forecast with bounds, for a product picked by a filter (line chart with band)
SELECT target_week, stock_code, country, forecast_units, p10, p90
FROM tillwise.gold.forecast_weekly
ORDER BY stock_code, country, target_week;

-- 7. Data-quality outcomes from the last pipeline update (table)
SELECT metric, row_count, pct_of_input
FROM tillwise.silver.dq_metrics
ORDER BY row_count DESC;
