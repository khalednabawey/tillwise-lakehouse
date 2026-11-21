# Data

Nothing in `data/raw/` is committed. Rebuild it with:

```bash
uv run python -m tillwise.ingest.download_sources
```

| File | Source | Licence |
|---|---|---|
| `online_retail_ii_2009_2010.csv`, `online_retail_ii_2010_2011.csv` | Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D | CC BY 4.0 |
| `weather_london_daily.csv` | Open-Meteo Historical Weather API (https://open-meteo.com) | CC BY 4.0 |
| `uk_holidays.csv` | `holidays` Python package (England bank holidays) | MIT |

Upload the retail CSVs to `/Volumes/<catalog>/raw/landing/retail/` and the other two to
`/Volumes/<catalog>/raw/landing/reference/` in your Databricks workspace.

The retail data is a real UK online gift retailer's transactions from 1 Dec 2009 to 9 Dec 2011.
It is used here for a non-commercial portfolio project, with the attribution above.
