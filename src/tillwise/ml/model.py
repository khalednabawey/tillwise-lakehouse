"""Gradient-boosted demand model (Spark ML), baselines, metrics and a time-based backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from tillwise.ml.dataset import FEATURES

DEFAULT_PARAMS = {"maxIter": 80, "maxDepth": 6, "stepSize": 0.1, "subsamplingRate": 0.8, "maxBins": 64,
                  "seed": 42}


def build_pipeline(params: dict | None = None) -> Pipeline:
    p = {**DEFAULT_PARAMS, **(params or {})}
    return Pipeline(stages=[
        StringIndexer(inputCols=["stock_code", "country"], outputCols=["stock_idx", "country_idx"],
                      handleInvalid="keep"),
        VectorAssembler(inputCols=FEATURES, outputCol="features"),
        # log1p target: errors on small and large sellers count more evenly while training
        GBTRegressor(featuresCol="features", labelCol="log_label", predictionCol="log_pred", **p),
    ])


def fit(train: DataFrame, params: dict | None = None) -> PipelineModel:
    return build_pipeline(params).fit(train.withColumn("log_label", F.log1p("label")))


def predict(model: PipelineModel, frame: DataFrame) -> DataFrame:
    out = model.transform(frame)
    return out.withColumn("prediction", F.greatest(F.expm1("log_pred"), F.lit(0.0))).drop(
        "features", "log_pred", "stock_idx", "country_idx")


def add_baselines(frame: DataFrame) -> DataFrame:
    """Seasonal naive (same week last year, or 13-week mean without history), 4-week mean, last value."""
    return (frame.withColumn("pred_seasonal_naive", F.col("units_same_week_last_year"))
            .withColumn("pred_ma4", F.col("ma4"))
            .withColumn("pred_last_value", F.col("last_units")))


def wape_and_bias(df: DataFrame, pred_col: str, label_col: str = "label") -> tuple[float, float]:
    """WAPE = sum|error| / sum actual; bias = sum(error) / sum actual (positive = over-forecast)."""
    r = df.agg(F.sum(F.abs(F.col(pred_col) - F.col(label_col))).alias("ae"),
               F.sum(F.col(pred_col) - F.col(label_col)).alias("e"),
               F.sum(label_col).alias("y")).first()
    y = float(r["y"] or 0.0)
    if y == 0:
        return float("nan"), float("nan")
    return float(r["ae"]) / y, float(r["e"]) / y


@dataclass
class BacktestResult:
    cutoff_idx: int
    metrics: dict[str, dict[str, float]]
    by_horizon: DataFrame
    predictions: DataFrame
    model: PipelineModel
    residual_quantiles: dict[int, tuple[float, float]] = field(default_factory=dict)


def backtest(frame: DataFrame, test_weeks: int = 12, params: dict | None = None) -> BacktestResult:
    """Train on targets up to the cutoff week, forecast weeks cutoff+1..cutoff+H from the cutoff origin.

    The cutoff is the last week minus `test_weeks`, so the test window is the most recent
    `test_weeks` of data, which the model never sees during training.
    """
    last_idx = frame.agg(F.max("target_idx")).first()[0]
    cutoff = last_idx - test_weeks
    train = frame.where(F.col("target_idx") <= cutoff)
    test = frame.where((F.col("origin_idx") == cutoff) & (F.col("h") <= test_weeks))

    model = fit(train, params)
    preds = add_baselines(predict(model, test)).cache()

    metrics = {}
    for name, col in [("gbt", "prediction"), ("seasonal_naive", "pred_seasonal_naive"),
                      ("ma4", "pred_ma4"), ("last_value", "pred_last_value")]:
        wape, bias = wape_and_bias(preds, col)
        metrics[name] = {"wape": round(wape, 4), "bias": round(bias, 4)}

    by_h = preds.groupBy("h").agg(
        (F.sum(F.abs(F.col("prediction") - F.col("label"))) / F.sum("label")).alias("wape_gbt"),
        (F.sum(F.abs(F.col("pred_seasonal_naive") - F.col("label"))) / F.sum("label")).alias("wape_seasonal"),
    ).orderBy("h")

    # Empirical 10th/90th percentiles of the *relative* error per horizon -> forecast intervals.
    # Relative (error / (prediction + 1)) so bands scale with the series: pooled absolute errors
    # are dominated by the many small series and give big series bands that are far too narrow.
    q = preds.withColumn("resid", (F.col("label") - F.col("prediction")) / (F.col("prediction") + 1)).groupBy("h").agg(
        F.percentile_approx("resid", 0.1).alias("q10"), F.percentile_approx("resid", 0.9).alias("q90"))
    quantiles = {int(r["h"]): (float(r["q10"]), float(r["q90"])) for r in q.collect()}

    return BacktestResult(cutoff_idx=cutoff, metrics=metrics, by_horizon=by_h, predictions=preds,
                          model=model, residual_quantiles=quantiles)


def with_intervals(preds: DataFrame, quantiles: dict[int, tuple[float, float]]) -> DataFrame:
    """Add p10/p90 bounds from backtest relative errors for each horizon (clipped at zero)."""
    spark = preds.sparkSession
    if not quantiles:
        return preds.withColumn("p10", F.lit(None).cast("double")).withColumn("p90", F.lit(None).cast("double"))
    qdf = spark.createDataFrame([(float(h), lo, hi) for h, (lo, hi) in quantiles.items()],
                                "h double, q10 double, q90 double")
    scale = F.col("prediction") + 1

    def bound(q: str) -> F.Column:
        return F.greatest(F.col("prediction") + F.coalesce(q, F.lit(0.0)) * scale, F.lit(0.0))

    return (preds.join(qdf, "h", "left")
            .withColumn("p10", bound("q10"))
            .withColumn("p90", bound("q90"))
            .drop("q10", "q90"))
