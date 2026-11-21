"""MLflow logging, kept separate so the core code runs without MLflow installed.

Locally, runs go to ./mlruns (view with `uv run mlflow ui`). On Databricks, runs go to
the workspace experiment and the final model is registered in Unity Catalog.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from tillwise.ml.model import DEFAULT_PARAMS, BacktestResult


def _mlflow():
    try:
        import mlflow

        return mlflow
    except ImportError:
        print("MLflow not installed; skipping tracking.")
        return None


def log_backtest(result: BacktestResult, experiment: str, extra_params: dict | None = None) -> str | None:
    """Log parameters, WAPE/bias for the model and each baseline, and per-horizon errors."""
    mlflow = _mlflow()
    if mlflow is None:
        return None
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="backtest_gbt_direct") as run:
        mlflow.log_params({**DEFAULT_PARAMS, "cutoff_idx": result.cutoff_idx, **(extra_params or {})})
        for name, m in result.metrics.items():
            mlflow.log_metric(f"wape_{name}", m["wape"])
            mlflow.log_metric(f"bias_{name}", m["bias"])
        with tempfile.TemporaryDirectory() as d:
            result.by_horizon.toPandas().to_csv(Path(d) / "wape_by_horizon.csv", index=False)
            (Path(d) / "residual_quantiles.json").write_text(json.dumps(result.residual_quantiles))
            mlflow.log_artifacts(d, artifact_path="backtest")
        return run.info.run_id


def log_and_register(model: PipelineModel, sample: DataFrame, experiment: str,
                     registered_model_name: str | None, alias: str = "champion") -> str | None:
    """Log the final model; on Databricks register it in Unity Catalog and point `alias` at it."""
    mlflow = _mlflow()
    if mlflow is None:
        return None
    import mlflow.spark
    from mlflow.models import infer_signature

    if registered_model_name:
        mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="final_gbt_all_history") as run:
        pdf = sample.limit(50).toPandas()
        inputs = pdf.drop(columns=[c for c in pdf.columns if c.startswith("pred") or c in ("label", "p10", "p90")])
        signature = infer_signature(inputs, pdf["prediction"]) if "prediction" in pdf else None
        info = mlflow.spark.log_model(model, artifact_path="model", signature=signature,
                                      registered_model_name=registered_model_name)
        if registered_model_name:
            from mlflow import MlflowClient

            version = info.registered_model_version
            MlflowClient().set_registered_model_alias(registered_model_name, alias, version)
            print(f"Registered {registered_model_name} v{version} as @{alias}")
        return run.info.run_id
