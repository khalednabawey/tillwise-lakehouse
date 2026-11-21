# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Create the Lakeflow pipeline and the daily job (from inside the workspace)
# MAGIC The same settings as `resources/*.yml` (the Asset Bundle), created with the Databricks SDK, which
# MAGIC authenticates automatically inside a notebook. Use this when you don't have the Databricks CLI set
# MAGIC up locally. Safe to re-run: it updates the existing pipeline and job by name.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service import pipelines as pl

w = WorkspaceClient()
repo_fs = os.path.abspath(os.path.join(os.getcwd(), ".."))          # /Workspace/Users/<you>/tillwise-lakehouse
repo_ws = repo_fs.removeprefix("/Workspace")                         # /Users/<you>/tillwise-lakehouse
print("Repo:", repo_fs)

PIPELINE_NAME, JOB_NAME = "tillwise-medallion", "tillwise-daily"
spec = dict(
    name=PIPELINE_NAME,
    catalog=CATALOG,
    schema="silver",
    serverless=True,
    channel="CURRENT",
    development=True,
    libraries=[pl.PipelineLibrary(file=pl.FileLibrary(path=f"{repo_ws}/databricks/pipelines/tillwise_pipeline.py"))],
    configuration={
        "tillwise.catalog": CATALOG,
        "tillwise.src_path": f"{repo_fs}/src",
        "tillwise.landing_path": T.landing_path,
    },
)
existing = [p for p in w.pipelines.list_pipelines(filter=f"name LIKE '{PIPELINE_NAME}'")]
if existing:
    pipeline_id = existing[0].pipeline_id
    w.pipelines.update(pipeline_id=pipeline_id, **spec)
    print("Updated pipeline", pipeline_id)
else:
    pipeline_id = w.pipelines.create(**spec).pipeline_id
    print("Created pipeline", pipeline_id)

# COMMAND ----------

tasks = [
    jobs.Task(task_key="refresh_pipeline", pipeline_task=jobs.PipelineTask(pipeline_id=pipeline_id)),
    jobs.Task(task_key="backtest_forecast", depends_on=[jobs.TaskDependency(task_key="refresh_pipeline")],
              notebook_task=jobs.NotebookTask(notebook_path=f"{repo_ws}/databricks/04_train_forecast",
                                              base_parameters={"catalog": CATALOG}, source=jobs.Source.WORKSPACE)),
    jobs.Task(task_key="forecast_next_weeks", depends_on=[jobs.TaskDependency(task_key="backtest_forecast")],
              notebook_task=jobs.NotebookTask(notebook_path=f"{repo_ws}/databricks/05_forecast_inference",
                                              base_parameters={"catalog": CATALOG}, source=jobs.Source.WORKSPACE)),
]
job_settings = dict(name=JOB_NAME, tasks=tasks, max_concurrent_runs=1,
                    schedule=jobs.CronSchedule(quartz_cron_expression="0 0 6 * * ?", timezone_id="Africa/Cairo",
                                               pause_status=jobs.PauseStatus.PAUSED))
found = [j for j in w.jobs.list(name=JOB_NAME)]
if found:
    job_id = found[0].job_id
    w.jobs.reset(job_id=job_id, new_settings=jobs.JobSettings(**job_settings))
    print("Updated job", job_id)
else:
    job_id = w.jobs.create(**job_settings).job_id
    print("Created job", job_id)

host = w.config.host
displayHTML(f'<a href="{host}/pipelines/{pipeline_id}" target="_blank">Open pipeline</a> · '
            f'<a href="{host}/jobs/{job_id}" target="_blank">Open job</a>')

# COMMAND ----------

# MAGIC %md Run the whole job now (pipeline → backtest → forecast). It takes a while on Free Edition.

# COMMAND ----------

run = w.jobs.run_now(job_id=job_id)
print("Started run", run.run_id, f"→ {host}/jobs/{job_id}/runs/{run.run_id}")
