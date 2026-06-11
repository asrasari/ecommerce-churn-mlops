"""Airflow DAG orchestrating the churn ML pipeline via the src/ modules.

The DAG only orchestrates — every step calls the same functions used standalone
(``src.train``, ``src.tune``, ``src.register``), so there is no duplicated logic.
The MLflow tracking URI is taken from the ``MLFLOW_TRACKING_URI`` env var set in
docker-compose (``http://host.docker.internal:8080``), never hardcoded.
"""
import sys
from datetime import datetime

from airflow.decorators import dag, task

# project source is mounted here and also on PYTHONPATH
sys.path.insert(0, "/opt/airflow/project")


@dag(
    dag_id="churn_pipeline",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "churn"],
)
def churn_pipeline():
    @task
    def ingest():
        from src.data_prep import load_data
        df = load_data()
        return int(len(df))

    @task
    def train_models(row_count: int):
        from src.train import main as train_main
        train_main()
        return "trained"

    @task
    def tune_models(_: str):
        from src.tune import main as tune_main
        tune_main(n_trials=10)
        return "tuned"

    @task
    def register_model(_: str):
        from src.register import main as register_main
        register_main()
        return "registered"

    rows = ingest()
    trained = train_models(rows)
    tuned = tune_models(trained)
    register_model(tuned)


churn_pipeline()
