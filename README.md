# E-Commerce Customer Churn — End-to-End MLOps with MLflow

**Course:** AIN-3009 Delivering AI Applications with MLOps — Bahçeşehir University
**Author:** Asra Sarı (Student No. 2101640)

An end-to-end machine-learning lifecycle system for predicting e-commerce customer
churn, built around **MLflow**: experiment tracking, model training, Optuna
hyperparameter tuning, a Model Registry with stage transitions, model serving, drift
monitoring, and an **Airflow** orchestration DAG.

## Rubric objectives → where they live

| # | Objective | Module |
|---|-----------|--------|
| 1 | Experiment Tracking | `src/train.py` — logs params, metrics, tags, plots, model per run |
| 2 | Model Training & Tuning | `src/train.py` (3 baselines) + `src/tune.py` (Optuna, each trial a run) |
| 3 | Model Deployment | `src/serve.py` (batch) + `src/serve_api.py` (real-time REST) |
| 4 | Performance Monitoring | `src/monitor.py` — live metrics + PSI/KS drift over batches |
| 5 | Model Registry | `src/register.py` — register + Staging→Production transitions |

## Dataset

`data/ecommerce_churn.csv` — 3,941 rows, 11 columns (8 numeric, 2 categorical, binary
`Churn` target, 17.1% positive, 576 missing values). A cleaned public version of the
Kaggle "E-Commerce Customer Churn" dataset (the course brief requires students to choose
their own dataset).

## Project structure

```
src/            config, data_prep, train, tune, register, serve, serve_api, monitor
tests/          pytest unit tests (data prep, metrics, PSI, best-run selection)
airflow/        docker-compose.yml + dags/churn_pipeline.py (TaskFlow DAG)
reports/        project report + figures
scripts/        package.sh (builds the <50 MB submission zip)
docs/           design spec + implementation plan
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 1. Start the MLflow tracking server

A local tracking server with a SQLite metadata DB and a local artifact store
(satisfies "tracking server + database + artifact storage"):

```bash
mlflow server --host 127.0.0.1 --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

Open the UI at **http://127.0.0.1:8080**.

## 2. Run the pipeline (in a second terminal)

```bash
source venv/bin/activate
python -m src.train       # train + log 3 baseline models
python -m src.tune        # Optuna tuning (20 trials, nested runs)
python -m src.register    # register best model, Staging -> Production
python -m src.serve --output predictions.csv   # batch predictions
python -m src.monitor     # 5 simulated batches + drift detection
```

## 3. Real-time serving (REST)

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:8080
python -m src.serve_api    # Flask service on http://127.0.0.1:1234
```

```bash
curl -X POST http://127.0.0.1:1234/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["Tenure","WarehouseToHome","NumberOfDeviceRegistered","SatisfactionScore","NumberOfAddress","Complain","DaySinceLastOrder","CashbackAmount","PreferedOrderCat","MaritalStatus"], "data": [[1,29,4,5,9,1,0,120.5,"Mobile Phone","Single"]]}}'
# -> {"predictions": [1]}
```

> **Note on serving:** MLflow's built-in `mlflow models serve` ships a scoring server
> that is incompatible with the modern Starlette/FastAPI versions resolved on Python
> 3.13 (it calls `@app.route`, removed in current Starlette). We therefore serve the
> exact same registered **Production** model through a small, version-proof Flask app
> (`src/serve_api.py`) that mirrors MLflow's `/invocations` JSON contract.

## 4. Tests

```bash
pytest -v
```

## 5. Airflow orchestration (optional, Docker)

A TaskFlow DAG runs the pipeline `ingest → train → tune → register`. The MLflow server
must be running on the host first, **bound to `0.0.0.0`** so the Airflow containers can
reach it via `host.docker.internal` (a `127.0.0.1` bind is only reachable locally):

```bash
mlflow server --host 0.0.0.0 --port 8080 \
  --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

```bash
cd airflow
docker compose up airflow-init
docker compose up -d
# Airflow UI at http://localhost:8088  (login: airflow / airflow)
# MLflow stays on 8080; Airflow's webserver is remapped to 8088 to avoid a clash.
```

Unpause and trigger `churn_pipeline` from the UI, or:

```bash
docker compose exec airflow-scheduler airflow dags unpause churn_pipeline
docker compose exec airflow-scheduler airflow dags trigger churn_pipeline
```

## Submission packaging

```bash
./scripts/package.sh <student-number>   # builds PRJ-asrasari-<number>.zip (<50 MB)
```

The zip **excludes** `venv/`, `mlruns/`, `mlartifacts/`, and `mlflow.db` (they grow large
per run, per the submission rules). The full MLflow runs/experiments are uploaded
separately to Google Drive — link in `reports/REPORT.md`.
