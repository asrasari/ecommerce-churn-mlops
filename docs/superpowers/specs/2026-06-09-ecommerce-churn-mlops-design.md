# Design: E-Commerce Customer Churn — End-to-End ML Lifecycle with MLflow

**Course:** AIN-3009 Delivering AI Applications with MLOps — Bahçeşehir University
**Student:** Asra Sarı (2101640)
**Date:** 2026-06-09
**Project type:** Term Project — Development and Evaluation of a Machine Learning Lifecycle Management System using MLflow

---

## 1. Goal & Rubric Mapping

Build a comprehensive ML system that manages the full lifecycle of a churn-prediction model
using MLflow. Domain: **retail / e-commerce customer churn** (same domain as course HW1/HW2).

The five graded objectives and where each is satisfied:

| # | Objective | Where satisfied |
|---|-----------|-----------------|
| 1 | Experiment Tracking | `train.py` — log params/metrics/tags/artifacts/model per run |
| 2 | Model Training & Tuning | `train.py` (3 baselines) + `tune.py` (Optuna, each trial = a run) |
| 3 | Model Deployment | `serve.py` (batch via `pyfunc`) + `mlflow models serve` (real-time REST) |
| 4 | Performance Monitoring | `monitor.py` — batch scoring, live metrics to MLflow, PSI/KS drift |
| 5 | Model Registry | `register.py` — register + `MlflowClient` Staging→Production transitions |

### Alignment with what the course actually taught
- MLflow Weeks 8–9 pattern reused verbatim: `set_tracking_uri` → `set_experiment` →
  `with start_run():` → `log_params`/`log_metric`/`set_tag` → `infer_signature` →
  `mlflow.sklearn.log_model(..., registered_model_name=...)` → `pyfunc.load_model`.
- Tracking server on **port 8080** (the port used in the lecture examples).
- **Optuna** for tuning (course Week 12 topic).
- **Airflow (TaskFlow API) + docker-compose** orchestration (Weeks 5–7, HW1/HW2, Week-6 lab).
- Explicitly framed as **own extensions** (not lectured): Registry stage transitions,
  drift monitoring, `mlflow models serve` REST deployment.

---

## 2. Architecture & Layers

Five independently runnable/testable units communicating through saved artifacts + the
MLflow tracking server:

1. **Data layer** (`data_prep.py`) — load, clean, split; build a reusable sklearn
   preprocessing `Pipeline`.
2. **Experiment & training layer** (`train.py`, `tune.py`) — train + log baselines; Optuna study.
3. **Registry & serving layer** (`register.py`, `serve.py`) — register best model, stage
   transitions, batch + REST serving.
4. **Monitoring layer** (`monitor.py`) — simulated incoming batches, live metrics, drift.
5. **Orchestration** (`airflow/`) — TaskFlow DAG wrapping the pipeline.

### Repo structure
```
goksin-proje/
├── data/                      # dataset (small, committed) or download script
├── src/
│   ├── config.py              # tracking URI, paths, experiment names, thresholds
│   ├── data_prep.py           # load, clean, split, preprocessing pipeline
│   ├── train.py               # train + log 3 baseline models
│   ├── tune.py                # Optuna study, each trial -> nested MLflow run
│   ├── register.py            # register best model, Staging->Production transitions
│   ├── serve.py               # load Production model, batch predict (pyfunc)
│   └── monitor.py             # batch scoring + PSI/KS drift -> MLflow
├── airflow/
│   ├── dags/churn_pipeline.py
│   └── docker-compose.yml
├── tests/                     # pytest
├── notebooks/                 # optional EDA / demo
├── reports/                   # project report (deliverable)
├── scripts/package.sh         # build the submission zip (<50 MB)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 3. MLflow Setup

Run a local tracking server (single command, all-local, no cloud/Docker for MLflow):

```bash
mlflow server \
  --host 127.0.0.1 --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

- **Tracking server** — satisfies "configure a tracking server".
- **SQLite `mlflow.db`** — single-file metadata DB; satisfies "database for experiment metadata".
- **`./mlartifacts`** — local artifact storage; satisfies "storage space for artifacts".

In code: `mlflow.set_tracking_uri("http://127.0.0.1:8080")`. Experiment names:
`ecommerce-churn` (training/tuning), `ecommerce-churn-monitoring` (live metrics).
Tracking URI read from `config.py` / env var (never hardcoded in Airflow tasks).

---

## 4. Data, Models & Tracking

**Dataset:** E-Commerce Customer Churn — `data/ecommerce_churn.csv`, **3,941 rows, 11
columns** (a cleaned public version of the Kaggle dataset; the prof provided no dataset, so
we selected our own per the brief). ~207 KB, committed to `data/`.

Columns:
- **Numeric (8):** `Tenure`, `WarehouseToHome`, `NumberOfDeviceRegistered`,
  `SatisfactionScore`, `NumberOfAddress`, `Complain`, `DaySinceLastOrder`, `CashbackAmount`
- **Categorical (2):** `PreferedOrderCat`, `MaritalStatus`
- **Target:** `Churn` (0/1), **17.1% positive** (mildly imbalanced)
- **576 missing values present** — makes the imputation step meaningful, not decorative.

Source mirror: https://github.com/delabrilliano/Ecommerce_Churn_Prediction

**Preprocessing** — one `ColumnTransformer` inside a `Pipeline`, reused everywhere:
- Numeric: median impute → `StandardScaler`
- Categorical: most-frequent impute → `OneHotEncoder(handle_unknown="ignore")`
- Preprocessing is wrapped *with* the model so serving accepts **raw** input (no
  train/serve skew).

**Baseline models** (3), each its own run under `ecommerce-churn`:
- Logistic Regression (the lecture's exact pattern)
- Random Forest
- Gradient Boosting

**Logged per run:**
- `log_params` — model hyperparameters
- `log_metric` — accuracy, precision, recall, F1, **ROC-AUC**
- `set_tag` — model family, dataset version
- `infer_signature` + `input_example` → `mlflow.sklearn.log_model(...)`
- Artifacts — confusion matrix PNG, ROC curve PNG, classification report text

**Hyperparameter tuning — Optuna:** a study on the strongest baseline; **each trial =
one nested MLflow run** logging trial params + AUC; best params produce a final logged
model. Satisfies objective 2.

---

## 5. Model Registry

- Registered model name: `ecommerce-churn-model`.
- `register.py` uses `MlflowClient` to: create a new version from the best run →
  transition to **Staging** → validate against a threshold (**AUC ≥ 0.85**, configurable in
  `config.py`) → transition to **Production**, archiving the prior Production version.
- The transitions are printed/logged so the demo shows real lifecycle governance.

---

## 6. Deployment / Serving

Both modes load the **Production** model by registry URI
(`models:/ecommerce-churn-model/Production`):

- **Batch** (`serve.py`): `mlflow.pyfunc.load_model(...)` → `.predict(df)` over a CSV
  (the exact lecture method).
- **Real-time REST**: `mlflow models serve -m models:/ecommerce-churn-model/Production -p 1234`,
  with an example client (`curl` + Python `requests`) posting JSON records and receiving
  churn predictions. No Flask/FastAPI needed — MLflow's native server.

---

## 7. Monitoring & Drift Detection

`monitor.py`:
- Simulates incoming production data as time-ordered batches (sampling/perturbing the
  held-out test set to induce realistic drift).
- Per batch: log live accuracy/F1/AUC to experiment `ecommerce-churn-monitoring` so metric
  drift shows as a trend in the MLflow UI.
- **Drift detection:** per feature, compute **Population Stability Index (PSI)** and a
  **Kolmogorov–Smirnov** test vs. the training baseline; log a `drift_score` and flag
  features breaching a threshold (PSI > 0.2). Pure numpy/scipy — no heavy deps.
- *Optional stretch:* an `evidently` HTML drift report if time allows.

---

## 8. Orchestration — Airflow

- `airflow/docker-compose.yml` runs Airflow (3.x, TaskFlow API) + Postgres metadata DB,
  mirroring the HW2 / Week-6 lab style.
- DAG `churn_pipeline`: `ingest → preprocess → train_and_tune → evaluate → register`,
  `schedule_interval='@daily'` (triggered manually for the demo).
- Tasks call the same `src/` functions — **no logic duplication**; Airflow only orchestrates.
- MLflow tracking URI injected via env var / Airflow Variable (not hardcoded), per the
  conventions HW1/HW2 emphasized.
- Built **last**, after the core MLflow pipeline is proven end-to-end.

---

## 9. Testing & Verification

Lightweight `pytest`:
- `data_prep` produces expected shapes; no target leakage; train/test split deterministic.
- The fitted pipeline round-trips a single raw record to a valid prediction.
- Registry transition logic selects the correct best version given mock runs.
- PSI computes the expected value on a known synthetic shift.

---

## 10. Packaging & Submission (prof's constraints)

- Final zip **< 50 MB** and downloadable, or it cannot be graded.
- **Exclude** `venv/`, `mlruns/`, `mlartifacts/`, `mlflow.db` (grow large per run).
- `scripts/package.sh` produces `PRJ-asrasari-<number>.zip` containing only: `src/`,
  `airflow/`, `tests/`, `data/` (or download script), `requirements.txt`, `README.md`,
  `reports/`. `.gitignore` enforces the same exclusions.
- To show real runs/experiments: upload the `mlruns`/`mlartifacts` to Google Drive and put
  **only the link** in the documentation. Report includes MLflow UI screenshots.

---

## 11. Build Order

So the core deliverable is never at risk:

1. `data_prep.py`
2. `train.py` (3 models logged)
3. `tune.py` (Optuna)
4. `register.py` (stage transitions)
5. `serve.py` (batch + REST)
6. `monitor.py` (drift)
7. Airflow DAG + docker-compose
8. Report + packaging script

---

## 12. Deliverables

1. **Code repository** — this repo (well-commented, PEP 8), pushed to GitHub.
2. **Project report** (`reports/`) — methodology, tools, model development, experiment
   results, registry/serving/monitoring, insights + reflection; MLflow UI screenshots;
   optional Drive link to full runs.
3. **Presentation/demo** — 5–8 min: MLflow UI walkthrough, run comparison, registry
   transition, a live prediction, drift trend.
