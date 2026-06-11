# E-Commerce Churn MLOps — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end ML lifecycle system for e-commerce customer churn using MLflow — experiment tracking, training, Optuna tuning, model registry with stage transitions, serving, drift monitoring, and an Airflow orchestration DAG.

**Architecture:** A `src/` package of focused, independently-runnable modules (`config`, `data_prep`, `train`, `tune`, `register`, `serve`, `monitor`) that communicate through saved artifacts and a local MLflow tracking server (SQLite backend + local artifact dir, port 8080). An Airflow TaskFlow DAG wraps the pipeline by calling the same `src/` functions. Tests cover pure logic (preprocessing shapes, metrics, PSI, best-run selection); MLflow-integration steps use run-and-observe verification.

**Tech Stack:** Python 3.10–3.12, MLflow 2.x (stages require 2.x), scikit-learn, pandas, numpy, scipy, Optuna, matplotlib, pytest; Apache Airflow 3.x via docker-compose.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/__init__.py` | Make `src` an importable package |
| `src/config.py` | All constants: tracking URI, paths, experiment names, feature lists, thresholds |
| `src/data_prep.py` | Load CSV, split train/test, build sklearn preprocessing `ColumnTransformer` |
| `src/train.py` | Metrics helper, plot helpers, build full pipeline, train + log 3 baseline models |
| `src/tune.py` | Optuna study (each trial = nested MLflow run), log + register best |
| `src/register.py` | Find best run, register model, `MlflowClient` Staging→Production transitions |
| `src/serve.py` | Load Production model via pyfunc, batch-predict a CSV |
| `src/monitor.py` | PSI/KS drift vs. baseline, simulate batches, log live metrics to MLflow |
| `tests/test_data_prep.py` | Split shapes, no leakage, preprocessor output |
| `tests/test_train.py` | Metrics correctness, pipeline round-trip |
| `tests/test_monitor.py` | PSI on known shift, drift flagging |
| `tests/test_register.py` | Best-run selection logic |
| `airflow/dags/churn_pipeline.py` | TaskFlow DAG: ingest → preprocess → train_and_tune → evaluate → register |
| `airflow/docker-compose.yml` | Airflow 3.x + Postgres |
| `scripts/package.sh` | Build the `<50 MB` submission zip |
| `requirements.txt` | Python deps |
| `README.md` | Setup + run instructions |
| `conftest.py` | Add project root to `sys.path` for tests |

---

## Task 0: Project Scaffolding & Environment

**Files:**
- Create: `requirements.txt`, `src/__init__.py`, `src/config.py`, `conftest.py`, `tests/__init__.py`

- [ ] **Step 1: Write `requirements.txt`**

```
mlflow>=2.16,<3
scikit-learn>=1.4,<1.6
pandas>=2.1,<2.3
numpy>=1.26,<2.2
scipy>=1.11,<1.15
optuna>=3.6,<4.2
matplotlib>=3.8,<3.10
pytest>=8.0,<9
```

- [ ] **Step 2: Create venv and install**

Run:
```bash
cd /Users/asra/goksin-proje
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Expected: all packages install without conflict. Confirm with `python -c "import mlflow, sklearn, optuna, scipy; print(mlflow.__version__)"` → prints a 2.x version.

- [ ] **Step 3: Create `src/__init__.py` and `tests/__init__.py` (both empty)**

```python
```

- [ ] **Step 4: Write `conftest.py`** (lets `from src... import` work in pytest from repo root)

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
```

- [ ] **Step 5: Write `src/config.py`**

```python
"""Central configuration for the e-commerce churn MLOps project."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_PATH = PROJECT_ROOT / "data" / "ecommerce_churn.csv"

# MLflow
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080")
EXPERIMENT_NAME = "ecommerce-churn"
MONITORING_EXPERIMENT = "ecommerce-churn-monitoring"
REGISTERED_MODEL_NAME = "ecommerce-churn-model"

# Schema
TARGET = "Churn"
NUMERIC_FEATURES = [
    "Tenure", "WarehouseToHome", "NumberOfDeviceRegistered",
    "SatisfactionScore", "NumberOfAddress", "Complain",
    "DaySinceLastOrder", "CashbackAmount",
]
CATEGORICAL_FEATURES = ["PreferedOrderCat", "MaritalStatus"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Registry promotion gate
PRIMARY_METRIC = "roc_auc"
AUC_THRESHOLD = 0.85
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/__init__.py src/config.py conftest.py tests/__init__.py
git commit -m "Scaffold project: config, requirements, test setup"
```

---

## Task 1: Data Loading & Splitting

**Files:**
- Create: `src/data_prep.py`
- Test: `tests/test_data_prep.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_prep.py
import pandas as pd
from src import config
from src.data_prep import load_data, split_data


def test_load_data_has_expected_columns():
    df = load_data()
    assert config.TARGET in df.columns
    for col in config.FEATURES:
        assert col in df.columns, f"missing {col}"
    assert len(df) > 3000


def test_split_data_shapes_and_no_leakage():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    # X has only feature columns, not the target
    assert config.TARGET not in X_train.columns
    assert list(X_train.columns) == config.FEATURES
    # split sizes line up
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_test) == len(X_test)
    # stratification keeps churn rate roughly equal in both splits
    assert abs(y_train.mean() - y_test.mean()) < 0.03
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_prep.py -v`
Expected: FAIL — `ModuleNotFoundError` / cannot import `load_data`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/data_prep.py
"""Load, split, and preprocess the e-commerce churn dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_data(path=config.DATA_PATH):
    """Read the churn CSV into a DataFrame."""
    return pd.read_csv(path)


def split_data(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):
    """Split into stratified train/test of features and target."""
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_prep.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/data_prep.py tests/test_data_prep.py
git commit -m "Add data loading and stratified splitting"
```

---

## Task 2: Preprocessing Pipeline

**Files:**
- Modify: `src/data_prep.py`
- Test: `tests/test_data_prep.py`

- [ ] **Step 1: Add the failing test**

```python
# append to tests/test_data_prep.py
from src.data_prep import build_preprocessor


def test_preprocessor_handles_missing_and_encodes():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X_train)
    # transformed matrix has no NaNs (imputation worked)
    import numpy as np
    arr = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert not np.isnan(arr).any()
    # one-hot expands columns beyond the raw feature count
    assert arr.shape[1] > len(config.FEATURES)
    # transform on unseen test data does not error (handle_unknown)
    pre.transform(X_test)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data_prep.py::test_preprocessor_handles_missing_and_encodes -v`
Expected: FAIL — cannot import `build_preprocessor`.

- [ ] **Step 3: Add the implementation**

```python
# append to src/data_prep.py
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor():
    """ColumnTransformer: median-impute+scale numerics, mode-impute+one-hot categoricals."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data_prep.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/data_prep.py tests/test_data_prep.py
git commit -m "Add reusable preprocessing ColumnTransformer"
```

---

## Task 3: Metrics & Plot Helpers

**Files:**
- Create: `src/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import numpy as np
from src.train import compute_metrics


def test_compute_metrics_perfect_prediction():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.8, 0.2])
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL — cannot import `compute_metrics`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/train.py
"""Train baseline churn models and log them to MLflow."""
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba):
    """Return the standard classification metric dict."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def save_plots(model, X_test, y_test, out_dir):
    """Write confusion-matrix and ROC-curve PNGs; return their paths."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cm_path = out / "confusion_matrix.png"
    roc_path = out / "roc_curve.png"

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    return str(cm_path), str(roc_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "Add metrics and plot helpers"
```

---

## Task 4: Full Pipeline & Round-Trip Test

**Files:**
- Modify: `src/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Add the failing test**

```python
# append to tests/test_train.py
from sklearn.linear_model import LogisticRegression
from src.data_prep import load_data, split_data
from src.train import build_pipeline


def test_pipeline_fits_and_predicts_from_raw_input():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)
    # predicts straight from raw (un-preprocessed) rows
    preds = pipe.predict(X_test.head(5))
    assert len(preds) == 5
    proba = pipe.predict_proba(X_test.head(5))
    assert proba.shape == (5, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py::test_pipeline_fits_and_predicts_from_raw_input -v`
Expected: FAIL — cannot import `build_pipeline`.

- [ ] **Step 3: Add the implementation**

```python
# append to src/train.py
from sklearn.pipeline import Pipeline as SkPipeline
from src.data_prep import build_preprocessor


def build_pipeline(model):
    """Wrap preprocessing + estimator so the model consumes raw input."""
    return SkPipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "Add full preprocessing+model pipeline builder"
```

---

## Task 5: Train & Log Three Baseline Models to MLflow

**Files:**
- Modify: `src/train.py`

> This task is verified by running against a live MLflow server (not a unit test).

- [ ] **Step 1: Add the training-run function and `main`**

```python
# append to src/train.py
import tempfile
import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src import config


def _model_zoo():
    """Name -> (estimator, param dict to log)."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            {"max_iter": 1000},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE),
            {"n_estimators": 200},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=config.RANDOM_STATE),
            {"n_estimators": 100, "learning_rate": 0.1},
        ),
    }


def train_and_log(name, estimator, params, X_train, X_test, y_train, y_test):
    """Fit one pipeline, log params/metrics/artifacts/model under a named run."""
    with mlflow.start_run(run_name=name):
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_params(params)
        mlflow.set_tag("model_family", name)
        mlflow.set_tag("dataset", "ecommerce_churn_v1")
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        with tempfile.TemporaryDirectory() as tmp:
            cm_path, roc_path = save_plots(pipe, X_test, y_test, tmp)
            mlflow.log_artifact(cm_path, artifact_path="plots")
            mlflow.log_artifact(roc_path, artifact_path="plots")

        signature = infer_signature(X_train, pipe.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
        )
        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        return metrics


def main():
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    for name, (estimator, params) in _model_zoo().items():
        train_and_log(name, estimator, params, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Start the MLflow tracking server (separate terminal, leave running)**

Run:
```bash
cd /Users/asra/goksin-proje && source venv/bin/activate
mlflow server --host 127.0.0.1 --port 8080 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```
Expected: logs "Listening at: http://127.0.0.1:8080". Open http://127.0.0.1:8080 in a browser.

- [ ] **Step 3: Run training**

Run (in the working terminal): `python -m src.train`
Expected: three lines printed (logistic_regression, random_forest, gradient_boosting) each with metrics; roc_auc values likely 0.85–0.99.

- [ ] **Step 4: Verify in the MLflow UI**

In the browser at http://127.0.0.1:8080 → experiment `ecommerce-churn`: confirm 3 runs, each with params, 5 metrics, a `model/` artifact, and `plots/` PNGs. Use the "Compare" button to view them side by side.

- [ ] **Step 5: Commit**

```bash
git add src/train.py
git commit -m "Train and log 3 baseline models to MLflow"
```

---

## Task 6: Hyperparameter Tuning with Optuna

**Files:**
- Create: `src/tune.py`

> Verified against the live MLflow server.

- [ ] **Step 1: Write `src/tune.py`**

```python
"""Optuna hyperparameter tuning for the churn model; each trial = a nested MLflow run."""
import mlflow
import optuna
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier

from src import config
from src.data_prep import load_data, split_data
from src.train import build_pipeline, compute_metrics


def _objective(trial, X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
    }
    with mlflow.start_run(nested=True):
        model = GradientBoostingClassifier(random_state=config.RANDOM_STATE, **params)
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_params(params)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.set_tag("phase", "tuning")
    return metrics[config.PRIMARY_METRIC]


def main(n_trials=20):
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    with mlflow.start_run(run_name="optuna-tuning"):
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda t: _objective(t, X_train, X_test, y_train, y_test),
            n_trials=n_trials,
        )
        best = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric(f"best_{config.PRIMARY_METRIC}", study.best_value)

        # fit + log the final tuned model so it is registrable
        model = GradientBoostingClassifier(random_state=config.RANDOM_STATE, **best)
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        signature = infer_signature(X_train, pipe.predict(X_train))
        mlflow.set_tag("model_family", "gradient_boosting_tuned")
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
        )
        # log the same metrics at top level so it competes in best-run search
        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)
        for key, value in compute_metrics(y_test, y_pred, y_proba).items():
            mlflow.log_metric(key, value)
        print(f"Best {config.PRIMARY_METRIC}={study.best_value:.4f} with {best}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tuning** (MLflow server still running)

Run: `python -m src.tune`
Expected: Optuna prints 20 trial logs, then `Best roc_auc=0.9x with {...}`.

- [ ] **Step 3: Verify in the MLflow UI**

Experiment `ecommerce-churn` → the `optuna-tuning` parent run expands to 20 nested runs. Open the parent → it has `best_*` params, a `model/` artifact, and top-level metrics.

- [ ] **Step 4: Commit**

```bash
git add src/tune.py
git commit -m "Add Optuna tuning with nested MLflow runs"
```

---

## Task 7: Best-Run Selection (unit-tested) + Registry Transitions

**Files:**
- Create: `src/register.py`
- Test: `tests/test_register.py`

- [ ] **Step 1: Write the failing test for best-run selection**

```python
# tests/test_register.py
from src.register import pick_best


class _FakeRun:
    def __init__(self, run_id, auc):
        self.info = type("I", (), {"run_id": run_id})()
        self.data = type("D", (), {"metrics": {"roc_auc": auc}})()


def test_pick_best_returns_highest_auc():
    runs = [_FakeRun("a", 0.81), _FakeRun("b", 0.94), _FakeRun("c", 0.88)]
    best = pick_best(runs, metric="roc_auc")
    assert best.info.run_id == "b"


def test_pick_best_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        pick_best([], metric="roc_auc")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_register.py -v`
Expected: FAIL — cannot import `pick_best`.

- [ ] **Step 3: Write `src/register.py`**

```python
"""Register the best run's model and walk it Staging -> Production."""
import mlflow
from mlflow.tracking import MlflowClient

from src import config


def pick_best(runs, metric=config.PRIMARY_METRIC):
    """Return the run with the highest value of `metric`."""
    if not runs:
        raise ValueError("no runs to choose from")
    return max(runs, key=lambda r: r.data.metrics.get(metric, float("-inf")))


def find_best_run(client, experiment_name=config.EXPERIMENT_NAME,
                  metric=config.PRIMARY_METRIC):
    """Search an experiment's runs (that logged a model) for the best metric."""
    exp = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=200,
    )
    # keep only runs that actually logged a model artifact
    runs = [r for r in runs if metric in r.data.metrics]
    return pick_best(runs, metric)


def main():
    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = MlflowClient()

    best = find_best_run(client)
    auc = best.data.metrics[config.PRIMARY_METRIC]
    model_uri = f"runs:/{best.info.run_id}/model"
    print(f"Best run {best.info.run_id} {config.PRIMARY_METRIC}={auc:.4f}")

    # register a new version
    mv = mlflow.register_model(model_uri, config.REGISTERED_MODEL_NAME)
    print(f"Registered version {mv.version}")

    # Staging
    client.transition_model_version_stage(
        name=config.REGISTERED_MODEL_NAME, version=mv.version, stage="Staging"
    )
    print(f"Version {mv.version} -> Staging")

    # validation gate -> Production
    if auc >= config.AUC_THRESHOLD:
        client.transition_model_version_stage(
            name=config.REGISTERED_MODEL_NAME,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Version {mv.version} passed gate (AUC>={config.AUC_THRESHOLD}) -> Production")
    else:
        print(f"Version {mv.version} held in Staging (AUC<{config.AUC_THRESHOLD})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_register.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run registration against the live server**

Run: `python -m src.register`
Expected: prints best run, "Registered version 1", "-> Staging", "-> Production".

- [ ] **Step 6: Verify in the MLflow UI**

UI → "Models" tab → `ecommerce-churn-model` exists with a version in **Production**.

- [ ] **Step 7: Commit**

```bash
git add src/register.py tests/test_register.py
git commit -m "Add model registration with Staging->Production transitions"
```

---

## Task 8: Batch & Real-Time Serving

**Files:**
- Create: `src/serve.py`

- [ ] **Step 1: Write `src/serve.py`**

```python
"""Load the Production churn model and score data (batch)."""
import argparse

import mlflow
import pandas as pd

from src import config

PRODUCTION_URI = f"models:/{config.REGISTERED_MODEL_NAME}/Production"


def load_production_model(uri=PRODUCTION_URI):
    """Load the current Production model as a pyfunc."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    return mlflow.pyfunc.load_model(uri)


def predict_csv(input_csv, output_csv):
    """Score a CSV of raw feature rows and write predictions."""
    model = load_production_model()
    df = pd.read_csv(input_csv)
    preds = model.predict(df[config.FEATURES])
    df["churn_prediction"] = preds
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} predictions to {output_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.DATA_PATH))
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    predict_csv(args.input, args.output)
```

- [ ] **Step 2: Run batch prediction**

Run: `python -m src.serve --output predictions.csv`
Expected: "Wrote 3941 predictions to predictions.csv"; the file has a `churn_prediction` column.

- [ ] **Step 3: Test real-time REST serving (separate terminal)**

Run:
```bash
source venv/bin/activate
mlflow models serve -m "models:/ecommerce-churn-model/Production" -p 1234 --no-conda
```
Then in another terminal:
```bash
curl -X POST http://127.0.0.1:1234/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["Tenure","WarehouseToHome","NumberOfDeviceRegistered","SatisfactionScore","NumberOfAddress","Complain","DaySinceLastOrder","CashbackAmount","PreferedOrderCat","MaritalStatus"], "data": [[15,29,4,3,2,0,7,143.32,"Laptop & Accessory","Single"]]}}'
```
Expected: JSON like `{"predictions": [0]}`.

- [ ] **Step 4: Commit**

```bash
git add src/serve.py
git commit -m "Add batch and REST serving from the Production model"
```

---

## Task 9: Drift Monitoring (PSI unit-tested) + Live Metric Logging

**Files:**
- Create: `src/monitor.py`
- Test: `tests/test_monitor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_monitor.py
import numpy as np
from src.monitor import calculate_psi


def test_psi_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    base = rng.normal(size=2000)
    assert calculate_psi(base, base) < 0.01


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, size=2000)
    shifted = rng.normal(3, 1, size=2000)
    assert calculate_psi(base, shifted) > 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_monitor.py -v`
Expected: FAIL — cannot import `calculate_psi`.

- [ ] **Step 3: Write `src/monitor.py`**

```python
"""Simulate incoming batches, log live metrics to MLflow, and detect drift (PSI/KS)."""
import numpy as np
import mlflow
from scipy.stats import ks_2samp

from src import config
from src.data_prep import load_data, split_data
from src.serve import load_production_model
from src.train import compute_metrics


def calculate_psi(expected, actual, buckets=10):
    """Population Stability Index between two 1-D numeric samples."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    e = np.histogram(expected, breakpoints)[0] / len(expected)
    a = np.histogram(actual, breakpoints)[0] / len(actual)
    e = np.where(e == 0, 1e-6, e)
    a = np.where(a == 0, 1e-6, a)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_drift(baseline_df, batch_df, features=config.NUMERIC_FEATURES, psi_threshold=0.2):
    """Per-feature PSI + KS p-value; flag features whose PSI exceeds the threshold."""
    report = {}
    for col in features:
        base = baseline_df[col].dropna()
        new = batch_df[col].dropna()
        psi = calculate_psi(base, new)
        ks_p = float(ks_2samp(base, new).pvalue)
        report[col] = {"psi": psi, "ks_pvalue": ks_p, "drifted": psi > psi_threshold}
    return report


def main(n_batches=5, drift_strength=0.5):
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.MONITORING_EXPERIMENT)
    model = load_production_model()

    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    baseline = X_train

    rng = np.random.default_rng(config.RANDOM_STATE)
    for i in range(n_batches):
        batch = X_test.sample(frac=1.0, replace=True, random_state=i).reset_index(drop=True)
        labels = y_test.sample(frac=1.0, replace=True, random_state=i).reset_index(drop=True)
        # inject increasing drift into CashbackAmount to simulate a shift over time
        batch = batch.copy()
        batch["CashbackAmount"] = batch["CashbackAmount"] * (1 + drift_strength * i)

        with mlflow.start_run(run_name=f"batch_{i}"):
            preds = model.predict(batch[config.FEATURES])
            proba = preds  # pyfunc sklearn returns class labels; use as proba proxy
            metrics = compute_metrics(labels, preds, proba)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            drift = feature_drift(baseline, batch)
            for col, stats in drift.items():
                mlflow.log_metric(f"psi_{col}", stats["psi"])
            n_drifted = sum(s["drifted"] for s in drift.values())
            mlflow.log_metric("n_drifted_features", n_drifted)
            mlflow.set_tag("batch_index", i)
            print(f"batch {i}: f1={metrics['f1']:.3f} drifted_features={n_drifted}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_monitor.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run monitoring against the live server**

Run: `python -m src.monitor`
Expected: 5 "batch i" lines; `n_drifted_features` rises in later batches as injected drift grows.

- [ ] **Step 6: Verify in the MLflow UI**

Experiment `ecommerce-churn-monitoring` → 5 runs; chart `psi_CashbackAmount` across batches to see the rising drift trend.

- [ ] **Step 7: Commit**

```bash
git add src/monitor.py tests/test_monitor.py
git commit -m "Add drift monitoring (PSI/KS) and live metric logging"
```

---

## Task 10: Run the Full Test Suite

**Files:** none (verification gate)

- [ ] **Step 1: Run all tests**

Run: `pytest -v`
Expected: all tests across `test_data_prep`, `test_train`, `test_register`, `test_monitor` pass.

- [ ] **Step 2: Commit any fixes**

```bash
git add -A && git commit -m "Green test suite" || echo "nothing to commit"
```

---

## Task 11: Airflow Orchestration DAG

**Files:**
- Create: `airflow/docker-compose.yml`, `airflow/dags/churn_pipeline.py`

> Built last. The MLflow server runs on the host; the Airflow container reaches it via `host.docker.internal`.

- [ ] **Step 1: Fetch the official Airflow compose file**

Run:
```bash
mkdir -p airflow/dags airflow/logs airflow/plugins
cd airflow
curl -sL "https://airflow.apache.org/docs/apache-airflow/3.0.0/docker-compose.yaml" -o docker-compose.yml
echo -e "AIRFLOW_UID=50000" > .env
```
Expected: `docker-compose.yml` downloaded; `.env` created.

- [ ] **Step 2: Add MLflow env + project mount to the compose `environment:` and `volumes:`**

In `airflow/docker-compose.yml`, under the `x-airflow-common` → `environment:` map, add:
```yaml
    MLFLOW_TRACKING_URI: "http://host.docker.internal:8080"
    PIP_ADDITIONAL_REQUIREMENTS: "mlflow>=2.16,<3 scikit-learn>=1.4,<1.6 optuna>=3.6,<4.2 pandas>=2.1,<2.3 scipy>=1.11,<1.15"
```
Under `x-airflow-common` → `volumes:`, add a mount of the project source and data:
```yaml
    - /Users/asra/goksin-proje/src:/opt/airflow/project/src
    - /Users/asra/goksin-proje/data:/opt/airflow/project/data
```
And add `extra_hosts` so the container can resolve the host:
```yaml
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

- [ ] **Step 3: Write the DAG**

```python
# airflow/dags/churn_pipeline.py
"""Airflow DAG orchestrating the churn ML pipeline via the src/ modules."""
import sys
from datetime import datetime

from airflow.decorators import dag, task

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
```

- [ ] **Step 4: Start Airflow**

Run:
```bash
cd /Users/asra/goksin-proje/airflow
docker compose up airflow-init
docker compose up -d
```
Expected: containers start; UI at http://localhost:8080 (login `airflow`/`airflow`).

> Note: Airflow UI and MLflow both default to 8080. Keep MLflow on 8080 and map Airflow's webserver to a different host port (e.g. edit the `airflow-apiserver` ports to `"8088:8080"`), then use http://localhost:8088.

- [ ] **Step 5: Trigger and verify**

In the Airflow UI, unpause `churn_pipeline` and trigger it. Expected: all 4 tasks succeed (green); new runs appear in the MLflow `ecommerce-churn` experiment and a new registered version.

- [ ] **Step 6: Commit**

```bash
cd /Users/asra/goksin-proje
git add airflow/docker-compose.yml airflow/dags/churn_pipeline.py
git commit -m "Add Airflow DAG orchestrating the churn pipeline"
```

---

## Task 12: README, Packaging Script & Report Scaffold

**Files:**
- Create: `README.md`, `scripts/package.sh`, `reports/REPORT.md`

- [ ] **Step 1: Write `scripts/package.sh`**

```bash
#!/usr/bin/env bash
# Build the submission zip (<50 MB), excluding venv, mlflow stores, lecture notes.
set -euo pipefail
NUMBER="${1:-XXXXXXX}"
OUT="PRJ-asrasari-${NUMBER}.zip"
rm -f "$OUT"
zip -r "$OUT" \
  src airflow/dags airflow/docker-compose.yml tests data \
  requirements.txt README.md conftest.py reports docs scripts \
  -x "*/__pycache__/*" "*.pyc" \
  >/dev/null
echo "Built $OUT"
du -h "$OUT"
```

- [ ] **Step 2: Verify the package is small enough**

Run:
```bash
chmod +x scripts/package.sh
./scripts/package.sh 12345
```
Expected: "Built PRJ-asrasari-12345.zip" and a size well under 50 MB (a few hundred KB).

- [ ] **Step 3: Write `README.md`**

Contents must cover: project overview; the 5 rubric objectives and which module hits each; setup (`python -m venv venv`, `pip install -r requirements.txt`); start MLflow server command; run order (`python -m src.train`, `tune`, `register`, `serve`, `monitor`); how to run tests (`pytest -v`); how to run Airflow; note that `mlruns/`, `mlartifacts/`, `mlflow.db`, `venv/` are excluded from the zip and a Drive link holds the full runs.

- [ ] **Step 4: Write `reports/REPORT.md` scaffold**

Sections (to fill with real numbers + MLflow screenshots after runs): Introduction & domain; Dataset & EDA; MLflow setup (server/SQLite/artifacts); Experiment tracking (3 baselines + comparison table); Hyperparameter tuning (Optuna results); Model Registry (Staging→Production); Deployment (batch + REST); Monitoring & drift (PSI trend); Airflow orchestration; Insights & reflection; Drive link to full runs.

- [ ] **Step 5: Commit**

```bash
git add README.md scripts/package.sh reports/REPORT.md
git commit -m "Add README, packaging script, and report scaffold"
```

---

## Self-Review Notes

- **Spec coverage:** Objective 1 (Task 5), 2 (Tasks 5+6), 3 (Task 8), 4 (Task 9), 5 (Task 7); MLflow setup (Task 5 step 2); Airflow (Task 11); packaging <50 MB (Task 12). All spec sections mapped.
- **Type consistency:** `build_pipeline`, `compute_metrics`, `load_data`, `split_data`, `build_preprocessor`, `load_production_model`, `calculate_psi`, `pick_best` names are used consistently across tasks.
- **Known risk flagged:** Airflow and MLflow both use port 8080 — Task 11 step 4 resolves it by remapping Airflow to 8088.
- **MLflow 2.x pin:** stages (`transition_model_version_stage`) require MLflow < 3; pinned in `requirements.txt`.
