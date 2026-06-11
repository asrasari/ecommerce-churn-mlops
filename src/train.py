"""Train baseline churn models and log them to MLflow."""
import tempfile

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline as SkPipeline

from src import config
from src.data_prep import build_preprocessor, load_data, split_data


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


def build_pipeline(model):
    """Wrap preprocessing + estimator so the model consumes raw input."""
    return SkPipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


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
