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
