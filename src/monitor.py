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

    for i in range(n_batches):
        batch = X_test.sample(frac=1.0, replace=True, random_state=i).reset_index(drop=True)
        labels = y_test.sample(frac=1.0, replace=True, random_state=i).reset_index(drop=True)
        # inject increasing drift into CashbackAmount to simulate a shift over time
        batch = batch.copy()
        batch["CashbackAmount"] = batch["CashbackAmount"] * (1 + drift_strength * i)

        with mlflow.start_run(run_name=f"batch_{i}"):
            preds = model.predict(batch[config.FEATURES])
            # pyfunc sklearn classifier returns class labels; use as a proba proxy for AUC
            metrics = compute_metrics(labels, preds, preds)
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
