import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data_prep import load_data, split_data
from src.train import build_pipeline, compute_metrics


def test_compute_metrics_perfect_prediction():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.8, 0.2])
    m = compute_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_pipeline_fits_and_predicts_from_raw_input():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test.head(5))
    assert len(preds) == 5
    proba = pipe.predict_proba(X_test.head(5))
    assert proba.shape == (5, 2)
