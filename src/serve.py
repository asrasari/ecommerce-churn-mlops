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
