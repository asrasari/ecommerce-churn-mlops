"""Minimal Flask REST service that serves the Production churn model.

Exposes the same JSON contract MLflow's own scoring server uses, so the demo is
familiar:

    GET  /health        -> {"status": "ok"}
    POST /invocations   -> {"predictions": [...]}

Request body accepts either MLflow's ``dataframe_split`` format::

    {"dataframe_split": {"columns": [...], "data": [[...]]}}

or a list of record dicts::

    {"dataframe_records": [{"Tenure": 1, ...}]}

Run with:  python -m src.serve_api   (model loaded once at startup)

We use Flask here because MLflow 2.22's built-in ``mlflow models serve`` scoring
server is incompatible with modern Starlette/FastAPI. This wrapper is
version-proof and loads the exact same registered Production model via pyfunc.
"""
import pandas as pd
from flask import Flask, jsonify, request

from src import config
from src.serve import load_production_model

app = Flask(__name__)
_model = None


def get_model():
    """Lazily load (and cache) the Production model."""
    global _model
    if _model is None:
        _model = load_production_model()
    return _model


# pandas dtype per MLflow signature type, so JSON ints don't trip schema enforcement
_MLFLOW_TO_PANDAS = {
    "double": "float64", "float": "float64",
    "long": "int64", "integer": "int32",
    "boolean": "bool", "string": "object",
}


def _schema_dtypes(model):
    """Map each input column to the pandas dtype the model expects."""
    schema = model.metadata.get_input_schema()
    return {
        col.name: _MLFLOW_TO_PANDAS.get(col.type.name, "object")
        for col in schema.inputs
    }


def _to_dataframe(payload, model):
    """Build a feature DataFrame coerced to the model's expected dtypes."""
    if "dataframe_split" in payload:
        split = payload["dataframe_split"]
        df = pd.DataFrame(data=split["data"], columns=split["columns"])
    elif "dataframe_records" in payload:
        df = pd.DataFrame(payload["dataframe_records"])
    else:
        raise ValueError("body must contain 'dataframe_split' or 'dataframe_records'")
    df = df[config.FEATURES]
    return df.astype(_schema_dtypes(model))


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/invocations")
def invocations():
    model = get_model()
    try:
        df = _to_dataframe(request.get_json(force=True), model)
    except (KeyError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    preds = model.predict(df)
    return jsonify({"predictions": [int(p) for p in preds]})


def main(host="127.0.0.1", port=1234):
    get_model()  # fail fast if the model can't load
    print(f"Serving Production model at http://{host}:{port}/invocations")
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()
