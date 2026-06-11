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
