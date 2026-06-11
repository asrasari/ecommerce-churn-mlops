"""Load, split, and preprocess the e-commerce churn dataset."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
