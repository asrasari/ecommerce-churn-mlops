import numpy as np

from src import config
from src.data_prep import build_preprocessor, load_data, split_data


def test_load_data_has_expected_columns():
    df = load_data()
    assert config.TARGET in df.columns
    for col in config.FEATURES:
        assert col in df.columns, f"missing {col}"
    assert len(df) > 3000


def test_split_data_shapes_and_no_leakage():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    assert config.TARGET not in X_train.columns
    assert list(X_train.columns) == config.FEATURES
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_test) == len(X_test)
    assert abs(y_train.mean() - y_test.mean()) < 0.03


def test_preprocessor_handles_missing_and_encodes():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X_train)
    arr = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    assert not np.isnan(arr).any()
    assert arr.shape[1] > len(config.FEATURES)
    pre.transform(X_test)
