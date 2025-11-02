import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor


def split_data(data: pd.DataFrame, target_label: str) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = data.drop(labels=target_label, axis="columns"), data[target_label]
    return X, y


def fit_model_search(X: pd.DataFrame, y: pd.Series, target_label: str) -> GridSearchCV:
    gscv = GridSearchCV(XGBRegressor(), {})
    gscv.fit(X, y)
    return gscv

