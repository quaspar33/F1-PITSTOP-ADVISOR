from __future__ import annotations

import pandas as pd

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRFRegressor, XGBRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from sklearn.base import clone

from typing import Dict

import time


DEFAULT_SEARCHES = {
    # Linear regression
    "LinearRegression": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), LinearRegression()),
        {"pca__n_components": [0.98, 0.95, 0.9]}
    ),

    "RidgeCV": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), RidgeCV(alphas=(0.1, 1.0, 10.0))),
        {"pca__n_components": [0.98, 0.95, 0.9]}
    ),

    "LassoCV": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), LassoCV(max_iter=100_000, alphas=[0.001, 0.01, 0.1, 1.0])),
        {"pca__n_components": [0.98, 0.95, 0.9]}
    ),

    "ElasticNetCV": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), ElasticNetCV(max_iter=100_000, l1_ratio=[0.2, 0.5, 0.8])),
        {"pca__n_components": [0.98, 0.95, 0.9]}
    ),

    # Polynomial regression
    "PolynomialLinearRegression": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), PolynomialFeatures(), LinearRegression()),
        {
            "polynomialfeatures__degree": [2, 3],
            "pca__n_components": [0.98, 0.95, 0.9]
        }
    ),

    "PolynomialRidgeCV": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), PolynomialFeatures(), RidgeCV(alphas=(0.1, 1.0, 10.0))),
        {
            "polynomialfeatures__degree": [2, 3],
            "pca__n_components": [0.98, 0.95, 0.9]
        }
    ),

    "PolynomialLassoCV": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), PolynomialFeatures(), LassoCV(max_iter=100_000, alphas=[0.01, 0.1, 1])),
        {
            "polynomialfeatures__degree": [2, 3],
            "pca__n_components": [0.98, 0.95, 0.9]
        }
    ),

    "PolynomialElasticNetCV": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), PolynomialFeatures(), ElasticNetCV(max_iter=100_000, l1_ratio=[0.2, 0.5, 0.8])),
        {
            "polynomialfeatures__degree": [2, 3],
            "pca__n_components": [0.98, 0.95, 0.9]
        }
    ),

    # Bagging models
    "RandomForestRegressor": GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10]
        }
    ),

    "ExtraTreesRegressor": GridSearchCV(
        ExtraTreesRegressor(random_state=42, n_jobs=-1),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10]
        }
    ),

    "XGBRFRegressor": GridSearchCV(
        XGBRFRegressor(
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
            verbosity=0
        ),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 6, 10],
            "colsample_bynode": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "subsample": [0.8, 1.0]
        }
    ),

    # Boosting models
    "AdaBoostRegressor": GridSearchCV(
        AdaBoostRegressor(random_state=42),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.5, 1.0]
        }
    ),

    "GradientBoostingRegressor": GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5],
            "subsample": [0.8, 1.0]
        }
    ),

    "XGBRegressor": GridSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror", verbosity=0),
        {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
    ),

    # Support vector models
    "SVR_linear": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), SVR(kernel="linear")),
        {
            "svr__C": [0.1, 1, 10, 100],
            "pca__n_components": [0.98, 0.95, 0.9]
        }
    ),

    "SVR_rbf": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), SVR(kernel="rbf")),
        {
            "svr__C": [0.1, 1, 10],
            "svr__gamma": ["scale", 0.01, 0.1, 1.0],
            "pca__n_components": [0.98, 0.95, 0.9]
        }
    ),

    # MLP
    "MLPRegressor": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), MLPRegressor(max_iter=100_000, random_state=42)),
        {
            "mlpregressor__hidden_layer_sizes": [(16,), (24,), (24, 12), (16, 16), (16, 8)],
            "mlpregressor__activation": ["relu", "tanh"],
            "mlpregressor__alpha": [0.0001, 0.001, 0.01],
            "mlpregressor__learning_rate_init": [0.001, 0.01],
            "pca__n_components": [0.98, 0.95, 0.9]
        }
    ),
}

def search(data: pd.DataFrame, target_label: str) -> Dict[str, GridSearchCV]:
    X, y = data.drop(target_label, axis="columns"), data[target_label]
    searches = {}
    for search_key, search_clone in DEFAULT_SEARCHES.items():
        start_time = time.time()
        print(f"Fitting {search_key}... ".ljust(50), end="")
        search_clone = clone(search_clone)
        search_clone.fit(X, y)
        searches[search_key] = search_clone
        print(f"Took {round(time.time() - start_time, 2)} seconds.")
        print(f"Best score: {round(search_clone.best_score_, 5)}")
        print()
    return searches