from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import pickle
import pathlib

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from sklearn.base import clone

from typing import Callable, Dict, List
from fastf1.core import Session
from datetime import datetime

import time

import f1_pitstop_advisor
import f1_pitstop_advisor.gather_data

def prepare_data(
        self, 
        session_path: str, 
        data_path: str,
        data_creation_function: Callable[[List[Session]], pd.DataFrame],
        cutoff_date: datetime) -> pd.DataFrame | Dict[str, pd.DataFrame]:
    
    if pathlib.Path(data_path).is_file():
        try:
            data = pd.read_csv(data_path, header=None)
        except pd.errors.ParserError:
            with open(data_path, "rb") as file:
                data = pickle.load(file)

    else:
        if pathlib.Path(session_path).is_file():
            with open(session_path, "rb") as file:
                sessions: List[Session] = pickle.load(file)
        else:
            sessions = f1_pitstop_advisor.gather_data._get_sessions(cutoff_date)
            with open(session_path, "wb") as file:
                pickle.dump(sessions, file)

        data = data_creation_function(sessions)
        with open(data_path, "wb") as file:
            pickle.dump(data, file)

    return data 


DEFAULT_SEARCHES = {
    # Linear regression
    "LinearRegression": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), LinearRegression()),
        {"pca__n_components": [0.98, 0.95, 0.9]}
    ),

    "RidgeCV": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), RidgeCV()),
        {
            "pca__n_components": [0.98, 0.95, 0.9],
            "ridgecv__alphas": [0.001, 0.01, 0.1, 1.0]
        }
    ),

    "LassoCV": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), LassoCV(max_iter=100_000)),
        {
            "pca__n_components": [0.98, 0.95, 0.9],
            "lassocv__alphas": [0.001, 0.01, 0.1, 1.0]
        }
    ),

    "ElasticNetCV": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), ElasticNetCV(max_iter=100_000)),
        {
            "pca__n_components": [0.98, 0.95, 0.9],
            "elasticnetcv__l1_ratio": [0.2, 0.5, 0.8]
        }
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
        make_pipeline(StandardScaler(), PCA(), PolynomialFeatures(), RidgeCV()),
        {
            "polynomialfeatures__degree": [2, 3],
            "pca__n_components": [0.98, 0.95, 0.9],
            "ridgecv__alphas": [0.01, 0.1, 1.0]
        }
    ),

    "PolynomialLassoCV": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), PolynomialFeatures(), LassoCV(max_iter=100_000)),
        {
            "polynomialfeatures__degree": [2, 3],
            "pca__n_components": [0.98, 0.95, 0.9],
            "lassocv__alphas": [0.01, 0.1, 1.0]
        }
    ),

    "PolynomialElasticNetCV": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), PolynomialFeatures(), ElasticNetCV(max_iter=100_000)),
        {
            "polynomialfeatures__degree": [2, 3],
            "pca__n_components": [0.98, 0.95, 0.9],
            "elasticnetcv__l1_ratio": [0.2, 0.5, 0.8]
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

class AbstractRegressionModelTest(ABC):

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def score(self) -> pd.DataFrame:
        pass

class RegressionModelTest(AbstractRegressionModelTest):
    def __init__(self, data: pd.DataFrame, target_label: str, searches: Dict[str, GridSearchCV] | None = None) -> None:
        self.data = data
        self.target_label = target_label
        self.searches = DEFAULT_SEARCHES if searches is None else searches

        if target_label not in self.data.columns:
            raise KeyError(f"Invalid target label. Column \"{target_label}\" is not present in data.")
    
    def fit(self) -> None:
        X, y = self.data.drop(self.target_label, axis="columns"), self.data[self.target_label]
        for search_key, search in self.searches.items():
            start_time = time.time()
            print(f"Fitting {search_key}... ".ljust(50), end="")
            search.fit(X, y)
            print(f"Took {round(time.time() - start_time, 2)} seconds.")
            print(f"Best score: {round(search.best_score_, 5)}")
            print()

    def score(self) -> pd.DataFrame:
        scores = {}
        for search_key, search in self.searches.items():
            scores[search_key] = search.best_score_
        return pd.DataFrame({
            "Score": scores
        }) 
    

class CircuitSeparatingModelTest(AbstractRegressionModelTest):
    def __init__(self, data: Dict[str, pd.DataFrame], target_label: str, searches: Dict[str, GridSearchCV] | None = None) -> None:
        self.data = data
        self.target_label = target_label
        self.searches = DEFAULT_SEARCHES if searches is None else searches

        for df in data.values():
            if target_label not in df.columns:
                raise KeyError(f"Invalid target label. Column \"{target_label}\" is not present in every circuit's data.")
    
    def fit(self) -> None:
        models_and_circuits = {}

        for search_key in self.searches.keys():
            models_and_circuits[search_key] = {}

        for circuit, data in self.data.items():
            print(f"Fitting models for {circuit}...")
            circuit_start_time = time.time()
            
            X, y = data.drop([self.target_label], axis="columns"), data[self.target_label]
            for search_key, search in self.searches.items():
                print(f"Fitting {search_key}... ".ljust(50), end="")
                model_start_time = time.time()

                model_search_copy = clone(search)
                model_search_copy.fit(X, y)
                models_and_circuits[search_key][circuit] = model_search_copy

                print(f"Took {round(time.time() - model_start_time, 2)} seconds.")
            
            print(f"Took a total of {round(time.time() - circuit_start_time, 2)} seconds to fit all models for circuit \"{circuit}\"")
            print()

    def score(self) -> pd.DataFrame:
        scores = {}
        for search_key, search in self.searches.items():
            scores[search_key] = search.best_score_
        return pd.DataFrame({
            "Score": scores
        }) 

        
def create_regression_model_test(
        data: pd.DataFrame | Dict[str, pd.DataFrame], 
        target_label: str, 
        searches: Dict[str, GridSearchCV] | None = None) -> AbstractRegressionModelTest:

        if isinstance(data, pd.DataFrame):
            return RegressionModelTest(data, target_label, searches)
        else:
            return CircuitSeparatingModelTest(data, target_label, searches)
            
