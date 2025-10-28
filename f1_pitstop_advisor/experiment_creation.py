from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import pickle
import pathlib

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRFRegressor, XGBRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import NotFittedError

from sklearn.base import BaseEstimator, clone

from typing import Callable, Dict, List
from fastf1.core import Session
from datetime import datetime

import time

import f1_pitstop_advisor
import f1_pitstop_advisor.gather_data

class SessionPreparer:
    def __init__(
        self,
        session_path: str,
        cutoff_date: datetime) -> None:

        self.session_path = pathlib.Path(session_path)
        self.cutoff_date = cutoff_date
        self.sessions: List[Session] | None = None

    def prepare_data(self) -> List[Session]:
        if self.sessions is None:
            print(f"Looking for session file at {self.session_path}...")
            if self.session_path.is_file():
                print(f"Supposed session file found.. Loading sessions...")
                with open(self.session_path, "rb") as file:
                    sessions: List[Session] = pickle.load(file)
                    print(f"Sessions loaded.")
            else:
                print(f"Session file not found. Loading sessions from FastF1...")
                sessions = f1_pitstop_advisor.gather_data._get_sessions(self.cutoff_date)
                
                self.session_path.parent.mkdir(exist_ok=True)
                with open(self.session_path, "wb") as file:
                    pickle.dump(sessions, file)
                    print(f"Sessions loaded and saved.")
            self.sessions = sessions
        
        return self.sessions

class DataPreparer:
    def __init__(
        self, 
        session_preparer: SessionPreparer,
        data_path: str,
        data_creation_function: Callable[[List[Session]], pd.DataFrame] | Callable[[List[Session]], Dict[str, pd.DataFrame]]) -> None:

        self.session_preparer = session_preparer
        self.data_path = pathlib.Path(data_path)
        self.data_creation_function = data_creation_function

    def prepare_data(self) -> pd.DataFrame | Dict[str, pd.DataFrame]:
        print(f"Looking for data file at {self.data_path}...")
        if self.data_path.is_file():
            print(f"Supposed data file found. Loading data...")
            try:
                data = pd.read_csv(self.data_path, header=None)
                print("Loaded data from csv.")
            except pd.errors.ParserError:
                with open(self.data_path, "rb") as file:
                    data = pickle.load(file)
                    print("Loaded data from pickle.")

        else:
            print(f"Data file not found. ", end="")
            sessions = self.session_preparer.prepare_data()

            print("Generating data from sessions...")
            data = self.data_creation_function(sessions)

            self.data_path.parent.mkdir(exist_ok=True)
            with open(self.data_path, "wb") as file:
                pickle.dump(data, file)
                print(f"Data has been generated and saved.")

        return data 


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


class AbstractRegressionModelTest(ABC):

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def score(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def best_model(self) -> BaseEstimator | Dict[str, BaseEstimator]:
        pass


class RegressionModelTest(AbstractRegressionModelTest):
    def __init__(self, data: pd.DataFrame, target_label: str, searches: Dict[str, GridSearchCV] | None = None) -> None:
        self.data = data
        self.target_label = target_label
        self.template_searches = DEFAULT_SEARCHES if searches is None else searches

        if target_label not in self.data.columns:
            raise KeyError(f"Invalid target label. Column \"{target_label}\" is not present in data.")
        
    def _check_fitted(self) -> None:
        if not hasattr(self, "searches_"):
            raise NotFittedError("Models have not been fitted yet. Call fit() before using this method.")
    
    def fit(self) -> None:
        X, y = self.data.drop(self.target_label, axis="columns"), self.data[self.target_label]
        self.searches_ = {}
        for search_key, search_clone in self.template_searches.items():
            start_time = time.time()
            print(f"Fitting {search_key}... ".ljust(50), end="")
            search_clone = clone(search_clone)
            search_clone.fit(X, y)
            self.searches_[search_key] = search_clone
            print(f"Took {round(time.time() - start_time, 2)} seconds.")
            print(f"Best score: {round(search_clone.best_score_, 5)}")
            print()

    def score(self) -> pd.DataFrame:
        self._check_fitted()

        scores = {}
        for search_key, search in self.searches_.items():
            scores[search_key] = search.best_score_
        return pd.DataFrame({
            "Score": scores
        }).sort_values(by="Score", axis="index", ascending=False)
    
    def best_model(self) -> BaseEstimator:
        self._check_fitted()

        return self.searches_[self.score().index[0]].best_estimator_
        
    

class CircuitSeparatingModelTest(AbstractRegressionModelTest):
    def __init__(self, data: Dict[str, pd.DataFrame], target_label: str, searches: Dict[str, GridSearchCV] | None = None) -> None:
        self.data = data
        self.target_label = target_label
        self.template_searches = DEFAULT_SEARCHES if searches is None else searches

        for df in data.values():
            if target_label not in df.columns:
                raise KeyError(f"Invalid target label. Column \"{target_label}\" is not present in every circuit's data.")
            
    def _check_fitted(self) -> None:
        if not hasattr(self, "circuits_and_searches_"):
            raise NotFittedError("Models have not been fitted yet. Call fit() before using this method.")
    
    def fit(self) -> None:
        self.circuits_and_searches_ = {}

        i = 1
        for circuit, data in self.data.items():
            print(f"==== Fitting models for {circuit} ({i}/{len(self.data)}) ====")
            circuit_start_time = time.time()
            self.circuits_and_searches_[circuit] = {}
            X, y = data.drop([self.target_label], axis="columns"), data[self.target_label]
            for search_key, search in self.template_searches.items():
                print(f"Fitting {search_key}... ".ljust(50), end="")
                model_start_time = time.time()

                search_clone = clone(search)
                search_clone.fit(X, y)
                self.circuits_and_searches_[circuit][search_key] = search_clone

                print(f"Took {round(time.time() - model_start_time, 2)} seconds.")
            
            print(f"Took a total of {round(time.time() - circuit_start_time, 2)} seconds to fit all models for circuit \"{circuit}\".")
            print()
            i += 1

    def score(self) -> pd.DataFrame:
        self._check_fitted()

        all_scores = self.all_scores()
        return pd.DataFrame({
            "Score": all_scores.mean(axis="index")
        }).sort_values(by="Score", axis="index", ascending=False)
    
    def all_scores(self) -> pd.DataFrame:
        self._check_fitted()

        all_scores = {}
        for circuit in self.circuits_and_searches_.keys():
            scores = {}
            for key, model in self.circuits_and_searches_[circuit].items():
                scores[key] = model.best_score_
            all_scores[circuit] = scores

        return pd.DataFrame(all_scores).T
    
    def score_statistics(self) -> pd.DataFrame:
        self._check_fitted()

        all_scores = self.all_scores()
        return pd.DataFrame({
            "MeanScore": all_scores.mean(axis="index"),
            "MedianScore": all_scores.median(axis="index"),
            "ScoreVariance": all_scores.var(axis="index"),
            "MinScore": all_scores.min(axis="index")
        }).sort_values(by="MeanScore", axis="index", ascending=False)
    
    def best_model(self) -> BaseEstimator | Dict[str, BaseEstimator]:
        self._check_fitted()

        chosen_estimators = {}
        all_scores = self.all_scores()
        all_scores["BestModelType"] = all_scores.idxmax(axis="columns")
        for circuit in self.circuits_and_searches_.keys():
            chosen_estimator_key = all_scores.loc[circuit, "BestModelType"]
            chosen_estimators[circuit] = self.circuits_and_searches_[circuit][chosen_estimator_key].best_estimator_

        return chosen_estimators

        
def create_regression_model_test(
        data: pd.DataFrame | Dict[str, pd.DataFrame], 
        target_label: str, 
        searches: Dict[str, GridSearchCV] | None = None) -> AbstractRegressionModelTest:

        if isinstance(data, pd.DataFrame):
            return RegressionModelTest(data, target_label, searches)
        else:
            return CircuitSeparatingModelTest(data, target_label, searches)
            
