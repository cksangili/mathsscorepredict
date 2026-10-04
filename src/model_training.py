import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

class ModelTraining:
    def __init__(self, config, preprocessor):
        self.config = config
        self.preprocessor = preprocessor
        self.random_state = config.get("random_state", 42)
        self.cv = config.get("cv", 5)
        self.scoring = config.get("scoring", "r2")
        self.val_test_size = config.get("val_test_size", 0.2)
        self.val_size = config.get("val_size", 0.5)
        self.results = []
        self.trained_models = {}

    def split_data(self, X, y):
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.val_test_size, random_state=self.random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=self.val_size, random_state=self.random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _build_model(self, name):
        if name == "linear_regression": return LinearRegression()
        if name == "ridge": return Ridge()
        if name == "random_forest":
            return RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        raise ValueError(f"Unsupported model: {name}")

    def train_model(self, name, X_train, y_train, X_val, y_val):
        pipe = Pipeline([("preprocessor", self.preprocessor), ("regressor", self._build_model(name))])
        search = GridSearchCV(pipe, self.config["models"][name]["param_grid"],
                              cv=self.cv, scoring=self.scoring, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        pred = search.best_estimator_.predict(X_val)
        result = {
            "Model": name, "CV_R2": search.best_score_,
            "Validation_MAE": mean_absolute_error(y_val, pred),
            "Validation_RMSE": np.sqrt(mean_squared_error(y_val, pred)),
            "Validation_R2": r2_score(y_val, pred),
            "Best_Params": search.best_params_,
        }
        self.results.append(result)
        self.trained_models[name] = search.best_estimator_
        return result

    def train_all_models(self, X_train, y_train, X_val, y_val):
        for name in self.config["models"]:
            self.train_model(name, X_train, y_train, X_val, y_val)
        return self.get_comparison()

    def get_comparison(self):
        return pd.DataFrame(self.results).sort_values("Validation_MAE").reset_index(drop=True)

    def select_best_model(self):
        comparison = self.get_comparison()
        name = comparison.iloc[0]["Model"]
        return name, self.trained_models[name]

    def evaluate_final_model(self, model, X_train, y_train, X_val, y_val, X_test, y_test):
        model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
        pred = model.predict(X_test)
        return {
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred),
        }
