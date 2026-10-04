import logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

logger = logging.getLogger(__name__)

class DataPreparation:
    def __init__(self, config_path="./src/config.yaml"):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.file_path = Path(self.config["file_path"])
        self.target_column = self.config["target_column"]
        self.numerical_features = self.config.get("numerical_features", [])
        self.nominal_features = self.config.get("nominal_features", [])
        self.ordinal_features = self.config.get("ordinal_features", [])
        self.passthrough_features = self.config.get("passthrough_features", [])

    @staticmethod
    def _time_to_minutes(value):
        if pd.isna(value):
            return np.nan
        try:
            hour, minute = map(int, str(value).split(":"))
            return hour * 60 + minute
        except (ValueError, TypeError):
            return np.nan

    def _add_sleep_duration(self, df):
        result = df.copy()
        sleep = result["sleep_time"].apply(self._time_to_minutes)
        wake = result["wake_time"].apply(self._time_to_minutes)
        duration = wake - sleep
        duration = duration.where(duration >= 0, duration + 24 * 60)
        result["sleep_duration_hours"] = duration / 60.0
        return result

    def load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.file_path.resolve()}")
        return pd.read_csv(self.file_path)

    def prepare_features(self, df):
        df = self._add_sleep_duration(df)
        drop_columns = [c for c in ["index", "student_id", "sleep_time", "wake_time"] if c in df.columns]
        df = df.drop(columns=drop_columns)
        feature_columns = self.numerical_features + self.nominal_features + self.ordinal_features + self.passthrough_features
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Features missing from dataset: {missing}")
        X = df[feature_columns].copy()
        y = pd.to_numeric(df[self.target_column], errors="coerce")
        valid = y.notna()
        return X.loc[valid], y.loc[valid]

    def create_preprocessor(self):
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
        transformers = [
            ("numerical", numeric, self.numerical_features),
            ("nominal", categorical, self.nominal_features),
        ]
        if self.ordinal_features:
            ordinal = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ])
            transformers.append(("ordinal", ordinal, self.ordinal_features))
        if self.passthrough_features:
            passthrough = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
            transformers.append(("passthrough", passthrough, self.passthrough_features))
        return ColumnTransformer(transformers=transformers, remainder="drop")
