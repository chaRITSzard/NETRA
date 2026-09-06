from pathlib import Path

import numpy as np
import pandas as pd
import yaml

class NETRApreprocessor:

    def __init__(self, config_path):
        self.config_path = Path(config_path)

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        features = self.config["features"]

        self.drop_features = features["drop"]
        self.keep_features = features["keep"]

        self.log1p_features = features["transform"]["log1p"]
        self.frequency_features = features["transform"]["frequency_encode"]

        self.target_column = self.config["target"]["output_column"]
        self.source_target_column = self.config["target"]["source_column"]

        self.frequency_maps = {}
        self.output_columns_ = None
        self.is_fitted = False

    def fit(self, X):
    #Learn preprocessing parameters from training data only.
        X = X.copy()
        for feature in self.frequency_features:
            frequencies = X[feature].value_counts(normalize=True)

            self.frequency_maps[feature] = frequencies.to_dict()

        self.is_fitted = True

        transformed = self._transform(X)
        self.output_columns_ = transformed.columns.tolist()

        return self

    def transform(self, X):
    #Apply previously learned preprocessing
        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor must be fitted before transform()"
            )
        X = X.copy()

        transformed = self._transform(X)

        transformed = transformed.reindex(
            columns = self.output_columns_,
            fill_value = 0
        )
        return transformed

    def fit_transform(self, X):
    #Fit on X and immediately transform it.
        self.fit(X)
        return self.transform(X)

    def _transform(self, X):
        X = X.copy()
        # Remove features selected during Phase 1.
        X = X.drop(columns=self.drop_features, errors="ignore")

        # Log-transform heavy-tailed numeric features.
        for feature in self.log1p_features:
            if feature in X.columns:
                X[feature] = np.log1p(X[feature])

        # Frequency encode high-cardinality categorical features.
        for feature in self.frequency_features:
            if feature in X.columns:
                mapping = self.frequency_maps[feature]

                # Unseen categories receive frequency 0.
                X[feature] = X[feature].map(mapping).fillna(0.0)

        # One-hot encode low-cardinality categorical features.
        categorical_features = [
            feature
            for feature in ["protocol_type", "flag"]
            if feature in X.columns
        ]

        if categorical_features:
            X = pd.get_dummies(
                X,
                columns=categorical_features,
                dtype=float
            )

        # Are all remaining values are numeric?
        X = X.astype(float)

        return X