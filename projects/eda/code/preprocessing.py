"""Pré-processamento reutilizável para o projeto Online Shoppers.

O módulo não ajusta nada ao ser importado. Chame ``fit`` somente com o treino;
depois use o mesmo objeto para transformar o teste e dados futuros.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


SKEWED_NUMERIC = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
]
BOUNDED_NUMERIC = ["BounceRates", "ExitRates", "SpecialDay"]
CATEGORICAL = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]
EXCLUDED = ["PageValues"]  # disponibilidade pré-desfecho não demonstrada
MODEL_FEATURES = SKEWED_NUMERIC + BOUNDED_NUMERIC + CATEGORICAL


class QuantileCapper(BaseEstimator, TransformerMixin):
    """Limita valores não negativos ao percentil superior aprendido no treino."""

    def __init__(self, upper_quantile: float = 0.99):
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        if not 0 < self.upper_quantile < 1:
            raise ValueError("upper_quantile deve estar entre 0 e 1")
        data = np.asarray(X, dtype=float)
        self.upper_ = np.nanquantile(data, self.upper_quantile, axis=0)
        self.n_features_in_ = data.shape[1]
        self.feature_names_in_ = np.asarray(
            getattr(X, "columns", [f"x{i}" for i in range(data.shape[1])]),
            dtype=object,
        )
        return self

    def transform(self, X):
        data = np.asarray(X, dtype=float)
        if data.shape[1] != self.n_features_in_:
            raise ValueError("Número de colunas diferente do usado em fit")
        return np.clip(data, 0, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            self.feature_names_in_ if input_features is None else input_features,
            dtype=object,
        )


def build_preprocessor() -> ColumnTransformer:
    """Constrói um pipeline não ajustado, seguro para categorias desconhecidas."""
    skewed = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("cap", QuantileCapper(upper_quantile=0.99)),
            (
                "log1p",
                FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            ),
            ("scale", StandardScaler()),
        ]
    )
    bounded = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("skew", skewed, SKEWED_NUMERIC),
            ("bounded", bounded, BOUNDED_NUMERIC),
            ("cat", categorical, CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
