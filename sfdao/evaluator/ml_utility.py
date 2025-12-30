"""Machine Learning Utility Evaluator module.

This module implements TSTR (Train on Synthetic, Test on Real) evaluation
to measure if synthetic data preserves ML utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import f1_score, roc_auc_score  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

__all__ = [
    "MLUtilityResult",
    "MLUtilityEvaluator",
]


@dataclass(frozen=True)
class MLUtilityResult:
    """Result of ML Utility evaluation.

    Attributes:
        tstr_auc: AUC-ROC score for Train-on-Synthetic, Test-on-Real.
        tstr_f1: F1 score for Train-on-Synthetic, Test-on-Real.
        trtr_auc: AUC-ROC baseline (Train-on-Real, Test-on-Real).
        trtr_f1: F1 baseline (Train-on-Real, Test-on-Real).
        utility_ratio: TSTR_AUC / TRTR_AUC ratio.
        model_type: Type of model used for evaluation.
        target_column: Name of the target column.
        n_features: Number of features used in the model.
    """

    tstr_auc: float
    tstr_f1: float
    trtr_auc: float
    trtr_f1: float
    utility_ratio: float
    model_type: str
    target_column: str
    n_features: int


ModelType = Literal["random_forest", "logistic_regression"]


class MLUtilityEvaluator:
    """Evaluator for measuring ML utility of synthetic data using TSTR approach.

    The TSTR (Train on Synthetic, Test on Real) approach:
    1. Train a model on synthetic data
    2. Test it on real data
    3. Compare with TRTR (Train on Real, Test on Real) baseline

    A high TSTR/TRTR ratio indicates the synthetic data preserves ML-relevant patterns.
    """

    VALID_MODEL_TYPES = {"random_forest", "logistic_regression"}

    def __init__(
        self,
        model_type: ModelType = "random_forest",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """Initialize the evaluator.

        Args:
            model_type: Type of model to use ("random_forest" or "logistic_regression").
            test_size: Fraction of real data to use for testing (default: 0.2).
            random_state: Random seed for reproducibility.

        Raises:
            ValueError: If model_type is not supported.
        """
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type '{model_type}'. "
                f"Supported types: {sorted(self.VALID_MODEL_TYPES)}"
            )
        self._model_type: ModelType = model_type
        self._test_size = test_size
        self._random_state = random_state

    def evaluate(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        target_column: str,
        feature_columns: Sequence[str] | None = None,
    ) -> MLUtilityResult:
        """Evaluate ML utility of synthetic data.

        Args:
            real_df: Real data DataFrame.
            synthetic_df: Synthetic data DataFrame.
            target_column: Name of the target column for classification.
            feature_columns: List of feature columns to use. If None, all numeric
                columns except target will be used.

        Returns:
            MLUtilityResult with TSTR and TRTR metrics.

        Raises:
            ValueError: If target_column is missing or data has fewer than 2 classes.
        """
        # Validate target column exists
        self._validate_target_column(real_df, synthetic_df, target_column)

        # Determine feature columns
        features = self._get_feature_columns(real_df, target_column, feature_columns)

        # Validate shared features exist in synthetic data
        features = [f for f in features if f in synthetic_df.columns]
        if not features:
            raise ValueError("No shared feature columns between real and synthetic data.")

        # Prepare data (drop NaN values)
        real_clean = real_df[features + [target_column]].dropna()
        synthetic_clean = synthetic_df[features + [target_column]].dropna()

        X_real = real_clean[features].to_numpy(dtype=float)
        y_real = real_clean[target_column].to_numpy()
        X_synthetic = synthetic_clean[features].to_numpy(dtype=float)
        y_synthetic = synthetic_clean[target_column].to_numpy()

        # Validate class counts
        self._validate_class_counts(y_real, y_synthetic, target_column)

        # Split real data for testing
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real,
            y_real,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=y_real,
        )

        # Train TRTR model (baseline)
        trtr_model = self._create_model()
        trtr_model.fit(X_real_train, y_real_train)
        trtr_auc, trtr_f1 = self._evaluate_model(trtr_model, X_real_test, y_real_test)

        # Train TSTR model (synthetic -> real)
        tstr_model = self._create_model()
        tstr_model.fit(X_synthetic, y_synthetic)
        tstr_auc, tstr_f1 = self._evaluate_model(tstr_model, X_real_test, y_real_test)

        # Calculate utility ratio
        utility_ratio = tstr_auc / trtr_auc if trtr_auc > 0 else 0.0

        return MLUtilityResult(
            tstr_auc=tstr_auc,
            tstr_f1=tstr_f1,
            trtr_auc=trtr_auc,
            trtr_f1=trtr_f1,
            utility_ratio=utility_ratio,
            model_type=self._model_type,
            target_column=target_column,
            n_features=len(features),
        )

    def _validate_target_column(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        target_column: str,
    ) -> None:
        """Validate that target column exists in both DataFrames."""
        if target_column not in real_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in real data.")
        if target_column not in synthetic_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in synthetic data.")

    def _get_feature_columns(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Sequence[str] | None,
    ) -> list[str]:
        """Get list of feature columns to use."""
        if feature_columns is not None:
            return list(feature_columns)

        # Use all numeric columns except target
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        return [col for col in numeric_cols if col != target_column]

    def _validate_class_counts(
        self,
        y_real: NDArray[np.generic],
        y_synthetic: NDArray[np.generic],
        target_column: str,
    ) -> None:
        """Validate that both datasets have at least 2 classes."""
        real_classes = len(np.unique(y_real))
        synthetic_classes = len(np.unique(y_synthetic))

        if real_classes < 2:
            raise ValueError(
                f"Real data must have at least 2 classes for '{target_column}'. "
                f"Found {real_classes} class(es)."
            )
        if synthetic_classes < 2:
            raise ValueError(
                f"Synthetic data must have at least 2 classes for '{target_column}'. "
                f"Found {synthetic_classes} class(es)."
            )

    def _create_model(self) -> RandomForestClassifier | LogisticRegression:
        """Create a new model instance based on model_type."""
        if self._model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self._random_state,
                n_jobs=-1,
            )
        else:
            return LogisticRegression(
                random_state=self._random_state,
                max_iter=1000,
                solver="lbfgs",
            )

    def _evaluate_model(
        self,
        model: RandomForestClassifier | LogisticRegression,
        X_test: NDArray[np.float64],
        y_test: NDArray[np.generic],
    ) -> tuple[float, float]:
        """Evaluate model and return AUC-ROC and F1 scores."""
        y_pred = model.predict(X_test)

        # For AUC-ROC, we need probability predictions
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            # Handle binary and multiclass
            if y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]
            try:
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
            except ValueError:
                # Fallback for edge cases
                auc = 0.5
        else:
            auc = 0.5

        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0.0)

        return float(auc), float(f1)
