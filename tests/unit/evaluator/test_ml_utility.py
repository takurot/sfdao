"""Unit tests for ML Utility Evaluator.

Tests cover TSTR (Train on Synthetic, Test on Real) evaluation logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sfdao.evaluator.ml_utility import (
    MLUtilityEvaluator,
    MLUtilityResult,
)


class TestMLUtilityResult:
    """Test cases for MLUtilityResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating a valid MLUtilityResult."""
        result = MLUtilityResult(
            tstr_auc=0.85,
            tstr_f1=0.80,
            trtr_auc=0.90,
            trtr_f1=0.88,
            utility_ratio=0.944,
            model_type="random_forest",
            target_column="Class",
            n_features=10,
        )
        assert result.tstr_auc == 0.85
        assert result.tstr_f1 == 0.80
        assert result.trtr_auc == 0.90
        assert result.trtr_f1 == 0.88
        assert result.utility_ratio == pytest.approx(0.944, rel=0.01)
        assert result.model_type == "random_forest"
        assert result.target_column == "Class"
        assert result.n_features == 10


class TestMLUtilityEvaluator:
    """Test cases for MLUtilityEvaluator class."""

    @pytest.fixture
    def sample_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create sample real and synthetic data for testing."""
        np.random.seed(42)
        n_samples = 200

        # Generate simple linearly separable data
        X_real = np.random.randn(n_samples, 3)
        y_real = (X_real[:, 0] + X_real[:, 1] > 0).astype(int)

        real_df = pd.DataFrame(X_real, columns=["f1", "f2", "f3"])
        real_df["Class"] = y_real

        # Synthetic data from similar distribution
        X_synthetic = np.random.randn(n_samples, 3)
        y_synthetic = (X_synthetic[:, 0] + X_synthetic[:, 1] > 0).astype(int)

        synthetic_df = pd.DataFrame(X_synthetic, columns=["f1", "f2", "f3"])
        synthetic_df["Class"] = y_synthetic

        return real_df, synthetic_df

    def test_evaluate_basic(self, sample_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """Test basic TSTR evaluation with similar distributions."""
        real_df, synthetic_df = sample_data

        evaluator = MLUtilityEvaluator()
        result = evaluator.evaluate(
            real_df=real_df,
            synthetic_df=synthetic_df,
            target_column="Class",
        )

        assert isinstance(result, MLUtilityResult)
        assert 0.0 <= result.tstr_auc <= 1.0
        assert 0.0 <= result.tstr_f1 <= 1.0
        assert 0.0 <= result.trtr_auc <= 1.0
        assert 0.0 <= result.trtr_f1 <= 1.0
        assert result.utility_ratio > 0
        assert result.model_type == "random_forest"
        assert result.target_column == "Class"
        assert result.n_features == 3

    def test_evaluate_identical_data(self, sample_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """Test evaluation when synthetic data equals real data (high utility)."""
        real_df, _ = sample_data

        evaluator = MLUtilityEvaluator()
        result = evaluator.evaluate(
            real_df=real_df,
            synthetic_df=real_df.copy(),
            target_column="Class",
        )

        # With identical data, TSTR should be close to TRTR
        assert result.utility_ratio > 0.8

    def test_evaluate_with_logistic_regression(
        self, sample_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test evaluation using logistic regression model."""
        real_df, synthetic_df = sample_data

        evaluator = MLUtilityEvaluator(model_type="logistic_regression")
        result = evaluator.evaluate(
            real_df=real_df,
            synthetic_df=synthetic_df,
            target_column="Class",
        )

        assert result.model_type == "logistic_regression"
        assert 0.0 <= result.tstr_auc <= 1.0

    def test_evaluate_with_feature_columns(
        self, sample_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test evaluation with explicit feature column specification."""
        real_df, synthetic_df = sample_data

        evaluator = MLUtilityEvaluator()
        result = evaluator.evaluate(
            real_df=real_df,
            synthetic_df=synthetic_df,
            target_column="Class",
            feature_columns=["f1", "f2"],  # Only use 2 features
        )

        assert result.n_features == 2

    def test_missing_target_column_error(
        self, sample_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test error when target column is missing."""
        real_df, synthetic_df = sample_data

        evaluator = MLUtilityEvaluator()
        with pytest.raises(ValueError, match="Target column 'missing' not found"):
            evaluator.evaluate(
                real_df=real_df,
                synthetic_df=synthetic_df,
                target_column="missing",
            )

    def test_target_column_missing_in_synthetic(
        self, sample_data: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
        """Test error when target column is missing in synthetic data."""
        real_df, synthetic_df = sample_data
        synthetic_df = synthetic_df.drop(columns=["Class"])

        evaluator = MLUtilityEvaluator()
        with pytest.raises(ValueError, match="Target column 'Class' not found"):
            evaluator.evaluate(
                real_df=real_df,
                synthetic_df=synthetic_df,
                target_column="Class",
            )

    def test_single_class_in_real_data(self) -> None:
        """Test error when real data has only one class."""
        real_df = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [5, 6, 7, 8], "Class": [0, 0, 0, 0]})
        synthetic_df = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [5, 6, 7, 8], "Class": [0, 1, 0, 1]})

        evaluator = MLUtilityEvaluator()
        with pytest.raises(ValueError, match="at least 2 classes"):
            evaluator.evaluate(
                real_df=real_df,
                synthetic_df=synthetic_df,
                target_column="Class",
            )

    def test_single_class_in_synthetic_data(self) -> None:
        """Test error when synthetic data has only one class."""
        real_df = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [5, 6, 7, 8], "Class": [0, 1, 0, 1]})
        synthetic_df = pd.DataFrame({"f1": [1, 2, 3, 4], "f2": [5, 6, 7, 8], "Class": [0, 0, 0, 0]})

        evaluator = MLUtilityEvaluator()
        with pytest.raises(ValueError, match="at least 2 classes"):
            evaluator.evaluate(
                real_df=real_df,
                synthetic_df=synthetic_df,
                target_column="Class",
            )

    def test_invalid_model_type(self) -> None:
        """Test error for invalid model type."""
        with pytest.raises(ValueError, match="Invalid model_type"):
            MLUtilityEvaluator(model_type="invalid_model")

    def test_evaluate_different_distributions(self) -> None:
        """Test evaluation when synthetic data differs significantly."""
        np.random.seed(42)
        n_samples = 200

        # Real data: linearly separable
        X_real = np.random.randn(n_samples, 3)
        y_real = (X_real[:, 0] + X_real[:, 1] > 0).astype(int)
        real_df = pd.DataFrame(X_real, columns=["f1", "f2", "f3"])
        real_df["Class"] = y_real

        # Synthetic data: random labels (no relationship with features)
        X_synthetic = np.random.randn(n_samples, 3)
        y_synthetic = np.random.randint(0, 2, n_samples)
        synthetic_df = pd.DataFrame(X_synthetic, columns=["f1", "f2", "f3"])
        synthetic_df["Class"] = y_synthetic

        evaluator = MLUtilityEvaluator()
        result = evaluator.evaluate(
            real_df=real_df,
            synthetic_df=synthetic_df,
            target_column="Class",
        )

        # With random labels in synthetic, TSTR should be much worse than TRTR
        assert result.utility_ratio < result.trtr_auc  # TSTR worse than baseline

    def test_evaluate_with_nan_values(self, sample_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """Test evaluation handles NaN values gracefully."""
        real_df, synthetic_df = sample_data

        # Add some NaN values
        real_df = real_df.copy()
        real_df.loc[0, "f1"] = np.nan
        synthetic_df = synthetic_df.copy()
        synthetic_df.loc[1, "f2"] = np.nan

        evaluator = MLUtilityEvaluator()
        result = evaluator.evaluate(
            real_df=real_df,
            synthetic_df=synthetic_df,
            target_column="Class",
        )

        # Should still produce valid results
        assert 0.0 <= result.tstr_auc <= 1.0

    def test_test_size_parameter(self, sample_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """Test evaluation with custom test_size parameter."""
        real_df, synthetic_df = sample_data

        evaluator = MLUtilityEvaluator(test_size=0.3)
        result = evaluator.evaluate(
            real_df=real_df,
            synthetic_df=synthetic_df,
            target_column="Class",
        )

        assert isinstance(result, MLUtilityResult)
