"""Integration tests for guard fill_to_target feature (Issue #38).

Tests that BaselineGenerator maintains n_samples when guard mode=exclude
and fill_to_target=True.
"""

import pandas as pd
import pytest
from sfdao.generator.baseline import BaselineGenerator
from sfdao.guard.base import GuardEngine, GuardPolicy, Rule, Violation


class NegativeValueRule(Rule):
    """Rule that flags rows where value < 0 as violations."""

    def validate(self, df: pd.DataFrame) -> list[Violation]:
        violations = []
        for i, val in enumerate(df["value"]):
            if val < 0:
                violations.append(
                    Violation(
                        column="value",
                        row_index=i,
                        rule_name="NegativeValueRule",
                        message=f"Value {val} is negative",
                    )
                )
        return violations


class RejectEvenIndexRule(Rule):
    """Rule that always rejects even-indexed rows (50% rejection rate)."""

    def validate(self, df: pd.DataFrame) -> list[Violation]:
        return [
            Violation(
                column="value",
                row_index=i,
                rule_name="RejectEvenIndex",
                message=f"Row {i} is even-indexed",
            )
            for i in range(0, len(df), 2)
        ]


class RejectNinetyNinePercentRule(Rule):
    """Rule that keeps only every 100th row (99% rejection rate)."""

    def validate(self, df: pd.DataFrame) -> list[Violation]:
        return [
            Violation(
                column="value",
                row_index=i,
                rule_name="RejectNinetyNinePercent",
                message=f"Row {i} rejected",
            )
            for i in range(len(df))
            if i % 100 != 0
        ]


@pytest.fixture
def real_df() -> pd.DataFrame:
    return pd.DataFrame({"value": [10.0, 20.0, 30.0, 40.0, 50.0] * 20})


def test_baseline_generator_exclude_without_fill_reduces_count(real_df: pd.DataFrame):
    """Without fill_to_target, EXCLUDE mode reduces the sample count."""
    guard = GuardEngine(
        rules=[RejectEvenIndexRule(columns=["value"])],
        policy=GuardPolicy.EXCLUDE,
        fill_to_target=False,
    )
    gen = BaselineGenerator(seed=42, guard=guard)
    gen.fit(real_df)

    n_samples = 100
    result = gen.sample(n_samples)

    # Approximately half the rows are rejected, so count < n_samples
    assert len(result) < n_samples


def test_baseline_generator_exclude_with_fill_maintains_count(real_df: pd.DataFrame):
    """With fill_to_target=True, EXCLUDE mode fills up to n_samples."""
    guard = GuardEngine(
        rules=[RejectEvenIndexRule(columns=["value"])],
        policy=GuardPolicy.EXCLUDE,
        fill_to_target=True,
    )
    gen = BaselineGenerator(seed=42, guard=guard)
    gen.fit(real_df)

    n_samples = 100
    result = gen.sample(n_samples)

    assert len(result) == n_samples


def test_baseline_generator_fill_to_target_with_low_rejection(real_df: pd.DataFrame):
    """fill_to_target works when rejection rate is low (a few rows removed)."""
    real_df_positive = pd.DataFrame({"value": list(range(1, 101))})

    guard = GuardEngine(
        rules=[NegativeValueRule(columns=["value"])],
        policy=GuardPolicy.EXCLUDE,
        fill_to_target=True,
    )
    gen = BaselineGenerator(seed=0, guard=guard)
    gen.fit(real_df_positive)

    n_samples = 50
    result = gen.sample(n_samples)

    assert len(result) == n_samples


def test_baseline_generator_fill_to_target_with_high_rejection(real_df: pd.DataFrame):
    """fill_to_target still reaches n_samples with a very high rejection rate."""
    guard = GuardEngine(
        rules=[RejectNinetyNinePercentRule(columns=["value"])],
        policy=GuardPolicy.EXCLUDE,
        fill_to_target=True,
    )
    gen = BaselineGenerator(seed=42, guard=guard)
    gen.fit(real_df)

    n_samples = 100
    result = gen.sample(n_samples)

    assert len(result) == n_samples


def test_baseline_generator_fill_to_target_config_driven():
    """fill_to_target via GuardSettings flows through the factory correctly."""
    from sfdao.config.models import GuardSettings
    from sfdao.guard.factory import create_guard_engine

    settings = GuardSettings(
        enabled=True,
        mode="exclude",
        fill_to_target=True,
        params={"non_negative": [{"columns": ["value"]}]},
    )
    engine = create_guard_engine(settings)

    assert engine.fill_to_target is True
    assert engine.policy == GuardPolicy.EXCLUDE


def test_baseline_generator_fill_to_target_false_config_driven():
    """fill_to_target defaults to False in GuardSettings."""
    from sfdao.config.models import GuardSettings
    from sfdao.guard.factory import create_guard_engine

    settings = GuardSettings(enabled=True, mode="exclude")
    engine = create_guard_engine(settings)

    assert engine.fill_to_target is False


def test_baseline_generator_fill_to_target_non_exclude_mode_no_fill(real_df: pd.DataFrame):
    """fill_to_target has no effect when mode is not 'exclude'."""
    guard = GuardEngine(
        rules=[RejectEvenIndexRule(columns=["value"])],
        policy=GuardPolicy.DETECT,
        fill_to_target=True,
    )
    gen = BaselineGenerator(seed=42, guard=guard)
    gen.fit(real_df)

    n_samples = 50
    result = gen.sample(n_samples)

    # DETECT mode doesn't remove rows
    assert len(result) == n_samples


def test_baseline_generator_fill_to_target_returns_typed_empty_frame_when_all_rows_rejected():
    """When all rows are rejected, fill_to_target stops and preserves schema."""
    from sfdao.guard.base import Rule

    class AllViolateRule(Rule):
        def validate(self, df: pd.DataFrame) -> list[Violation]:
            return [
                Violation(
                    column="value",
                    row_index=i,
                    rule_name="AllViolate",
                    message="always",
                )
                for i in range(len(df))
            ]

    guard = GuardEngine(
        rules=[AllViolateRule(columns=["value"])],
        policy=GuardPolicy.EXCLUDE,
        fill_to_target=True,
    )
    real_df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    gen = BaselineGenerator(seed=42, guard=guard)
    gen.fit(real_df)

    # Should not raise, should return empty DataFrame
    result = gen.sample(10)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert result["value"].dtype == real_df["value"].dtype
