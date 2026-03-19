"""Unit tests for GuardEngine fill_to_target feature (Issue #38)."""

import pandas as pd
from sfdao.guard.base import GuardEngine, GuardPolicy, Rule, Violation


class AlwaysViolateFirstHalfRule(Rule):
    """Test rule that marks first half of rows as violations."""

    def validate(self, df: pd.DataFrame) -> list[Violation]:
        violations = []
        half = len(df) // 2
        for i in range(half):
            violations.append(
                Violation(
                    column="value",
                    row_index=i,
                    rule_name="AlwaysViolateFirstHalf",
                    message=f"Row {i} always violates",
                )
            )
        return violations


class AllViolateRule(Rule):
    """Test rule that marks all rows as violations."""

    def validate(self, df: pd.DataFrame) -> list[Violation]:
        return [
            Violation(
                column="value",
                row_index=i,
                rule_name="AllViolate",
                message=f"Row {i} always violates",
            )
            for i in range(len(df))
        ]


def test_guard_engine_fill_to_target_default_false():
    engine = GuardEngine(rules=[], policy=GuardPolicy.EXCLUDE)
    assert engine.fill_to_target is False


def test_guard_engine_fill_to_target_can_be_set_true():
    engine = GuardEngine(rules=[], policy=GuardPolicy.EXCLUDE, fill_to_target=True)
    assert engine.fill_to_target is True


def test_guard_engine_exclude_without_fill_reduces_rows():
    """Without fill_to_target, EXCLUDE mode reduces row count."""
    rule = AlwaysViolateFirstHalfRule(columns=["value"])
    engine = GuardEngine(rules=[rule], policy=GuardPolicy.EXCLUDE, fill_to_target=False)

    df = pd.DataFrame({"value": list(range(10))})
    result_df, violations = engine.apply(df)

    assert len(result_df) < 10
    assert len(violations) == 5


def test_guard_engine_fill_to_target_does_not_affect_apply_directly():
    """fill_to_target is a flag for generators to use; apply() itself doesn't fill."""
    rule = AlwaysViolateFirstHalfRule(columns=["value"])
    engine = GuardEngine(rules=[rule], policy=GuardPolicy.EXCLUDE, fill_to_target=True)

    df = pd.DataFrame({"value": list(range(10))})
    result_df, violations = engine.apply(df)

    # apply() still removes rows — the filling is done by the generator
    assert len(result_df) == 5
    assert len(violations) == 5
