import pandas as pd
from sfdao.guard.base import Rule, Violation


def test_violation_creation():
    v = Violation(
        column="amount", row_index=5, rule_name="NonNegative", message="Value -10 is negative"
    )
    assert v.column == "amount"
    assert v.row_index == 5
    assert v.rule_name == "NonNegative"


def test_rule_base_class():
    class MyRule(Rule):
        def validate(self, df: pd.DataFrame) -> list[Violation]:
            return []

    rule = MyRule(columns=["col1"])
    assert rule.columns == ["col1"]
    assert rule.validate(pd.DataFrame()) == []
