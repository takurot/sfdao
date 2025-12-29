import pandas as pd
from sfdao.guard.rules.numeric import NumericRangeRule, NonNegativeRule


def test_numeric_range_rule():
    df = pd.DataFrame({"amount": [10, 20, -5, 100, 30]})
    rule = NumericRangeRule(columns=["amount"], min_value=0, max_value=50)
    violations = rule.validate(df)

    assert len(violations) == 2
    assert violations[0].row_index == 2  # -5 is below min
    assert violations[1].row_index == 3  # 100 is above max


def test_non_negative_rule():
    df = pd.DataFrame({"balance": [100, 0, -1]})
    rule = NonNegativeRule(columns=["balance"])
    violations = rule.validate(df)

    assert len(violations) == 1
    assert violations[0].row_index == 2
