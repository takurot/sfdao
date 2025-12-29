import pandas as pd
from sfdao.guard.rules.uniqueness import UniqueRule
from sfdao.guard.rules.datetime import MonotonicDatetimeRule


def test_unique_rule():
    df = pd.DataFrame({"id": [1, 2, 3, 2, 4]})
    rule = UniqueRule(columns=["id"])
    violations = rule.validate(df)

    # Depending on implementation, it might flag one or both '2's.
    # Usually we flag the subsequent occurrences.
    assert len(violations) == 1
    assert violations[0].row_index == 3
    assert "not unique" in violations[0].message


def test_monotonic_datetime_rule():
    df = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-03"])}
    )
    rule = MonotonicDatetimeRule(columns=["timestamp"])
    violations = rule.validate(df)

    assert len(violations) == 1
    assert violations[0].row_index == 2
    assert "not monotonic" in violations[0].message
