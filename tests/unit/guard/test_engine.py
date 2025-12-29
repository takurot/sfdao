import pandas as pd
from sfdao.guard.base import GuardPolicy
from sfdao.guard.engine import GuardEngine
from sfdao.guard.rules.numeric import NumericRangeRule, NonNegativeRule


def test_guard_engine_detect_only():
    df = pd.DataFrame({"amount": [10, -5, 20], "balance": [100, 50, -10]})

    rules = [NonNegativeRule(columns=["amount"]), NonNegativeRule(columns=["balance"])]

    engine = GuardEngine(rules=rules, policy=GuardPolicy.DETECT)
    cleaned_df, violations = engine.apply(df)

    assert len(violations) == 2
    assert cleaned_df.equals(df)  # DETECT should not modify data


def test_guard_engine_exclude_policy():
    df = pd.DataFrame({"amount": [10, -5, 20], "balance": [100, 50, -10]})

    rules = [NonNegativeRule(columns=["amount", "balance"])]

    engine = GuardEngine(rules=rules, policy=GuardPolicy.EXCLUDE)
    cleaned_df, violations = engine.apply(df)

    assert len(cleaned_df) == 1
    assert cleaned_df.index.tolist() == [0]
    assert len(violations) == 2


def test_guard_engine_clip_policy():
    df = pd.DataFrame(
        {
            "amount": [10, -5, 100],
        }
    )

    # Clip only works if rule supports it. For now let's see how we implement it.
    # We might need a separate interface for rules that support clipping.
    rules = [NumericRangeRule(columns=["amount"], min_value=0, max_value=50)]

    engine = GuardEngine(rules=rules, policy=GuardPolicy.CLIP)
    cleaned_df, violations = engine.apply(df)

    assert cleaned_df.at[1, "amount"] == 0
    assert cleaned_df.at[2, "amount"] == 50
    assert len(violations) == 2
