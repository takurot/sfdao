from sfdao.config.models import GuardSettings
from sfdao.guard.base import GuardEngine, GuardPolicy, Rule
from sfdao.guard.rules.numeric import NumericRangeRule, NonNegativeRule
from sfdao.guard.rules.uniqueness import UniqueRule
from sfdao.guard.rules.datetime import MonotonicDatetimeRule


def create_guard_engine(settings: GuardSettings) -> GuardEngine:
    if not settings.enabled:
        return GuardEngine(rules=[], policy=GuardPolicy.DETECT)

    rules: list[Rule] = []
    params = settings.params or {}

    if "numeric_range" in params:
        for p in params["numeric_range"]:
            rules.append(
                NumericRangeRule(
                    columns=p["columns"], min_value=p.get("min"), max_value=p.get("max")
                )
            )

    if "non_negative" in params:
        for p in params["non_negative"]:
            rules.append(NonNegativeRule(columns=p["columns"]))

    if "unique" in params:
        for p in params["unique"]:
            rules.append(UniqueRule(columns=p["columns"]))

    if "monotonic_datetime" in params:
        for p in params["monotonic_datetime"]:
            rules.append(MonotonicDatetimeRule(columns=p["columns"]))

    policy = GuardPolicy(settings.mode)
    return GuardEngine(rules=rules, policy=policy)
