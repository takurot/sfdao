from enum import Enum
from typing import List, Optional, Tuple
from pydantic import BaseModel
import pandas as pd


class GuardPolicy(Enum):
    DETECT = "detect"
    EXCLUDE = "exclude"
    CLIP = "clip"
    CORRECT = "correct"


class Violation(BaseModel):
    column: str
    row_index: int
    rule_name: str
    message: str


class Rule:
    def __init__(self, columns: List[str], name: Optional[str] = None):
        self.columns = columns
        self.name = name or self.__class__.__name__

    def validate(self, df: pd.DataFrame) -> List[Violation]:
        """Validate the dataframe against the rule and return a list of violations."""
        raise NotImplementedError("Subclasses must implement validate()")


class GuardEngine:
    def __init__(self, rules: List[Rule], policy: GuardPolicy = GuardPolicy.DETECT):
        self.rules = rules
        self.policy = policy

    def apply(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Violation]]:
        """Apply rules and policy to the dataframe."""
        all_violations = []
        for rule in self.rules:
            violations = rule.validate(df)
            all_violations.extend(violations)

        if self.policy == GuardPolicy.DETECT:
            return df, all_violations

        cleaned_df = df.copy()

        if self.policy == GuardPolicy.EXCLUDE:
            # Drop rows with any violations
            bad_indices = set(v.row_index for v in all_violations)
            cleaned_df = cleaned_df.drop(index=list(bad_indices)).reset_index(drop=True)

        elif self.policy == GuardPolicy.CLIP:
            # CLIP only applies to rules that support it (like NumericRangeRule)
            # We import here or check by attribute to avoid circularity if needed
            from sfdao.guard.rules.numeric import NumericRangeRule

            for rule in self.rules:
                if isinstance(rule, NumericRangeRule):
                    for col in rule.columns:
                        if col in cleaned_df.columns:
                            if rule.min_value is not None:
                                cleaned_df[col] = cleaned_df[col].clip(lower=rule.min_value)
                            if rule.max_value is not None:
                                cleaned_df[col] = cleaned_df[col].clip(upper=rule.max_value)

        return cleaned_df, all_violations
