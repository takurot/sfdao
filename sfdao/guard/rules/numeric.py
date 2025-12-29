from typing import List, Optional
import pandas as pd
from sfdao.guard.base import Rule, Violation


class NumericRangeRule(Rule):
    def __init__(
        self,
        columns: List[str],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        name: Optional[str] = None,
    ):
        super().__init__(columns, name)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, df: pd.DataFrame) -> List[Violation]:
        violations = []
        for col in self.columns:
            if col not in df.columns:
                continue

            series = df[col]
            if self.min_value is not None:
                invalid_min = series < self.min_value
                for idx in series.index[invalid_min]:
                    violations.append(
                        Violation(
                            column=col,
                            row_index=int(idx),
                            rule_name=self.name,
                            message=f"Value {series[idx]} is below minimum {self.min_value}",
                        )
                    )

            if self.max_value is not None:
                invalid_max = series > self.max_value
                for idx in series.index[invalid_max]:
                    violations.append(
                        Violation(
                            column=col,
                            row_index=int(idx),
                            rule_name=self.name,
                            message=f"Value {series[idx]} is above maximum {self.max_value}",
                        )
                    )
        return violations


class NonNegativeRule(NumericRangeRule):
    def __init__(self, columns: List[str], name: Optional[str] = None):
        super().__init__(columns, min_value=0, name=name or "NonNegativeRule")
