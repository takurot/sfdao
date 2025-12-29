from typing import List
import pandas as pd
from sfdao.guard.base import Rule, Violation


class MonotonicDatetimeRule(Rule):
    def validate(self, df: pd.DataFrame) -> List[Violation]:
        violations = []
        for col in self.columns:
            if col not in df.columns:
                continue

            series = pd.to_datetime(df[col])
            # For simplicity, let's just check if any value is less than the maximum seen so far
            current_max = None
            for idx, val in series.items():
                if current_max is not None and val < current_max:
                    violations.append(
                        Violation(
                            column=col,
                            row_index=int(str(idx)),  # Fixed for mypy
                            rule_name=self.name,
                            message=(
                                f"Value {val} in column '{col}' is not monotonic "
                                f"(less than previous max {current_max})"
                            ),
                        )
                    )
                if current_max is None or val > current_max:
                    current_max = val

        return violations
