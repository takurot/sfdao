from typing import List
import pandas as pd
from sfdao.guard.base import Rule, Violation


class UniqueRule(Rule):
    def validate(self, df: pd.DataFrame) -> List[Violation]:
        violations = []
        for col in self.columns:
            if col not in df.columns:
                continue

            # Find duplicates
            duplicated = df[col].duplicated(keep="first")
            for idx in df.index[duplicated]:
                violations.append(
                    Violation(
                        column=col,
                        row_index=int(idx),
                        rule_name=self.name,
                        message=f"Value {df.at[idx, col]} in column '{col}' is not unique",
                    )
                )
        return violations
