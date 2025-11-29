from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["CSVLoader"]


class CSVLoader:
    """Load CSV files into pandas DataFrames with light type coercion."""

    def load(self, filepath: str | Path) -> pd.DataFrame:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        try:
            df = pd.read_csv(path, on_bad_lines="error", engine="python")
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise ValueError(f"Failed to parse CSV: {path}") from exc

        return self._coerce_types(df)

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attempt to coerce common column types (datetime, numeric)."""

        datetime_cols = [col for col in df.columns if self._looks_like_datetime(col)]
        for col in datetime_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

        numeric_cols = [
            col
            for col in df.columns
            if col not in datetime_cols and self._looks_like_numeric(df[col])
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    @staticmethod
    def _looks_like_datetime(column_name: str) -> bool:
        lowered = column_name.lower()
        return any(token in lowered for token in ("time", "date", "timestamp"))

    @staticmethod
    def _looks_like_numeric(series: pd.Series) -> bool:
        sample = series.dropna().head(20)
        if sample.empty:
            return True
        try:
            pd.to_numeric(sample)
            return True
        except (TypeError, ValueError):
            return False
