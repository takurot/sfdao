from __future__ import annotations

import re
from enum import Enum
from typing import Callable

import pandas as pd

__all__ = ["ColumnType", "TypeDetector"]


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    PII = "pii"
    FREE_TEXT = "free_text"


class TypeDetector:
    """Detect column semantic types with lightweight heuristics."""

    EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.IGNORECASE)
    PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-\s]?)?(?:\(?\d{2,4}\)?[-\s]?){2,3}\d{2,4}")
    CREDIT_CARD_PATTERN = re.compile(r"(?:\d{4}[-\s]?){3}\d{4}")
    SAMPLE_SIZE = 100
    DATETIME_VALID_RATIO = 0.8
    NUMERIC_VALID_RATIO = 0.9
    MIN_UNIX_TIMESTAMP_MEDIAN = 1e9

    def detect(self, series: pd.Series, column_name: str) -> ColumnType:  # noqa: ARG002
        clean_series = series.dropna()
        if clean_series.empty:
            return ColumnType.FREE_TEXT

        checks: list[tuple[ColumnType, Callable[[pd.Series], bool]]] = [
            (ColumnType.PII, self._is_pii),
            (ColumnType.DATETIME, self._is_datetime),
            (ColumnType.NUMERIC, self._is_numeric),
            (ColumnType.CATEGORICAL, self._is_categorical),
        ]
        for column_type, checker in checks:
            if checker(clean_series):
                return column_type

        return ColumnType.FREE_TEXT

    @staticmethod
    def _valid_ratio(series: pd.Series) -> float:
        return float(series.notna().mean())

    @classmethod
    def _string_sample(cls, series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().head(cls.SAMPLE_SIZE)

    def _is_pii(self, series: pd.Series) -> bool:
        sample = self._string_sample(series)

        if sample.str.fullmatch(self.EMAIL_PATTERN).any():
            return True

        digit_counts = sample.str.count(r"\d")

        phone_candidates = sample[digit_counts >= 9]
        # Require non-digit separators to avoid false positives on raw numeric IDs/timestamps.
        phone_candidates = phone_candidates[phone_candidates.str.contains(r"[^\d]", regex=True)]
        if not phone_candidates.empty and phone_candidates.str.fullmatch(self.PHONE_PATTERN).any():
            return True

        credit_card_candidates = sample[digit_counts >= 15]
        if credit_card_candidates.str.fullmatch(self.CREDIT_CARD_PATTERN).any():
            return True

        return False

    def _is_datetime(self, series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        if pd.api.types.is_numeric_dtype(series):
            numeric_series = pd.to_numeric(series, errors="coerce")
            median_abs = numeric_series.abs().median()
            if (
                median_abs < self.MIN_UNIX_TIMESTAMP_MEDIAN
            ):  # below plausible UNIX timestamp (seconds)
                return False

            parsed = pd.to_datetime(numeric_series, errors="coerce", utc=True, unit="s")
            return self._valid_ratio(parsed) >= self.DATETIME_VALID_RATIO

        parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
        return self._valid_ratio(parsed) >= self.DATETIME_VALID_RATIO

    def _is_numeric(self, series: pd.Series) -> bool:
        if pd.api.types.is_numeric_dtype(series):
            return True

        normalized = series.astype(str).str.replace(",", "", regex=False).str.strip()
        coerced = pd.to_numeric(normalized, errors="coerce")
        return self._valid_ratio(coerced) >= self.NUMERIC_VALID_RATIO

    def _is_categorical(self, series: pd.Series) -> bool:
        total = len(series)
        if total == 0:
            return False

        unique = series.nunique(dropna=True)
        unique_ratio = unique / total
        return unique <= 20 and unique_ratio <= 0.5
