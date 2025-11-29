from __future__ import annotations

import re
from enum import Enum

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

    def detect(self, series: pd.Series, column_name: str) -> ColumnType:  # noqa: ARG002
        clean_series = series.dropna()
        if clean_series.empty:
            return ColumnType.FREE_TEXT

        if self._is_pii(clean_series):
            return ColumnType.PII

        if self._is_datetime(clean_series):
            return ColumnType.DATETIME

        if self._is_numeric(clean_series):
            return ColumnType.NUMERIC

        if self._is_categorical(clean_series):
            return ColumnType.CATEGORICAL

        return ColumnType.FREE_TEXT

    def _is_pii(self, series: pd.Series) -> bool:
        sample = series.astype(str).str.strip().head(100)

        if sample.str.fullmatch(self.EMAIL_PATTERN).any():
            return True

        digit_counts = sample.str.count(r"\d")

        phone_candidates = sample[digit_counts >= 9]
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
            if median_abs < 1e9:  # below plausible UNIX timestamp (seconds)
                return False

            parsed = pd.to_datetime(numeric_series, errors="coerce", utc=True, unit="s")
            return parsed.notna().mean() >= 0.8

        parsed = pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
        return parsed.notna().mean() >= 0.8

    def _is_numeric(self, series: pd.Series) -> bool:
        if pd.api.types.is_numeric_dtype(series):
            return True

        normalized = series.astype(str).str.replace(",", "", regex=False).str.strip()
        coerced = pd.to_numeric(normalized, errors="coerce")
        return coerced.notna().mean() >= 0.9

    def _is_categorical(self, series: pd.Series) -> bool:
        total = len(series)
        if total == 0:
            return False

        unique = series.nunique(dropna=True)
        unique_ratio = unique / total
        return unique <= 20 and unique_ratio <= 0.5
