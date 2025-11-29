from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

__all__ = ["ColumnInfo", "DataSchema", "SchemaExtractor"]


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int


class DataSchema(BaseModel):
    num_rows: int
    num_columns: int
    columns: list[ColumnInfo]


class SchemaExtractor:
    """Extract a lightweight schema summary from a DataFrame."""

    @staticmethod
    def extract(df: pd.DataFrame) -> DataSchema:
        columns = [SchemaExtractor._build_column_info(df, col) for col in df.columns]
        return DataSchema(
            num_rows=len(df),
            num_columns=len(df.columns),
            columns=columns,
        )

    @staticmethod
    def _build_column_info(df: pd.DataFrame, column: str) -> ColumnInfo:
        series = df[column]
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        return ColumnInfo(
            name=column,
            dtype=str(series.dtype),
            null_count=null_count,
            unique_count=unique_count,
        )
