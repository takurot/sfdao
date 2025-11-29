from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sfdao.ingestion.loader import CSVLoader
from sfdao.ingestion.schema import DataSchema, SchemaExtractor


@pytest.fixture()
def sample_transactions_df() -> pd.DataFrame:
    loader = CSVLoader()
    return loader.load(Path("tests/fixtures/sample_transactions.csv"))


def test_extract_schema_basic(sample_transactions_df: pd.DataFrame):
    schema = SchemaExtractor.extract(sample_transactions_df)

    assert isinstance(schema, DataSchema)
    assert schema.num_rows == len(sample_transactions_df)
    assert schema.num_columns == len(sample_transactions_df.columns)
    assert [col.name for col in schema.columns] == list(sample_transactions_df.columns)

    amount_info = next(col for col in schema.columns if col.name == "amount")
    assert amount_info.null_count == 1
    assert amount_info.unique_count == sample_transactions_df["amount"].nunique(dropna=True)
    assert amount_info.dtype.startswith("float")


def test_extract_schema_with_nulls():
    df = pd.DataFrame(
        {
            "account_id": [1, 2, None],
            "balance": [100.0, None, 150.5],
            "note": ["ok", None, "ok"],
        }
    )

    schema = SchemaExtractor.extract(df)
    balances = next(col for col in schema.columns if col.name == "balance")
    assert balances.null_count == 1
    assert balances.unique_count == 2

    account = next(col for col in schema.columns if col.name == "account_id")
    assert account.null_count == 1
    assert account.unique_count == 2


def test_extract_schema_empty_dataframe():
    df = pd.DataFrame(columns=["a", "b"])

    schema = SchemaExtractor.extract(df)
    assert schema.num_rows == 0
    assert schema.num_columns == 2
    assert all(col.null_count == 0 for col in schema.columns)
    assert all(col.unique_count == 0 for col in schema.columns)
