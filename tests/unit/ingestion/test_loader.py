from pathlib import Path

import pandas as pd
import pytest

from sfdao.ingestion.loader import CSVLoader


def test_load_csv_file_parses_expected_columns():
    loader = CSVLoader()
    df = loader.load(Path("tests/fixtures/sample_transactions.csv"))

    assert not df.empty
    assert list(df.columns) == [
        "transaction_id",
        "amount",
        "balance",
        "timestamp",
        "customer_id",
        "description",
    ]
    assert pd.api.types.is_numeric_dtype(df["amount"])
    assert pd.api.types.is_numeric_dtype(df["balance"])
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_load_csv_missing_file():
    loader = CSVLoader()
    missing_path = Path("tests/fixtures/does_not_exist.csv")

    with pytest.raises(FileNotFoundError):
        loader.load(missing_path)


def test_load_csv_malformed_file():
    loader = CSVLoader()
    malformed_path = Path("tests/fixtures/malformed_transactions.csv")

    with pytest.raises(ValueError, match="Failed to parse CSV"):
        loader.load(malformed_path)
