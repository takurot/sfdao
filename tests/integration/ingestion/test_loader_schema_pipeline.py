from pathlib import Path

from sfdao.ingestion.loader import CSVLoader
from sfdao.ingestion.schema import SchemaExtractor


def test_loader_and_schema_pipeline():
    loader = CSVLoader()
    df = loader.load(Path("tests/fixtures/sample_transactions.csv"))

    schema = SchemaExtractor.extract(df)

    assert schema.num_rows == len(df)
    assert schema.num_columns == len(df.columns)
    # unique counts should match the dataframe's calculations
    amount_info = next(col for col in schema.columns if col.name == "amount")
    assert amount_info.unique_count == df["amount"].nunique(dropna=True)
    timestamp_info = next(col for col in schema.columns if col.name == "timestamp")
    assert timestamp_info.dtype.startswith("datetime")
