import pandas as pd

from sfdao.ingestion.type_detector import ColumnType, TypeDetector


def test_detect_numeric_column():
    data = pd.Series([100, 200.5, 300, 0])
    detector = TypeDetector()

    col_type = detector.detect(data, "amount")

    assert col_type == ColumnType.NUMERIC


def test_detect_categorical_column():
    data = pd.Series(["A"] * 5 + ["B"] * 5 + ["C"] * 5)
    detector = TypeDetector()

    col_type = detector.detect(data, "category")

    assert col_type == ColumnType.CATEGORICAL


def test_detect_datetime_column():
    data = pd.Series(["2023-01-01", "2023-01-02 12:00:00", "2023-02-15"])
    detector = TypeDetector()

    col_type = detector.detect(data, "timestamp")

    assert col_type == ColumnType.DATETIME


def test_detect_pii_column_email_and_phone():
    detector = TypeDetector()

    email_series = pd.Series(["user1@example.com", "user.two@test.co.jp", None])
    col_type = detector.detect(email_series, "email")
    assert col_type == ColumnType.PII

    phone_series = pd.Series(["090-1234-5678", "03-9876-5432", None])
    col_type = detector.detect(phone_series, "phone")
    assert col_type == ColumnType.PII


def test_detect_free_text_column():
    data = pd.Series(
        [
            "Payment for invoice #123",
            "Refund issued after double charge",
            "Chargeback processed by bank",
            "Customer requested statement copy",
        ]
    )
    detector = TypeDetector()

    col_type = detector.detect(data, "description")

    assert col_type == ColumnType.FREE_TEXT


def test_detect_all_null_column_as_free_text():
    data = pd.Series([None, float("nan"), None])
    detector = TypeDetector()

    col_type = detector.detect(data, "empty_col")

    assert col_type == ColumnType.FREE_TEXT


def test_detect_credit_card_column_as_pii():
    data = pd.Series(["4111 1111 1111 1111", "4242-4242-4242-4242", None])
    detector = TypeDetector()

    col_type = detector.detect(data, "card_number")

    assert col_type == ColumnType.PII


def test_detect_unix_second_timestamp_column_as_datetime():
    data = pd.Series([1_700_000_000, 1_700_086_400, 1_700_172_800])
    detector = TypeDetector()

    col_type = detector.detect(data, "event_ts")

    assert col_type == ColumnType.DATETIME


def test_detect_comma_formatted_numbers_as_numeric():
    data = pd.Series(["1,200.50", "3,000", "42"])
    detector = TypeDetector()

    col_type = detector.detect(data, "amount")

    assert col_type == ColumnType.NUMERIC
