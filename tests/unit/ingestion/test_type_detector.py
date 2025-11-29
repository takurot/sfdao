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
