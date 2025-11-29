from sfdao.ingestion.financial_domain import FinancialDomainMapper, FinancialEntity, FinancialRole
from sfdao.ingestion.schema import ColumnInfo, DataSchema


def _sample_schema() -> DataSchema:
    return DataSchema(
        num_rows=5,
        num_columns=6,
        columns=[
            ColumnInfo(name="transaction_id", dtype="int64", null_count=0, unique_count=5),
            ColumnInfo(name="amount", dtype="float64", null_count=0, unique_count=5),
            ColumnInfo(name="balance_after", dtype="float64", null_count=0, unique_count=5),
            ColumnInfo(name="timestamp", dtype="datetime64[ns]", null_count=0, unique_count=5),
            ColumnInfo(name="customer_id", dtype="object", null_count=0, unique_count=4),
            ColumnInfo(name="description", dtype="object", null_count=1, unique_count=5),
        ],
    )


def test_infer_financial_roles_from_column_names():
    schema = _sample_schema()
    mapper = FinancialDomainMapper()

    roles = mapper.infer_roles(schema)

    assert roles["amount"] == FinancialRole.TRANSACTION_AMOUNT
    assert roles["balance_after"] == FinancialRole.BALANCE
    assert roles["timestamp"] == FinancialRole.TIMESTAMP
    assert roles["customer_id"] == FinancialRole.CUSTOMER_ID
    assert roles["description"] == FinancialRole.DESCRIPTION
    assert roles["transaction_id"] == FinancialRole.TRANSACTION_ID


def test_set_custom_role_overrides_inference():
    schema = _sample_schema()
    mapper = FinancialDomainMapper()

    mapper.set_role("description", FinancialRole.COUNTERPARTY)

    roles = mapper.infer_roles(schema)

    assert roles["description"] == FinancialRole.COUNTERPARTY
    assert mapper.get_role("description") == FinancialRole.COUNTERPARTY


def test_entity_identifier_mapping():
    mapper = FinancialDomainMapper()

    mapper.set_entity_identifier(FinancialEntity.CUSTOMER, "customer_id")
    mapper.set_entity_identifier(FinancialEntity.TRANSACTION, "transaction_id")

    assert mapper.get_entity_identifier(FinancialEntity.CUSTOMER) == "customer_id"
    assert mapper.get_entity_identifier(FinancialEntity.TRANSACTION) == "transaction_id"
