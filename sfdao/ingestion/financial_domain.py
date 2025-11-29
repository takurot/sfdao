from __future__ import annotations

from enum import Enum
from typing import Dict, Mapping

from .schema import DataSchema

__all__ = ["FinancialRole", "FinancialEntity", "FinancialDomainMapper"]


class FinancialRole(str, Enum):
    TRANSACTION_ID = "transaction_id"
    TRANSACTION_AMOUNT = "transaction_amount"
    BALANCE = "balance"
    TIMESTAMP = "timestamp"
    CUSTOMER_ID = "customer_id"
    COUNTERPARTY = "counterparty"
    DESCRIPTION = "description"
    UNKNOWN = "unknown"


class FinancialEntity(str, Enum):
    CUSTOMER = "customer"
    TRANSACTION = "transaction"


class FinancialDomainMapper:
    """Map schema columns to financial roles with simple heuristics and manual overrides."""

    def __init__(self) -> None:
        self._role_overrides: Dict[str, FinancialRole] = {}
        self._resolved_roles: Dict[str, FinancialRole] = {}
        self._entity_identifiers: Dict[FinancialEntity, str] = {}

    def set_role(self, column_name: str, role: FinancialRole) -> None:
        self._role_overrides[column_name] = role
        self._resolved_roles[column_name] = role

    def get_role(self, column_name: str) -> FinancialRole | None:
        return self._resolved_roles.get(column_name) or self._role_overrides.get(column_name)

    def infer_roles(self, schema: DataSchema) -> Mapping[str, FinancialRole]:
        roles: Dict[str, FinancialRole] = {}
        for column in schema.columns:
            column_name = column.name
            role = self._role_overrides.get(column_name)
            if role is None:
                role = self._infer_role_from_name(column_name)
            roles[column_name] = role

        self._resolved_roles = roles
        return roles

    def set_entity_identifier(self, entity: FinancialEntity, column_name: str) -> None:
        self._entity_identifiers[entity] = column_name

    def get_entity_identifier(self, entity: FinancialEntity) -> str | None:
        return self._entity_identifiers.get(entity)

    def _infer_role_from_name(self, column_name: str) -> FinancialRole:
        name = column_name.lower()

        if "transaction" in name and "id" in name:
            return FinancialRole.TRANSACTION_ID
        if "amount" in name or name.endswith("_amt") or name.endswith("amt"):
            return FinancialRole.TRANSACTION_AMOUNT
        if "balance" in name or name.startswith("bal"):
            return FinancialRole.BALANCE
        if "time" in name or "date" in name or "timestamp" in name:
            return FinancialRole.TIMESTAMP
        if "customer" in name or "account" in name or "client" in name:
            return FinancialRole.CUSTOMER_ID
        if "counterparty" in name or "merchant" in name or "payee" in name:
            return FinancialRole.COUNTERPARTY
        if "desc" in name or "memo" in name or "note" in name or "description" in name:
            return FinancialRole.DESCRIPTION

        return FinancialRole.UNKNOWN
