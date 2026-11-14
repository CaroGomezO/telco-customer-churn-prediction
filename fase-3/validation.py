from __future__ import annotations

import math
from typing import Any, Sequence

from pipeline import NUMERIC_COLUMNS

ID_FIELD = "customerID"
BINARY_FIELDS = frozenset({"SeniorCitizen"})
INTEGER_FIELDS = frozenset({"tenure"})
NON_NEGATIVE_FIELDS = frozenset({"tenure", "MonthlyCharges"})


def _describe(value: Any) -> str:
    if isinstance(value, bool):
        return f"el booleano {str(value).lower()}"
    if isinstance(value, str):
        shown = value if len(value) <= 30 else value[:27] + "..."
        return f'el texto "{shown}"'
    if isinstance(value, (int, float)):
        return f"el número {value}"
    if isinstance(value, dict):
        return "un objeto"
    if isinstance(value, list):
        return "una lista"
    return type(value).__name__


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_value(column: str, value: Any) -> str | None:
    if column in BINARY_FIELDS:
        if not _is_number(value) or value not in (0, 1):
            return f"debe ser el número 0 o 1 (se recibió {_describe(value)})"
        return None
    if column in NUMERIC_COLUMNS:
        if not _is_number(value):
            return f"debe ser un número (se recibió {_describe(value)})"
        try:
            finite = math.isfinite(value)
        except OverflowError:  # entero demasiado grande para convertirse a decimal
            finite = False
        if not finite:
            return "debe ser un número finito"
        if column in INTEGER_FIELDS and value != int(value):
            return f"debe ser un número entero (se recibió {value})"
        if column in NON_NEGATIVE_FIELDS and value < 0:
            return f"no puede ser negativo (se recibió {value})"
        return None
    if not isinstance(value, str):
        return f"debe ser texto (se recibió {_describe(value)})"
    if not value.strip():
        return "no puede ser un texto vacío; use null para indicar un dato faltante"
    return None


def validate_record(payload: dict, feature_columns: Sequence[str]) -> tuple[dict, list[dict]]:
    clean: dict[str, Any] = {}
    problems: list[dict] = []
    for column in feature_columns:
        if column not in payload:
            problems.append({"campo": column, "problema": "campo obligatorio ausente"})
            continue
        value = payload[column]
        if value is None:
            clean[column] = None
            continue
        problem = _check_value(column, value)
        if problem:
            problems.append({"campo": column, "problema": problem})
        else:
            clean[column] = value
    return clean, problems


def problems_message(problems: list[dict]) -> str:
    detail = "; ".join(f"{p['campo']}: {p['problema']}" for p in problems)
    return f"Entrada inválida. {detail}."


def extract_identifier(payload: dict) -> str | int | None:
    value = payload.get(ID_FIELD)
    if isinstance(value, str) and value.strip():
        return value
    if _is_number(value) and isinstance(value, int):
        return value
    return None