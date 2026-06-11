"""Minimal cleaning helpers for Fangraphs CSV exports."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


IDENTIFIER_COLUMNS = {
    "name",
    "nameascii",
    "team",
    "handedness",
    "bats",
    "throws",
    "position",
    "pos",
    "playerid",
    "fangraphs_id",
    "mlbamid",
    "mlb_id",
}

OBVIOUS_IDENTIFIER_COLUMNS = {
    "name",
    "team",
    "playerid",
    "fangraphs_id",
    "mlbamid",
    "mlb_id",
}

MISSING_VALUE_WARNING_THRESHOLD = 0.30
NUMERIC_SUCCESS_THRESHOLD = 0.80


def standardize_column_name(column: object) -> str:
    """Turn a raw Fangraphs column name into a predictable snake_case name."""
    name = str(column).lstrip("\ufeff").strip().lower()
    name = name.replace("%", "_pct")
    name = name.replace("+", "_plus")
    name = re.sub(r"-$", "_minus", name)
    name = re.sub(r"[\s\-/]+", "_", name)
    name = re.sub(r"[^a-z0-9_]+", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed"


def standardize_columns(columns: Iterable[object]) -> list[str]:
    """Standardize columns and keep names unique after cleaning."""
    counts: dict[str, int] = {}
    clean_names: list[str] = []

    for column in columns:
        base_name = standardize_column_name(column)
        counts[base_name] = counts.get(base_name, 0) + 1

        if counts[base_name] == 1:
            clean_names.append(base_name)
        else:
            clean_names.append(f"{base_name}_{counts[base_name]}")

    return clean_names


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply light, explainable cleaning without dropping raw records."""
    cleaned = df.copy()
    cleaned.columns = standardize_columns(cleaned.columns)

    for column in cleaned.columns:
        if column in IDENTIFIER_COLUMNS:
            cleaned[column] = _preserve_identifier(cleaned[column])

    for column in cleaned.columns:
        if column in IDENTIFIER_COLUMNS:
            continue

        if column.endswith("_pct"):
            cleaned[column] = _convert_percentage_column(cleaned[column])
        else:
            cleaned[column] = _convert_numeric_looking_column(cleaned[column])

    return cleaned


def validate_cleaned_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    missing_threshold: float = MISSING_VALUE_WARNING_THRESHOLD,
) -> list[str]:
    """Return validation warnings while allowing the pipeline to continue."""
    warnings: list[str] = []

    if not any(column in df.columns for column in OBVIOUS_IDENTIFIER_COLUMNS):
        warnings.append(
            f"{dataset_name}: no obvious player/team identifier column found"
        )

    total_cells = df.shape[0] * df.shape[1]
    missing_values = int(df.isna().sum().sum())
    missing_rate = missing_values / total_cells if total_cells else 0

    if total_cells and missing_rate >= missing_threshold:
        warnings.append(
            f"{dataset_name}: {missing_rate:.1%} of values are missing"
        )

    return warnings


def summarize_dataframe(df: pd.DataFrame) -> dict[str, int]:
    """Create the simple summary requested by the ingestion script."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
    }


def _preserve_identifier(series: pd.Series) -> pd.Series:
    """Keep identifiers as labels instead of treating them like model features."""
    return series.astype("string").str.strip()


def _convert_percentage_column(series: pd.Series) -> pd.Series:
    """Convert values like '25.4%' to 0.254 while preserving decimal rates."""
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any() and numeric.abs().max() > 1:
            return numeric / 100
        return numeric

    text = series.astype("string").str.strip()
    has_percent_symbol = text.str.contains("%", regex=False, na=False)
    cleaned_text = _clean_numeric_text(text).str.replace("%", "", regex=False)
    numeric = pd.to_numeric(cleaned_text, errors="coerce")

    if has_percent_symbol.any():
        numeric.loc[has_percent_symbol] = numeric.loc[has_percent_symbol] / 100

    without_symbol = ~has_percent_symbol & numeric.notna()
    if without_symbol.any() and numeric.loc[without_symbol].abs().max() > 1:
        numeric.loc[without_symbol] = numeric.loc[without_symbol] / 100

    return numeric


def _convert_numeric_looking_column(series: pd.Series) -> pd.Series:
    """Convert mostly numeric text columns while leaving true text alone."""
    if pd.api.types.is_numeric_dtype(series):
        return series

    text = series.astype("string").str.strip()
    cleaned_text = _clean_numeric_text(text)
    numeric = pd.to_numeric(cleaned_text, errors="coerce")

    non_empty = cleaned_text.notna()
    non_empty_count = int(non_empty.sum())
    if non_empty_count == 0:
        return series

    success_rate = numeric.notna().sum() / non_empty_count
    if success_rate >= NUMERIC_SUCCESS_THRESHOLD:
        return numeric

    return series


def _clean_numeric_text(series: pd.Series) -> pd.Series:
    return (
        series.str.replace(",", "", regex=False)
        .replace("", pd.NA)
        .replace("-", pd.NA)
        .replace("--", pd.NA)
        .replace("NA", pd.NA)
        .replace("N/A", pd.NA)
    )
