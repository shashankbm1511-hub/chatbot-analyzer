"""Load, validate and clean conversation CSV files.

Only two columns are required: ``conversation_id`` and ``conversation_text``.
Every other column is optional; when it is missing or partly empty the loader
fills a safe default and records what it did in a :class:`LoadReport`, so the
UI can tell the user exactly how their data was changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import pandas as pd

from src.preprocessing import parse_transcript

REQUIRED_COLUMNS = ["conversation_id", "conversation_text"]
OPTIONAL_COLUMNS = [
    "timestamp",
    "customer_id",
    "intent",
    "channel",
    "resolved",
    "escalated",
    "response_time_seconds",
    "conversation_length",
]
BOOLEAN_COLUMNS = ["resolved", "escalated"]

_TRUE_VALUES = {"true", "t", "yes", "y", "1", "1.0"}
_FALSE_VALUES = {"false", "f", "no", "n", "0", "0.0"}

CsvSource = str | Path | IO[bytes] | IO[str]


class DataValidationError(ValueError):
    """Raised when an input file cannot be used at all."""


@dataclass
class LoadReport:
    """Summary of what the loader changed, shown to the user in the app."""

    rows_in: int = 0
    rows_out: int = 0
    dropped_empty_text: int = 0
    dropped_duplicate_ids: int = 0
    filled_values: dict[str, int] = field(default_factory=dict)
    missing_optional_columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def to_bool(value: object) -> bool | None:
    """Convert common truthy/falsy spellings to ``bool``; return ``None`` if unknown."""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    return None


def load_conversations(source: CsvSource) -> tuple[pd.DataFrame, LoadReport]:
    """Read a CSV file (path or file-like object) and return a cleaned frame.

    Raises:
        DataValidationError: if the file cannot be parsed, is empty, or lacks
            the required columns.
    """
    try:
        raw = pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The file is empty.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Could not parse the file as CSV: {exc}") from exc
    return validate_and_clean(raw)


def validate_and_clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, LoadReport]:
    """Validate required columns and normalise types and missing values."""
    report = LoadReport(rows_in=len(raw))
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise DataValidationError(
            "Missing required column(s): " + ", ".join(missing_required)
            + ". Expected at least: " + ", ".join(REQUIRED_COLUMNS) + "."
        )
    if df.empty:
        raise DataValidationError("The file has a header but no rows.")

    # --- required columns -------------------------------------------------
    df["conversation_text"] = df["conversation_text"].fillna("").astype(str).str.strip()
    empty_mask = df["conversation_text"] == ""
    report.dropped_empty_text = int(empty_mask.sum())
    df = df.loc[~empty_mask]

    df["conversation_id"] = df["conversation_id"].astype(str).str.strip()
    duplicate_mask = df["conversation_id"].duplicated(keep="first")
    report.dropped_duplicate_ids = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].reset_index(drop=True)

    if df.empty:
        raise DataValidationError("No usable conversations after removing empty rows.")

    # --- optional columns -------------------------------------------------
    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            report.missing_optional_columns.append(column)

    _clean_timestamp(df, report)
    _fill_text_column(df, "customer_id", "unknown", report)
    _fill_text_column(df, "channel", "unknown", report)
    if "intent" in df.columns:
        # Keep missing labels as NaN: they must not be used for model training.
        df["intent"] = df["intent"].where(df["intent"].notna(), None)
        df["intent"] = df["intent"].map(lambda v: str(v).strip() if v is not None else None)

    for column in BOOLEAN_COLUMNS:
        if column in df.columns:
            converted = df[column].map(to_bool)
            unknown = int(converted.isna().sum())
            if unknown:
                report.warnings.append(
                    f"{unknown} value(s) in '{column}' were missing or not recognised as true/false."
                )
            df[column] = converted.astype("boolean")

    if "response_time_seconds" in df.columns:
        rt = pd.to_numeric(df["response_time_seconds"], errors="coerce")
        negative = int((rt < 0).sum())
        if negative:
            report.warnings.append(f"{negative} negative response time(s) were treated as missing.")
        rt = rt.mask(rt < 0)
        report.filled_values["response_time_seconds (left blank)"] = int(rt.isna().sum())
        df["response_time_seconds"] = rt

    # Conversation length = number of messages. A provided value is kept; missing
    # values (or a missing column) are computed from the parsed transcript.
    computed_length = df["conversation_text"].map(lambda t: len(parse_transcript(t)))
    if "conversation_length" in df.columns:
        provided = pd.to_numeric(df["conversation_length"], errors="coerce")
        filled = int(provided.isna().sum())
        if filled:
            report.filled_values["conversation_length"] = filled
        df["conversation_length"] = provided.fillna(computed_length).astype(int)
    else:
        df["conversation_length"] = computed_length.astype(int)

    report.filled_values = {k: v for k, v in report.filled_values.items() if v}
    report.rows_out = len(df)
    return df, report


def _clean_timestamp(df: pd.DataFrame, report: LoadReport) -> None:
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.NaT
        return
    parsed = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    bad = int(parsed.isna().sum() - df["timestamp"].isna().sum())
    if bad > 0:
        report.warnings.append(f"{bad} timestamp(s) could not be parsed and were left blank.")
    df["timestamp"] = parsed


def _fill_text_column(df: pd.DataFrame, column: str, default: str, report: LoadReport) -> None:
    if column not in df.columns:
        df[column] = default
        return
    missing = df[column].isna() | (df[column].astype(str).str.strip() == "")
    if missing.any():
        report.filled_values[column] = int(missing.sum())
    df[column] = df[column].where(~missing, default).astype(str).str.strip()
