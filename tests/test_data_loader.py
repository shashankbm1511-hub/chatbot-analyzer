"""Tests for CSV loading, validation and missing-value handling."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from src.data_loader import DataValidationError, load_conversations, to_bool, validate_and_clean
from tests.conftest import SAMPLE_CSV


def _csv(text: str) -> io.StringIO:
    return io.StringIO(text)


def test_sample_dataset_loads_cleanly():
    df, report = load_conversations(SAMPLE_CSV)
    assert 100 <= len(df) <= 300
    assert report.rows_in == report.rows_out == len(df)
    assert df["conversation_id"].is_unique
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["resolved"].dtype == "boolean"
    assert df["channel"].notna().all()  # blanks in the sample are filled with "unknown"


def test_missing_required_column_raises():
    with pytest.raises(DataValidationError, match="conversation_text"):
        load_conversations(_csv("conversation_id,intent\nC1,Billing\n"))


def test_empty_file_raises():
    with pytest.raises(DataValidationError):
        load_conversations(_csv(""))


def test_header_only_raises():
    with pytest.raises(DataValidationError):
        load_conversations(_csv("conversation_id,conversation_text\n"))


def test_empty_text_and_duplicate_ids_are_dropped():
    raw = pd.DataFrame(
        {
            "conversation_id": ["A", "B", "B", "C"],
            "conversation_text": ["Customer: hi", "Customer: first", "Customer: dup", "   "],
        }
    )
    df, report = validate_and_clean(raw)
    assert list(df["conversation_id"]) == ["A", "B"]
    assert report.dropped_duplicate_ids == 1
    assert report.dropped_empty_text == 1
    assert df.loc[df["conversation_id"] == "B", "conversation_text"].iat[0] == "Customer: first"


def test_all_rows_empty_raises():
    raw = pd.DataFrame({"conversation_id": ["A"], "conversation_text": [None]})
    with pytest.raises(DataValidationError):
        validate_and_clean(raw)


def test_missing_values_are_filled_and_reported():
    raw = pd.DataFrame(
        {
            "Conversation ID": ["A", "B"],  # header normalisation: spaces / case
            "conversation_text": ["Customer: hi\nBot: hello", "Customer: help"],
            "channel": ["web", None],
            "customer_id": [None, "C-1"],
            "response_time_seconds": [3.2, -1],
            "conversation_length": [None, 1],
        }
    )
    df, report = validate_and_clean(raw)
    assert df["channel"].tolist() == ["web", "unknown"]
    assert df["customer_id"].tolist() == ["unknown", "C-1"]
    assert pd.isna(df.loc[1, "response_time_seconds"])  # negative -> missing
    assert any("negative" in w for w in report.warnings)
    assert df["conversation_length"].tolist() == [2, 1]  # missing length computed from transcript
    assert report.filled_values["channel"] == 1


def test_optional_columns_are_reported_when_absent():
    df, report = load_conversations(_csv('conversation_id,conversation_text\nC1,"Customer: hi"\n'))
    assert "resolved" in report.missing_optional_columns
    assert df["conversation_length"].iat[0] == 1
    assert df["timestamp"].isna().all()


def test_unparseable_timestamp_is_blank_with_warning():
    csv = 'conversation_id,conversation_text,timestamp\nC1,"Customer: hi",not-a-date\nC2,"Customer: yo",2026-05-01\n'
    df, report = load_conversations(_csv(csv))
    assert df["timestamp"].isna().sum() == 1
    assert any("timestamp" in w for w in report.warnings)


@pytest.mark.parametrize(
    "value, expected",
    [(True, True), ("TRUE", True), ("yes", True), (1, True), ("0", False), ("No", False),
     (False, False), (None, None), (float("nan"), None), ("maybe", None)],
)
def test_to_bool(value, expected):
    assert to_bool(value) is expected


def test_unrecognised_booleans_warn():
    raw = pd.DataFrame({"conversation_id": ["A", "B"], "conversation_text": ["Customer: a", "Customer: b"],
                        "resolved": ["yes", "perhaps"]})
    df, report = validate_and_clean(raw)
    assert bool(df["resolved"].iat[0]) is True
    assert pd.isna(df["resolved"].iat[1])
    assert any("resolved" in w for w in report.warnings)
