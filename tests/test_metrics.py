"""Tests for KPI and aggregation logic."""

from __future__ import annotations

import pandas as pd
import pytest

from src import metrics


@pytest.fixture()
def small_enriched() -> pd.DataFrame:
    """Hand-built frame with the derived columns the metrics module expects."""
    return pd.DataFrame(
        {
            "conversation_id": ["1", "2", "3", "4"],
            "timestamp": pd.to_datetime(["2026-05-04", "2026-05-05", "2026-05-12", "2026-05-13"]),
            "resolved": pd.array([True, False, True, False], dtype="boolean"),
            "escalated": pd.array([False, True, False, False], dtype="boolean"),
            "num_messages": [4, 8, 4, 6],
            "sentiment_score": [0.5, -0.6, 0.2, 0.1],
            "sentiment_label": ["Positive", "Negative", "Positive", "Positive"],
            "handoff_detected": [False, True, False, False],
            "resolution_estimate": ["Likely resolved", "Likely unresolved", "Likely resolved", "Unclear"],
            "escalation_risk": ["Low", "High", "Low", "Medium"],
            "escalation_score": [0, 9, 1, 3],
            "topic": ["billing", "billing", "login", "login"],
            "intent_final": ["Billing", "Billing", "Account/Login", "Account/Login"],
            "customer_text": ["charged twice", "charged twice again refund", "password reset", "password locked"],
        }
    )


def test_safe_rate_handles_zero_denominator():
    assert metrics.safe_rate(3, 0) == 0.0
    assert metrics.safe_rate(1, 4) == 0.25


def test_compute_kpis(small_enriched):
    k = metrics.compute_kpis(small_enriched)
    assert k["total_conversations"].value == 4
    assert k["resolution_rate"].value == pytest.approx(0.5)
    assert k["resolution_rate"].source == "data column"
    assert k["escalation_rate"].value == pytest.approx(0.25)
    assert k["human_handoff_rate"].value == pytest.approx(0.25)
    assert k["avg_conversation_length"].value == pytest.approx(5.5)
    assert k["avg_sentiment"].value == pytest.approx(0.05)
    assert k["unresolved_conversations"].value == 2
    assert k["negative_conversations"].value == 1


def test_kpis_fall_back_to_heuristics_without_ground_truth(small_enriched):
    df = small_enriched.drop(columns=["resolved", "escalated"])
    k = metrics.compute_kpis(df)
    assert k["resolution_rate"].value == pytest.approx(0.5)  # two "Likely resolved"
    assert "heuristic" in k["resolution_rate"].source
    assert k["escalation_rate"].value == pytest.approx(0.25)  # one "High"
    assert "heuristic" in k["escalation_rate"].source


def test_missing_ground_truth_values_count_as_not_resolved(small_enriched):
    df = small_enriched.copy()
    df["resolved"] = pd.array([True, None, None, None], dtype="boolean")
    assert metrics.compute_kpis(df)["resolution_rate"].value == pytest.approx(0.25)


def test_compute_kpis_on_empty_frame(small_enriched):
    k = metrics.compute_kpis(small_enriched.iloc[0:0])
    assert k["total_conversations"].value == 0
    assert k["resolution_rate"].value == 0.0


def test_conversations_over_time_weekly(small_enriched):
    timeline = metrics.conversations_over_time(small_enriched, "W")
    assert timeline["conversations"].tolist() == [2, 2]
    assert timeline["resolution_rate"].tolist() == [0.5, 0.5]
    assert timeline["period"].iloc[0] == pd.Timestamp("2026-05-04")  # weeks start on Monday


def test_conversations_over_time_without_timestamps(small_enriched):
    df = small_enriched.assign(timestamp=pd.NaT)
    assert metrics.conversations_over_time(df).empty


def test_distribution_shares_sum_to_one(small_enriched):
    dist = metrics.distribution(small_enriched, "sentiment_label", ["Positive", "Neutral", "Negative"])
    assert dist["share"].sum() == pytest.approx(1.0)
    assert dist["sentiment_label"].tolist() == ["Positive", "Negative"]


def test_unresolved_by_topic(small_enriched):
    table = metrics.unresolved_by(small_enriched, "topic").set_index("topic")
    assert table.loc["billing", "unresolved"] == 1
    assert table.loc["login", "unresolved_rate"] == pytest.approx(0.5)


def test_escalation_by_intent(small_enriched):
    table = metrics.escalation_by(small_enriched, "intent_final").set_index("intent_final")
    assert table.loc["Billing", "escalation_rate"] == pytest.approx(0.5)
    assert table.loc["Account/Login", "escalated"] == 0


def test_needs_review_orders_by_score(small_enriched):
    assert metrics.needs_review(small_enriched, "High")["conversation_id"].tolist() == ["2"]
    assert metrics.needs_review(small_enriched, "Medium")["conversation_id"].tolist() == ["2", "4"]


def test_top_pain_points_uses_problem_conversations(small_enriched):
    terms = [t for t, _ in metrics.top_pain_points(small_enriched)]
    assert terms  # unresolved/negative conversations exist
    assert any("charged" in t or "password" in t for t in terms)


def test_kpis_on_sample_data(enriched_df):
    k = metrics.compute_kpis(enriched_df)
    assert k["total_conversations"].value == len(enriched_df)
    for name in ("resolution_rate", "escalation_rate", "human_handoff_rate"):
        assert 0.0 <= k[name].value <= 1.0
    assert -1.0 <= k["avg_sentiment"].value <= 1.0
