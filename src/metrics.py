"""Dataset-level KPIs and aggregations (pure pandas, no UI code).

Where a ground-truth column exists (``resolved``, ``escalated``) it is used.
When it does not, the KPI falls back to the heuristic estimate and the returned
``source`` field says so, so the dashboard never presents an estimate as fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.topic_model import corpus_keywords


def safe_rate(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, or 0.0 when the denominator is zero."""
    return float(numerator) / float(denominator) if denominator else 0.0


@dataclass
class KPI:
    value: float
    source: str  # "data column", "derived from text", or "heuristic estimate"


def resolved_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Boolean resolved flag: the ``resolved`` column if usable, else the heuristic."""
    if "resolved" in df.columns and df["resolved"].notna().any():
        return df["resolved"].fillna(False).astype(bool), "data column"
    return df["resolution_estimate"].eq("Likely resolved"), "heuristic estimate"


def escalated_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Boolean escalated flag: the ``escalated`` column if usable, else High risk."""
    if "escalated" in df.columns and df["escalated"].notna().any():
        return df["escalated"].fillna(False).astype(bool), "data column"
    return df["escalation_risk"].eq("High"), "heuristic estimate (High risk)"


def compute_kpis(df: pd.DataFrame) -> dict[str, KPI]:
    """Headline KPIs for an enriched frame (see ``ConversationAnalyzer.enrich``)."""
    total = len(df)
    resolved, resolved_src = resolved_series(df)
    escalated, escalated_src = escalated_series(df)
    return {
        "total_conversations": KPI(total, "data"),
        "resolution_rate": KPI(safe_rate(resolved.sum(), total), resolved_src),
        "escalation_rate": KPI(safe_rate(escalated.sum(), total), escalated_src),
        "avg_conversation_length": KPI(float(df["num_messages"].mean()) if total else 0.0, "derived from text"),
        "avg_sentiment": KPI(float(df["sentiment_score"].mean()) if total else 0.0, "VADER, customer messages"),
        "unresolved_conversations": KPI(int(total - resolved.sum()), resolved_src),
        "negative_conversations": KPI(int(df["sentiment_label"].eq("Negative").sum()), "VADER, customer messages"),
        "human_handoff_rate": KPI(safe_rate(df["handoff_detected"].sum(), total), "derived from text"),
    }


def conversations_over_time(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Conversation counts and resolution rate per period (``freq``: 'D', 'W' or 'M')."""
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        return pd.DataFrame(columns=["period", "conversations", "resolution_rate"])
    data = df.dropna(subset=["timestamp"]).copy()
    resolved, _ = resolved_series(data)
    data["_resolved"] = resolved
    period_freq = {"D": "D", "W": "W-SUN", "M": "M"}.get(freq, "W-SUN")
    data["period"] = data["timestamp"].dt.to_period(period_freq).dt.start_time
    grouped = data.groupby("period").agg(conversations=("conversation_id", "count"), resolved=("_resolved", "sum"))
    grouped["resolution_rate"] = grouped["resolved"] / grouped["conversations"]
    return grouped.drop(columns="resolved").reset_index()


def distribution(df: pd.DataFrame, column: str, order: list[str] | None = None) -> pd.DataFrame:
    """Counts and shares of each value in ``column``."""
    counts = df[column].fillna("Unknown").value_counts()
    if order:
        counts = counts.reindex([o for o in order if o in counts.index] + [i for i in counts.index if i not in order])
    result = counts.rename_axis(column).reset_index(name="count")
    result["share"] = result["count"] / result["count"].sum() if len(result) else 0.0
    return result


def rate_by_group(df: pd.DataFrame, group_column: str, flag: pd.Series, rate_name: str = "rate") -> pd.DataFrame:
    """Share of rows where ``flag`` is True within each group, with counts."""
    data = pd.DataFrame({group_column: df[group_column].fillna("Unknown"), "_flag": flag.astype(bool)})
    grouped = data.groupby(group_column)["_flag"].agg(["sum", "count"]).rename(columns={"sum": "flagged", "count": "total"})
    grouped[rate_name] = grouped["flagged"] / grouped["total"]
    return grouped.sort_values(rate_name, ascending=False).reset_index()


def unresolved_by(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Unresolved conversation counts and rates per group, most unresolved first."""
    resolved, _ = resolved_series(df)
    table = rate_by_group(df, group_column, ~resolved, rate_name="unresolved_rate")
    table = table.rename(columns={"flagged": "unresolved"})
    return table.sort_values(["unresolved", "unresolved_rate"], ascending=False).reset_index(drop=True)


def escalation_by(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    escalated, _ = escalated_series(df)
    table = rate_by_group(df, group_column, escalated, rate_name="escalation_rate")
    return table.rename(columns={"flagged": "escalated"})


def top_pain_points(df: pd.DataFrame, top_n: int = 12) -> list[tuple[str, float]]:
    """Most characteristic terms in unresolved or negative conversations.

    Uses TF-IDF weights over the customer text of problem conversations — a
    keyword summary of "what customers are struggling with", not a causal claim.
    """
    resolved, _ = resolved_series(df)
    problem = df[(~resolved) | df["sentiment_label"].eq("Negative")]
    return corpus_keywords(problem["customer_text"].tolist(), top_n=top_n)


def needs_review(df: pd.DataFrame, min_level: str = "High") -> pd.DataFrame:
    """Conversations at or above ``min_level`` escalation risk, highest score first."""
    levels = {"Low": 0, "Medium": 1, "High": 2}
    mask = df["escalation_risk"].map(levels) >= levels[min_level]
    return df[mask].sort_values("escalation_score", ascending=False)
