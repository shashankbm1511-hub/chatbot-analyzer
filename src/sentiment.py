"""Sentiment scoring with VADER.

VADER (Hutto & Gilbert, 2014) is a *pretrained, lexicon- and rule-based*
sentiment model: a human-rated word list plus rules for negation, intensifiers
and punctuation. It is not trained on this project's data. Each message gets a
compound score in [-1, 1]; we use the standard VADER cut-offs (+/-0.05) to map
scores to Positive / Neutral / Negative.

Only customer messages are scored: bot and agent replies are scripted to be
polite, so including them would pull every conversation towards "Positive".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.preprocessing import Turn, customer_turns

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05
STRONG_NEGATIVE_THRESHOLD = -0.5

SENTIMENT_LABELS = ["Positive", "Neutral", "Negative"]


@lru_cache(maxsize=1)
def _analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def score_text(text: str) -> float:
    """Return the VADER compound score for ``text`` (0.0 for empty text)."""
    if not text or not str(text).strip():
        return 0.0
    return float(_analyzer().polarity_scores(str(text))["compound"])


def label_from_score(score: float) -> str:
    """Map a compound score to Positive / Neutral / Negative."""
    if score >= POSITIVE_THRESHOLD:
        return "Positive"
    if score <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


@dataclass
class SentimentResult:
    """Conversation-level sentiment summary."""

    score: float  # mean compound score of customer messages
    label: str
    min_score: float  # most negative customer message
    final_score: float  # last customer message
    turn_scores: list[float] = field(default_factory=list)

    @property
    def ended_negative(self) -> bool:
        return self.final_score <= NEGATIVE_THRESHOLD


def analyze_sentiment(turns: list[Turn]) -> SentimentResult:
    """Score each customer message and aggregate to the conversation level."""
    scores = [score_text(t.text) for t in customer_turns(turns)]
    if not scores:
        return SentimentResult(score=0.0, label="Neutral", min_score=0.0, final_score=0.0)
    mean = sum(scores) / len(scores)
    return SentimentResult(
        score=round(mean, 4),
        label=label_from_score(mean),
        min_score=min(scores),
        final_score=scores[-1],
        turn_scores=scores,
    )
