"""Explainable, rule-based escalation-risk score.

This is a transparent **heuristic**, not a trained model. Each signal adds a
fixed number of points; the total maps to Low / Medium / High. Every point is
traceable to a signal and the text that triggered it, so a reviewer can see
exactly why a conversation was flagged.

It estimates whether a conversation *shows signs* that a human should look at
it. It is not a prediction of churn, complaints or any business outcome, and
the weights and thresholds are hand-set defaults that would need tuning on
real, labelled data.

Only customer messages (plus overall length) are used, so the score does not
simply read off whether a bot already transferred the chat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.preprocessing import Turn, count_repeated_messages, customer_turns, normalize_text
from src.sentiment import NEGATIVE_THRESHOLD, STRONG_NEGATIVE_THRESHOLD, SentimentResult

# --- signal weights (points) -------------------------------------------------
WEIGHT_HUMAN_REQUEST = 3
WEIGHT_STRONG_NEGATIVE = 2
WEIGHT_NEGATIVE_OVERALL = 1
WEIGHT_COMPLAINT_LANGUAGE = 2
WEIGHT_REPEATED_MESSAGE = 1  # per repeat, capped
MAX_REPEAT_POINTS = 2
WEIGHT_FAILED_ATTEMPTS = 1  # 2 if several failure phrases
WEIGHT_LONG_CONVERSATION = 1
WEIGHT_CANCEL_REFUND = 1
WEIGHT_NEGATIVE_ENDING = 1

LONG_CONVERSATION_TURNS = 8

# --- level thresholds ----------------------------------------------------------
MEDIUM_THRESHOLD = 3
HIGH_THRESHOLD = 6
RISK_LEVELS = ["Low", "Medium", "High"]

HUMAN_REQUEST_PHRASES = [
    "human", "real person", "live agent", "speak to someone", "talk to someone",
    "speak to a person", "talk to a person", "customer service", "representative",
    "supervisor", "manager", "speak with a", "talk to an agent", "speak to an agent",
]
COMPLAINT_PHRASES = [
    "worst", "ridiculous", "useless", "unacceptable", "terrible", "awful", "frustrated",
    "frustrating", "unhappy", "not happy", "rude", "complaint", "not listening",
    "nobody is helping", "waste of time", "disappointed",
]
FAILED_ATTEMPT_PHRASES = [
    "still not", "still haven't", "still getting", "still isn't", "still crashes", "still there",
    "keep getting", "already tried", "already checked", "already updated", "third time", "twice now",
    "doesn't work", "not working", "same problem", "same error", "same answer", "nothing has changed",
    "times already",
]
CANCEL_REFUND_PHRASES = ["cancel", "refund", "money back", "close my account", "chargeback"]
ABANDON_PHRASES = ["give up", "forget it", "never mind", "useless", "not helping", "try again later"]


def _compile(phrases: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    return [(p, re.compile(r"\b" + re.escape(normalize_text(p)) + r"\b")) for p in phrases]


_HUMAN = _compile(HUMAN_REQUEST_PHRASES)
_COMPLAINT = _compile(COMPLAINT_PHRASES)
_FAILED = _compile(FAILED_ATTEMPT_PHRASES)
_CANCEL = _compile(CANCEL_REFUND_PHRASES)
_ABANDON = _compile(ABANDON_PHRASES)


def find_phrases(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    """Return the phrases from ``patterns`` that occur in ``text`` (normalised)."""
    normalized = normalize_text(text)
    return [phrase for phrase, pattern in patterns if pattern.search(normalized)]


@dataclass
class EscalationSignal:
    """One reason that contributed points to the escalation score."""

    name: str
    points: int
    evidence: str


@dataclass
class EscalationResult:
    score: int
    level: str
    signals: list[EscalationSignal] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return "; ".join(f"{s.name} (+{s.points})" for s in self.signals) or "No escalation signals"


def level_from_score(score: int) -> str:
    """Map a point total to Low / Medium / High."""
    if score >= HIGH_THRESHOLD:
        return "High"
    if score >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def trigger_phrases(text: str) -> list[str]:
    """All escalation-related phrases found in ``text`` (for highlighting in the UI)."""
    found: list[str] = []
    for patterns in (_HUMAN, _COMPLAINT, _FAILED, _CANCEL, _ABANDON):
        found.extend(find_phrases(text, patterns))
    return sorted(set(found), key=len, reverse=True)


def human_requested(turns: list[Turn]) -> bool:
    """True if any customer message explicitly asks for a person."""
    return any(find_phrases(t.text, _HUMAN) for t in customer_turns(turns))


def score_escalation(turns: list[Turn], sentiment: SentimentResult) -> EscalationResult:
    """Compute the escalation score and the signals behind it."""
    signals: list[EscalationSignal] = []
    cust = customer_turns(turns)
    cust_text = " ".join(t.text for t in cust)

    human_hits = find_phrases(cust_text, _HUMAN)
    if human_hits:
        signals.append(EscalationSignal("Explicit request for a human", WEIGHT_HUMAN_REQUEST,
                                        f"Customer used: {', '.join(human_hits)}"))

    if sentiment.min_score <= STRONG_NEGATIVE_THRESHOLD:
        signals.append(EscalationSignal("Strongly negative message", WEIGHT_STRONG_NEGATIVE,
                                        f"Most negative customer message scored {sentiment.min_score:.2f} "
                                        f"(threshold {STRONG_NEGATIVE_THRESHOLD})"))
    elif sentiment.score <= NEGATIVE_THRESHOLD:
        signals.append(EscalationSignal("Negative overall sentiment", WEIGHT_NEGATIVE_OVERALL,
                                        f"Mean customer sentiment {sentiment.score:.2f}"))

    complaint_hits = find_phrases(cust_text, _COMPLAINT)
    if complaint_hits:
        signals.append(EscalationSignal("Complaint language", WEIGHT_COMPLAINT_LANGUAGE,
                                        f"Customer used: {', '.join(complaint_hits)}"))

    repeats = count_repeated_messages(turns)
    if repeats:
        points = min(repeats * WEIGHT_REPEATED_MESSAGE, MAX_REPEAT_POINTS)
        signals.append(EscalationSignal("Repeated customer message", points,
                                        f"{repeats} message(s) repeated an earlier one"))

    failed_hits = find_phrases(cust_text, _FAILED)
    if failed_hits:
        points = WEIGHT_FAILED_ATTEMPTS * (2 if len(failed_hits) >= 2 else 1)
        signals.append(EscalationSignal("Failed attempts / issue persists", points,
                                        f"Customer used: {', '.join(failed_hits)}"))

    if len(turns) >= LONG_CONVERSATION_TURNS:
        signals.append(EscalationSignal("Long conversation", WEIGHT_LONG_CONVERSATION,
                                        f"{len(turns)} messages (threshold {LONG_CONVERSATION_TURNS})"))

    cancel_hits = find_phrases(cust_text, _CANCEL)
    if cancel_hits:
        signals.append(EscalationSignal("Cancellation / refund language", WEIGHT_CANCEL_REFUND,
                                        f"Customer used: {', '.join(cancel_hits)}"))

    if cust:
        last = cust[-1].text
        abandon_hits = find_phrases(last, _ABANDON)
        if abandon_hits or sentiment.ended_negative:
            evidence = (f"Last customer message: “{last}”")
            signals.append(EscalationSignal("Conversation ended negatively", WEIGHT_NEGATIVE_ENDING, evidence))

    score = sum(s.points for s in signals)
    return EscalationResult(score=score, level=level_from_score(score), signals=signals)
