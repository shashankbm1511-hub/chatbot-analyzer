"""Per-conversation analysis and dataset enrichment.

:class:`ConversationAnalyzer` holds the models fitted on the loaded dataset
(ML intent classifier, topic model) and combines them with the heuristic
components (sentiment, resolution estimate, escalation score) to produce:

* :meth:`ConversationAnalyzer.analyze` — a full, explainable analysis of one
  conversation for the drill-down view;
* :meth:`ConversationAnalyzer.enrich` — derived columns for every row, used by
  the dashboard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from src.escalation import EscalationResult, human_requested, score_escalation
from src.intent import (
    IntentModelEvaluation,
    IntentPrediction,
    MLIntentClassifier,
    RuleBasedIntentClassifier,
    train_intent_model,
)
from src.preprocessing import Turn, count_repeated_messages, customer_text, customer_turns, parse_transcript
from src.sentiment import SentimentResult, analyze_sentiment
from src.topic_model import TopicModel, fit_topics

RESOLUTION_LABELS = ["Likely resolved", "Likely unresolved", "Unclear"]

_CLOSURE_PATTERN = re.compile(
    r"\b(thanks|thank you|that worked|worked|solved|sorted|resolved|perfect|great|got it|helpful|fixed)\b",
    re.IGNORECASE,
)
_UNRESOLVED_PATTERN = re.compile(
    r"\b(give up|forget it|never mind|not helping|useless|still not|doesn't work|not working|"
    r"not happy|hope it actually|try again later|doesn't answer)\b",
    re.IGNORECASE,
)
_HANDOFF_PATTERN = re.compile(
    r"\b(transferring you|connecting you|transfer you|human agent now|live agent now)\b",
    re.IGNORECASE,
)


@dataclass
class ResolutionEstimate:
    label: str
    evidence: str


@dataclass
class ConversationStats:
    total_messages: int
    customer_messages: int
    bot_messages: int
    agent_messages: int
    customer_words: int
    avg_customer_message_words: float
    repeated_customer_messages: int


@dataclass
class ConversationAnalysis:
    conversation_id: str
    turns: list[Turn]
    stats: ConversationStats
    sentiment: SentimentResult
    rule_intent: IntentPrediction
    ml_intent: IntentPrediction | None
    topic_id: int | None
    topic_name: str | None
    keywords: list[str]
    resolution: ResolutionEstimate
    escalation: EscalationResult
    human_requested: bool
    handoff_detected: bool
    notes: list[str] = field(default_factory=list)

    @property
    def intent(self) -> IntentPrediction:
        """The intent shown as primary: ML when available, otherwise the rule-based one."""
        return self.ml_intent or self.rule_intent


def detect_handoff(turns: list[Turn]) -> bool:
    """True if a human agent took part, or the bot announced a transfer to one."""
    return any(t.role == "agent" for t in turns) or any(
        t.role == "bot" and _HANDOFF_PATTERN.search(t.text) for t in turns
    )


def estimate_resolution(turns: list[Turn]) -> ResolutionEstimate:
    """Heuristic resolution estimate from the customer's final messages.

    The last customer message decides: thanks / "that worked" suggests resolved;
    giving up or saying the issue persists suggests unresolved. Anything else is
    "Unclear". This is a heuristic — use a ``resolved`` column when you have one.
    """
    cust = customer_turns(turns)
    if not cust:
        return ResolutionEstimate("Unclear", "No customer messages found.")
    last = cust[-1].text
    unresolved = _UNRESOLVED_PATTERN.search(last)
    if unresolved:
        return ResolutionEstimate("Likely unresolved", f"Last customer message contains “{unresolved.group(0)}”.")
    closure = _CLOSURE_PATTERN.search(last)
    if closure:
        return ResolutionEstimate("Likely resolved", f"Last customer message contains “{closure.group(0)}”.")
    return ResolutionEstimate("Unclear", "Last customer message has no clear closure or failure phrase.")


def compute_stats(turns: list[Turn]) -> ConversationStats:
    cust = customer_turns(turns)
    words = [len(t.text.split()) for t in cust]
    return ConversationStats(
        total_messages=len(turns),
        customer_messages=len(cust),
        bot_messages=sum(t.role == "bot" for t in turns),
        agent_messages=sum(t.role == "agent" for t in turns),
        customer_words=sum(words),
        avg_customer_message_words=round(sum(words) / len(words), 1) if words else 0.0,
        repeated_customer_messages=count_repeated_messages(turns),
    )


class ConversationAnalyzer:
    """Fits dataset-level models once and analyses conversations with them."""

    def __init__(self, n_topics: int = 8, random_state: int = 42) -> None:
        self.n_topics = n_topics
        self.random_state = random_state
        self.rule_classifier = RuleBasedIntentClassifier()
        self.ml_classifier: MLIntentClassifier | None = None
        self.intent_model_note: str | None = None
        self.topic_model: TopicModel | None = None
        self.topic_model_note: str | None = None

    # ------------------------------------------------------------------ fitting
    def fit(self, df: pd.DataFrame) -> ConversationAnalyzer:
        """Train the ML intent model (if labels exist) and the topic model."""
        texts = [customer_text(parse_transcript(t)) or t for t in df["conversation_text"]]
        labels = df["intent"].tolist() if "intent" in df.columns else []
        if labels:
            self.ml_classifier, self.intent_model_note = train_intent_model(texts, labels)
        else:
            self.intent_model_note = "No 'intent' column — using the rule-based classifier only."
        try:
            self.topic_model = fit_topics(texts, n_topics=self.n_topics, random_state=self.random_state)
        except ValueError as exc:
            self.topic_model = None
            self.topic_model_note = f"Topic model not fitted: {exc}"
        return self

    @property
    def intent_evaluation(self) -> IntentModelEvaluation | None:
        return self.ml_classifier.evaluation if self.ml_classifier else None

    # ----------------------------------------------------------------- analysis
    def analyze(self, text: str, conversation_id: str = "custom") -> ConversationAnalysis:
        """Run the full analysis on one transcript."""
        turns = parse_transcript(text)
        cust_text = customer_text(turns) or text
        sentiment = analyze_sentiment(turns)

        topic_id = topic_name = None
        keywords: list[str] = []
        if self.topic_model is not None:
            topic_id = self.topic_model.assign(cust_text)
            topic_name = self.topic_model.topic_name(topic_id)
            keywords = self.topic_model.keywords(cust_text)

        return ConversationAnalysis(
            conversation_id=conversation_id,
            turns=turns,
            stats=compute_stats(turns),
            sentiment=sentiment,
            rule_intent=self.rule_classifier.predict(cust_text),
            ml_intent=self.ml_classifier.predict(cust_text) if self.ml_classifier else None,
            topic_id=topic_id,
            topic_name=topic_name,
            keywords=keywords,
            resolution=estimate_resolution(turns),
            escalation=score_escalation(turns, sentiment),
            human_requested=human_requested(turns),
            handoff_detected=detect_handoff(turns),
        )

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with derived analysis columns added."""
        out = df.copy()
        rows = []
        for text in out["conversation_text"]:
            turns = parse_transcript(text)
            cust_text = customer_text(turns) or text
            sentiment = analyze_sentiment(turns)
            escalation = score_escalation(turns, sentiment)
            rows.append(
                {
                    "customer_text": cust_text,
                    "num_messages": len(turns),
                    "customer_messages": len(customer_turns(turns)),
                    "sentiment_score": sentiment.score,
                    "sentiment_label": sentiment.label,
                    "rule_intent": self.rule_classifier.predict(cust_text).label,
                    "escalation_score": escalation.score,
                    "escalation_risk": escalation.level,
                    "escalation_signals": escalation.summary,
                    "human_requested": human_requested(turns),
                    "handoff_detected": detect_handoff(turns),
                    "resolution_estimate": estimate_resolution(turns).label,
                }
            )
        derived = pd.DataFrame(rows, index=out.index)
        out = pd.concat([out, derived], axis=1)

        if self.ml_classifier is not None:
            out["predicted_intent"] = self.ml_classifier.predict_many(out["customer_text"].tolist())
        else:
            out["predicted_intent"] = out["rule_intent"]

        if self.topic_model is not None and len(self.topic_model.labels) == len(out):
            out["topic_id"] = self.topic_model.labels
            out["topic"] = out["topic_id"].map(self.topic_model.topic_names)
        else:
            out["topic_id"] = -1
            out["topic"] = "Not available"

        # The intent used for reporting: the provided label when present, otherwise the prediction.
        if "intent" in out.columns:
            out["intent_final"] = out["intent"].where(out["intent"].notna(), out["predicted_intent"])
        else:
            out["intent_final"] = out["predicted_intent"]
        return out
