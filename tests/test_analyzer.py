"""Tests for parsing, sentiment, intent, topics, escalation and the analyzer."""

from __future__ import annotations

import pandas as pd
import pytest

from src.conversation_analyzer import ConversationAnalyzer, detect_handoff, estimate_resolution
from src.escalation import level_from_score, score_escalation, trigger_phrases
from src.intent import INTENT_TAXONOMY, MLIntentClassifier, RuleBasedIntentClassifier, train_intent_model
from src.preprocessing import count_repeated_messages, customer_text, parse_transcript, tokenize
from src.sentiment import analyze_sentiment, label_from_score, score_text
from src.topic_model import corpus_keywords, fit_topics
from tests.conftest import ANGRY, HANDOFF, HAPPY

# --- preprocessing -------------------------------------------------------------


def test_parse_transcript_roles_and_aliases():
    turns = parse_transcript("User: hi\nAssistant: hello\nHuman agent: taking over\nCustomer: thanks")
    assert [t.role for t in turns] == ["customer", "bot", "agent", "customer"]
    assert turns[0].text == "hi"


def test_parse_transcript_pipe_separator_and_continuation_lines():
    turns = parse_transcript("Customer: my order || Bot: which one?\nit was placed monday")
    assert [t.role for t in turns] == ["customer", "bot"]
    assert turns[1].text == "which one? it was placed monday"


def test_parse_transcript_without_prefixes_is_one_customer_message():
    turns = parse_transcript("the app keeps crashing")
    assert len(turns) == 1 and turns[0].role == "customer"


@pytest.mark.parametrize("text", ["", "   ", None])
def test_parse_transcript_empty(text):
    assert parse_transcript(text) == []


def test_tokenize_keeps_contractions_whole_and_drops_stopwords():
    tokens = tokenize("I don't want a refund, I can't log in!")
    assert tokens == ["refund", "log"]  # "don't"/"can't" are stop words, never split into "don"/"can"


def test_count_repeated_messages():
    turns = parse_transcript(ANGRY)
    assert count_repeated_messages(turns) == 1


# --- sentiment -----------------------------------------------------------------


def test_sentiment_labels():
    assert label_from_score(0.5) == "Positive"
    assert label_from_score(0.0) == "Neutral"
    assert label_from_score(-0.5) == "Negative"
    assert score_text("") == 0.0
    assert score_text("This is the worst, most useless service") < -0.05
    assert score_text("Great, thank you so much!") > 0.05


def test_sentiment_uses_customer_messages_only():
    turns = parse_transcript("Customer: this is terrible\nBot: Wonderful! Happy to help, great day!")
    result = analyze_sentiment(turns)
    assert result.label == "Negative"
    assert len(result.turn_scores) == 1


def test_sentiment_of_conversation_without_customer_messages():
    result = analyze_sentiment(parse_transcript("Bot: hello"))
    assert result.label == "Neutral" and result.turn_scores == []


# --- intent --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, intent",
    [
        ("I was charged twice on my invoice", "Billing"),
        ("I can't log in, my password reset email never came", "Account/Login"),
        ("The app keeps crashing with an error", "Technical Issue"),
        ("I want a refund, money back please", "Refund"),
        ("Please cancel my subscription", "Cancellation"),
        ("xyz", "Other"),
    ],
)
def test_rule_based_intent(text, intent):
    prediction = RuleBasedIntentClassifier().predict(text)
    assert prediction.label == intent
    assert prediction.method == "rule-based"
    if intent != "Other":
        assert prediction.evidence


def test_ml_intent_trains_and_explains(sample_df):
    texts = [customer_text(parse_transcript(t)) for t in sample_df["conversation_text"]]
    model, note = train_intent_model(texts, sample_df["intent"].tolist())
    assert model is not None and note is None
    assert model.evaluation is not None and 0.0 <= model.evaluation.accuracy <= 1.0
    prediction = model.predict("I was charged twice for my subscription this month")
    assert prediction.label in INTENT_TAXONOMY
    assert prediction.method == "ml"
    assert 0.0 <= prediction.confidence <= 1.0


def test_ml_intent_not_trained_without_labels():
    model, note = train_intent_model(["a", "b"], [None, None])
    assert model is None and "rule-based" in note


def test_ml_intent_requires_two_classes():
    with pytest.raises(ValueError):
        MLIntentClassifier().fit(["refund please"] * 12, ["Refund"] * 12)


def test_ml_intent_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        MLIntentClassifier().predict("hello")


# --- topics --------------------------------------------------------------------


def test_fit_topics_assigns_every_document(sample_df):
    texts = [customer_text(parse_transcript(t)) for t in sample_df["conversation_text"]]
    model = fit_topics(texts, n_topics=6)
    assert len(model.labels) == len(texts)
    assert len(model.topic_terms) == 6
    assert all(" / " in name or name.startswith("Topic") for tid, name in model.topic_names.items() if tid >= 0)


def test_fit_topics_clamps_k_and_rejects_empty():
    model = fit_topics(["refund my order", "password reset link"], n_topics=10)
    assert len(model.topic_terms) <= 2
    with pytest.raises(ValueError):
        fit_topics([])


def test_corpus_keywords_edge_cases():
    assert corpus_keywords([]) == []
    assert corpus_keywords(["the and of"]) == []


# --- escalation ----------------------------------------------------------------


def test_escalation_low_for_happy_conversation():
    turns = parse_transcript(HAPPY)
    result = score_escalation(turns, analyze_sentiment(turns))
    assert result.level == "Low"
    assert result.score < 3


def test_escalation_high_with_explained_signals():
    turns = parse_transcript(ANGRY)
    result = score_escalation(turns, analyze_sentiment(turns))
    assert result.level == "High"
    names = {s.name for s in result.signals}
    assert {"Explicit request for a human", "Complaint language", "Repeated customer message",
            "Cancellation / refund language", "Long conversation", "Conversation ended negatively"} <= names
    assert result.score == sum(s.points for s in result.signals)
    assert all(s.evidence for s in result.signals)


def test_escalation_ignores_bot_text():
    turns = parse_transcript("Customer: What are your hours?\nBot: You can talk to a human agent or request a refund.")
    result = score_escalation(turns, analyze_sentiment(turns))
    assert result.signals == []


@pytest.mark.parametrize("score, level", [(0, "Low"), (2, "Low"), (3, "Medium"), (5, "Medium"), (6, "High"), (14, "High")])
def test_level_thresholds(score, level):
    assert level_from_score(score) == level


def test_trigger_phrases():
    assert "real person" in trigger_phrases("Can I speak to a real person? This is ridiculous")
    assert trigger_phrases("What time do you open?") == []


# --- resolution / handoff ------------------------------------------------------


def test_resolution_estimates():
    assert estimate_resolution(parse_transcript(HAPPY)).label == "Likely resolved"
    assert estimate_resolution(parse_transcript(ANGRY)).label == "Likely unresolved"
    assert estimate_resolution(parse_transcript("Customer: ok\nBot: bye")).label == "Unclear"
    assert estimate_resolution([]).label == "Unclear"


def test_handoff_detection():
    assert detect_handoff(parse_transcript(HANDOFF))
    assert not detect_handoff(parse_transcript(HAPPY))


# --- analyzer end to end -------------------------------------------------------


def test_analyze_single_conversation(fitted_analyzer):
    analysis = fitted_analyzer.analyze(ANGRY, "test")
    assert analysis.conversation_id == "test"
    assert analysis.stats.total_messages == 9
    assert analysis.stats.customer_messages == 5
    assert analysis.stats.repeated_customer_messages == 1
    assert analysis.escalation.level == "High"
    assert analysis.human_requested and not analysis.handoff_detected
    assert analysis.intent.label in INTENT_TAXONOMY
    assert analysis.keywords


def test_enrich_adds_complete_derived_columns(enriched_df, sample_df):
    derived = ["sentiment_score", "sentiment_label", "predicted_intent", "rule_intent", "topic", "escalation_score",
               "escalation_risk", "escalation_signals", "human_requested", "handoff_detected",
               "resolution_estimate", "intent_final", "num_messages"]
    assert len(enriched_df) == len(sample_df)
    for column in derived:
        assert column in enriched_df.columns
        assert enriched_df[column].notna().all(), column
    assert set(enriched_df["escalation_risk"]) <= {"Low", "Medium", "High"}
    assert set(enriched_df["sentiment_label"]) <= {"Positive", "Neutral", "Negative"}


def test_analyzer_without_intent_labels_uses_rules(sample_df):
    df = sample_df.drop(columns=["intent"]).head(40)
    analyzer = ConversationAnalyzer(n_topics=4).fit(df)
    assert analyzer.ml_classifier is None
    enriched = analyzer.enrich(df)
    assert (enriched["predicted_intent"] == enriched["rule_intent"]).all()


def test_analyzer_handles_unlabelled_rows(sample_df):
    df = sample_df.head(60).copy()
    df.loc[df.index[:5], "intent"] = None
    enriched = ConversationAnalyzer(n_topics=4).fit(df).enrich(df)
    assert enriched["intent_final"].notna().all()
    assert (enriched.loc[df.index[:5], "intent_final"] == enriched.loc[df.index[:5], "predicted_intent"]).all()
    assert isinstance(enriched, pd.DataFrame)
