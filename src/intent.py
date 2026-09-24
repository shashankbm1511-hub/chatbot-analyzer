"""Intent classification: an explainable keyword baseline and a trained model.

Two classifiers are provided and both are shown in the app:

* :class:`RuleBasedIntentClassifier` — *heuristic*. Counts matches against a
  hand-written keyword list per intent. No training; always available.
* :class:`MLIntentClassifier` — *supervised machine learning*. TF-IDF
  (unigrams + bigrams) followed by multinomial logistic regression, trained at
  start-up on whatever labelled ``intent`` column is in the loaded CSV. With the
  bundled data that means it is trained on **synthetic, template-generated**
  conversations, so its held-out score is only a sanity check, not evidence of
  real-world accuracy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import STOPWORDS, TOKEN_PATTERN, normalize_text

INTENT_TAXONOMY = [
    "Billing",
    "Account/Login",
    "Technical Issue",
    "Order/Transaction",
    "Refund",
    "Cancellation",
    "Product Information",
    "Complaint",
    "General Query",
    "Other",
]

# Keywords are matched as whole words/phrases on normalised text.
INTENT_KEYWORDS: dict[str, list[str]] = {
    "Billing": ["bill", "billing", "invoice", "charged", "charge", "late fee", "payment went", "unpaid", "overcharged"],
    "Account/Login": ["log in", "login", "sign in", "password", "locked", "reset", "two factor", "verification code", "email address", "account settings"],
    "Technical Issue": ["crash", "crashing", "error", "bug", "slow", "loading", "blank screen", "not working", "export", "notifications", "update"],
    "Order/Transaction": ["order", "delivery", "deliver", "shipment", "tracking", "track", "parcel", "wrong item", "payment failed", "deducted"],
    "Refund": ["refund", "money back", "returned", "reimburse", "store credit"],
    "Cancellation": ["cancel", "cancelling", "cancellation", "close my account", "unsubscribe", "renewal"],
    "Product Information": ["plan", "plans", "premium", "pro plan", "size", "warranty", "ship internationally", "offline", "feature", "difference between"],
    "Complaint": ["worst", "unhappy", "rude", "frustrated", "ridiculous", "unacceptable", "terrible", "complaint", "nobody is helping", "four times"],
    "General Query": ["support hours", "hours", "contact", "loyalty", "privacy policy", "student discount", "email you"],
    "Other": ["feedback", "partnership", "anyone there"],
}

_KEYWORD_PATTERNS = {
    intent: [(kw, re.compile(r"\b" + re.escape(kw) + r"\b")) for kw in keywords]
    for intent, keywords in INTENT_KEYWORDS.items()
}


@dataclass
class IntentPrediction:
    """A predicted intent plus the evidence behind it."""

    label: str
    confidence: float  # rule: share of matched keywords; ML: class probability
    method: str  # "rule-based" or "ml"
    evidence: list[str] = field(default_factory=list)  # matched keywords or top terms


class RuleBasedIntentClassifier:
    """Heuristic keyword matcher. Ties go to the intent listed first in the taxonomy."""

    def predict(self, text: str) -> IntentPrediction:
        normalized = normalize_text(text)
        hits: dict[str, list[str]] = {}
        for intent, patterns in _KEYWORD_PATTERNS.items():
            matched = [kw for kw, pattern in patterns if pattern.search(normalized)]
            if matched:
                hits[intent] = matched
        if not hits:
            return IntentPrediction(label="Other", confidence=0.0, method="rule-based")
        best = max(hits, key=lambda i: (len(hits[i]), -INTENT_TAXONOMY.index(i)))
        total = sum(len(v) for v in hits.values())
        return IntentPrediction(
            label=best,
            confidence=round(len(hits[best]) / total, 3),
            method="rule-based",
            evidence=hits[best],
        )


@dataclass
class IntentModelEvaluation:
    """Held-out evaluation of the ML intent model on the loaded data."""

    n_train: int
    n_test: int
    accuracy: float
    macro_f1: float
    classes: list[str]
    note: str


class MLIntentClassifier:
    """TF-IDF + logistic regression intent classifier."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=normalize_text,
                        stop_words=sorted(STOPWORDS),
                        token_pattern=TOKEN_PATTERN,
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=1,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=5.0)),
            ]
        )
        self.is_fitted = False
        self.evaluation: IntentModelEvaluation | None = None

    def fit(self, texts: list[str], labels: list[str], test_size: float = 0.25) -> IntentModelEvaluation:
        """Evaluate on a stratified hold-out split, then refit on all rows.

        Raises:
            ValueError: if there are fewer than two classes or too few rows.
        """
        labels_series = pd.Series(labels)
        if labels_series.nunique() < 2:
            raise ValueError("Need at least two different intent labels to train.")
        if len(texts) < 10:
            raise ValueError("Need at least 10 labelled conversations to train.")

        counts = labels_series.value_counts()
        can_stratify = counts.min() >= 2 and int(len(texts) * test_size) >= labels_series.nunique()
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=labels if can_stratify else None,
        )
        self.pipeline.fit(x_train, y_train)
        predictions = self.pipeline.predict(x_test)
        self.evaluation = IntentModelEvaluation(
            n_train=len(x_train),
            n_test=len(x_test),
            accuracy=float(accuracy_score(y_test, predictions)),
            macro_f1=float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            classes=sorted(set(labels)),
            note=(
                "Measured on a held-out split of the loaded file. On the bundled synthetic "
                "data the conversations come from templates, so this score is optimistic and "
                "says nothing about performance on real conversations."
            ),
        )
        # Refit on all labelled rows for the final model.
        self.pipeline.fit(texts, labels)
        self.is_fitted = True
        return self.evaluation

    def predict(self, text: str, top_n_terms: int = 5) -> IntentPrediction:
        """Predict an intent and return the terms that pushed hardest towards it."""
        if not self.is_fitted:
            raise RuntimeError("MLIntentClassifier.predict called before fit().")
        probabilities = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        best_index = int(np.argmax(probabilities))
        return IntentPrediction(
            label=str(classes[best_index]),
            confidence=round(float(probabilities[best_index]), 3),
            method="ml",
            evidence=self._top_terms(text, best_index, top_n_terms),
        )

    def predict_many(self, texts: list[str]) -> list[str]:
        if not self.is_fitted:
            raise RuntimeError("MLIntentClassifier.predict_many called before fit().")
        return [str(p) for p in self.pipeline.predict(texts)]

    def _top_terms(self, text: str, class_index: int, top_n: int) -> list[str]:
        """Terms in ``text`` with the largest (tf-idf x coefficient) contribution."""
        vectorizer: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        model: LogisticRegression = self.pipeline.named_steps["clf"]
        vector = vectorizer.transform([text])
        coefficients = model.coef_[class_index] if model.coef_.shape[0] > 1 else model.coef_[0]
        contributions = vector.multiply(coefficients).toarray()[0]
        order = np.argsort(contributions)[::-1]
        terms = vectorizer.get_feature_names_out()
        return [str(terms[i]) for i in order[:top_n] if contributions[i] > 0]


def train_intent_model(texts: list[str], labels: list[object]) -> tuple[MLIntentClassifier | None, str | None]:
    """Train the ML intent model if usable labels exist.

    Returns ``(model, None)`` on success or ``(None, reason)`` when the model
    cannot be trained; the app then falls back to the rule-based classifier.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    pairs = [
        (text, str(label))
        for text, label in zip(texts, labels, strict=True)
        if label is not None and str(label).strip() and str(label) != "nan"
    ]
    if not pairs:
        return None, "No 'intent' labels in the file — using the rule-based classifier only."
    model = MLIntentClassifier()
    try:
        model.fit([p[0] for p in pairs], [p[1] for p in pairs])
    except ValueError as exc:
        return None, f"ML intent model not trained: {exc}"
    return model, None
