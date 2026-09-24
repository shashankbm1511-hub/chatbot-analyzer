"""Transcript parsing and text normalisation.

Transcripts are stored as plain text with one message per line, each prefixed
by a speaker label, for example::

    Customer: I can't log in
    Bot: I've sent a reset link
    Agent: Hi, I'm from the support team

Messages may also be separated by ``||`` instead of newlines. Lines without a
recognised speaker prefix are appended to the previous message; if the whole
text has no prefixes it is treated as a single customer message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROLE_ALIASES = {
    "customer": "customer",
    "user": "customer",
    "client": "customer",
    "bot": "bot",
    "chatbot": "bot",
    "assistant": "bot",
    "virtual assistant": "bot",
    "agent": "agent",
    "human agent": "agent",
    "support": "agent",
    "support agent": "agent",
}

_SPEAKER_PATTERN = re.compile(
    r"^\s*(" + "|".join(sorted(map(re.escape, ROLE_ALIASES), key=len, reverse=True)) + r")\s*:\s*",
    re.IGNORECASE,
)
_NON_ALPHA = re.compile(r"[^a-z\s']")
_WHITESPACE = re.compile(r"\s+")

# Words are runs of letters that may contain an inner apostrophe ("don't").
# Shared with every TfidfVectorizer so contractions are never split into
# fragments such as "don" or "ll".
TOKEN_PATTERN = r"(?u)\b[a-z][a-z']*[a-z]\b"
_TOKEN_RE = re.compile(TOKEN_PATTERN)

# Words that carry no information about *what the issue is*: greetings,
# politeness, contractions, and generic chat/handoff vocabulary. Added on top of
# scikit-learn's English stop-word list for topic and keyword extraction.
DOMAIN_STOPWORDS = frozenset(
    {
        # greetings and politeness
        "hi", "hello", "hey", "thanks", "thank", "please", "ok", "okay", "yes", "sorry",
        "great", "perfect", "helpful", "finally", "sorted", "solved", "worked", "fixing",
        # contractions
        "can't", "don't", "i'm", "it's", "i've", "i'd", "i'll", "didn't", "doesn't",
        "isn't", "haven't", "won't", "wasn't", "aren't", "couldn't", "that's", "you're",
        "we're", "we'll", "you'll", "there's", "what's",
        # generic conversational verbs and fillers
        "just", "really", "want", "need", "help", "got", "tried", "like", "know", "way",
        "did", "does", "today", "time", "able", "instead", "exactly", "specifically",
        "use", "try", "later", "forget", "mind", "never", "hope", "actually", "wait",
        "asked", "ask", "meant", "answer", "question", "thing", "things", "happy",
        "helping", "gets", "sure", "let", "check",
        # handoff vocabulary (describes the chat, not the issue)
        "human", "agent", "person", "real", "speak", "talk", "live", "connect",
        "transfer", "bot", "urgent", "customer", "service", "follow",
    }
)
STOPWORDS = frozenset(ENGLISH_STOP_WORDS) | DOMAIN_STOPWORDS


@dataclass(frozen=True)
class Turn:
    """A single message in a conversation."""

    role: str  # "customer", "bot" or "agent"
    text: str


def parse_transcript(text: str) -> list[Turn]:
    """Split a raw transcript into a list of :class:`Turn` objects."""
    if not isinstance(text, str) or not text.strip():
        return []
    lines = [line for chunk in text.split("||") for line in chunk.splitlines()]
    turns: list[Turn] = []
    for line in lines:
        if not line.strip():
            continue
        match = _SPEAKER_PATTERN.match(line)
        if match:
            role = ROLE_ALIASES[match.group(1).lower()]
            turns.append(Turn(role=role, text=line[match.end():].strip()))
        elif turns:
            previous = turns[-1]
            turns[-1] = Turn(role=previous.role, text=f"{previous.text} {line.strip()}".strip())
        else:
            turns.append(Turn(role="customer", text=line.strip()))
    return [t for t in turns if t.text]


def customer_turns(turns: list[Turn]) -> list[Turn]:
    """Return only the messages written by the customer."""
    return [t for t in turns if t.role == "customer"]


def customer_text(turns: list[Turn]) -> str:
    """Join all customer messages into one string."""
    return " ".join(t.text for t in customer_turns(turns))


def normalize_text(text: str) -> str:
    """Lower-case, strip punctuation and digits, and collapse whitespace."""
    lowered = _NON_ALPHA.sub(" ", str(text).lower())
    return _WHITESPACE.sub(" ", lowered).strip()


def tokenize(text: str) -> list[str]:
    """Return content tokens: normalised, stop words removed, length >= 3."""
    return [tok for tok in _TOKEN_RE.findall(normalize_text(text)) if len(tok) >= 3 and tok not in STOPWORDS]


def jaccard_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity between two strings (0 when both are empty)."""
    set_a, set_b = set(tokenize(a)), set(tokenize(b))
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def count_repeated_messages(turns: list[Turn], threshold: float = 0.8) -> int:
    """Count customer messages that repeat an earlier customer message.

    Two messages count as a repeat when they are identical after normalisation
    or when their token-set Jaccard similarity is at least ``threshold``.
    """
    seen: list[str] = []
    repeats = 0
    for turn in customer_turns(turns):
        normalized = normalize_text(turn.text)
        if not normalized:
            continue
        if any(normalized == prev or jaccard_similarity(normalized, prev) >= threshold for prev in seen):
            repeats += 1
        seen.append(normalized)
    return repeats
