"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.conversation_analyzer import ConversationAnalyzer
from src.data_loader import load_conversations

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "sample_conversations.csv"


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    df, _ = load_conversations(SAMPLE_CSV)
    return df


@pytest.fixture(scope="session")
def fitted_analyzer(sample_df: pd.DataFrame) -> ConversationAnalyzer:
    return ConversationAnalyzer(n_topics=8).fit(sample_df)


@pytest.fixture(scope="session")
def enriched_df(sample_df: pd.DataFrame, fitted_analyzer: ConversationAnalyzer) -> pd.DataFrame:
    return fitted_analyzer.enrich(sample_df)


HAPPY = "Customer: I can't log in to my account.\nBot: I've sent a reset link.\nCustomer: Great, that worked. Thanks!"
ANGRY = (
    "Customer: I was charged twice for my subscription.\n"
    "Bot: Could you share more details?\n"
    "Customer: I already checked the invoices page, the extra charge is still there. This is ridiculous.\n"
    "Bot: Sorry, I didn't get that.\n"
    "Customer: I was charged twice for my subscription.\n"
    "Bot: Here is an article that might help.\n"
    "Customer: This is the worst service ever. I want to talk to a human agent and get a refund.\n"
    "Bot: All our agents are busy.\n"
    "Customer: Useless. I give up."
)
HANDOFF = (
    "Customer: Where is my order?\n"
    "Bot: I'm transferring you to a human agent now.\n"
    "Agent: Hi, I've contacted the courier.\n"
    "Customer: Thank you."
)
