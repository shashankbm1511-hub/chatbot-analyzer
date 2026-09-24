"""Smoke test: every page of the Streamlit app renders without exceptions."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["Dashboard", "Conversation Analyzer", "Topics & Intents", "Data & Methodology"]


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app: AppTest, page: str):
    app.sidebar.radio(key="nav").set_value(page).run()
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_analyzer_paste_mode(app: AppTest):
    app.sidebar.radio(key="nav").set_value("Conversation Analyzer").run()
    # Set via session state: older AppTest versions mishandle segmented_control.set_value().
    app.session_state["analyzer_mode"] = "Paste a transcript"
    app.run()
    assert not app.exception
    assert any("Escalation risk" in m.value for m in app.markdown)
