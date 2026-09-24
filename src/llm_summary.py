"""Optional LLM-generated conversation summary.

Disabled unless *both* are true:
  1. the ``openai`` package is installed (``pip install -r requirements-optional.txt``), and
  2. ``OPENAI_API_KEY`` is set (environment variable or a local ``.env`` file).

Everything else in the app works without it. When enabled, the selected
transcript is sent to the OpenAI API — only use it with data you are allowed
to share with a third-party service.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "gpt-4o-mini"
MAX_TRANSCRIPT_CHARS = 8000

_PROMPT = (
    "You are reviewing a customer-support chatbot conversation. In at most four short "
    "bullet points, state: the customer's issue, what the bot did, whether the issue "
    "appears resolved, and one suggested next action for a human reviewer. Do not "
    "invent facts that are not in the transcript.\n\nTranscript:\n{transcript}"
)

try:  # python-dotenv is optional at runtime too
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def llm_status() -> tuple[bool, str]:
    """Return ``(available, human-readable reason)``."""
    try:
        import openai  # noqa: F401
    except ImportError:
        return False, "Optional: install `openai` (requirements-optional.txt) to enable AI summaries."
    if not os.getenv("OPENAI_API_KEY"):
        return False, "Optional: set OPENAI_API_KEY in a .env file to enable AI summaries."
    return True, f"AI summary enabled (model: {os.getenv('OPENAI_MODEL', DEFAULT_MODEL)})."


def summarize_conversation(transcript: str) -> str:
    """Summarise a transcript with the OpenAI API.

    Raises:
        RuntimeError: if the integration is not configured or the API call fails.
    """
    available, reason = llm_status()
    if not available:
        raise RuntimeError(reason)
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            messages=[{"role": "user", "content": _PROMPT.format(transcript=transcript[:MAX_TRANSCRIPT_CHARS])}],
            temperature=0.2,
            max_tokens=300,
        )
    except Exception as exc:  # network, auth, quota, etc.
        raise RuntimeError(f"OpenAI API call failed: {exc}") from exc
    return (response.choices[0].message.content or "").strip()
