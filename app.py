"""Chatbot Analyzer — Streamlit application.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from src import metrics
from src import visualization as viz
from src.conversation_analyzer import ConversationAnalysis, ConversationAnalyzer
from src.data_loader import REQUIRED_COLUMNS, DataValidationError, LoadReport, load_conversations
from src.escalation import (
    HIGH_THRESHOLD,
    LONG_CONVERSATION_TURNS,
    MEDIUM_THRESHOLD,
    WEIGHT_CANCEL_REFUND,
    WEIGHT_COMPLAINT_LANGUAGE,
    WEIGHT_FAILED_ATTEMPTS,
    WEIGHT_HUMAN_REQUEST,
    WEIGHT_LONG_CONVERSATION,
    WEIGHT_NEGATIVE_ENDING,
    WEIGHT_NEGATIVE_OVERALL,
    WEIGHT_REPEATED_MESSAGE,
    WEIGHT_STRONG_NEGATIVE,
    MAX_REPEAT_POINTS,
    trigger_phrases,
)
from src.llm_summary import llm_status, summarize_conversation
from src.sentiment import label_from_score

APP_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = APP_DIR / "data" / "sample_conversations.csv"
PAGES = ["Dashboard", "Conversation Analyzer", "Topics & Intents", "Data & Methodology"]
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

st.set_page_config(
    page_title="Chatbot Analyzer",
    page_icon=":material/forum:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
  --ca-surface: #ffffff; --ca-border: #e7e6e2; --ca-text: #0b0b0b; --ca-muted: #52514e;
  --ca-primary: #2a78d6; --ca-good: #0a7d0a; --ca-warn: #8a5a00; --ca-bad: #b42323;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.01em; }
.ca-hero h1 { font-size: 1.9rem; margin: 0 0 .15rem 0; }
.ca-hero p { color: var(--ca-muted); margin: 0; font-size: 1rem; }
.ca-section { margin: 1.6rem 0 .4rem 0; }
.ca-section h3 { font-size: 1.15rem; margin: 0; }
.ca-section p { color: var(--ca-muted); margin: .15rem 0 0 0; font-size: .9rem; }
.ca-card { background: var(--ca-surface); border: 1px solid var(--ca-border); border-radius: 12px;
  padding: 14px 16px; height: 100%; box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.ca-kpi-label { color: var(--ca-muted); font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
.ca-kpi-value { color: var(--ca-text); font-size: 1.75rem; font-weight: 700; line-height: 1.2; margin-top: 4px; }
.ca-kpi-note { color: var(--ca-muted); font-size: .75rem; margin-top: 4px; }
.ca-kpi-accent { border-top: 3px solid var(--ca-primary); }
.ca-badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
  border: 1px solid transparent; white-space: nowrap; }
.ca-badge-good { background: #e8f6e8; color: var(--ca-good); border-color: #bfe5bf; }
.ca-badge-warn { background: #fff4db; color: var(--ca-warn); border-color: #f7dc9a; }
.ca-badge-bad  { background: #fdeaea; color: var(--ca-bad); border-color: #f5c2c2; }
.ca-badge-info { background: #eaf2fc; color: #1c5cab; border-color: #c4dbf6; }
.ca-badge-muted{ background: #f2f1ee; color: var(--ca-muted); border-color: #e2e0db; }
.ca-chat { display: flex; flex-direction: column; gap: 10px; }
.ca-msg { max-width: 86%; padding: 10px 14px; border-radius: 14px; line-height: 1.45; font-size: .95rem; }
.ca-msg-meta { font-size: .72rem; font-weight: 600; color: var(--ca-muted); margin-bottom: 3px; text-transform: uppercase; letter-spacing: .04em; }
.ca-customer { align-self: flex-end; background: #eaf2fc; border: 1px solid #c4dbf6; }
.ca-bot { align-self: flex-start; background: #f5f4f1; border: 1px solid #e7e6e2; }
.ca-agent { align-self: flex-start; background: #ecf7ef; border: 1px solid #c7e6cf; }
.ca-chat mark.ca-kw { background: #fff1c2; padding: 0 2px; border-radius: 3px; }
.ca-chat mark.ca-trigger { background: #fbdada; padding: 0 2px; border-radius: 3px; }
.ca-keyline { font-size: .8rem; color: var(--ca-muted); margin-top: 6px; }
.ca-card h4 { margin: 0 0 6px 0; font-size: .95rem; }
.ca-big { font-size: 1.35rem; font-weight: 700; color: var(--ca-text); }
.ca-small { font-size: .82rem; color: var(--ca-muted); }
.ca-demo-note { background: #fff8e6; border: 1px solid #f7dc9a; border-radius: 10px; padding: 8px 12px;
  font-size: .85rem; color: #6b4a00; }
section[data-testid="stSidebar"] .ca-brand { font-weight: 700; font-size: 1.15rem; }
section[data-testid="stSidebar"] .ca-brand-sub { color: var(--ca-muted); font-size: .8rem; margin-bottom: .6rem; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def section(title: str, subtitle: str = "") -> None:
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    st.markdown(f'<div class="ca-section"><h3>{html.escape(title)}</h3>{sub}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, note: str = "", accent: bool = False) -> str:
    accent_class = " ca-kpi-accent" if accent else ""
    return (
        f'<div class="ca-card{accent_class}"><div class="ca-kpi-label">{html.escape(label)}</div>'
        f'<div class="ca-kpi-value">{html.escape(value)}</div>'
        f'<div class="ca-kpi-note">{html.escape(note)}</div></div>'
    )


def badge(text: str, kind: str = "info") -> str:
    return f'<span class="ca-badge ca-badge-{kind}">{html.escape(text)}</span>'


RISK_BADGE = {"Low": "good", "Medium": "warn", "High": "bad"}
SENTIMENT_BADGE = {"Positive": "good", "Neutral": "muted", "Negative": "bad"}
RESOLUTION_BADGE = {"Likely resolved": "good", "Likely unresolved": "bad", "Unclear": "muted"}


def chart(fig) -> None:
    st.plotly_chart(fig, config=PLOT_CONFIG, theme=None)


def pct(value: float) -> str:
    return f"{value:.1%}"


def markdown_table(rows: list[list[object]], headers: list[str]) -> None:
    """Render a small reference table as Markdown so long text wraps instead of truncating."""
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    st.markdown("\n".join(lines))


# -----------------------------------------------------------------------------
# Data pipeline (cached per uploaded file + settings)
# -----------------------------------------------------------------------------
@dataclass
class PipelineResult:
    data: pd.DataFrame
    analyzer: ConversationAnalyzer
    report: LoadReport


@st.cache_resource(show_spinner="Analyzing conversations…", max_entries=4)
def run_pipeline(file_bytes: bytes, n_topics: int) -> PipelineResult:
    import io

    df, report = load_conversations(io.BytesIO(file_bytes))
    analyzer = ConversationAnalyzer(n_topics=n_topics).fit(df)
    return PipelineResult(data=analyzer.enrich(df), analyzer=analyzer, report=report)


def sidebar() -> tuple[str, bytes, bool, int]:
    with st.sidebar:
        st.markdown('<div class="ca-brand">Chatbot Analyzer</div>'
                    '<div class="ca-brand-sub">Conversational analytics &amp; intelligence</div>',
                    unsafe_allow_html=True)
        page = st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")
        st.divider()
        st.markdown("**Data source**")
        source = st.radio("Data source", ["Sample dataset (synthetic)", "Upload CSV"],
                          label_visibility="collapsed", key="source")
        using_sample = True
        file_bytes = SAMPLE_PATH.read_bytes()
        if source == "Upload CSV":
            uploaded = st.file_uploader("CSV with conversation_id and conversation_text", type=["csv"])
            if uploaded is not None:
                file_bytes = uploaded.getvalue()
                using_sample = False
            else:
                st.caption("No file uploaded yet — showing the sample dataset.")
        with st.expander("Model settings"):
            n_topics = st.slider("Number of topics (NMF)", 4, 15, 8,
                                 help="How many topics the unsupervised topic model should find.")
        st.divider()
        available, reason = llm_status()
        st.caption(("✓ " if available else "") + reason)
        st.caption("Portfolio prototype · synthetic demo data · not a production system.")
    return page, file_bytes, using_sample, n_topics


# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Render a single row of filters above the charts and return the filtered frame."""
    with st.container(border=True):
        cols = st.columns([1.4, 1, 1.3, 1, 1])
        mask = pd.Series(True, index=df.index)

        timestamps = df["timestamp"].dropna() if "timestamp" in df.columns else pd.Series(dtype="datetime64[ns]")
        if not timestamps.empty:
            start, end = timestamps.min().date(), timestamps.max().date()
            chosen = cols[0].date_input("Date range", (start, end), min_value=start, max_value=end, key=f"{key}_dates")
            if isinstance(chosen, tuple) and len(chosen) == 2 and chosen != (start, end):
                ts = df["timestamp"].dt.date
                mask &= ts.between(chosen[0], chosen[1])
        else:
            cols[0].caption("No timestamps in data")

        def multiselect(col, label: str, column: str) -> None:
            nonlocal mask
            options = sorted(df[column].dropna().astype(str).unique())
            chosen_values = col.multiselect(label, options, placeholder="All", key=f"{key}_{column}")
            if chosen_values:
                mask &= df[column].astype(str).isin(chosen_values)

        multiselect(cols[1], "Channel", "channel")
        multiselect(cols[2], "Intent", "intent_final")
        multiselect(cols[3], "Sentiment", "sentiment_label")
        multiselect(cols[4], "Escalation risk", "escalation_risk")
    filtered = df[mask]
    if len(filtered) < len(df):
        st.caption(f"Showing {len(filtered):,} of {len(df):,} conversations.")
    return filtered


def go_to_conversation(conversation_id: str) -> None:
    st.session_state["jump_to"] = conversation_id
    st.session_state["nav"] = "Conversation Analyzer"
    st.session_state["analyzer_mode"] = "From dataset"


def _on_review_select(table_key: str, ids: list[str]) -> None:
    rows = st.session_state[table_key].selection.rows
    if rows:
        go_to_conversation(ids[rows[0]])


# -----------------------------------------------------------------------------
# Page: Dashboard
# -----------------------------------------------------------------------------
def page_dashboard(df: pd.DataFrame, using_sample: bool) -> None:
    st.markdown(
        '<div class="ca-hero"><h1>Conversation intelligence dashboard</h1>'
        "<p>What customers ask the chatbot, how those conversations end, and which ones need a human.</p></div>",
        unsafe_allow_html=True,
    )
    if using_sample:
        st.markdown('<div class="ca-demo-note" style="margin-top:10px">You are viewing the bundled '
                    "<b>synthetic demo dataset</b> (generated from templates; no real customers). "
                    "Upload your own CSV from the sidebar.</div>", unsafe_allow_html=True)
    st.write("")
    data = apply_filters(df, "dash")
    if data.empty:
        st.warning("No conversations match the current filters.")
        return

    k = metrics.compute_kpis(data)
    row1 = st.columns(4)
    row1[0].markdown(kpi_card("Total conversations", f"{int(k['total_conversations'].value):,}",
                              "in the current selection", accent=True), unsafe_allow_html=True)
    row1[1].markdown(kpi_card("Resolution rate", pct(k["resolution_rate"].value),
                              f"source: {k['resolution_rate'].source}", accent=True), unsafe_allow_html=True)
    row1[2].markdown(kpi_card("Escalation rate", pct(k["escalation_rate"].value),
                              f"source: {k['escalation_rate'].source}", accent=True), unsafe_allow_html=True)
    row1[3].markdown(kpi_card("Human handoff rate", pct(k["human_handoff_rate"].value),
                              "live agent joined (detected in transcript)", accent=True), unsafe_allow_html=True)
    st.write("")
    row2 = st.columns(4)
    avg_sent = k["avg_sentiment"].value
    row2[0].markdown(kpi_card("Avg conversation length", f"{k['avg_conversation_length'].value:.1f} msgs",
                              "all speakers"), unsafe_allow_html=True)
    row2[1].markdown(kpi_card("Avg customer sentiment", f"{avg_sent:+.2f}",
                              f"{label_from_score(avg_sent)} · VADER compound, −1 to +1"), unsafe_allow_html=True)
    row2[2].markdown(kpi_card("Unresolved conversations", f"{int(k['unresolved_conversations'].value):,}",
                              f"source: {k['unresolved_conversations'].source}"), unsafe_allow_html=True)
    row2[3].markdown(kpi_card("Negative conversations", f"{int(k['negative_conversations'].value):,}",
                              "mean customer score ≤ −0.05"), unsafe_allow_html=True)

    # --- volume and outcomes
    section("Volume & outcomes", "How many conversations arrive, and how many end resolved.")
    c1, c2 = st.columns([2, 1])
    with c1:
        with st.container(border=True):
            freq_label = st.segmented_control("Granularity", ["Day", "Week", "Month"], default="Week",
                                              key="dash_freq", label_visibility="collapsed") or "Week"
            timeline = metrics.conversations_over_time(data, {"Day": "D", "Week": "W", "Month": "M"}[freq_label])
            st.markdown(f"**Conversations per {freq_label.lower()}**")
            chart(viz.conversations_over_time(timeline, freq_label))
    with c2:
        with st.container(border=True):
            resolved, _ = metrics.resolved_series(data)
            st.markdown("**Resolution status**")
            chart(viz.status_donut({"Resolved": int(resolved.sum()), "Unresolved": int((~resolved).sum())},
                                   viz.RESOLUTION_COLORS))

    # --- what customers talk about
    section("What customers are talking about",
            "Intents use the file's labels where present (otherwise the ML prediction); topics are discovered by the model.")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Intent distribution**")
            chart(viz.category_bar(metrics.distribution(data, "intent_final"), "intent_final"))
    with c2:
        with st.container(border=True):
            st.markdown("**Topic distribution** · unsupervised NMF")
            chart(viz.category_bar(metrics.distribution(data, "topic"), "topic"))

    # --- customer experience
    section("Customer experience", "Sentiment of customer messages, escalation-risk mix and conversation length.")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**Sentiment distribution**")
            chart(viz.sentiment_bar(metrics.distribution(data, "sentiment_label", ["Positive", "Neutral", "Negative"])))
    with c2:
        with st.container(border=True):
            st.markdown("**Escalation risk** · rule-based score")
            chart(viz.risk_bar(metrics.distribution(data, "escalation_risk", ["Low", "Medium", "High"])))
    with c3:
        with st.container(border=True):
            st.markdown("**Conversation length**")
            chart(viz.length_histogram(data))

    # --- where things go wrong
    section("Where conversations go wrong", "Topics that leave customers unresolved, and intents most often escalated.")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Unresolved conversations by topic**")
            chart(viz.unresolved_by_topic(metrics.unresolved_by(data, "topic")))
    with c2:
        with st.container(border=True):
            st.markdown("**Escalation rate by intent**")
            chart(viz.rate_bar(metrics.escalation_by(data, "intent_final"), "intent_final",
                               "escalation_rate", "escalated", "Escalation rate"))

    # --- pain points & review queue
    section("Customer pain points & review queue",
            "Terms that stand out in unresolved or negative conversations, and the conversations most likely to need a human.")
    c1, c2 = st.columns([1, 1.5])
    with c1:
        with st.container(border=True):
            st.markdown("**Top customer issues** · TF-IDF keywords")
            chart(viz.keyword_bar(metrics.top_pain_points(data)))
            st.caption("Keyword summary of problem conversations — descriptive, not causal.")
    with c2:
        with st.container(border=True):
            review = metrics.needs_review(data, "High")
            st.markdown(f"**Needs human review** · {len(review)} high-risk conversation(s) — select a row to open it")
            if review.empty:
                st.info("No high-risk conversations in the current selection.")
            else:
                table = review[["conversation_id", "intent_final", "escalation_score",
                                "escalation_signals"]].head(50)
                ids = table["conversation_id"].tolist()
                st.dataframe(
                    table, hide_index=True, height=420, key="review_table",
                    on_select=lambda: _on_review_select("review_table", ids), selection_mode="single-row",
                    column_config={
                        "conversation_id": "Conversation",
                        "intent_final": "Intent",
                        "escalation_score": st.column_config.NumberColumn("Score", format="%d", width="small"),
                        "escalation_signals": st.column_config.TextColumn("Signals", width="large"),
                    },
                )


# -----------------------------------------------------------------------------
# Page: Conversation Analyzer
# -----------------------------------------------------------------------------
def highlight(text: str, keywords: list[str], triggers: list[str]) -> str:
    """HTML-escape ``text`` and mark keywords (yellow) and escalation triggers (red)."""
    escaped = html.escape(text, quote=False)
    spans: list[tuple[int, int, str]] = []
    for css, terms in (("ca-trigger", triggers), ("ca-kw", keywords)):
        for term in sorted(set(terms), key=len, reverse=True):
            pattern = r"\b" + r"\W+".join(re.escape(w) for w in term.split()) + r"\b"
            for m in re.finditer(pattern, escaped, flags=re.IGNORECASE):
                if not any(s < m.end() and m.start() < e for s, e, _ in spans):
                    spans.append((m.start(), m.end(), css))
    for start, end, css in sorted(spans, reverse=True):
        escaped = f'{escaped[:start]}<mark class="{css}">{escaped[start:end]}</mark>{escaped[end:]}'
    return escaped


def render_transcript(analysis: ConversationAnalysis) -> None:
    sentiment_iter = iter(analysis.sentiment.turn_scores)
    parts = ['<div class="ca-chat">']
    for turn in analysis.turns:
        meta = {"customer": "Customer", "bot": "Chatbot", "agent": "Human agent"}[turn.role]
        extra = ""
        triggers: list[str] = []
        if turn.role == "customer":
            score = next(sentiment_iter, 0.0)
            label = label_from_score(score)
            extra = " · " + f"{label} ({score:+.2f})"
            triggers = trigger_phrases(turn.text)
        body = highlight(turn.text, analysis.keywords if turn.role == "customer" else [], triggers)
        parts.append(f'<div class="ca-msg ca-{turn.role}"><div class="ca-msg-meta">{meta}{html.escape(extra)}</div>{body}</div>')
    parts.append("</div>")
    parts.append('<div class="ca-keyline"><mark class="ca-kw" style="background:#fff1c2">keyword</mark> TF-IDF keyword &nbsp; '
                 '<mark style="background:#fbdada">phrase</mark> escalation trigger phrase</div>')
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_analysis(analysis: ConversationAnalysis, row: pd.Series | None, analyzer: ConversationAnalyzer) -> None:
    esc = analysis.escalation
    # --- escalation
    with st.container(border=True):
        st.markdown(f"#### Escalation risk &nbsp; {badge(esc.level, RISK_BADGE[esc.level])} "
                    f'<span class="ca-small">score {esc.score} · Medium ≥ {MEDIUM_THRESHOLD}, High ≥ {HIGH_THRESHOLD}</span>',
                    unsafe_allow_html=True)
        if esc.signals:
            st.dataframe(pd.DataFrame([{"Signal": s.name, "Points": s.points, "Evidence": s.evidence} for s in esc.signals]),
                         hide_index=True, column_config={"Evidence": st.column_config.TextColumn(width="large")})
        else:
            st.caption("No escalation signals found.")
        with st.expander("How is this calculated?"):
            st.markdown("A transparent **rule-based score**: each signal found in the customer's messages adds fixed "
                        "points (see *Data & Methodology* for the full table). It flags conversations that *show signs* "
                        "of needing a human; it is not a trained model and does not predict churn.")

    c1, c2 = st.columns(2)
    # --- sentiment
    with c1:
        with st.container(border=True):
            s = analysis.sentiment
            st.markdown(f"#### Sentiment &nbsp; {badge(s.label, SENTIMENT_BADGE[s.label])}", unsafe_allow_html=True)
            st.markdown(f'<span class="ca-small">Mean {s.score:+.2f} · most negative {s.min_score:+.2f} · '
                        f"last message {s.final_score:+.2f}</span>", unsafe_allow_html=True)
            chart(viz.turn_sentiment_line(s.turn_scores))
            with st.expander("How is this calculated?"):
                st.markdown("**VADER** — a pretrained lexicon and rule-based sentiment model (not trained on this data). "
                            "Each customer message gets a compound score from −1 to +1. The conversation score is the "
                            "mean; ≥ +0.05 is Positive, ≤ −0.05 is Negative. Bot replies are excluded.")
    # --- intent
    with c2:
        with st.container(border=True):
            primary = analysis.intent
            st.markdown(f"#### Intent &nbsp; {badge(primary.label, 'info')}", unsafe_allow_html=True)
            if analysis.ml_intent:
                st.markdown(f'<span class="ca-small"><b>ML model:</b> {html.escape(analysis.ml_intent.label)} '
                            f"({analysis.ml_intent.confidence:.0%} probability) · top terms: "
                            f"{html.escape(', '.join(analysis.ml_intent.evidence) or '—')}</span>", unsafe_allow_html=True)
            r = analysis.rule_intent
            st.markdown(f'<br><span class="ca-small"><b>Keyword rules:</b> {html.escape(r.label)} · matched: '
                        f"{html.escape(', '.join(r.evidence) or 'no keywords')}</span>", unsafe_allow_html=True)
            if row is not None and "intent" in row and pd.notna(row.get("intent")):
                st.markdown(f'<br><span class="ca-small"><b>Label in file:</b> {html.escape(str(row["intent"]))}</span>',
                            unsafe_allow_html=True)
            with st.expander("How is this calculated?"):
                note = analyzer.intent_model_note or ""
                st.markdown("**ML model** — TF-IDF (1–2-grams) + logistic regression, trained on the `intent` labels in the "
                            "loaded file. *Top terms* are the words in this conversation that contributed most to the "
                            "predicted class.\n\n**Keyword rules** — a heuristic baseline that counts matches against a "
                            "hand-written keyword list per intent. " + note)

    c1, c2 = st.columns(2)
    # --- topic
    with c1:
        with st.container(border=True):
            st.markdown(f"#### Topic &nbsp; {badge(analysis.topic_name or 'n/a', 'muted')}", unsafe_allow_html=True)
            if analyzer.topic_model is not None and analysis.topic_id is not None and analysis.topic_id >= 0:
                terms = analyzer.topic_model.topic_terms.get(analysis.topic_id, [])
                st.markdown(f'<span class="ca-small">Topic terms: {html.escape(", ".join(terms))}</span>',
                            unsafe_allow_html=True)
            st.markdown(f'<br><span class="ca-small"><b>Keywords in this conversation:</b> '
                        f"{html.escape(', '.join(analysis.keywords) or '—')}</span>", unsafe_allow_html=True)
            with st.expander("How is this calculated?"):
                st.markdown("**Unsupervised topic model** — TF-IDF over customer messages, factorised with NMF. The "
                            "conversation is assigned the topic with the largest weight. Topic names are the "
                            "model's top terms, not human labels. Keywords are this conversation's highest TF-IDF terms.")
    # --- resolution & handoff
    with c2:
        with st.container(border=True):
            res = analysis.resolution
            st.markdown(f"#### Resolution &nbsp; {badge(res.label, RESOLUTION_BADGE[res.label])}", unsafe_allow_html=True)
            st.markdown(f'<span class="ca-small">{html.escape(res.evidence)}</span>', unsafe_allow_html=True)
            facts = []
            if row is not None and "resolved" in row and pd.notna(row.get("resolved")):
                facts.append(f"Resolved (file): <b>{'Yes' if row['resolved'] else 'No'}</b>")
            if row is not None and "escalated" in row and pd.notna(row.get("escalated")):
                facts.append(f"Escalated (file): <b>{'Yes' if row['escalated'] else 'No'}</b>")
            facts.append(f"Human requested: <b>{'Yes' if analysis.human_requested else 'No'}</b>")
            facts.append(f"Live agent handoff: <b>{'Yes' if analysis.handoff_detected else 'No'}</b>")
            st.markdown('<br><span class="ca-small">' + " · ".join(facts) + "</span>", unsafe_allow_html=True)
            with st.expander("How is this calculated?"):
                st.markdown("**Heuristic** — looks at the customer's last message: thanks / *that worked* → likely "
                            "resolved; *give up*, *still not working* → likely unresolved; otherwise unclear. Handoff is "
                            "detected when a human agent speaks or the bot announces a transfer. Values marked *(file)* "
                            "come from your data, not the model.")


def page_analyzer(df: pd.DataFrame, analyzer: ConversationAnalyzer) -> None:
    st.markdown('<div class="ca-hero"><h1>Conversation analyzer</h1>'
                "<p>Drill into a single conversation: transcript, sentiment, intent, topic, resolution and escalation signals.</p></div>",
                unsafe_allow_html=True)
    st.write("")
    mode = st.segmented_control("Source", ["From dataset", "Paste a transcript"], default="From dataset",
                                key="analyzer_mode", label_visibility="collapsed") or "From dataset"

    row: pd.Series | None = None
    if mode == "From dataset":
        jump = st.session_state.pop("jump_to", None)
        if jump is not None:
            st.session_state["an_risk"] = "All"
            st.session_state["an_intent"] = "All"
            st.session_state["an_search"] = ""
            st.session_state["an_conv"] = jump
        with st.container(border=True):
            f1, f2, f3 = st.columns([1, 1, 2])
            risk = f1.selectbox("Escalation risk", ["All", "High", "Medium", "Low"], key="an_risk")
            intents = ["All"] + sorted(df["intent_final"].dropna().astype(str).unique())
            intent = f2.selectbox("Intent", intents, key="an_intent")
            search = f3.text_input("Search transcripts", key="an_search", placeholder="e.g. refund, password, crash")
            subset = df
            if risk != "All":
                subset = subset[subset["escalation_risk"] == risk]
            if intent != "All":
                subset = subset[subset["intent_final"] == intent]
            if search.strip():
                subset = subset[subset["conversation_text"].str.contains(re.escape(search.strip()), case=False)]
            if subset.empty:
                st.warning("No conversations match these filters.")
                return
            labels = {r.conversation_id: f"{r.conversation_id} · {r.intent_final} · {r.escalation_risk} risk · {r.sentiment_label}"
                      for r in subset.itertuples()}
            ids = list(labels)
            if st.session_state.get("an_conv") not in ids:
                st.session_state["an_conv"] = ids[0]
            conv_id = st.selectbox(f"Conversation ({len(ids)} match)", ids, format_func=labels.get, key="an_conv")
        row = df.loc[df["conversation_id"] == conv_id].iloc[0]
        text = row["conversation_text"]
    else:
        conv_id = "custom"
        text = st.text_area(
            "Transcript — one message per line, prefixed with Customer:, Bot: or Agent:",
            height=180,
            value="Customer: I was charged twice for my subscription this month.\n"
                  "Bot: Could you share a few more details?\n"
                  "Customer: I already checked the invoices page, the extra charge is still there. This is ridiculous.\n"
                  "Customer: I want to talk to a human agent.",
        )
        if not text.strip():
            st.info("Paste a transcript to analyze it.")
            return

    analysis = analyzer.analyze(text, conv_id)
    if not analysis.turns:
        st.warning("Could not find any messages in this transcript.")
        return

    stats = analysis.stats
    cols = st.columns(6)
    cols[0].metric("Messages", stats.total_messages, border=True)
    cols[1].metric("From customer", stats.customer_messages, border=True)
    cols[2].metric("Customer words", stats.customer_words, border=True)
    cols[3].metric("Words / msg", stats.avg_customer_message_words, border=True,
                   help="Average words per customer message")
    cols[4].metric("Repeats", stats.repeated_customer_messages, border=True,
                   help="Customer messages that repeat an earlier one")
    if row is not None and pd.notna(row.get("response_time_seconds")):
        cols[5].metric("Bot response", f"{row['response_time_seconds']:.1f}s", border=True,
                       help="Average bot response time (from the file)")
    else:
        cols[5].metric("Channel", str(row["channel"]) if row is not None else "—", border=True)

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        with st.container(border=True):
            meta = ""
            if row is not None:
                when = row["timestamp"].strftime("%d %b %Y, %H:%M") if pd.notna(row.get("timestamp")) else "no timestamp"
                meta = f" · {html.escape(str(row.get('channel', '')))} · {when}"
            st.markdown(f"#### Transcript <span class='ca-small'>{html.escape(conv_id)}{meta}</span>",
                        unsafe_allow_html=True)
            render_transcript(analysis)
        available, reason = llm_status()
        with st.container(border=True):
            st.markdown("#### AI summary <span class='ca-small'>optional · OpenAI</span>", unsafe_allow_html=True)
            if available:
                if st.button("Generate summary", key=f"sum_{conv_id}"):
                    try:
                        with st.spinner("Summarizing…"):
                            st.markdown(summarize_conversation(text))
                    except RuntimeError as exc:
                        st.error(str(exc))
                st.caption("Sends this transcript to the OpenAI API. Only use with data you may share.")
            else:
                st.caption(reason)
    with right:
        render_analysis(analysis, row, analyzer)


# -----------------------------------------------------------------------------
# Page: Topics & Intents
# -----------------------------------------------------------------------------
def page_topics(df: pd.DataFrame, analyzer: ConversationAnalyzer) -> None:
    st.markdown('<div class="ca-hero"><h1>Topics &amp; intents</h1>'
                "<p>What the unsupervised topic model found, how it lines up with intents, and how the intent models compare.</p></div>",
                unsafe_allow_html=True)
    st.write("")
    data = apply_filters(df, "topics")
    if data.empty:
        st.warning("No conversations match the current filters.")
        return

    section("Topic overview", "One row per discovered topic. Sorted by number of conversations.")
    resolved, _ = metrics.resolved_series(data)
    escalated, _ = metrics.escalated_series(data)
    summary = (
        data.assign(_resolved=resolved, _escalated=escalated)
        .groupby("topic")
        .agg(conversations=("conversation_id", "count"), unresolved_rate=("_resolved", lambda s: 1 - s.mean()),
             escalation_rate=("_escalated", "mean"), avg_sentiment=("sentiment_score", "mean"),
             top_intent=("intent_final", lambda s: s.mode().iat[0] if not s.mode().empty else ""))
        .sort_values("conversations", ascending=False)
        .reset_index()
    )
    summary[["unresolved_rate", "escalation_rate"]] *= 100
    if analyzer.topic_model is not None:
        name_to_terms = {analyzer.topic_model.topic_name(t): ", ".join(v[:6]) for t, v in analyzer.topic_model.topic_terms.items()}
        summary.insert(1, "top_terms", summary["topic"].map(name_to_terms).fillna(""))
    st.dataframe(
        summary, hide_index=True,
        column_config={
            "topic": "Topic", "top_terms": st.column_config.TextColumn("Top terms", width="large"),
            "conversations": "Conversations", "top_intent": "Most common intent",
            "unresolved_rate": st.column_config.ProgressColumn("Unresolved", format="%.0f%%", min_value=0, max_value=100),
            "escalation_rate": st.column_config.ProgressColumn("Escalated", format="%.0f%%", min_value=0, max_value=100),
            "avg_sentiment": st.column_config.NumberColumn("Avg sentiment", format="%+.2f"),
        },
    )

    section("Topics × intents", "Conversation counts. A topic concentrated in one column maps cleanly onto an intent.")
    with st.container(border=True):
        chart(viz.intent_topic_heatmap(data))
    section("Sentiment mix by intent", "Share of each intent's conversations that are positive, neutral or negative.")
    with st.container(border=True):
        chart(viz.sentiment_by_group(data, "intent_final"))

    section("Intent model comparison", "The trained ML model versus the keyword-rule baseline.")
    evaluation = analyzer.intent_evaluation
    c = st.columns(4)
    agreement = (df["rule_intent"] == df["predicted_intent"]).mean()
    c[0].metric("Rules vs ML agreement", pct(agreement), border=True,
                help="Share of all conversations where both classifiers predict the same intent.")
    if "intent" in df.columns and df["intent"].notna().any():
        labelled = df[df["intent"].notna()]
        c[1].metric("Rules vs file labels", pct((labelled["rule_intent"] == labelled["intent"]).mean()), border=True,
                    help="Keyword rules are not trained, so this is a fair check of the heuristic on this file.")
    if evaluation:
        c[2].metric("ML hold-out accuracy", pct(evaluation.accuracy), border=True,
                    help=f"{evaluation.n_test} held-out conversations; model trained on {evaluation.n_train}.")
        c[3].metric("ML hold-out macro-F1", f"{evaluation.macro_f1:.2f}", border=True)
        st.caption("⚠ " + evaluation.note)
    else:
        c[2].info(analyzer.intent_model_note or "ML model not trained.")

    section("Topic explorer", "Read example conversations from one topic.")
    topics = summary["topic"].tolist()
    chosen = st.selectbox("Topic", topics, key="topic_explore")
    examples = data[data["topic"] == chosen].head(8)
    for r in examples.itertuples():
        with st.expander(f"{r.conversation_id} · {r.intent_final} · {r.sentiment_label} · {r.escalation_risk} risk"):
            st.text(r.conversation_text)
            st.button("Open in analyzer", key=f"open_{r.conversation_id}", on_click=go_to_conversation,
                      args=(r.conversation_id,))


# -----------------------------------------------------------------------------
# Page: Data & Methodology
# -----------------------------------------------------------------------------
def page_methodology(df: pd.DataFrame, report: LoadReport, using_sample: bool) -> None:
    st.markdown('<div class="ca-hero"><h1>Data &amp; methodology</h1>'
                "<p>What was loaded, what each component does, and where its limits are.</p></div>",
                unsafe_allow_html=True)

    section("Data quality report")
    c = st.columns(4)
    c[0].metric("Rows in file", report.rows_in, border=True)
    c[1].metric("Rows analyzed", report.rows_out, border=True)
    c[2].metric("Dropped: empty text", report.dropped_empty_text, border=True)
    c[3].metric("Dropped: duplicate IDs", report.dropped_duplicate_ids, border=True)
    if report.filled_values:
        st.markdown("**Missing values handled:** " + ", ".join(f"`{k}`: {v}" for k, v in report.filled_values.items())
                    + " — text fields are set to `unknown`, numeric fields left blank and excluded from averages.")
    if report.missing_optional_columns:
        st.info("Optional columns not in the file: " + ", ".join(report.missing_optional_columns)
                + ". Related KPIs fall back to heuristic estimates and are labelled as such.")
    for warning in report.warnings:
        st.warning(warning)
    if using_sample:
        st.markdown('<div class="ca-demo-note">The sample file is <b>synthetic</b>: generated by '
                    "<code>scripts/generate_synthetic_data.py</code> from hand-written templates with a fixed seed. "
                    "A few blanks are included on purpose to exercise missing-value handling.</div>",
                    unsafe_allow_html=True)

    section("Methodology", "Every component, the technique behind it, and what kind of method it is.")
    methods = [
        ["Sentiment", "VADER compound score on customer messages", "Pretrained lexicon + rules", "Not trained on this data; tuned for short social/informal text."],
        ["Intent (primary)", "TF-IDF 1–2-grams + logistic regression", "Supervised ML, trained on the loaded file", "Needs an `intent` column. On synthetic data its score is optimistic."],
        ["Intent (baseline)", "Keyword match counts per intent", "Heuristic (rule-based)", "Always available; transparent; misses paraphrases."],
        ["Topics", "TF-IDF + NMF, dominant topic per conversation", "Unsupervised", "Topic names are top terms, not human labels; number of topics is a setting."],
        ["Keywords / pain points", "Highest TF-IDF terms", "Statistical, unsupervised", "Descriptive only."],
        ["Resolution estimate", "Closure vs failure phrases in the last customer message", "Heuristic (rule-based)", "Used only when no `resolved` column exists."],
        ["Escalation risk", "Weighted signal points → Low / Medium / High", "Heuristic (rule-based)", "Explainable; weights are hand-set, not learned."],
        ["Human handoff", "Agent turn or bot transfer message in transcript", "Heuristic (rule-based)", "Depends on transcript format."],
    ]
    markdown_table(methods, ["Component", "Technique", "Type", "Notes"])

    section("Escalation scoring rules", f"Low < {MEDIUM_THRESHOLD} ≤ Medium < {HIGH_THRESHOLD} ≤ High")
    rules = [
        ["Explicit request for a human", WEIGHT_HUMAN_REQUEST, "e.g. “talk to a human”, “real person”, “live agent”"],
        ["Strongly negative message", WEIGHT_STRONG_NEGATIVE, "any customer message with VADER ≤ −0.5"],
        ["Negative overall sentiment", WEIGHT_NEGATIVE_OVERALL, "mean ≤ −0.05 (only if no strongly negative message)"],
        ["Complaint language", WEIGHT_COMPLAINT_LANGUAGE, "e.g. “worst”, “ridiculous”, “unacceptable”, “rude”"],
        ["Repeated customer message", f"{WEIGHT_REPEATED_MESSAGE} each (max {MAX_REPEAT_POINTS})", "same message, or ≥ 80% token overlap"],
        ["Failed attempts / issue persists", f"{WEIGHT_FAILED_ATTEMPTS} (2 if several)", "e.g. “still not”, “already tried”, “same error”"],
        ["Long conversation", WEIGHT_LONG_CONVERSATION, f"≥ {LONG_CONVERSATION_TURNS} messages"],
        ["Cancellation / refund language", WEIGHT_CANCEL_REFUND, "e.g. “cancel”, “refund”, “money back”"],
        ["Conversation ended negatively", WEIGHT_NEGATIVE_ENDING, "last customer message negative or giving up"],
    ]
    markdown_table(rules, ["Signal", "Points", "Triggered by"])

    section("Limitations")
    st.markdown(
        "- **Synthetic data.** The bundled conversations are template-generated, so patterns are cleaner than real chats "
        "and model scores on them are optimistic.\n"
        "- **Heuristics produce false positives.** Phrase rules can fire on innocent text (e.g. “manager” in “account "
        "manager”) and miss sarcasm, typos or non-English messages.\n"
        "- **No real-world validation.** Thresholds and weights are sensible defaults; they need calibration against "
        "labelled production conversations before any operational use.\n"
        "- **English only**, and tuned for short support-chat messages."
    )

    section("Expected CSV format")
    schema = [
        ["conversation_id", "required", "Unique ID per conversation"],
        ["conversation_text", "required", "One message per line (or `||`), prefixed `Customer:`, `Bot:` or `Agent:`"],
        ["timestamp", "optional", "Any parseable date/time"],
        ["customer_id", "optional", "Customer identifier (use pseudonymous IDs)"],
        ["intent", "optional", "Label used to train the ML intent model"],
        ["channel", "optional", "e.g. web, mobile_app, messaging"],
        ["resolved", "optional", "true/false — otherwise estimated"],
        ["escalated", "optional", "true/false — otherwise estimated from risk"],
        ["response_time_seconds", "optional", "Average bot response time"],
        ["conversation_length", "optional", "Number of messages — computed if missing"],
    ]
    markdown_table(schema, ["Column", "Status", "Description"])
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Download sample CSV", SAMPLE_PATH.read_bytes(), "sample_conversations.csv", "text/csv")
    export_cols = [c for c in df.columns if c != "customer_text"]
    d2.download_button("Download analyzed data", df[export_cols].to_csv(index=False).encode(),
                       "analyzed_conversations.csv", "text/csv")

    section("Data preview")
    st.dataframe(df.drop(columns=["customer_text"]).head(200), hide_index=True, height=360)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    page, file_bytes, using_sample, n_topics = sidebar()
    try:
        result = run_pipeline(file_bytes, n_topics)
    except DataValidationError as exc:
        st.error(f"Could not load this file: {exc}")
        st.info("Required columns: " + ", ".join(REQUIRED_COLUMNS) + ". See *Data & Methodology* for the full format.")
        st.stop()

    if page == "Dashboard":
        page_dashboard(result.data, using_sample)
    elif page == "Conversation Analyzer":
        page_analyzer(result.data, result.analyzer)
    elif page == "Topics & Intents":
        page_topics(result.data, result.analyzer)
    else:
        page_methodology(result.data, result.report, using_sample)


if __name__ == "__main__":
    main()
