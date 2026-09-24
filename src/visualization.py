"""Plotly chart builders with one consistent visual style.

Colour rules:
* one hue (blue) for magnitude-only charts — categories are identified by their
  axis labels, not by colour;
* a diverging blue / grey / red scheme for sentiment (positive / neutral / negative);
* reserved status colours for resolution and escalation risk, always paired with
  a text label so meaning never depends on colour alone.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- palette -----------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e7e6e2"
PRIMARY = "#2a78d6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

SENTIMENT_COLORS = {"Positive": "#2a78d6", "Neutral": "#a3a19b", "Negative": "#e34948"}
STATUS_GOOD, STATUS_WARNING, STATUS_SERIOUS, STATUS_CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
RESOLUTION_COLORS = {"Resolved": STATUS_GOOD, "Unresolved": STATUS_CRITICAL}
RISK_COLORS = {"Low": STATUS_GOOD, "Medium": STATUS_WARNING, "High": STATUS_CRITICAL}
SEQUENTIAL_BLUE = ["#f0f6fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

FONT_FAMILY = "Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


def apply_layout(fig: go.Figure, height: int = 340, show_legend: bool = False) -> go.Figure:
    """Shared layout: light surface, recessive grid, compact margins."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=16, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_FAMILY, size=13, color=TEXT_SECONDARY),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family=FONT_FAMILY),
        bargap=0.3,
    )
    # automargin makes room for long category names and tick labels (off by default in Plotly.js).
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=TEXT_SECONDARY),
                     title_font=dict(size=12), automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=TEXT_SECONDARY),
                     title_font=dict(size=12), automargin=True)
    return fig


def _hbar(data: pd.DataFrame, value: str, category: str, value_label: str, text: str | None = None,
          color: str = PRIMARY, height: int | None = None, hover: dict | None = None) -> go.Figure:
    """Horizontal bar chart, largest value on top."""
    data = data.sort_values(value, ascending=True)
    fig = px.bar(data, x=value, y=category, orientation="h", text=text, hover_data=hover,
                 labels={value: value_label, category: ""})
    fig.update_traces(marker_color=color, marker_cornerradius=4, textposition="outside",
                      cliponaxis=False, textfont=dict(color=TEXT_SECONDARY))
    fig.update_yaxes(showgrid=False)
    fig = apply_layout(fig, height=height or max(260, 34 * len(data) + 60))
    # Leave headroom on the right so outside value labels are never clipped.
    max_value = float(data[value].max()) if len(data) else 0.0
    fig.update_xaxes(showgrid=True, gridcolor=GRID, range=[0, max_value * (1.45 if text else 1.08) or 1])
    return fig


def empty_figure(message: str = "No data for the current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color=TEXT_SECONDARY),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_layout(fig, height=260)


# --- dashboard charts ---------------------------------------------------------
def conversations_over_time(timeline: pd.DataFrame, period_label: str = "Week") -> go.Figure:
    if timeline.empty:
        return empty_figure("No timestamps available")
    fig = go.Figure(
        go.Scatter(
            x=timeline["period"], y=timeline["conversations"], mode="lines+markers",
            line=dict(color=PRIMARY, width=2), marker=dict(size=8, color=PRIMARY, line=dict(color=SURFACE, width=2)),
            customdata=timeline["resolution_rate"],
            hovertemplate=f"{period_label} of %{{x|%d %b %Y}}<br>Conversations: %{{y}}<br>"
                          "Resolved: %{customdata:.0%}<extra></extra>",
        )
    )
    fig.update_yaxes(title="Conversations", rangemode="tozero")
    fig.update_layout(hovermode="x unified")
    return apply_layout(fig)


def category_bar(dist: pd.DataFrame, column: str, value_label: str = "Conversations") -> go.Figure:
    """Counts per category (intent, topic, channel) with share labels."""
    if dist.empty:
        return empty_figure()
    data = dist.copy()
    data["label"] = data.apply(lambda r: f"{int(r['count'])}  ({r['share']:.0%})", axis=1)
    return _hbar(data, "count", column, value_label, text="label")


def sentiment_bar(dist: pd.DataFrame) -> go.Figure:
    if dist.empty:
        return empty_figure()
    data = dist.copy()
    data["label"] = data.apply(lambda r: f"{int(r['count'])} ({r['share']:.0%})", axis=1)
    fig = px.bar(data, x="sentiment_label", y="count", text="label", color="sentiment_label",
                 color_discrete_map=SENTIMENT_COLORS, labels={"sentiment_label": "", "count": "Conversations"},
                 category_orders={"sentiment_label": ["Positive", "Neutral", "Negative"]})
    fig.update_traces(marker_cornerradius=4, textposition="outside", cliponaxis=False,
                      textfont=dict(color=TEXT_SECONDARY))
    fig.update_yaxes(range=[0, data["count"].max() * 1.18])
    return apply_layout(fig)


def status_donut(counts: dict[str, int], colors: dict[str, str]) -> go.Figure:
    """Part-to-whole donut for a small set of statuses, labelled with name and share."""
    labels = [k for k in colors if counts.get(k, 0) > 0]
    if not labels:
        return empty_figure()
    fig = go.Figure(
        go.Pie(
            labels=labels, values=[counts[k] for k in labels], hole=0.62, sort=False,
            marker=dict(colors=[colors[k] for k in labels], line=dict(color=SURFACE, width=2)),
            textinfo="label+percent", textposition="inside", insidetextorientation="horizontal",
            textfont=dict(color="white", size=13),
            hovertemplate="%{label}: %{value} conversations (%{percent})<extra></extra>",
        )
    )
    total = sum(counts[k] for k in labels)
    fig.add_annotation(text=f"<b>{total}</b><br>total", showarrow=False, font=dict(size=15, color=TEXT_PRIMARY))
    fig = apply_layout(fig)
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=8))
    return fig


def risk_bar(dist: pd.DataFrame) -> go.Figure:
    if dist.empty:
        return empty_figure()
    data = dist.copy()
    data["label"] = data.apply(lambda r: f"{int(r['count'])} ({r['share']:.0%})", axis=1)
    fig = px.bar(data, x="escalation_risk", y="count", text="label", color="escalation_risk",
                 color_discrete_map=RISK_COLORS, labels={"escalation_risk": "Escalation risk", "count": "Conversations"},
                 category_orders={"escalation_risk": ["Low", "Medium", "High"]})
    fig.update_traces(marker_cornerradius=4, textposition="outside", cliponaxis=False,
                      textfont=dict(color=TEXT_SECONDARY))
    fig.update_yaxes(range=[0, data["count"].max() * 1.18])
    return apply_layout(fig)


def rate_bar(table: pd.DataFrame, category: str, rate_column: str, count_column: str, rate_label: str) -> go.Figure:
    """Rate per category, labelled with 'rate (n of total)'."""
    if table.empty:
        return empty_figure()
    data = table.copy()
    data["label"] = data.apply(lambda r: f"{r[rate_column]:.0%}  ({int(r[count_column])} of {int(r['total'])})", axis=1)
    fig = _hbar(data, rate_column, category, rate_label, text="label", color=STATUS_SERIOUS)
    fig.update_xaxes(tickformat=".0%")
    return fig


def unresolved_by_topic(table: pd.DataFrame) -> go.Figure:
    if table.empty:
        return empty_figure()
    data = table.copy()
    data["label"] = data.apply(lambda r: f"{int(r['unresolved'])} · {r['unresolved_rate']:.0%}", axis=1)
    return _hbar(data, "unresolved", "topic", "Unresolved conversations (label: count · share of topic)",
                 text="label", color=STATUS_CRITICAL, hover={"total": True, "unresolved_rate": ":.0%"})


def length_histogram(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure()
    fig = px.histogram(df, x="num_messages", labels={"num_messages": "Messages per conversation"},
                       color_discrete_sequence=[PRIMARY])
    fig.update_traces(marker_line_color=SURFACE, marker_line_width=2, marker_cornerradius=4,
                      hovertemplate="%{x} messages: %{y} conversations<extra></extra>", xbins=dict(size=1))
    fig.update_yaxes(title="Conversations")
    return apply_layout(fig)


def keyword_bar(keywords: list[tuple[str, float]]) -> go.Figure:
    if not keywords:
        return empty_figure("Not enough text to extract keywords")
    data = pd.DataFrame(keywords, columns=["term", "weight"])
    return _hbar(data, "weight", "term", "Summed TF-IDF weight", color=STATUS_SERIOUS,
                 hover={"weight": ":.2f"})


def sentiment_by_group(df: pd.DataFrame, group: str) -> go.Figure:
    """100%-stacked sentiment mix per group (e.g. intent)."""
    if df.empty:
        return empty_figure()
    shares = (pd.crosstab(df[group], df["sentiment_label"], normalize="index")
              .reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0))
    shares = shares.sort_values("Negative")
    fig = go.Figure()
    for label in ["Positive", "Neutral", "Negative"]:
        fig.add_bar(y=shares.index, x=shares[label], name=label, orientation="h",
                    marker=dict(color=SENTIMENT_COLORS[label], line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{y}<br>" + label + ": %{x:.0%}<extra></extra>")
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    fig.update_xaxes(tickformat=".0%", range=[0, 1], showgrid=True, gridcolor=GRID)
    return apply_layout(fig, height=max(280, 34 * len(shares) + 80), show_legend=True)


def intent_topic_heatmap(df: pd.DataFrame, intent_column: str = "intent_final") -> go.Figure:
    """How discovered topics line up with intents (counts)."""
    if df.empty:
        return empty_figure()
    table = pd.crosstab(df["topic"], df[intent_column])
    fig = go.Figure(
        go.Heatmap(
            z=table.values, x=table.columns, y=table.index, colorscale=SEQUENTIAL_BLUE,
            xgap=2, ygap=2, colorbar=dict(title=dict(text="Count", font=dict(size=12)), thickness=10),
            hovertemplate="Topic: %{y}<br>Intent: %{x}<br>Conversations: %{z}<extra></extra>",
            text=table.values, texttemplate="%{text}", textfont=dict(size=11),
        )
    )
    fig.update_xaxes(tickangle=-25, tickmode="array", tickvals=list(table.columns), ticktext=list(table.columns))
    fig.update_yaxes(tickmode="array", tickvals=list(table.index), ticktext=list(table.index))
    return apply_layout(fig, height=max(360, 40 * len(table.index) + 150))


def turn_sentiment_line(scores: list[float]) -> go.Figure:
    """Sentiment of each customer message in order, with the neutral band shaded."""
    if not scores:
        return empty_figure("No customer messages")
    x = list(range(1, len(scores) + 1))
    fig = go.Figure()
    fig.add_hrect(y0=-0.05, y1=0.05, fillcolor=GRID, opacity=0.6, line_width=0)
    fig.add_trace(
        go.Scatter(x=x, y=scores, mode="lines+markers", line=dict(color=PRIMARY, width=2),
                   marker=dict(size=10, color=[SENTIMENT_COLORS["Positive"] if s >= 0.05 else
                                               SENTIMENT_COLORS["Negative"] if s <= -0.05 else
                                               SENTIMENT_COLORS["Neutral"] for s in scores],
                               line=dict(color=SURFACE, width=2)),
                   hovertemplate="Customer message %{x}<br>Score: %{y:.2f}<extra></extra>")
    )
    fig.update_xaxes(title="Customer message #", dtick=1)
    fig.update_yaxes(title="VADER compound", range=[-1.05, 1.05])
    return apply_layout(fig, height=230)
