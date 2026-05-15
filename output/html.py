"""
HTML report generator for the market intelligence scanner.
Produces dark-theme (#0d1117) newsletter-style reports with embedded
interactive Plotly charts. The Plotly JS library is loaded once via
CDN at the top of each report (so individual chart divs stay small).

Exports:
    build_daily_report(...)  -> str
    build_weekly_report(...) -> str
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import date, datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from output.explainers import EXPLAINERS, explainer_html, glossary_html

# Plotly.js CDN — loaded once per report. Charts emit only their <div>.
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

# ── Colour palette (matches scanner.py) ────────────────────────────────────
BG      = "#0d1117"
SURFACE = "#161b22"
BORDER  = "#30363d"
BLUE    = "#1f6feb"
BLUE_LT = "#58a6ff"
GREEN   = "#3fb950"
AMBER   = "#f0883e"
RED     = "#ff7b72"
TEXT    = "#e6edf3"
MUTED   = "#8b949e"

# ── Plotly helpers ───────────────────────────────────────────────────────────

_DARK_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=SURFACE,
    font=dict(
        family='-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
        color=TEXT,
        size=12,
    ),
    margin=dict(l=10, r=10, t=40, b=30),
    hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(color=TEXT)),
)


def fig_to_html(fig) -> str:
    """Convert a Plotly Figure to a self-contained HTML div.

    The plotly.js library is *not* embedded — it is loaded once via a
    <script src=...> tag at the top of the report.
    """
    return fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
        div_id=None,
    )


# ── Chart: market state gauge + driver bars ─────────────────────────────────

def chart_market_state(score: int, drivers: list[dict]) -> str:
    """
    Two-panel interactive Plotly figure:
      Top    — Indicator gauge (0-100) for the Market State Score.
      Bottom — Horizontal bar chart of the top 6 drivers by |contribution|.

    Args:
        score:   Integer 0-100.
        drivers: List of dicts with keys 'label' and 'contribution' (float).
                 Optional key 'description' is surfaced in the hover.

    Returns:
        Plotly HTML div string (no full <html> wrapper, no plotly.js).
    """
    # Score colour and regime label
    if score < 35:
        score_color, regime = RED, "Risk-Off"
    elif score < 50:
        score_color, regime = AMBER, "Caution"
    elif score < 70:
        score_color, regime = "#d8a657", "Neutral"
    else:
        score_color, regime = GREEN, "Risk-On"

    top_drivers = sorted(
        drivers or [], key=lambda d: abs(d.get("contribution", 0)), reverse=True
    )[:6]

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.18,
        specs=[[{"type": "indicator"}], [{"type": "xy"}]],
        subplot_titles=("", "Driver contributions"),
    )

    # ── Gauge ──
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=int(score),
            number=dict(font=dict(color=score_color, size=42)),
            title=dict(
                text=f"<span style='font-size:11px;color:{MUTED}'>"
                     f"Market State · {regime}</span>",
                font=dict(color=MUTED, size=12),
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickwidth=1,
                    tickcolor=BORDER,
                    tickfont=dict(color=MUTED, size=10),
                    tickvals=[0, 35, 50, 70, 100],
                ),
                bar=dict(color=score_color, thickness=0.28),
                bgcolor=SURFACE,
                borderwidth=0,
                steps=[
                    dict(range=[0, 35],  color="rgba(248, 81, 73, 0.18)"),
                    dict(range=[35, 50], color="rgba(210, 153, 34, 0.18)"),
                    dict(range=[50, 70], color="rgba(216, 166, 87, 0.18)"),
                    dict(range=[70, 100], color="rgba(86, 211, 100, 0.18)"),
                ],
                threshold=dict(
                    line=dict(color=TEXT, width=2),
                    thickness=0.75,
                    value=int(score),
                ),
            ),
        ),
        row=1,
        col=1,
    )

    # ── Driver bars ──
    if top_drivers:
        # Reverse so largest |contribution| ends up at the TOP of the chart
        bars = list(reversed(top_drivers))
        labels = [d.get("name") or d.get("label", "?") for d in bars]
        vals = [float(d.get("contribution", 0)) for d in bars]
        colors = [GREEN if v >= 0 else RED for v in vals]
        descs = [d.get("description") or d.get("direction", "") for d in bars]

        hover = [
            f"<b>{lbl}</b><br>Contribution: {v:+.2f}"
            + (f"<br>{desc}" if desc else "")
            for lbl, v, desc in zip(labels, vals, descs)
        ]

        fig.add_trace(
            go.Bar(
                x=vals,
                y=labels,
                orientation="h",
                marker=dict(color=colors, line=dict(width=0)),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor=BORDER,
            zerolinecolor=BORDER,
            zerolinewidth=1,
            tickfont=dict(color=MUTED, size=10),
            row=2,
            col=1,
        )
        fig.update_yaxes(
            tickfont=dict(color=TEXT, size=11),
            automargin=True,
            row=2,
            col=1,
        )
    else:
        fig.add_annotation(
            text="No driver data",
            xref="x2",
            yref="y2",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=MUTED, size=11),
            row=2,
            col=1,
        )
        fig.update_xaxes(visible=False, row=2, col=1)
        fig.update_yaxes(visible=False, row=2, col=1)

    fig.update_layout(
        **_DARK_LAYOUT,
        height=520,
        showlegend=False,
    )
    # Sub-plot title styling
    for ann in fig.layout.annotations:
        if ann.text == "Driver contributions":
            ann.font = dict(color=MUTED, size=11)
    return fig_to_html(fig)


# ── Chart: sector rotation ───────────────────────────────────────────────────

def _sector_score(d: dict) -> float:
    """Return the canonical rank score (prefers rank_score, falls back to score)."""
    v = d.get("rank_score", d.get("score"))
    return float(v) if v is not None else 0.0


def chart_sector_rotation(sector_data: list[dict]) -> str:
    """
    Horizontal bar chart of sectors ranked by rank_score (or score).

    Each dict can have:
        symbol/name     str   — sector identifier / label
        rank_score      float — composite RS score (preferred)
        score           float — fallback if rank_score absent
        rs_3m, rs_6m    float — surfaced in hover
        ytd_return      float — surfaced in hover
        rank_delta      int   — rendered as ▲n / ▼n / = next to bar

    Returns:
        Plotly HTML div string.
    """
    if not sector_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No sector data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=MUTED, size=12),
        )
        fig.update_layout(**_DARK_LAYOUT, height=180,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig_to_html(fig)

    # Sort ascending so highest score is at TOP in horizontal layout
    data = sorted(sector_data, key=_sector_score)
    n = len(data)

    symbols = [d.get("symbol", "") for d in data]
    names = [d.get("name", d.get("symbol", "?")) for d in data]
    labels = [
        f"{s} · {nm}" if s and nm and s != nm else (nm or s or "?")
        for s, nm in zip(symbols, names)
    ]
    scores = [_sector_score(d) for d in data]
    deltas = [int(d.get("rank_delta", 0) or 0) for d in data]
    colors = [GREEN if s >= 0 else RED for s in scores]

    def _fmt_opt(v, suffix=""):
        if v is None:
            return "—"
        try:
            return f"{float(v):+.2f}{suffix}"
        except (TypeError, ValueError):
            return str(v)

    hover = []
    for d, lbl, sc in zip(data, labels, scores):
        rs3 = _fmt_opt(d.get("rs_3m"))
        rs6 = _fmt_opt(d.get("rs_6m"))
        ytd = _fmt_opt(d.get("ytd_return"))
        rd = d.get("rank_delta")
        rd_str = (f"+{rd}" if rd and rd > 0 else (str(rd) if rd else "0"))
        hover.append(
            f"<b>{lbl}</b><br>"
            f"Rank score: {sc:+.2f}<br>"
            f"RS 3m: {rs3}<br>"
            f"RS 6m: {rs6}<br>"
            f"YTD return: {ytd}<br>"
            f"Rank Δ: {rd_str}"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
            showlegend=False,
        )
    )

    # Rank-delta annotations at bar end
    if scores:
        rng = (max(scores) - min(scores)) or 1.0
        offset = rng * 0.03
        for i, (sc, delta) in enumerate(zip(scores, deltas)):
            if delta > 0:
                txt, col = f"▲{delta}", GREEN
            elif delta < 0:
                txt, col = f"▼{abs(delta)}", RED
            else:
                txt, col = "=", MUTED
            x_pos = sc + (offset if sc >= 0 else -offset)
            anchor = "left" if sc >= 0 else "right"
            fig.add_annotation(
                x=x_pos,
                y=labels[i],
                text=txt,
                showarrow=False,
                font=dict(color=col, size=11),
                xanchor=anchor,
                yanchor="middle",
            )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=BORDER,
        zeroline=True,
        zerolinecolor=BORDER,
        zerolinewidth=1,
        tickfont=dict(color=MUTED, size=10),
    )
    fig.update_yaxes(
        tickfont=dict(color=TEXT, size=11),
        automargin=True,
    )
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(
            text="Sector Rotation — Ranked by Relative Strength",
            font=dict(color=MUTED, size=12),
            x=0.02,
        ),
        height=max(220, n * 34 + 80),
        bargap=0.35,
    )
    return fig_to_html(fig)


# ── Chart: theme heatmap ─────────────────────────────────────────────────────

def chart_theme_heatmap(ranked_themes: list[dict]) -> str:
    """
    Horizontal bars showing all themes by theme_score (z-scored).

    Colouring: top 5 green, middle yellow, bottom red.
    Hover surfaces description, all 6 components, and a truncated member list.

    Returns:
        Plotly HTML div string.
    """
    if not ranked_themes:
        fig = go.Figure()
        fig.add_annotation(
            text="No theme data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=MUTED, size=12),
        )
        fig.update_layout(**_DARK_LAYOUT, height=180,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig_to_html(fig)

    # Ascending sort so top scorers appear at TOP in horizontal layout
    data = sorted(ranked_themes, key=lambda d: float(d.get("theme_score", d.get("score", 0)) or 0))
    n = len(data)
    labels = [d.get("name", "?") for d in data]
    scores = [float(d.get("theme_score", d.get("score", 0)) or 0) for d in data]

    # Tier colours: top 5 green, bottom 5 red, middle yellow.
    # data is ascending, so the LAST 5 are the top tier.
    colors = [MUTED] * n
    top_n = min(5, n)
    bot_n = min(5, max(0, n - top_n))
    for i in range(n):
        if i >= n - top_n:
            colors[i] = GREEN
        elif i < bot_n:
            colors[i] = RED
        else:
            colors[i] = "#d8a657"  # yellow

    hover = []
    for d in data:
        desc = d.get("description", "")
        members = d.get("members") or d.get("available_members") or []
        member_str = ", ".join(str(m) for m in members[:5])
        if len(members) > 5:
            member_str += f", … (+{len(members) - 5})"

        components = d.get("components") or {}
        comp_lines = []
        # Show up to 6 components (the spec's "all 6 components")
        for k, v in list(components.items())[:6]:
            if isinstance(v, (int, float)):
                comp_lines.append(f"{k}: {v:+.2f}")
            else:
                comp_lines.append(f"{k}: {v}")
        comp_str = "<br>".join(comp_lines)

        parts = [f"<b>{d.get('name', '?')}</b>"]
        if desc:
            parts.append(f"<i>{desc}</i>")
        parts.append(f"Theme score: {float(d.get('theme_score', 0) or 0):+.2f}")
        if comp_str:
            parts.append(comp_str)
        if member_str:
            parts.append(f"Members: {member_str}")
        hover.append("<br>".join(parts))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
            showlegend=False,
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=BORDER,
        zeroline=True,
        zerolinecolor=BORDER,
        zerolinewidth=1,
        tickfont=dict(color=MUTED, size=10),
        title=dict(
            text="Theme score (z-scored)",
            font=dict(color=MUTED, size=10),
        ),
    )
    fig.update_yaxes(
        tickfont=dict(color=TEXT, size=11),
        automargin=True,
    )
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(
            text="Theme Heatmap — Sorted by Relative Strength Score",
            font=dict(color=MUTED, size=12),
            x=0.02,
        ),
        height=max(240, n * 32 + 100),
        bargap=0.35,
    )
    return fig_to_html(fig)


# ── Chart: 52-week range ─────────────────────────────────────────────────────

def chart_52w_range(candidates: list[dict]) -> str:
    """
    Horizontal range bars showing [lo52, hi52] for up to 15 candidates,
    with the current price drawn as a marker dot on the range.

    Each dict needs: ticker, lo52, hi52, price; optional: name.

    Returns:
        Plotly HTML div string.
    """
    items = [
        c for c in (candidates or [])[:15]
        if c.get("lo52") and c.get("hi52") and c.get("price")
    ]
    if not items:
        fig = go.Figure()
        fig.add_annotation(
            text="No range data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=MUTED, size=12),
        )
        fig.update_layout(**_DARK_LAYOUT, height=180,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig_to_html(fig)

    # Reverse so first candidate is at TOP in horizontal layout
    items_ord = list(reversed(items))
    tickers = [c["ticker"] for c in items_ord]
    lo = [float(c["lo52"]) for c in items_ord]
    hi = [float(c["hi52"]) for c in items_ord]
    price = [float(c["price"]) for c in items_ord]
    names = [c.get("name", "") for c in items_ord]

    range_hover = []
    for c, t, l, h, p, nm in zip(items_ord, tickers, lo, hi, price, names):
        span = (h - l) if h != l else 1.0
        from_hi = (p - h) / h * 100 if h else 0.0
        from_lo = (p - l) / l * 100 if l else 0.0
        range_hover.append(
            f"<b>{t}</b>"
            + (f" — {nm}" if nm else "")
            + f"<br>Price: ${p:,.2f}"
            + f"<br>52w high: ${h:,.2f}"
            + f"<br>52w low:  ${l:,.2f}"
            + f"<br>From high: {from_hi:+.1f}%"
            + f"<br>From low:  {from_lo:+.1f}%"
        )

    fig = go.Figure()

    # Background bar: full 52w range (lo → hi)
    fig.add_trace(
        go.Bar(
            x=[h - l for h, l in zip(hi, lo)],
            base=lo,
            y=tickers,
            orientation="h",
            marker=dict(color=SURFACE, line=dict(color=BORDER, width=1)),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=range_hover,
            showlegend=False,
            name="52w range",
        )
    )
    # Filled portion: lo → current price (intensity of where we are in range)
    fig.add_trace(
        go.Bar(
            x=[p - l for p, l in zip(price, lo)],
            base=lo,
            y=tickers,
            orientation="h",
            marker=dict(color=BLUE, opacity=0.55, line=dict(width=0)),
            hoverinfo="skip",
            showlegend=False,
            name="lo→price",
        )
    )

    # Current price marker dots
    fig.add_trace(
        go.Scatter(
            x=price,
            y=tickers,
            mode="markers",
            marker=dict(
                color=BLUE_LT,
                size=11,
                line=dict(color=TEXT, width=1.2),
                symbol="circle",
            ),
            customdata=range_hover,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
            name="current price",
        )
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=BORDER,
        tickfont=dict(color=MUTED, size=10),
        tickprefix="$",
    )
    fig.update_yaxes(
        tickfont=dict(color=TEXT, size=11),
        automargin=True,
    )
    fig.update_layout(
        **_DARK_LAYOUT,
        title=dict(
            text="52-week range  ·  ● = current price",
            font=dict(color=MUTED, size=12),
            x=0.02,
        ),
        barmode="overlay",
        height=max(220, len(items_ord) * 34 + 80),
        bargap=0.35,
    )
    return fig_to_html(fig)


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _chart(html_div: str) -> str:
    """Wrap a Plotly chart HTML div in a styled, dark container.

    Returns '' if the chart string is empty.
    """
    if not html_div:
        return ""
    return f'<div class="chart-container">{html_div}</div>'


def _badge(label: str, bg: str, fg: str) -> str:
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:20px;font-size:11px;font-weight:700">{label}</span>'
    )


def _pct_color(v, *, invert: bool = False) -> str:
    """Return green/red color string for a percentage value."""
    if v is None:
        return MUTED
    positive = v > 0
    if invert:
        positive = not positive
    return GREEN if positive else RED


def _fmt_price(v) -> str:
    return f"${v:,.2f}" if v else "—"


def _fmt_pct(v, plus: bool = True) -> str:
    if v is None:
        return "—"
    return f"{'+'if v>0 and plus else ''}{v:.1f}%"


def _fmt_mcap(v) -> str:
    if not v:
        return "—"
    if v >= 1e12:
        return f"${v/1e12:.1f}T"
    if v >= 1e9:
        return f"${v/1e9:.1f}B"
    return f"${v/1e6:.0f}M"


# ── Shared CSS ────────────────────────────────────────────────────────────────

def _css() -> str:
    return f"""
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",
              Arial,sans-serif;
  background:{BG};color:{TEXT};line-height:1.55;font-size:14px
}}
.wrap{{max-width:1100px;margin:0 auto;padding:0 16px 56px}}
.sec{{margin-bottom:36px}}
.sec-title{{
  font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
  color:{MUTED};border-bottom:1px solid {BORDER};
  padding-bottom:9px;margin-bottom:18px
}}
.card{{
  background:{SURFACE};border:1px solid {BORDER};
  border-radius:8px;padding:16px;margin-bottom:12px
}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{
  color:{MUTED};font-weight:600;font-size:10px;text-transform:uppercase;
  letter-spacing:.06em;padding:7px 8px;border-bottom:1px solid {BORDER};
  text-align:left
}}
td{{padding:8px;border-bottom:1px solid {SURFACE};vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:#21262d}}
.badge{{
  display:inline-block;padding:2px 8px;border-radius:20px;
  font-size:11px;font-weight:700
}}
details.explainer{{
  background:{BG};border:1px solid {BORDER};border-radius:6px;
  padding:8px 12px;margin:8px 0 14px;font-size:12px
}}
details.explainer summary{{
  cursor:pointer;color:{BLUE_LT};font-weight:600;list-style:none;
  outline:none
}}
details.explainer summary::-webkit-details-marker{{display:none}}
details.explainer p{{
  margin-top:8px;color:{MUTED};line-height:1.6
}}
.grid-4{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
.stat-box{{background:{BG};padding:10px;border-radius:6px;font-size:12px}}
.stat-label{{color:{MUTED};margin-bottom:3px;font-size:11px}}
.stat-val{{font-weight:700;color:{TEXT}}}
.held-badge{{
  font-size:10px;font-weight:700;color:{AMBER};
  background:#3d2800;padding:1px 6px;border-radius:10px;
  margin-left:5px
}}
.pulse-strip{{
  display:flex;flex-wrap:wrap;background:{SURFACE};
  border:1px solid {BORDER};border-radius:8px;overflow:hidden
}}
.pulse-cell{{
  flex:1 1 120px;text-align:center;padding:14px 20px;
  border-right:1px solid {BORDER}
}}
.pulse-cell:last-child{{border-right:none}}
.glossary dl{{display:block}}
.glossary-term{{
  background:{SURFACE};border:1px solid {BORDER};border-radius:6px;
  padding:12px 14px;margin-bottom:8px
}}
.glossary-term dt{{color:{TEXT};font-weight:700;margin-bottom:4px}}
.glossary-term dd{{color:{MUTED};font-size:13px;line-height:1.6;margin-left:0}}
.chart-container{{
  background:{BG};border:1px solid {BORDER};border-radius:8px;
  padding:12px;margin:16px 0;overflow:hidden
}}
.chart-container .js-plotly-plot{{width:100% !important}}
@media(max-width:600px){{
  .grid-4{{grid-template-columns:repeat(2,1fr)}}
  .wrap{{padding:0 10px 32px}}
  .chart-container{{padding:6px}}
}}
"""


def _html_shell(title: str, body: str, report_date: date = None) -> str:
    """Wrap body content in a complete HTML document."""
    d = (report_date or date.today()).strftime("%A, %B %d, %Y")
    now_str = datetime.now().strftime("%H:%M ET")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<script src="{PLOTLY_CDN}" charset="utf-8"></script>
<style>{_css()}</style>
</head>
<body><div class="wrap">
{body}
<div style="border-top:1px solid {BORDER};padding-top:16px;
            font-size:11px;color:#6e7681;text-align:center;margin-top:32px">
  Not financial advice &middot; data via yfinance / Yahoo Finance
  &middot; generated {now_str}
</div>
</div></body></html>"""


def _header(label: str, date_str: str, subtitle: str = "") -> str:
    sub = f'<div style="font-size:12px;color:rgba(255,255,255,.5);margin-top:4px">{subtitle}</div>' if subtitle else ""
    return f"""
<div style="background:#0c2d6b;border-radius:10px;
            padding:28px 28px 22px;margin:24px 0 28px">
  <div style="font-size:10px;font-weight:700;letter-spacing:.18em;
              text-transform:uppercase;color:rgba(255,255,255,.5);
              margin-bottom:6px">{label}</div>
  <div style="font-size:26px;font-weight:700;color:#fff">{date_str}</div>
  {sub}
</div>"""


def _pulse_strip(state_score: dict) -> str:
    """Render the market pulse / breadth strip from state_score data."""
    cells = ""
    indicators = [
        ("S&P 500",  state_score.get("sp500_val"),   state_score.get("sp500_pct"),   False),
        ("Nasdaq",   state_score.get("nasdaq_val"),  state_score.get("nasdaq_pct"),  False),
        ("VIX",      state_score.get("vix_val"),     state_score.get("vix_pct"),     True),
        ("10Y Yield",state_score.get("yield_val"),   state_score.get("yield_pct"),   True),
    ]
    for name, val, pct, invert in indicators:
        if val is None:
            continue
        positive = (pct or 0) > 0
        good = not positive if invert else positive
        color = GREEN if good else RED
        sign = "+" if (pct or 0) > 0 else ""
        fmt_val = f"{val:,.2f}" if val else "—"
        fmt_change = f"{sign}{pct:.2f}%" if pct is not None else "—"
        cells += f"""
  <div class="pulse-cell">
    <div style="font-size:10px;color:{MUTED};text-transform:uppercase;
                letter-spacing:.07em;margin-bottom:4px">{name}</div>
    <div style="font-size:20px;font-weight:700;color:{TEXT}">{fmt_val}</div>
    <div style="font-size:13px;color:{color};font-weight:600">{fmt_change}</div>
  </div>"""

    score_val = state_score.get("score", state_score.get("total_score"))
    if score_val is not None:
        if score_val < 35:
            sc, sl = RED, "Risk-Off"
        elif score_val < 50:
            sc, sl = AMBER, "Caution"
        elif score_val < 70:
            sc, sl = "#e3b341", "Neutral"
        else:
            sc, sl = GREEN, "Risk-On"
        cells += f"""
  <div class="pulse-cell">
    <div style="font-size:10px;color:{MUTED};text-transform:uppercase;
                letter-spacing:.07em;margin-bottom:4px">Market State</div>
    <div style="font-size:20px;font-weight:700;color:{sc}">{score_val}</div>
    <div style="font-size:13px;color:{sc};font-weight:600">{sl}</div>
  </div>"""

    return f'<div class="pulse-strip">{cells}</div>' if cells else ""


def _breadth_strip(new_highs_lows: dict) -> str:
    """Render a small breadth indicator row."""
    nh = new_highs_lows.get("new_highs_52w", 0)
    nl = new_highs_lows.get("new_lows_52w", 0)
    pct50 = new_highs_lows.get("pct_above_50dma")
    pct200 = new_highs_lows.get("pct_above_200dma")

    items = []
    if nh or nl:
        ratio_color = GREEN if nh > nl else RED
        items.append(
            f'<span>52w highs <strong style="color:{GREEN}">{nh}</strong> '
            f'/ lows <strong style="color:{RED}">{nl}</strong></span>'
        )
    if pct50 is not None:
        c = GREEN if pct50 > 60 else (RED if pct50 < 40 else AMBER)
        items.append(
            f'<span>% &gt;50 DMA <strong style="color:{c}">{pct50:.0f}%</strong></span>'
        )
    if pct200 is not None:
        c = GREEN if pct200 > 60 else (RED if pct200 < 40 else AMBER)
        items.append(
            f'<span>% &gt;200 DMA <strong style="color:{c}">{pct200:.0f}%</strong></span>'
        )

    if not items:
        return ""
    inner = "  &nbsp;&middot;&nbsp;  ".join(items)
    return f"""
<div class="sec">
  <div class="sec-title">Market breadth</div>
  <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;
              padding:14px 18px;font-size:13px;color:{MUTED}">
    {inner}
  </div>
</div>"""


def _gap_movers_section(gap_movers: list[dict]) -> str:
    if not gap_movers:
        return ""
    rows = ""
    for g in gap_movers:
        pct = g.get("pct_change", 0)
        c = GREEN if pct >= 0 else RED
        news = g.get("news_title", "")
        news_td = (f'<td style="font-size:11px;color:{MUTED}">{news[:80]}</td>'
                   if news else "<td>—</td>")
        rows += f"""<tr>
  <td style="font-weight:700;color:{TEXT}">{g.get('ticker','?')}</td>
  <td style="color:{MUTED};font-size:12px">{g.get('name','')[:24]}</td>
  <td style="color:{c};font-weight:600">{_fmt_pct(pct)}</td>
  {news_td}
</tr>"""
    return f"""
<div class="sec">
  <div class="sec-title">Gap movers today</div>
  <div style="background:{SURFACE};border:1px solid {BORDER};
              border-radius:8px;overflow:hidden">
  <table>
    <thead><tr>
      <th>Ticker</th><th>Name</th><th>Change</th><th>Headline</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</div>"""


def _new_candidates_section(new_candidates: list[dict]) -> str:
    if not new_candidates:
        return ""
    cards = ""
    for c in new_candidates:
        ticker = c.get("ticker", "?")
        name = c.get("name", "")
        price = _fmt_price(c.get("price"))
        reason = c.get("reason", "")
        screen = c.get("screen", "")
        cards += f"""
<div class="card" style="display:flex;justify-content:space-between;
                          align-items:center;gap:10px;margin-bottom:8px">
  <div>
    <span style="font-weight:700;font-size:16px;color:{TEXT}">{ticker}</span>
    <span style="font-size:12px;color:{MUTED};margin-left:8px">{name}</span>
    {f'<span style="font-size:11px;color:{BLUE_LT};margin-left:6px">{screen}</span>' if screen else ''}
  </div>
  <div style="text-align:right;font-size:13px">
    <div style="font-weight:700;color:{TEXT}">{price}</div>
    <div style="color:{MUTED};font-size:11px">{reason[:60]}</div>
  </div>
</div>"""
    return f"""
<div class="sec">
  <div class="sec-title">New candidates since yesterday</div>
  {cards}
</div>"""


def _earnings_reactions_section(earnings_reactions: list[dict]) -> str:
    if not earnings_reactions:
        return ""
    rows = ""
    for e in earnings_reactions:
        gap = e.get("gap_pct", 0)
        c = GREEN if gap >= 0 else RED
        beat = e.get("beat_miss", "")
        bm_c = GREEN if "beat" in beat.lower() else RED if "miss" in beat.lower() else MUTED
        rows += f"""<tr>
  <td style="font-weight:700;color:{TEXT}">{e.get('ticker','?')}</td>
  <td style="color:{bm_c};font-weight:600">{beat}</td>
  <td style="color:{c};font-weight:600">{_fmt_pct(gap)}</td>
  <td>{_fmt_pct(e.get('revenue_growth'))}</td>
</tr>"""
    return f"""
<div class="sec">
  <div class="sec-title">Recent earnings reactions</div>
  <div style="background:{SURFACE};border:1px solid {BORDER};
              border-radius:8px;overflow:hidden">
  <table>
    <thead><tr>
      <th>Ticker</th><th>Beat/Miss</th><th>Gap</th><th>Rev growth</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
</div>"""


# ── Daily report ─────────────────────────────────────────────────────────────

def build_daily_report(
    state_score: dict,
    sector_rotation: list[dict],
    new_highs_lows: dict,
    gap_movers: list[dict],
    new_candidates: list[dict],
    earnings_reactions: list[dict],
    report_date: date = None,
) -> str:
    """
    Generates a compact daily report (~1 page):
      - Header with date + Market State gauge
      - Market pulse strip
      - Breadth (% above 50/200 DMA, new highs/lows)
      - Sector moves chart
      - Gap movers (if any)
      - New candidates (if any)
      - Earnings reactions (if any)

    Returns:
        Complete HTML document string.
    """
    rd = report_date or date.today()
    date_str = rd.strftime("%A, %B %d, %Y")

    score_val = state_score.get("score", state_score.get("total_score", 0)) or 0
    drivers = state_score.get("drivers", [])

    gauge_div = chart_market_state(int(score_val), drivers)
    sector_div = chart_sector_rotation(sector_rotation)

    body = _header("Daily Market Brief", date_str,
                   f"Market State Score: {score_val}/100")

    # Pulse strip
    pulse_html = _pulse_strip(state_score)
    if pulse_html:
        body += f"""
<div class="sec">
  <div class="sec-title">Market pulse</div>
  {pulse_html}
</div>"""

    # Gauge chart
    if gauge_div:
        body += f"""
<div class="sec">
  <div class="sec-title">Market state</div>
  {_chart(gauge_div)}
  {explainer_html("market_state_score", "Market State Score")}
</div>"""

    # Breadth
    body += _breadth_strip(new_highs_lows)

    # Sector rotation chart
    if sector_div:
        body += f"""
<div class="sec">
  <div class="sec-title">Sector rotation</div>
  {_chart(sector_div)}
  {explainer_html("sector_rotation", "Sector Rotation")}
</div>"""

    # Gap movers
    body += _gap_movers_section(gap_movers)

    # New candidates
    body += _new_candidates_section(new_candidates)

    # Earnings reactions
    body += _earnings_reactions_section(earnings_reactions)

    return _html_shell(f"Daily Market Brief — {date_str}", body, rd)


# ── Weekly report ─────────────────────────────────────────────────────────────

def _theme_section(ranked_themes: list[dict], emerging_clusters: list[dict]) -> str:
    heatmap_div = chart_theme_heatmap(ranked_themes)

    # Top 5 themes with members
    top5 = ranked_themes[:5] if ranked_themes else []
    theme_cards = ""
    for t in top5:
        members = t.get("members", [])
        mem_str = ", ".join(str(m) for m in members[:8])
        score = t.get("theme_score", t.get("score", 0))
        c = GREEN if score > 0 else RED
        theme_cards += f"""
<div class="card" style="margin-bottom:8px">
  <div style="display:flex;justify-content:space-between;align-items:baseline">
    <span style="font-weight:700;font-size:15px;color:{TEXT}">{t.get('name','?')}</span>
    <span style="font-size:12px;color:{c};font-weight:600">
      score {score:+.2f}</span>
  </div>
  {f'<div style="font-size:12px;color:{MUTED};margin-top:5px">{mem_str}</div>' if mem_str else ''}
</div>"""

    # Emerging clusters
    cluster_html = ""
    if emerging_clusters:
        cluster_html = f'<div class="sec-title" style="margin-top:20px">Emerging clusters</div>'
        for cl in emerging_clusters:
            members = cl.get("members") or cl.get("tickers") or []
            tickers = ", ".join(str(m) for m in members[:6])
            label = cl.get("label") or cl.get("theme") or "Unnamed cluster"
            delta = cl.get("delta")
            note = cl.get("note")
            if delta is not None and not note:
                note = f"Internal correlation rose by {delta:+.2f} over the last 60 days"
            cluster_html += f"""
<div class="card" style="margin-bottom:8px">
  <span style="font-weight:700;color:{AMBER}">{label}</span>
  <span style="font-size:12px;color:{MUTED};margin-left:10px">{tickers}</span>
  {f'<div style="font-size:11px;color:#6e7681;margin-top:4px">{note}</div>' if note else ''}
</div>"""

    return f"""
<div class="sec">
  <div class="sec-title">Theme heatmap</div>
  {_chart(heatmap_div)}
  {explainer_html("theme_strength", "Theme Strength Score")}
  {theme_cards}
  {cluster_html}
</div>"""


def _sector_section(sector_rotation: list[dict]) -> str:
    sector_div = chart_sector_rotation(sector_rotation)
    rotation_call = ""
    for s in sector_rotation:
        if s.get("rotation_call"):
            rotation_call = s["rotation_call"]
            break
    stovall = ""
    for s in sector_rotation:
        if s.get("stovall_phase"):
            stovall = s["stovall_phase"]
            break

    extra = ""
    if rotation_call:
        extra += f'<div style="font-size:13px;color:{BLUE_LT};margin-bottom:6px">&#128260; {rotation_call}</div>'
    if stovall:
        extra += f'<div style="font-size:12px;color:{MUTED}">Stovall phase: <strong style="color:{TEXT}">{stovall}</strong></div>'

    return f"""
<div class="sec">
  <div class="sec-title">Sector rotation</div>
  {_chart(sector_div)}
  {extra}
  {explainer_html("sector_rotation", "Sector Rotation")}
</div>"""


def _candidate_table_row(c: dict, held_tickers: set) -> str:
    ticker = c.get("ticker", "?")
    is_held = ticker.upper() in held_tickers
    held_badge = '<span class="held-badge">HELD</span>' if is_held else ""
    name = c.get("name", "")
    price = _fmt_price(c.get("price"))
    rsi = c.get("rsi")
    rsi_c = RED if (rsi or 0) > 75 else (GREEN if (rsi or 0) < 35 else TEXT)
    # screens.py stores it as pct_from_52wh (negative value: 0 = at high, -10 = 10% below)
    # Older API used "from_hi" (positive value: 10 = 10% below)
    pct_from_52wh = c.get("pct_from_52wh")
    if pct_from_52wh is not None:
        from_hi_display = abs(pct_from_52wh)  # convert to positive for display
    else:
        from_hi_display = c.get("from_hi")
    fh_c = GREEN if from_hi_display and 10 <= from_hi_display <= 40 else AMBER
    rev = c.get("rev_growth")
    buy_pct = c.get("buy_pct")
    reason = c.get("reason", c.get("verdict", ""))
    rsi_display = f"{rsi:.0f}" if isinstance(rsi, (int, float)) else "—"
    buy_pct_display = f"{buy_pct:.0f}%" if isinstance(buy_pct, (int, float)) else "—"

    return f"""<tr>
  <td>
    <span style="font-weight:700;color:{TEXT}">{ticker}</span>{held_badge}
    <div style="font-size:11px;color:{MUTED}">{name[:22]}</div>
  </td>
  <td style="font-size:11px;color:{MUTED}">{reason[:60]}</td>
  <td style="font-weight:700">{price}</td>
  <td style="color:{rsi_c}">{rsi_display}</td>
  <td style="color:{fh_c}">{_fmt_pct(from_hi_display) if from_hi_display is not None else '—'}</td>
  <td>{_fmt_pct(rev) if rev else '—'}</td>
  <td style="color:{MUTED}">{buy_pct_display}</td>
</tr>"""


_SCREEN_ID_TO_EXPLAINER = {
    1: "52wH_proximity",
    2: "quality_pullback",
    3: "risk_adj_momentum",
    4: "quality_momentum",
    5: "pead",
    6: "analyst_revision",
    7: "insider_buys",
    8: "quality_oversold",
}


def _screen_section(
    screen_id,
    candidates: list[dict],
    screen_meta: dict,
    held_tickers: set,
    range_div: str,
) -> str:
    meta = screen_meta.get(screen_id, {})
    title = meta.get("name", str(screen_id).replace("_", " ").title())
    description = meta.get("description", "")
    citation = meta.get("citation", meta.get("evidence", ""))

    expl_key = _SCREEN_ID_TO_EXPLAINER.get(screen_id, str(screen_id))
    expl = explainer_html(expl_key, title)

    if not candidates:
        body = f'<div style="color:{MUTED};padding:12px">No candidates this week.</div>'
    else:
        rows = "".join(
            _candidate_table_row(c, held_tickers) for c in candidates
        )
        body = f"""
<div style="background:{SURFACE};border:1px solid {BORDER};
            border-radius:8px;overflow:hidden;margin-top:8px">
<table>
  <thead><tr>
    <th>Ticker</th><th>Reason</th><th>Price</th>
    <th>RSI</th><th>% off 52wH</th><th>Rev growth</th><th>Buy%</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table></div>"""

    citation_html = (
        f'<div style="font-size:11px;color:#6e7681;margin-bottom:6px">'
        f'&#128218; {citation}</div>'
        if citation else ""
    )
    desc_html = (
        f'<div style="font-size:12px;color:{MUTED};margin-bottom:8px">'
        f'{description}</div>'
        if description else ""
    )

    return f"""
<div class="card" style="margin-bottom:18px">
  <div style="font-size:16px;font-weight:700;color:{TEXT};margin-bottom:6px">
    {title}</div>
  {desc_html}
  {citation_html}
  {expl}
  {body}
</div>"""


def _watchlist_section(watchlist: dict, report_date: date,
                        watch_prices: dict | None = None) -> str:
    if not watchlist:
        return ""
    today = report_date or date.today()
    watch_prices = watch_prices or {}
    cards = ""
    for ticker, cfg in watchlist.items():
        if isinstance(cfg, dict):
            target = cfg.get("buy_at")
            direction = cfg.get("direction", "below")
            note = cfg.get("note", "")
            price = watch_prices.get(ticker) or cfg.get("current_price")
        else:
            target = cfg
            direction = "below"
            note = ""
            price = watch_prices.get(ticker)

        target_str = _fmt_price(target)
        price_str = _fmt_price(price) if price else "—"

        if price and target:
            if direction == "below":
                pct_away = (price - target) / target * 100
                label = f"{pct_away:+.1f}% vs target"
                prox_c = GREEN if abs(pct_away) <= 5 else MUTED
            else:
                pct_away = (target - price) / price * 100
                label = f"{pct_away:.1f}% to go"
                prox_c = GREEN if pct_away <= 5 else MUTED
            prox_html = f'<span style="color:{prox_c};font-weight:600">{label}</span>'
        else:
            prox_html = ""

        cards += f"""
<div class="card" style="margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;align-items:baseline">
    <span style="font-size:17px;font-weight:700;color:{TEXT}">{ticker}</span>
    {prox_html}
  </div>
  <div style="margin-top:6px;font-size:13px;color:{MUTED}">
    Current <strong style="color:{TEXT}">{price_str}</strong>
    &nbsp;&middot;&nbsp;
    Target <strong style="color:{BLUE_LT}">{target_str}</strong>
    ({direction})
  </div>
  {f'<div style="margin-top:5px;font-size:12px;color:#6e7681">{note}</div>' if note else ''}
</div>"""

    return f"""
<div class="sec">
  <div class="sec-title">Watch list — entry targets</div>
  {cards}
</div>"""


def _pre_ipo_section(pre_ipo: list[dict]) -> str:
    if not pre_ipo:
        return ""
    cards = ""
    for item in pre_ipo:
        name = item.get("name", "?")
        note = item.get("note", "")
        expected = item.get("expected", "")
        cards += f"""
<div class="card" style="margin-bottom:8px">
  <div style="font-weight:700;color:{TEXT}">{name}
    {f'<span style="font-size:11px;color:{MUTED};margin-left:8px">{expected}</span>' if expected else ''}
  </div>
  {f'<div style="font-size:12px;color:{MUTED};margin-top:4px">{note}</div>' if note else ''}
</div>"""

    return f"""
<div class="sec">
  <div class="sec-title">Pre-IPO watch</div>
  {cards}
</div>"""


def build_weekly_report(
    state_score: dict,
    sector_rotation: list[dict],
    ranked_themes: list[dict],
    emerging_clusters: list[dict],
    screen_results: dict,
    screen_meta: dict,
    held_tickers: set,
    watchlist: dict,
    pre_ipo: list[dict],
    new_highs_lows: dict,
    report_date: date = None,
    watch_prices: dict | None = None,
) -> str:
    """
    Generates the full weekly report (~5 sections):
      1. Header + Market State (gauge + driver bar chart + regime label)
      2. Themes section (theme heatmap chart + top 5 themes + emerging clusters)
      3. Sector rotation (chart + rotation_call + Stovall phase)
      4. Candidate screens — per-screen card with explainer + table
      5. Watchlist — current price vs target + note from config
      6. Pre-IPO watch (static from config)
      7. Breadth detail (new highs/lows)

    Returns:
        Complete HTML document string.
    """
    rd = report_date or date.today()
    date_str = rd.strftime("%A, %B %d, %Y")
    score_val = state_score.get("score", state_score.get("total_score", 0)) or 0
    drivers = state_score.get("drivers", [])

    gauge_div = chart_market_state(int(score_val), drivers)

    # Collect all candidates for 52w range chart
    all_candidates: list[dict] = []
    for clist in screen_results.values():
        all_candidates.extend(clist)
    # De-duplicate by ticker
    seen_tickers: set = set()
    unique_candidates: list[dict] = []
    for c in all_candidates:
        t = c.get("ticker", "")
        if t not in seen_tickers:
            seen_tickers.add(t)
            unique_candidates.append(c)
    range_div = chart_52w_range(unique_candidates)

    # ── 1. Header ──────────────────────────────────────────────────────────
    body = _header("Weekly Market Brief", date_str,
                   f"Market State Score: {score_val}/100")

    # Pulse strip
    pulse_html = _pulse_strip(state_score)
    if pulse_html:
        body += f"""
<div class="sec">
  <div class="sec-title">Market pulse</div>
  {pulse_html}
</div>"""

    # Gauge
    if gauge_div:
        body += f"""
<div class="sec">
  <div class="sec-title">Market state</div>
  {_chart(gauge_div)}
  {explainer_html("market_state_score", "Market State Score")}
</div>"""

    # ── 2. Themes ──────────────────────────────────────────────────────────
    body += _theme_section(ranked_themes, emerging_clusters)

    # ── 3. Sector rotation ─────────────────────────────────────────────────
    body += _sector_section(sector_rotation)

    # ── 4. Screens ─────────────────────────────────────────────────────────
    if screen_results:
        screens_inner = ""
        for screen_id, candidates in screen_results.items():
            screens_inner += _screen_section(
                screen_id, candidates, screen_meta, held_tickers, range_div
            )
        body += f"""
<div class="sec">
  <div class="sec-title">Candidate screens</div>
  {screens_inner}
</div>"""

    # 52w range chart (after screens, referencing all candidates)
    if range_div:
        body += f"""
<div class="sec">
  <div class="sec-title">52-week range — all candidates</div>
  {_chart(range_div)}
</div>"""

    # ── 5. Watchlist ───────────────────────────────────────────────────────
    body += _watchlist_section(watchlist, rd, watch_prices)

    # ── 6. Pre-IPO ─────────────────────────────────────────────────────────
    body += _pre_ipo_section(pre_ipo)

    # ── 7. Breadth ─────────────────────────────────────────────────────────
    body += _breadth_strip(new_highs_lows)

    # Glossary at the bottom
    body += glossary_html()

    return _html_shell(f"Weekly Market Brief — {date_str}", body, rd)
