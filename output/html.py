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

# ── Colour palette (editorial / research-note aesthetic) ───────────────────
# Ink black background, paper-cream typography, single amber accent.
# Sharp signal colors only used on data — chrome stays monochrome.
BG       = "#0a0b0d"       # near-pure ink
SURFACE  = "#111316"       # subtle layer
SURFACE2 = "#16191d"       # second-level surface (tables, nav)
BORDER   = "#21252b"       # hairlines
BORDER_2 = "#2c3037"       # stronger hairlines (section dividers)
TEXT     = "#e8e2d4"       # paper cream (warm, not pure white)
TEXT_DIM = "#c4bda9"       # softened text
MUTED    = "#7a7669"       # supporting copy
FAINT    = "#4a4842"       # captions / metadata

# Signal colors — used SPARSELY for data emphasis only
AMBER    = "#c9a449"       # primary accent (gold/amber)
AMBER_DK = "#8a701f"       # darker amber for backgrounds
GREEN    = "#7ea668"       # earthy green (positive)
RED      = "#b4544a"       # earthy red (negative)
BLUE_LT  = "#7a93a8"       # cool grey-blue (links)

# Legacy aliases (some functions still reference these)
BLUE     = BLUE_LT

# ── Plotly helpers ───────────────────────────────────────────────────────────

_DARK_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,                # no chart-bg fill — keep it flat
    font=dict(
        family='"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace',
        color=TEXT_DIM,
        size=11,
    ),
    margin=dict(l=12, r=12, t=36, b=28),
    hoverlabel=dict(
        bgcolor=SURFACE2,
        bordercolor=AMBER_DK,
        font=dict(family='"JetBrains Mono", monospace', color=TEXT, size=12),
    ),
)

# Axis defaults applied via update_xaxes / update_yaxes — avoids
# kwarg collisions when callers also pass xaxis/yaxis to update_layout.
_AXIS_DEFAULTS = dict(
    gridcolor=BORDER,
    zerolinecolor=BORDER,
    linecolor=BORDER,
    tickfont=dict(
        family='"JetBrains Mono", monospace', size=10, color=MUTED
    ),
)


def _apply_axis_theme(fig):
    """Apply the dark axis theme to every axis on the figure."""
    try:
        fig.update_xaxes(**_AXIS_DEFAULTS)
        fig.update_yaxes(**_AXIS_DEFAULTS)
    except Exception:
        pass
    return fig


def fig_to_html(fig) -> str:
    """Convert a Plotly Figure to a self-contained HTML div.

    The plotly.js library is *not* embedded — it is loaded once via a
    <script src=...> tag at the top of the report.
    """
    _apply_axis_theme(fig)
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


# ── Shared CSS — editorial research-note aesthetic ────────────────────────────

# Fonts loaded from Google Fonts:
#   Newsreader (transitional serif, editorial display) — body + display
#   JetBrains Mono (data, numbers, ticker symbols)
#   Inter Tight (UI chrome, small labels — used sparingly)
FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;'
    '0,6..72,700;0,6..72,800;1,6..72,400&'
    'family=JetBrains+Mono:wght@400;500;600;700&'
    'family=Inter+Tight:wght@400;500;600;700&'
    'display=swap" rel="stylesheet">'
)


def _css() -> str:
    return f"""
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html{{scroll-behavior:smooth;scroll-padding-top:88px}}
body{{
  font-family:"Newsreader","Iowan Old Style","Apple Garamond",Georgia,serif;
  background:{BG};color:{TEXT};
  font-size:16px;line-height:1.55;font-weight:400;
  font-feature-settings:"kern" 1,"liga" 1,"calt" 1,"onum" 1,"pnum" 1;
  -webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;
  background-image:radial-gradient(at 20% 0%,rgba(201,164,73,0.025) 0%,transparent 50%),
                   radial-gradient(at 80% 100%,rgba(122,147,168,0.02) 0%,transparent 50%);
  background-attachment:fixed;
}}

/* ── Typography rhythm ─────────────────────────────────────── */
.mono,th[data-sort],td.num,td.tk,.pill-num{{
  font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums;
}}
.ui{{font-family:"Inter Tight",-apple-system,system-ui,sans-serif}}

/* ── Layout ────────────────────────────────────────────────── */
.wrap{{max-width:1180px;margin:0 auto;padding:0 32px 88px}}
@media(max-width:720px){{.wrap{{padding:0 16px 56px}}}}

/* ── Top nav: sticky scrollspy ─────────────────────────────── */
.topnav{{
  position:sticky;top:0;z-index:50;
  background:rgba(10,11,13,0.82);
  backdrop-filter:saturate(180%) blur(20px);
  -webkit-backdrop-filter:saturate(180%) blur(20px);
  border-bottom:1px solid {BORDER};
  margin:0 -32px 0;padding:0 32px;
}}
@media(max-width:720px){{.topnav{{margin:0 -16px;padding:0 16px}}}}
.topnav-inner{{
  max-width:1180px;margin:0 auto;
  display:flex;align-items:center;gap:24px;height:56px;
  overflow-x:auto;scrollbar-width:none;
}}
.topnav-inner::-webkit-scrollbar{{display:none}}
.topnav-brand{{
  font-family:"Newsreader",Georgia,serif;font-weight:600;font-size:15px;
  color:{TEXT};letter-spacing:-0.01em;white-space:nowrap;
  display:flex;align-items:center;gap:10px;
}}
.topnav-brand::before{{
  content:"";width:7px;height:7px;border-radius:50%;
  background:{AMBER};box-shadow:0 0 8px rgba(201,164,73,0.6);
  animation:pulse 2.4s ease-in-out infinite;
}}
@keyframes pulse{{
  0%,100%{{opacity:1;transform:scale(1)}}
  50%{{opacity:.55;transform:scale(.78)}}
}}
.topnav-links{{
  display:flex;align-items:center;gap:2px;flex:1;
}}
.topnav-link{{
  font-family:"Inter Tight",sans-serif;
  font-size:11.5px;font-weight:500;letter-spacing:0.04em;text-transform:uppercase;
  color:{MUTED};text-decoration:none;padding:8px 12px;border-radius:0;
  position:relative;transition:color .15s ease;white-space:nowrap;
}}
.topnav-link:hover{{color:{TEXT_DIM}}}
.topnav-link.active{{color:{AMBER}}}
.topnav-link.active::after{{
  content:"";position:absolute;left:12px;right:12px;bottom:-1px;height:1px;
  background:{AMBER};
}}
.topnav-meta{{
  font-family:"JetBrains Mono",monospace;font-size:11px;
  color:{FAINT};white-space:nowrap;letter-spacing:-0.01em;
}}

/* ── Masthead / hero ───────────────────────────────────────── */
.masthead{{
  padding:56px 0 40px;border-bottom:1px solid {BORDER_2};margin-bottom:48px;
  position:relative;
}}
.masthead-kicker{{
  font-family:"Inter Tight",sans-serif;
  font-size:11px;font-weight:600;letter-spacing:0.22em;text-transform:uppercase;
  color:{AMBER};margin-bottom:18px;
}}
.masthead-title{{
  font-family:"Newsreader",serif;font-weight:500;
  font-size:clamp(38px,5.5vw,68px);line-height:1.02;letter-spacing:-0.025em;
  color:{TEXT};margin-bottom:18px;font-variation-settings:"opsz" 72;
}}
.masthead-title em{{
  font-style:italic;font-weight:500;color:{TEXT_DIM};
  font-feature-settings:"swsh" 1,"salt" 1;
}}
.masthead-sub{{
  font-family:"Newsreader",serif;font-style:italic;font-size:19px;color:{MUTED};
  margin-bottom:28px;max-width:680px;line-height:1.5;
}}
.masthead-strip{{
  display:grid;grid-template-columns:repeat(4,1fr);gap:0;
  border-top:1px solid {BORDER};border-bottom:1px solid {BORDER};
  margin-top:32px;
}}
@media(max-width:720px){{
  .masthead-strip{{grid-template-columns:repeat(2,1fr)}}
  .masthead-strip > *:nth-child(2){{border-right:none}}
}}
.masthead-stat{{
  padding:18px 22px;border-right:1px solid {BORDER};
}}
.masthead-stat:last-child{{border-right:none}}
.masthead-stat-label{{
  font-family:"Inter Tight",sans-serif;font-size:9.5px;font-weight:600;
  letter-spacing:0.18em;text-transform:uppercase;color:{FAINT};
  margin-bottom:8px;
}}
.masthead-stat-val{{
  font-family:"JetBrains Mono",monospace;font-size:24px;font-weight:500;
  color:{TEXT};letter-spacing:-0.02em;line-height:1;
  font-variant-numeric:tabular-nums;
}}
.masthead-stat-val.amber{{color:{AMBER}}}
.masthead-stat-val.green{{color:{GREEN}}}
.masthead-stat-val.red{{color:{RED}}}
.masthead-stat-sub{{
  font-family:"Inter Tight",sans-serif;font-size:11px;color:{MUTED};
  margin-top:5px;letter-spacing:0.01em;
}}

/* ── Sections ─────────────────────────────────────────────── */
.sec{{margin-bottom:64px;scroll-margin-top:88px}}
.sec-head{{
  display:flex;align-items:baseline;gap:18px;
  margin-bottom:24px;padding-bottom:14px;
  border-bottom:1px solid {BORDER_2};
}}
.sec-num{{
  font-family:"JetBrains Mono",monospace;font-size:11px;font-weight:500;
  color:{FAINT};letter-spacing:0;line-height:1;
}}
.sec-title{{
  font-family:"Newsreader",serif;font-weight:500;font-size:30px;
  letter-spacing:-0.02em;color:{TEXT};line-height:1.1;
  font-variation-settings:"opsz" 48;
}}
.sec-title em{{font-style:italic;color:{TEXT_DIM}}}
.sec-lede{{
  font-family:"Newsreader",serif;font-style:italic;font-size:16px;
  color:{MUTED};margin:-6px 0 22px;max-width:720px;line-height:1.6;
}}
.subsec-title{{
  font-family:"Inter Tight",sans-serif;font-size:11px;font-weight:600;
  letter-spacing:0.16em;text-transform:uppercase;color:{TEXT_DIM};
  margin:24px 0 12px;
}}

/* ── Cards (now hairline rules, not bubbles) ──────────────── */
.card{{
  background:transparent;border:none;
  border-top:1px solid {BORDER};
  padding:18px 0 6px;margin-bottom:0;
}}
.card-row{{
  display:grid;grid-template-columns:160px 1fr auto;gap:24px;
  align-items:baseline;padding:14px 0;
  border-top:1px solid {BORDER};
}}
.card-row:first-child{{border-top:none}}

/* ── Tables ───────────────────────────────────────────────── */
table{{
  width:100%;border-collapse:collapse;font-size:13px;
}}
thead{{border-bottom:1px solid {BORDER_2}}}
th{{
  font-family:"Inter Tight",sans-serif;
  color:{FAINT};font-weight:600;font-size:10px;text-transform:uppercase;
  letter-spacing:0.12em;padding:11px 10px 11px 0;text-align:left;
  position:relative;user-select:none;
}}
th.num,td.num{{text-align:right;padding-right:10px}}
th[data-sort]{{cursor:pointer;transition:color .12s ease}}
th[data-sort]:hover{{color:{TEXT_DIM}}}
th[data-sort]::after{{
  content:"⇅";color:{FAINT};font-size:9px;
  margin-left:6px;opacity:.5;
}}
th[data-sort].sort-asc::after{{content:"↑";color:{AMBER};opacity:1}}
th[data-sort].sort-desc::after{{content:"↓";color:{AMBER};opacity:1}}
td{{
  padding:11px 10px 11px 0;
  border-bottom:1px solid {BORDER};
  vertical-align:baseline;color:{TEXT};
}}
tbody tr:last-child td{{border-bottom:none}}
tbody tr{{transition:background .12s ease}}
tbody tr:hover{{background:rgba(201,164,73,0.04)}}
td.tk{{
  font-weight:600;letter-spacing:0;color:{TEXT};
}}
td.tk .ticker-symbol{{
  font-family:"JetBrains Mono",monospace;font-weight:600;
  font-size:13px;letter-spacing:0.01em;
}}
td .row-sub{{
  font-family:"Inter Tight",sans-serif;font-size:10.5px;color:{FAINT};
  margin-top:3px;letter-spacing:0.01em;
}}
td.reason{{
  font-family:"Newsreader",serif;font-style:italic;font-size:13px;
  color:{TEXT_DIM};line-height:1.45;
}}

/* ── Numeric color signal ────────────────────────────────── */
.pos{{color:{GREEN}}}
.neg{{color:{RED}}}
.muted{{color:{MUTED}}}
.faint{{color:{FAINT}}}
.amber{{color:{AMBER}}}

/* ── Ticker pill / held badge ────────────────────────────── */
.held-badge{{
  display:inline-block;font-family:"Inter Tight",sans-serif;
  font-size:9px;font-weight:700;color:{AMBER};
  background:rgba(201,164,73,0.10);
  border:1px solid rgba(201,164,73,0.32);
  padding:1px 5px;border-radius:2px;margin-left:6px;
  letter-spacing:0.08em;text-transform:uppercase;
  vertical-align:middle;line-height:1.2;
}}
.tag{{
  display:inline-block;font-family:"Inter Tight",sans-serif;
  font-size:10px;font-weight:600;color:{MUTED};
  background:{SURFACE2};border:1px solid {BORDER};
  padding:2px 8px;border-radius:0;letter-spacing:0.04em;
  text-transform:uppercase;
}}
.tag.amber{{color:{AMBER};border-color:rgba(201,164,73,0.32);
            background:rgba(201,164,73,0.08)}}

/* ── Explainers ─────────────────────────────────────────── */
details.explainer{{
  background:transparent;border:none;border-left:2px solid {AMBER_DK};
  padding:6px 0 6px 16px;margin:12px 0 18px;font-size:13.5px;
}}
details.explainer summary{{
  cursor:pointer;color:{AMBER};font-weight:500;list-style:none;
  outline:none;font-family:"Inter Tight",sans-serif;
  font-size:11px;letter-spacing:0.12em;text-transform:uppercase;
}}
details.explainer summary::-webkit-details-marker,
details.explainer summary::marker{{display:none}}
details.explainer summary::before{{
  content:"+";display:inline-block;width:14px;
  font-family:"JetBrains Mono",monospace;color:{AMBER};
  transition:transform .15s ease;
}}
details.explainer[open] summary::before{{content:"−"}}
details.explainer p{{
  margin-top:10px;color:{TEXT_DIM};line-height:1.65;
  font-family:"Newsreader",serif;font-size:14px;
  max-width:680px;
}}

/* ── Pulse / market state strip (used by daily) ──────────── */
.pulse-strip{{
  display:grid;grid-template-columns:repeat(4,1fr);gap:0;
  border-top:1px solid {BORDER};border-bottom:1px solid {BORDER};
  margin:18px 0 0;
}}
.pulse-cell{{
  padding:18px 20px;border-right:1px solid {BORDER};text-align:left;
}}
.pulse-cell:last-child{{border-right:none}}
@media(max-width:720px){{
  .pulse-strip{{grid-template-columns:repeat(2,1fr)}}
}}

/* ── Chart containers ────────────────────────────────────── */
.chart-container{{
  background:transparent;border:none;
  border-top:1px solid {BORDER};border-bottom:1px solid {BORDER};
  padding:24px 0;margin:24px 0;overflow:hidden;
}}
.chart-container .js-plotly-plot{{width:100% !important}}

/* ── Backtest panel ──────────────────────────────────────── */
details.backtest-panel{{
  background:transparent;border-top:1px solid {BORDER};
  padding:16px 0;margin:14px 0 0;
}}
details.backtest-panel summary{{
  cursor:pointer;list-style:none;outline:none;
  font-family:"Inter Tight",sans-serif;font-size:11px;font-weight:600;
  letter-spacing:0.14em;text-transform:uppercase;color:{TEXT_DIM};
  display:flex;align-items:center;gap:10px;
}}
details.backtest-panel summary::-webkit-details-marker,
details.backtest-panel summary::marker{{display:none}}
details.backtest-panel summary::before{{
  content:"▸";font-size:10px;color:{AMBER};
  transition:transform .15s ease;display:inline-block;
}}
details.backtest-panel[open] summary::before{{transform:rotate(90deg)}}
.backtest-grid{{
  display:grid;grid-template-columns:repeat(6,1fr);gap:24px;
  margin-top:14px;padding-top:14px;border-top:1px dashed {BORDER};
}}
@media(max-width:720px){{.backtest-grid{{grid-template-columns:repeat(3,1fr)}}}}
.backtest-cell-label{{
  font-family:"Inter Tight",sans-serif;font-size:9px;font-weight:600;
  letter-spacing:0.14em;text-transform:uppercase;color:{FAINT};
  margin-bottom:6px;
}}
.backtest-cell-val{{
  font-family:"JetBrains Mono",monospace;font-size:18px;font-weight:500;
  color:{TEXT};font-variant-numeric:tabular-nums;letter-spacing:-0.02em;
}}

/* ── Watchlist / pre-IPO / glossary cards ────────────────── */
.list-row{{
  display:grid;grid-template-columns:180px 1fr;gap:32px;
  padding:18px 0;border-top:1px solid {BORDER};align-items:baseline;
}}
.list-row:first-child{{border-top:none}}
.list-row-label{{
  font-family:"JetBrains Mono",monospace;font-size:14px;font-weight:600;
  color:{TEXT};letter-spacing:0.01em;
}}
.list-row-body p{{
  font-family:"Newsreader",serif;font-size:14.5px;color:{TEXT_DIM};
  line-height:1.55;
}}
.list-row-meta{{
  font-family:"Inter Tight",sans-serif;font-size:11px;color:{MUTED};
  margin-top:6px;letter-spacing:0.02em;
}}

.glossary dl{{display:block}}
.glossary-term{{
  display:grid;grid-template-columns:200px 1fr;gap:32px;
  padding:20px 0;border-top:1px solid {BORDER};
}}
@media(max-width:720px){{
  .glossary-term{{grid-template-columns:1fr;gap:8px}}
}}
.glossary-term:first-child{{border-top:1px solid {BORDER_2}}}
.glossary-term dt{{
  font-family:"Newsreader",serif;font-weight:500;font-size:18px;color:{TEXT};
  letter-spacing:-0.01em;line-height:1.3;
}}
.glossary-term dd{{
  font-family:"Newsreader",serif;color:{TEXT_DIM};
  font-size:14.5px;line-height:1.65;margin-left:0;max-width:680px;
}}

/* ── Footer ──────────────────────────────────────────────── */
.footer{{
  border-top:1px solid {BORDER_2};margin-top:80px;padding:32px 0 0;
  font-family:"Inter Tight",sans-serif;font-size:11px;color:{FAINT};
  letter-spacing:0.04em;display:flex;justify-content:space-between;
  align-items:center;flex-wrap:wrap;gap:12px;
}}
.footer-mark{{
  font-family:"Newsreader",serif;font-style:italic;font-size:13px;
  color:{MUTED};
}}

/* ── Mobile ──────────────────────────────────────────────── */
@media(max-width:720px){{
  body{{font-size:15px}}
  .sec-title{{font-size:24px}}
  .masthead{{padding:36px 0 24px;margin-bottom:32px}}
  .backtest-cell-val{{font-size:15px}}
  th,td{{font-size:12px;padding:8px 6px 8px 0}}
  .topnav-meta{{display:none}}
}}

/* ── Print (because real research notes get printed) ─────── */
@media print{{
  .topnav{{display:none}}
  body{{background:white;color:black}}
  .masthead-stat-val{{color:black !important}}
}}
"""


def _scrollspy_js() -> str:
    """Inline JavaScript for sortable tables + scrollspy navigation."""
    return r"""
(function(){
  'use strict';

  /* ── Sortable tables ──────────────────────────────────── */
  function parseSortValue(cell) {
    // Prefer data-sort-value if present, else parse text
    var v = cell.getAttribute('data-sort-value');
    if (v !== null) return parseFloat(v) || v;
    var t = (cell.innerText || cell.textContent || '').trim();
    // Strip currency, commas, percent, plus/minus
    var cleaned = t.replace(/[$,%]/g, '').replace(/—|—/g, '').trim();
    if (cleaned === '' || cleaned === '-' || cleaned === '–') return -Infinity;
    var n = parseFloat(cleaned);
    return isNaN(n) ? t.toLowerCase() : n;
  }
  function makeSortable(table) {
    var thead = table.querySelector('thead');
    if (!thead) return;
    var headers = thead.querySelectorAll('th[data-sort]');
    headers.forEach(function(th, colIdx) {
      th.addEventListener('click', function() {
        var tbody = table.querySelector('tbody');
        if (!tbody) return;
        var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
        var asc = !th.classList.contains('sort-asc');
        // Reset other headers
        headers.forEach(function(h){
          h.classList.remove('sort-asc','sort-desc');
        });
        th.classList.add(asc ? 'sort-asc' : 'sort-desc');
        // Find actual column index in row cells
        var idx = Array.prototype.indexOf.call(thead.querySelectorAll('th'), th);
        rows.sort(function(a, b){
          var av = parseSortValue(a.cells[idx]);
          var bv = parseSortValue(b.cells[idx]);
          if (av === bv) return 0;
          if (typeof av === 'number' && typeof bv === 'number') {
            return asc ? av - bv : bv - av;
          }
          return asc ? String(av).localeCompare(String(bv))
                     : String(bv).localeCompare(String(av));
        });
        rows.forEach(function(r){ tbody.appendChild(r); });
      });
    });
  }
  document.querySelectorAll('table.sortable').forEach(makeSortable);

  /* ── Scrollspy: highlight nav link of current section ─── */
  var sections = Array.prototype.slice.call(document.querySelectorAll('section[id], div.sec[id]'));
  var navLinks = Array.prototype.slice.call(document.querySelectorAll('.topnav-link'));
  if (sections.length && navLinks.length && 'IntersectionObserver' in window) {
    var byId = {};
    navLinks.forEach(function(l){
      var id = (l.getAttribute('href') || '').replace('#','');
      if (id) byId[id] = l;
    });
    var observer = new IntersectionObserver(function(entries) {
      // Find the entry closest to the top that's intersecting
      var visible = entries.filter(function(e){ return e.isIntersecting; });
      if (!visible.length) return;
      visible.sort(function(a,b){ return a.boundingClientRect.top - b.boundingClientRect.top; });
      var id = visible[0].target.id;
      var link = byId[id];
      if (link) {
        navLinks.forEach(function(l){ l.classList.remove('active'); });
        link.classList.add('active');
      }
    }, { rootMargin: '-80px 0px -65% 0px', threshold: 0 });
    sections.forEach(function(s){ observer.observe(s); });
  }

  /* ── Smooth scroll for in-page links (browser fallback) ── */
  document.querySelectorAll('a[href^="#"]').forEach(function(a){
    a.addEventListener('click', function(e){
      var href = a.getAttribute('href');
      if (href === '#' || href.length < 2) return;
      var el = document.querySelector(href);
      if (el) {
        e.preventDefault();
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        history.pushState(null, '', href);
      }
    });
  });
})();
"""


def _html_shell(title: str, body: str, report_date: date = None,
                 nav_links: list = None, nav_meta: str = None) -> str:
    """Wrap body content in a complete HTML document with masthead + sticky nav."""
    d = (report_date or date.today()).strftime("%A, %B %d, %Y")
    now_str = datetime.now().strftime("%H:%M ET")
    nav_links = nav_links or []
    nav_links_html = "".join(
        f'<a class="topnav-link" href="#{anchor}">{label}</a>'
        for anchor, label in nav_links
    )
    meta_html = (
        f'<span class="topnav-meta">{nav_meta}</span>' if nav_meta else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
{FONTS_LINK}
<script src="{PLOTLY_CDN}" charset="utf-8"></script>
<style>{_css()}</style>
</head>
<body>
<nav class="topnav">
  <div class="topnav-inner">
    <div class="topnav-brand">Market Brief</div>
    <div class="topnav-links">{nav_links_html}</div>
    {meta_html}
  </div>
</nav>
<div class="wrap">
{body}
<div class="footer">
  <div>
    <span class="footer-mark">An automated market intelligence brief.</span>
    &nbsp;Not financial advice. Data via yfinance, FRED, AAII, SEC EDGAR.
  </div>
  <div>Generated {now_str}</div>
</div>
</div>
<script>{_scrollspy_js()}</script>
</body></html>"""


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
    inner = "  &nbsp;·&nbsp;  ".join(items)
    return f"""
<section id="breadth" class="sec">
  {_section_head("08", "Market", "breadth", "Universe-level health check — new highs, new lows, percentage above key moving averages.")}
  <div class="mono" style="font-size:13px;color:{TEXT_DIM};padding:18px 0;
                            line-height:1.8;letter-spacing:0.01em">
    {inner}
  </div>
</section>"""


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

    # Top 5 themes with members — as a numbered list
    top5 = ranked_themes[:5] if ranked_themes else []
    theme_cards = ""
    for i, t in enumerate(top5, 1):
        members = t.get("members", [])
        mem_str = ", ".join(str(m) for m in members[:10])
        score = t.get("theme_score", t.get("score", 0))
        score_class = "pos" if score > 0 else "neg"
        theme_cards += f"""
<div class="list-row">
  <div>
    <div class="ui faint" style="font-size:9.5px;letter-spacing:0.18em;
                                    text-transform:uppercase;margin-bottom:6px">
      #{i:02d} &nbsp;·&nbsp; <span class="mono {score_class}">{score:+.2f}</span>
    </div>
    <div style="font-family:'Newsreader',serif;font-weight:500;font-size:18px;
                color:{TEXT};letter-spacing:-0.01em">
      {t.get('name','?')}
    </div>
  </div>
  <div class="list-row-body">
    <p class="mono" style="font-size:12px;color:{TEXT_DIM};line-height:1.7;letter-spacing:0.02em">
      {mem_str}
    </p>
  </div>
</div>"""

    # Emerging clusters
    cluster_html = ""
    if emerging_clusters:
        cluster_html = '<div class="subsec-title" style="margin-top:36px">Emerging clusters</div>'
        for cl in emerging_clusters:
            members = cl.get("members") or cl.get("tickers") or []
            tickers = ", ".join(str(m) for m in members[:8])
            label = cl.get("label") or cl.get("theme") or "Unnamed cluster"
            delta = cl.get("delta")
            note = cl.get("note")
            if delta is not None and not note:
                note = f"Internal correlation rose by {delta:+.2f} over the last 60 days"
            cluster_html += f"""
<div class="list-row">
  <div>
    <div class="ui amber" style="font-size:9.5px;letter-spacing:0.18em;
                                    text-transform:uppercase;margin-bottom:6px">Emerging</div>
    <div style="font-family:'Newsreader',serif;font-weight:500;font-size:17px;
                color:{AMBER};letter-spacing:-0.01em">
      {label}
    </div>
    {f'<div class="list-row-meta">{note}</div>' if note else ''}
  </div>
  <div class="list-row-body">
    <p class="mono" style="font-size:12px;color:{TEXT_DIM};line-height:1.7;letter-spacing:0.02em">
      {tickers}
    </p>
  </div>
</div>"""

    return f"""
<section id="themes" class="sec">
  {_section_head("02", "Thematic", "strength", "Sixteen curated baskets z-scored across six dimensions; emerging clusters auto-detected via Mantegna correlation distance.")}
  {_chart(heatmap_div)}
  {explainer_html("theme_strength", "Theme Strength Score")}
  <div class="subsec-title">Top five themes by composite score</div>
  {theme_cards}
  {cluster_html}
</section>"""


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
        extra += (
            f'<div style="font-family:\'Newsreader\',serif;font-style:italic;'
            f'font-size:16px;color:{AMBER};margin:14px 0 6px;line-height:1.4;'
            f'max-width:680px">↺ &nbsp;{rotation_call}</div>'
        )
    if stovall:
        extra += (
            f'<div class="ui" style="font-size:11px;color:{MUTED};'
            f'letter-spacing:0.04em">Inferred cycle phase: '
            f'<strong style="color:{TEXT}">{stovall}</strong></div>'
        )

    return f"""
<section id="sectors" class="sec">
  {_section_head("03", "Sector", "rotation", "Eleven SPDR sector ETFs ranked by Faber-style momentum: 0.6 × 3-month relative strength + 0.4 × 6-month.")}
  {_chart(sector_div)}
  {extra}
  {explainer_html("sector_rotation", "Sector Rotation")}
</section>"""


def _candidate_table_row(c: dict, held_tickers: set) -> str:
    ticker = c.get("ticker", "?")
    is_held = ticker.upper() in held_tickers
    held_badge = '<span class="held-badge">Held</span>' if is_held else ""
    name = c.get("name", "")
    price_val = c.get("price")
    price = _fmt_price(price_val)
    rsi = c.get("rsi")
    rsi_c = "neg" if (rsi or 0) > 75 else ("pos" if (rsi or 0) < 35 else "")
    pct_from_52wh = c.get("pct_from_52wh")
    if pct_from_52wh is not None:
        from_hi_display = abs(pct_from_52wh)
    else:
        from_hi_display = c.get("from_hi")
    fh_c = "pos" if from_hi_display and 10 <= from_hi_display <= 40 else "amber"
    rev = c.get("rev_growth")
    buy_pct = c.get("buy_pct")
    reason = c.get("reason", c.get("verdict", ""))
    rsi_display = f"{rsi:.0f}" if isinstance(rsi, (int, float)) else "—"
    buy_pct_display = f"{buy_pct:.0f}%" if isinstance(buy_pct, (int, float)) else "—"

    # data-sort-value for proper numeric sorting; cells render formatted text
    rsi_sort = f"{rsi:.2f}" if isinstance(rsi, (int, float)) else "-9999"
    fhi_sort = f"{from_hi_display:.2f}" if isinstance(from_hi_display, (int, float)) else "-9999"
    price_sort = f"{price_val:.4f}" if isinstance(price_val, (int, float)) else "-9999"
    rev_sort = f"{rev:.2f}" if isinstance(rev, (int, float)) else "-9999"
    buy_sort = f"{buy_pct:.2f}" if isinstance(buy_pct, (int, float)) else "-9999"
    rev_disp = _fmt_pct(rev) if rev else "—"
    fhi_disp = _fmt_pct(from_hi_display) if from_hi_display is not None else "—"

    name_sub = (
        f'<div class="row-sub">{name[:32]}</div>' if name else ""
    )
    return f"""<tr>
  <td class="tk" data-sort-value="{ticker}">
    <span class="ticker-symbol">{ticker}</span>{held_badge}{name_sub}
  </td>
  <td class="reason">{reason}</td>
  <td class="num" data-sort-value="{price_sort}">{price}</td>
  <td class="num {rsi_c}" data-sort-value="{rsi_sort}">{rsi_display}</td>
  <td class="num {fh_c}" data-sort-value="{fhi_sort}">{fhi_disp}</td>
  <td class="num pos" data-sort-value="{rev_sort}">{rev_disp}</td>
  <td class="num muted" data-sort-value="{buy_sort}">{buy_pct_display}</td>
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


def _fmt_signed_pct(v) -> str:
    """Format a percentage with sign, e.g. +8.2% or -1.4%. '—' if None."""
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{f:+.1f}%"


def _fmt_signed_pp(v) -> str:
    """Format alpha (percentage points) with sign, e.g. +4.1pp."""
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    return f"{f:+.1f}pp"


def _backtest_panel(screen_id, backtest_data: dict | None) -> str:
    """
    Render an inline collapsible block summarizing the 5-year walk-forward
    backtest for one screen. If `backtest_data` is None or empty, returns
    an empty string. If it indicates `skipped`, renders an explanatory note.

    Reads these exact keys (per the field-name contract):
      name, n_observations, n_unique_tickers,
      median_fwd_1m/3m/6m, spy_baseline_1m/3m/6m,
      alpha_3m, alpha_6m, hit_rate_3m, sharpe_3m,
      lookback_weeks, as_of, skipped, reason.
    """
    if not backtest_data:
        return ""

    if backtest_data.get("skipped"):
        reason = backtest_data.get("reason", "")
        reason_html = (
            f' — <em>{reason}</em>' if reason else ""
        )
        return f"""
<details class="backtest-panel">
  <summary>Historical performance · 5y walk-forward</summary>
  <div style="margin-top:14px;padding-top:14px;border-top:1px dashed {BORDER};
              color:{MUTED};font-size:13.5px;line-height:1.6;
              font-family:'Newsreader',serif;font-style:italic;max-width:680px">
    Historical backtest not available for this screen — requires fundamentals
    snapshots or non-price data.{reason_html}
  </div>
</details>"""

    med_3m = backtest_data.get("median_fwd_3m")
    spy_3m = backtest_data.get("spy_baseline_3m")
    alpha_3m = backtest_data.get("alpha_3m")
    hit_rate = backtest_data.get("hit_rate_3m")
    sharpe = backtest_data.get("sharpe_3m")
    n_obs = backtest_data.get("n_observations")
    n_uniq = backtest_data.get("n_unique_tickers")
    lookback_weeks = backtest_data.get("lookback_weeks")
    as_of = backtest_data.get("as_of", "")

    # No-data case (zero observations)
    if not n_obs:
        return f"""
<details class="backtest-panel">
  <summary>Historical performance · 5y walk-forward</summary>
  <div style="margin-top:14px;padding-top:14px;border-top:1px dashed {BORDER};
              color:{MUTED};font-size:13.5px;
              font-family:'Newsreader',serif;font-style:italic;max-width:680px">
    Backtest produced no observations over the lookback window — insufficient
    data or no signals fired.
  </div>
</details>"""

    med_color = "pos" if (med_3m or 0) >= 0 else "neg"
    alpha_color = "pos" if (alpha_3m or 0) >= 0 else "neg"
    sharpe_str = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "—"
    hit_color = "pos" if (hit_rate or 0) >= 50 else "amber"
    hit_str = f"{hit_rate:.0f}%" if isinstance(hit_rate, (int, float)) else "—"

    lookback_label = (
        f"{lookback_weeks // 52}y walk-forward"
        if isinstance(lookback_weeks, int) and lookback_weeks >= 52
        else "Walk-forward"
    )

    return f"""
<details class="backtest-panel">
  <summary>Historical performance · {lookback_label}</summary>
  <div class="backtest-grid">
    <div>
      <div class="backtest-cell-label">Median 3m fwd</div>
      <div class="backtest-cell-val {med_color}">{_fmt_signed_pct(med_3m)}</div>
    </div>
    <div>
      <div class="backtest-cell-label">vs SPY 3m</div>
      <div class="backtest-cell-val">{_fmt_signed_pct(spy_3m)}</div>
    </div>
    <div>
      <div class="backtest-cell-label">3m alpha</div>
      <div class="backtest-cell-val {alpha_color}">{_fmt_signed_pp(alpha_3m)}</div>
    </div>
    <div>
      <div class="backtest-cell-label">Hit rate</div>
      <div class="backtest-cell-val {hit_color}">{hit_str}</div>
    </div>
    <div>
      <div class="backtest-cell-label">Sharpe 3m</div>
      <div class="backtest-cell-val">{sharpe_str}</div>
    </div>
    <div>
      <div class="backtest-cell-label">Observations</div>
      <div class="backtest-cell-val">{n_obs:,}</div>
    </div>
  </div>
  <div class="ui faint" style="margin-top:10px;font-size:10.5px;letter-spacing:0.04em">
    {n_uniq or 0} unique tickers · as of {as_of}
  </div>
</details>"""


def _screen_section(
    screen_id,
    candidates: list[dict],
    screen_meta: dict,
    held_tickers: set,
    range_div: str,
    backtest_data: dict | None = None,
) -> str:
    meta = screen_meta.get(screen_id, {})
    title = meta.get("name", str(screen_id).replace("_", " ").title())
    description = meta.get("description", "")
    citation = meta.get("citation", meta.get("evidence", ""))

    expl_key = _SCREEN_ID_TO_EXPLAINER.get(screen_id, str(screen_id))
    expl = explainer_html(expl_key, title)

    if not candidates:
        body = f'<div class="faint" style="padding:18px 0;font-style:italic">No candidates surfaced this week.</div>'
    else:
        rows = "".join(
            _candidate_table_row(c, held_tickers) for c in candidates
        )
        body = f"""
<table class="sortable">
  <thead><tr>
    <th data-sort>Ticker</th>
    <th>Reason</th>
    <th class="num" data-sort>Price</th>
    <th class="num" data-sort>RSI</th>
    <th class="num" data-sort>Off 52wH</th>
    <th class="num" data-sort>Rev Growth</th>
    <th class="num" data-sort>Buy %</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>"""

    citation_html = (
        f'<div class="ui faint" style="font-size:11px;letter-spacing:0.04em;'
        f'margin:8px 0 12px;font-style:italic">'
        f'<span class="amber">●</span> &nbsp;{citation}</div>'
        if citation else ""
    )
    desc_html = (
        f'<div style="font-family:\'Newsreader\',serif;font-style:italic;'
        f'font-size:15px;color:{TEXT_DIM};margin-bottom:10px;max-width:680px">'
        f'{description}</div>'
        if description else ""
    )
    n_count = len(candidates)
    count_pill = (
        f'<span class="tag amber" style="margin-left:auto">{n_count} '
        f'name{"s" if n_count != 1 else ""}</span>'
        if candidates else ""
    )

    backtest_html = _backtest_panel(screen_id, backtest_data)

    return f"""
<div class="card">
  <div style="display:flex;align-items:baseline;gap:14px;margin-bottom:4px">
    <h3 style="font-family:'Newsreader',serif;font-weight:500;font-size:22px;
               color:{TEXT};letter-spacing:-0.015em;line-height:1.2;margin:0">
      {title}</h3>
    {count_pill}
  </div>
  {desc_html}
  {citation_html}
  {expl}
  {body}
  {backtest_html}
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
<div class="list-row">
  <div>
    <div class="ui faint" style="font-size:9.5px;letter-spacing:0.18em;
                                    text-transform:uppercase;margin-bottom:6px">
      Entry · {direction}
    </div>
    <div class="mono" style="font-size:20px;font-weight:600;color:{TEXT};
                              letter-spacing:0.01em">
      {ticker}
    </div>
    <div class="ui" style="font-size:11px;color:{MUTED};margin-top:6px;
                            letter-spacing:0.02em">
      Target <strong class="mono amber">{target_str}</strong>
      &nbsp;&middot;&nbsp; Now <strong class="mono">{price_str}</strong>
      &nbsp;&nbsp;{prox_html}
    </div>
  </div>
  <div class="list-row-body">
    <p>{note}</p>
  </div>
</div>"""

    return f"""
<section id="watchlist" class="sec">
  {_section_head("06", "Watch", "list", "Personal entry targets. Live prices versus configured levels with proximity flags.")}
  {cards}
</section>"""


def _pre_ipo_section(pre_ipo: list[dict]) -> str:
    if not pre_ipo:
        return ""
    cards = ""
    for item in pre_ipo:
        name = item.get("name", "?")
        note = item.get("note", "")
        expected = item.get("expected", "")
        expected_html = (
            f'<div class="list-row-meta">{expected}</div>' if expected else ""
        )
        cards += f"""
<div class="list-row">
  <div>
    <div class="ui faint" style="font-size:9.5px;letter-spacing:0.18em;
                                    text-transform:uppercase;margin-bottom:6px">Private</div>
    <div style="font-family:'Newsreader',serif;font-weight:500;font-size:18px;
                color:{TEXT};letter-spacing:-0.01em">
      {name}
    </div>
    {expected_html}
  </div>
  <div class="list-row-body">
    <p>{note}</p>
  </div>
</div>"""

    return f"""
<section id="preipo" class="sec">
  {_section_head("07", "Pre-IPO", "watch", "Notable private companies tracked manually — informational only, no live data.")}
  {cards}
</section>"""


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
    backtest_results: dict | None = None,
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
    short_date = rd.strftime("%b %d, %Y").upper()
    score_val = state_score.get("score", state_score.get("total_score", 0)) or 0
    drivers = state_score.get("drivers", [])
    regime = state_score.get("regime") or _regime_for(int(score_val))

    gauge_div = chart_market_state(int(score_val), drivers)

    # Aggregate candidates for the 52w range chart + stats counts
    all_candidates: list[dict] = []
    for clist in screen_results.values():
        all_candidates.extend(clist)
    seen_tickers: set = set()
    unique_candidates: list[dict] = []
    for c in all_candidates:
        t = c.get("ticker", "")
        if t not in seen_tickers:
            seen_tickers.add(t)
            unique_candidates.append(c)
    range_div = chart_52w_range(unique_candidates)

    # ── Compute the masthead stats strip ──────────────────────────────────
    breadth = state_score.get("breadth", {}) or {}
    pct_above_200 = breadth.get("pct_above_200dma")
    n_themes_tracked = len(ranked_themes) if ranked_themes else 0
    n_candidates = len(unique_candidates)

    score_class = ("amber" if score_val < 50 else
                   "green" if score_val >= 70 else "")
    breadth_class = ""
    if pct_above_200 is not None:
        if pct_above_200 >= 60:
            breadth_class = "green"
        elif pct_above_200 < 40:
            breadth_class = "red"

    masthead_stats = f"""
<div class="masthead-strip">
  <div class="masthead-stat">
    <div class="masthead-stat-label">Market State</div>
    <div class="masthead-stat-val {score_class}">{score_val}<span style="font-size:14px;color:{FAINT}">/100</span></div>
    <div class="masthead-stat-sub">{regime}</div>
  </div>
  <div class="masthead-stat">
    <div class="masthead-stat-label">Breadth · &gt; 200 DMA</div>
    <div class="masthead-stat-val {breadth_class}">{f'{pct_above_200:.0f}%' if pct_above_200 is not None else '—'}</div>
    <div class="masthead-stat-sub">% R1000 above 200-day MA</div>
  </div>
  <div class="masthead-stat">
    <div class="masthead-stat-label">Themes Tracked</div>
    <div class="masthead-stat-val">{n_themes_tracked}</div>
    <div class="masthead-stat-sub">Curated baskets ranked</div>
  </div>
  <div class="masthead-stat">
    <div class="masthead-stat-label">Names Surfaced</div>
    <div class="masthead-stat-val">{n_candidates}</div>
    <div class="masthead-stat-sub">Unique across 8 screens</div>
  </div>
</div>"""

    body = f"""
<div class="masthead">
  <div class="masthead-kicker">Weekly Brief · {short_date}</div>
  <h1 class="masthead-title">
    A read of the market<br>
    <em>at the close.</em>
  </h1>
  <p class="masthead-sub">
    Market regime, sector rotation, thematic strength, and eight academically-validated
    candidate screens — synthesized from Russell 1000 prices, FRED macro series, AAII sentiment,
    and SEC EDGAR Form 4 filings.
  </p>
  {masthead_stats}
</div>"""

    # ── 1. Market State ──────────────────────────────────────────────────
    body += f"""
<section id="state" class="sec">
  {_section_head("01", "Market", "state", "Composite score from nine indicators — breadth, volatility regime, credit, cross-asset ratios.")}
  {_chart(gauge_div) if gauge_div else ''}
  {explainer_html("market_state_score", "Market State Score")}
</section>"""

    # ── 2. Themes ────────────────────────────────────────────────────────
    body += _theme_section(ranked_themes, emerging_clusters)

    # ── 3. Sector rotation ───────────────────────────────────────────────
    body += _sector_section(sector_rotation)

    # ── 4. Candidate screens ─────────────────────────────────────────────
    if screen_results:
        screens_inner = ""
        for screen_id, candidates in screen_results.items():
            bt = None
            if backtest_results:
                bt = backtest_results.get(screen_id)
                if bt is None:
                    bt = backtest_results.get(str(screen_id))
                    if bt is None:
                        try:
                            bt = backtest_results.get(int(screen_id))
                        except (TypeError, ValueError):
                            bt = None
            screens_inner += _screen_section(
                screen_id, candidates, screen_meta, held_tickers, range_div,
                backtest_data=bt,
            )
        body += f"""
<section id="screens" class="sec">
  {_section_head("04", "Candidate", "screens", "Eight evidence-tiered screens. Held names annotated. Sort any column. Tables are click-sortable.")}
  {screens_inner}
</section>"""

    # ── 5. 52-week range overview ─────────────────────────────────────────
    if range_div:
        body += f"""
<section id="ranges" class="sec">
  {_section_head("05", "Where they trade", "", "Every surfaced candidate plotted on its 52-week range. Hover to inspect.")}
  {_chart(range_div)}
</section>"""

    # ── 6. Watchlist ──────────────────────────────────────────────────────
    body += _watchlist_section(watchlist, rd, watch_prices)

    # ── 7. Pre-IPO watch ──────────────────────────────────────────────────
    body += _pre_ipo_section(pre_ipo)

    # ── 8. Breadth detail ─────────────────────────────────────────────────
    body += _breadth_strip(new_highs_lows)

    # ── 9. Glossary ───────────────────────────────────────────────────────
    body += glossary_html()

    nav_links = [
        ("state",     "Market State"),
        ("themes",    "Themes"),
        ("sectors",   "Sectors"),
        ("screens",   "Screens"),
        ("watchlist", "Watchlist"),
        ("preipo",    "Pre-IPO"),
        ("glossary",  "Glossary"),
    ]
    nav_meta = f"Score {score_val} · {regime} · {short_date}"
    return _html_shell(
        f"Weekly Market Brief — {date_str}", body, rd,
        nav_links=nav_links, nav_meta=nav_meta,
    )


def _regime_for(score: int) -> str:
    if score < 35:
        return "Risk-Off"
    if score < 50:
        return "Caution"
    if score < 70:
        return "Neutral"
    return "Risk-On"


def _section_head(num: str, title: str, em_word: str = "", lede: str = "") -> str:
    """Render a section header. If em_word is given, it's set in italic serif."""
    if em_word:
        title_html = f'{title} <em>{em_word}</em>'
    else:
        title_html = title
    lede_html = (
        f'<p class="sec-lede">{lede}</p>' if lede else ""
    )
    return f"""
  <div class="sec-head">
    <span class="sec-num">/ {num}</span>
    <h2 class="sec-title">{title_html}</h2>
  </div>
  {lede_html}"""
