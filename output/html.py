"""
HTML report generator for the market intelligence scanner.
Produces dark-theme (0d1117) newsletter-style reports with embedded
base64 matplotlib charts. No external CSS/JS dependencies.

Exports:
    build_daily_report(...)  -> str
    build_weekly_report(...) -> str
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import base64
import io
import json
from datetime import date, datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

from output.explainers import EXPLAINERS, explainer_html, glossary_html

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

# ── Matplotlib helpers ──────────────────────────────────────────────────────

def _apply_dark(fig, ax_or_axes):
    """Apply consistent dark styling to a figure and its axes."""
    fig.patch.set_facecolor(BG)
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    for ax in np.array(axes).ravel():
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[:].set_visible(False)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)


def fig_to_b64(fig) -> str:
    """Convert a matplotlib Figure to a base64 PNG string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── Chart: market state gauge + driver bars ─────────────────────────────────

def chart_market_state(score: int, drivers: list[dict]) -> str:
    """
    Two-panel figure:
      Left  — semicircular gauge showing the 0-100 market state score.
      Right — horizontal bar chart of the top 6 driver contributions.

    Args:
        score:   Integer 0-100.
        drivers: List of dicts with keys 'label' and 'contribution' (float).

    Returns:
        base64 PNG string.
    """
    plt.style.use("dark_background")

    # Pick score colour
    if score < 35:
        score_color = RED
    elif score < 50:
        score_color = AMBER
    elif score < 70:
        score_color = "#e3b341"   # yellow
    else:
        score_color = GREEN

    fig, (ax_gauge, ax_bars) = plt.subplots(
        1, 2,
        figsize=(10, 3.8),
        gridspec_kw={"width_ratios": [1, 1.6]},
    )
    _apply_dark(fig, [ax_gauge, ax_bars])

    # ── Gauge (semicircle) ──────────────────────────────────────────────────
    ax_gauge.set_xlim(-1.2, 1.2)
    ax_gauge.set_ylim(-0.25, 1.15)
    ax_gauge.set_aspect("equal")
    ax_gauge.axis("off")

    # Background arc (full 180°)
    theta_bg = np.linspace(np.pi, 0, 200)
    ax_gauge.plot(np.cos(theta_bg), np.sin(theta_bg),
                  color=SURFACE, linewidth=22, solid_capstyle="butt", zorder=1)

    # Filled arc — score/100 of the semicircle
    frac = max(0, min(1, score / 100))
    theta_fill = np.linspace(np.pi, np.pi - frac * np.pi, 200)
    ax_gauge.plot(np.cos(theta_fill), np.sin(theta_fill),
                  color=score_color, linewidth=22, solid_capstyle="butt", zorder=2)

    # Zone tick marks at 35, 50, 70
    for zone in (0.35, 0.50, 0.70):
        angle = np.pi - zone * np.pi
        ax_gauge.plot(
            [0.78 * np.cos(angle), 0.94 * np.cos(angle)],
            [0.78 * np.sin(angle), 0.94 * np.sin(angle)],
            color=BORDER, linewidth=1.5, zorder=3,
        )

    # Score text in centre
    ax_gauge.text(0, 0.25, str(score), ha="center", va="center",
                  fontsize=38, fontweight="700", color=score_color, zorder=4)

    # Regime label
    if score < 35:
        regime = "Risk-Off"
    elif score < 50:
        regime = "Caution"
    elif score < 70:
        regime = "Neutral"
    else:
        regime = "Risk-On"
    ax_gauge.text(0, -0.05, regime, ha="center", va="center",
                  fontsize=11, color=MUTED, zorder=4)
    ax_gauge.text(0, 1.08, "Market State", ha="center", va="center",
                  fontsize=9, color=MUTED, zorder=4)

    # Zone labels
    for val, label in ((0, "0"), (35, "35"), (50, "50"), (70, "70"), (100, "100")):
        angle = np.pi - (val / 100) * np.pi
        r = 1.12
        ax_gauge.text(r * np.cos(angle), r * np.sin(angle), label,
                      ha="center", va="center", fontsize=7, color=MUTED)

    # ── Driver bars ─────────────────────────────────────────────────────────
    top_drivers = sorted(drivers, key=lambda d: abs(d.get("contribution", 0)),
                         reverse=True)[:6]
    if not top_drivers:
        ax_bars.axis("off")
        ax_bars.text(0.5, 0.5, "No driver data", ha="center", va="center",
                     color=MUTED, fontsize=9, transform=ax_bars.transAxes)
    else:
        labels = [d.get("label", "?")[:28] for d in top_drivers]
        vals = [float(d.get("contribution", 0)) for d in top_drivers]
        colors = [GREEN if v >= 0 else RED for v in vals]
        y = np.arange(len(labels))

        ax_bars.barh(y, vals, color=colors, height=0.55,
                     edgecolor="none", zorder=2)
        ax_bars.axvline(0, color=BORDER, linewidth=0.8, zorder=3)
        ax_bars.set_yticks(y)
        ax_bars.set_yticklabels(labels, color=TEXT, fontsize=9)
        ax_bars.tick_params(axis="x", colors=MUTED, labelsize=8)
        ax_bars.invert_yaxis()
        ax_bars.set_title("Driver contributions", color=MUTED,
                          fontsize=9, pad=6)

    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)


# ── Chart: sector rotation ───────────────────────────────────────────────────

def chart_sector_rotation(sector_data: list[dict]) -> str:
    """
    Horizontal bar chart of sectors ranked by Faber score (or relative strength).

    Each dict in sector_data should have keys:
        name        str   — sector name (may be abbreviated)
        score       float — Faber/RS score
        rank_delta  int   — change in rank vs prior period (positive = improved)

    Returns:
        base64 PNG string.
    """
    plt.style.use("dark_background")

    if not sector_data:
        fig, ax = plt.subplots(figsize=(8, 2))
        _apply_dark(fig, ax)
        ax.text(0.5, 0.5, "No sector data available",
                ha="center", va="center", color=MUTED, fontsize=9,
                transform=ax.transAxes)
        fig.tight_layout()
        return fig_to_b64(fig)

    # Sort by score descending
    data = sorted(sector_data, key=lambda d: d.get("score", 0), reverse=True)
    n = len(data)
    labels = [d.get("name", "?")[:22] for d in data]
    scores = [float(d.get("score", 0)) for d in data]
    deltas = [int(d.get("rank_delta", 0)) for d in data]
    colors = [GREEN if s >= 0 else RED for s in scores]

    fig, ax = plt.subplots(figsize=(9, max(3, n * 0.55 + 0.8)))
    _apply_dark(fig, ax)
    ax.set_facecolor(BG)

    y = np.arange(n)
    ax.barh(y, scores, color=colors, height=0.55, edgecolor="none", zorder=2)
    ax.axvline(0, color=BORDER, linewidth=0.8, zorder=3)

    # Rank delta arrows on the right
    for i, (delta, score) in enumerate(zip(deltas, scores)):
        if delta > 0:
            arrow = f"▲{abs(delta)}"
            ac = GREEN
        elif delta < 0:
            arrow = f"▼{abs(delta)}"
            ac = RED
        else:
            arrow = "–"
            ac = MUTED
        # Place the arrow just past the bar
        x_pos = score + (max(scores) - min(scores)) * 0.02 if scores else 0.5
        ax.text(x_pos, i, f" {arrow}", ha="left", va="center",
                fontsize=9, color=ac, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=TEXT, fontsize=9.5, fontweight="600")
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.invert_yaxis()
    ax.set_title("Sector Rotation — Ranked by Relative Strength",
                 color=MUTED, fontsize=10, pad=8)
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)


# ── Chart: theme heatmap ─────────────────────────────────────────────────────

def chart_theme_heatmap(ranked_themes: list[dict]) -> str:
    """
    Horizontal bar chart of all themes sorted by theme_score (z-scored).
    Top third coloured green, middle yellow, bottom red.

    Each dict should have keys:
        name         str   — theme name
        theme_score  float — z-scored score (may be negative)

    Returns:
        base64 PNG string.
    """
    plt.style.use("dark_background")

    if not ranked_themes:
        fig, ax = plt.subplots(figsize=(8, 2))
        _apply_dark(fig, ax)
        ax.text(0.5, 0.5, "No theme data available",
                ha="center", va="center", color=MUTED, fontsize=9,
                transform=ax.transAxes)
        fig.tight_layout()
        return fig_to_b64(fig)

    data = sorted(ranked_themes, key=lambda d: d.get("theme_score", 0), reverse=True)
    n = len(data)
    labels = [d.get("name", "?")[:30] for d in data]
    scores = [float(d.get("theme_score", 0)) for d in data]

    # Colour tiers
    third = max(1, n // 3)
    colors = (
        [GREEN] * third
        + ["#e3b341"] * (n - 2 * third)
        + [RED] * third
    )
    # Trim to n if rounding makes it longer
    colors = colors[:n]
    while len(colors) < n:
        colors.append(MUTED)

    fig, ax = plt.subplots(figsize=(9, max(3, n * 0.52 + 0.8)))
    _apply_dark(fig, ax)

    y = np.arange(n)
    ax.barh(y, scores, color=colors, height=0.55, edgecolor="none", zorder=2)
    ax.axvline(0, color=BORDER, linewidth=0.8, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=TEXT, fontsize=9.5)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.invert_yaxis()
    ax.set_title("Theme Heatmap — Sorted by Relative Strength Score",
                 color=MUTED, fontsize=10, pad=8)
    ax.set_xlabel("Theme score (z-scored)", color=MUTED, fontsize=8)
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)


# ── Chart: 52-week range ─────────────────────────────────────────────────────

def chart_52w_range(candidates: list[dict]) -> str:
    """
    Horizontal range bars for the top ~15 candidates.
    Each dict needs: ticker, lo52, hi52, price.

    Returns:
        base64 PNG string.
    """
    plt.style.use("dark_background")

    items = [
        (s["ticker"], s["lo52"], s["hi52"], s["price"])
        for s in candidates[:15]
        if s.get("lo52") and s.get("hi52") and s.get("price")
    ]
    if not items:
        fig, ax = plt.subplots(figsize=(8, 2))
        _apply_dark(fig, ax)
        ax.text(0.5, 0.5, "No range data available",
                ha="center", va="center", color=MUTED, fontsize=9,
                transform=ax.transAxes)
        fig.tight_layout()
        return fig_to_b64(fig)

    n = len(items)
    fig, ax = plt.subplots(figsize=(9, max(3, n * 0.58)))
    _apply_dark(fig, ax)

    for i, (ticker, lo, hi, curr) in enumerate(items):
        span = hi - lo if hi != lo else 1.0
        ax.barh(i, span, left=lo, height=0.45, color=SURFACE, zorder=2)
        ax.barh(i, curr - lo, left=lo, height=0.45, color=BLUE, alpha=0.75, zorder=3)
        ax.plot(curr, i, "|", color=BLUE_LT, markersize=16,
                markeredgewidth=2.5, zorder=5)
        ax.text(lo - span * 0.01, i, f"${lo:,.0f}",
                ha="right", va="center", fontsize=7.5, color=MUTED)
        ax.text(hi + span * 0.01, i, f"${hi:,.0f}",
                ha="left", va="center", fontsize=7.5, color=MUTED)
        ax.text(curr, i + 0.30, f"${curr:,.2f}",
                ha="center", va="bottom", fontsize=8,
                color=BLUE_LT, fontweight="bold")

    ax.set_yticks(range(n))
    ax.set_yticklabels([t for t, *_ in items], color=TEXT,
                       fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.set_title("52-week range  ·  ▎ = current price",
                 color=MUTED, fontsize=10, pad=8)
    ax.invert_yaxis()
    fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _img(b64: str, style: str = "") -> str:
    """Wrap a base64 PNG in an <img> tag, or return '' if empty."""
    if not b64:
        return ""
    default = "width:100%;border-radius:8px;margin-bottom:20px"
    s = style or default
    return f'<img src="data:image/png;base64,{b64}" style="{s}">'


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
@media(max-width:600px){{
  .grid-4{{grid-template-columns:repeat(2,1fr)}}
  .wrap{{padding:0 10px 32px}}
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

    gauge_b64 = chart_market_state(int(score_val), drivers)
    sector_b64 = chart_sector_rotation(sector_rotation)

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
    if gauge_b64:
        body += f"""
<div class="sec">
  <div class="sec-title">Market state</div>
  {_img(gauge_b64)}
  {explainer_html("market_state_score", "Market State Score")}
</div>"""

    # Breadth
    body += _breadth_strip(new_highs_lows)

    # Sector rotation chart
    if sector_b64:
        body += f"""
<div class="sec">
  <div class="sec-title">Sector rotation</div>
  {_img(sector_b64)}
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
    heatmap_b64 = chart_theme_heatmap(ranked_themes)

    # Top 5 themes with members
    top5 = ranked_themes[:5] if ranked_themes else []
    theme_cards = ""
    for t in top5:
        members = t.get("members", [])
        mem_str = ", ".join(str(m) for m in members[:8])
        score = t.get("theme_score", 0)
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
            tickers = ", ".join(cl.get("tickers", [])[:6])
            cluster_html += f"""
<div class="card" style="margin-bottom:8px">
  <span style="font-weight:700;color:{AMBER}">{cl.get('theme','?')}</span>
  <span style="font-size:12px;color:{MUTED};margin-left:10px">{tickers}</span>
  {f'<div style="font-size:11px;color:#6e7681;margin-top:4px">{cl.get("note","")}</div>' if cl.get('note') else ''}
</div>"""

    return f"""
<div class="sec">
  <div class="sec-title">Theme heatmap</div>
  {_img(heatmap_b64)}
  {explainer_html("theme_strength", "Theme Strength Score")}
  {theme_cards}
  {cluster_html}
</div>"""


def _sector_section(sector_rotation: list[dict]) -> str:
    sector_b64 = chart_sector_rotation(sector_rotation)
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
  {_img(sector_b64)}
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
    from_hi = c.get("from_hi")
    fh_c = GREEN if from_hi and 10 <= from_hi <= 40 else AMBER
    rev = c.get("rev_growth")
    buy_pct = c.get("buy_pct")
    reason = c.get("reason", c.get("verdict", ""))

    return f"""<tr>
  <td>
    <span style="font-weight:700;color:{TEXT}">{ticker}</span>{held_badge}
    <div style="font-size:11px;color:{MUTED}">{name[:22]}</div>
  </td>
  <td style="font-size:11px;color:{MUTED}">{reason[:40]}</td>
  <td style="font-weight:700">{price}</td>
  <td style="color:{rsi_c}">{rsi if rsi else '—'}</td>
  <td style="color:{fh_c}">{_fmt_pct(from_hi) if from_hi else '—'}</td>
  <td>{_fmt_pct(rev) if rev else '—'}</td>
  <td style="color:{MUTED}">{f'{buy_pct}%' if buy_pct else '—'}</td>
</tr>"""


def _screen_section(
    screen_id: str,
    candidates: list[dict],
    screen_meta: dict,
    held_tickers: set,
    range_b64: str,
) -> str:
    meta = screen_meta.get(screen_id, {})
    title = meta.get("name", screen_id.replace("_", " ").title())
    description = meta.get("description", "")
    citation = meta.get("citation", "")

    expl = explainer_html(screen_id, title)

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


def _watchlist_section(watchlist: dict, report_date: date) -> str:
    if not watchlist:
        return ""
    today = report_date or date.today()
    cards = ""
    for ticker, cfg in watchlist.items():
        if isinstance(cfg, dict):
            target = cfg.get("buy_at")
            direction = cfg.get("direction", "below")
            note = cfg.get("note", "")
            price = cfg.get("current_price")  # may not be present
        else:
            target = cfg
            direction = "below"
            note = ""
            price = None

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

    gauge_b64 = chart_market_state(int(score_val), drivers)

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
    range_b64 = chart_52w_range(unique_candidates)

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
    if gauge_b64:
        body += f"""
<div class="sec">
  <div class="sec-title">Market state</div>
  {_img(gauge_b64)}
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
                screen_id, candidates, screen_meta, held_tickers, range_b64
            )
        body += f"""
<div class="sec">
  <div class="sec-title">Candidate screens</div>
  {screens_inner}
</div>"""

    # 52w range chart (after screens, referencing all candidates)
    if range_b64:
        body += f"""
<div class="sec">
  <div class="sec-title">52-week range — all candidates</div>
  {_img(range_b64)}
</div>"""

    # ── 5. Watchlist ───────────────────────────────────────────────────────
    body += _watchlist_section(watchlist, rd)

    # ── 6. Pre-IPO ─────────────────────────────────────────────────────────
    body += _pre_ipo_section(pre_ipo)

    # ── 7. Breadth ─────────────────────────────────────────────────────────
    body += _breadth_strip(new_highs_lows)

    # Glossary at the bottom
    body += glossary_html()

    return _html_shell(f"Weekly Market Brief — {date_str}", body, rd)
