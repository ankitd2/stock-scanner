#!/usr/bin/env python3
"""
Free market scanner — yfinance + matplotlib + HTML report.
No AI APIs. No paid services.

Outputs
  1. report.html  — full newsletter-style HTML (saved as Actions artifact)
  2. GitHub Actions job summary (visible in Actions tab, markdown)
  3. Discord webhook — urgent alerts only (watch price hit / big moves)
"""

import json, os, sys, io, base64, warnings, traceback
from datetime import datetime, timedelta, date
from pathlib import Path
import requests

warnings.filterwarnings("ignore")
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────
def load_config():
    with open("config.json") as f:
        return json.load(f)

NOW  = datetime.now()
DATE = NOW.strftime("%A, %B %d, %Y")

# ── Screener universe ──────────────────────────────────────────────────────
UNIVERSE = [
    "MSFT","GOOGL","AMZN","META","NVDA","TSLA","AAPL",
    "CRM","NOW","SNOW","DDOG","NET","ZS","CRWD","PANW",
    "PLTR","APP","TTD","HUBS","SHOP","RDDT","SPOT","NFLX",
    "AMD","AVGO","MU","AMAT","KLAC","MRVL","ARM","TSM",
    "ANET","CRDO","VRT","GEV","CEG","VST","NBIS","APLD","CRWV",
    "UBER","ABNB","BKNG","DASH","ISRG","DXCM","HIMS","VEEV",
    "INTU","ADBE","RKLB","KTOS","AXON","LMT",
    "COIN","HOOD","SOFI","NU","MELI",
]

# ── Formatters ─────────────────────────────────────────────────────────────
def fmt_price(v):
    return f"${v:,.2f}" if v else "—"

def fmt_mcap(v):
    if not v: return "—"
    if v >= 1e12: return f"${v/1e12:.1f}T"
    if v >= 1e9:  return f"${v/1e9:.1f}B"
    return f"${v/1e6:.0f}M"

def fmt_pct(v, plus=True):
    if v is None: return "—"
    return f"{'+'if v>0 and plus else ''}{v:.1f}%"

# ── Data helpers ───────────────────────────────────────────────────────────
def calc_rsi(prices, period=14):
    try:
        delta = prices.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        return round(float((100 - 100/(1+gain/loss)).iloc[-1]), 1)
    except: return None

def get_rec_counts(t):
    buy=hold=sell=0
    try:
        recs = t.recommendations
        if recs is None or recs.empty: return buy,hold,sell
        for col in recs.tail(30).columns:
            cl  = col.lower()
            val = int(recs.tail(30)[col].sum())
            if "buy" in cl:   buy  += val
            elif "hold" in cl or "neutral" in cl: hold += val
            elif "sell" in cl: sell += val
    except: pass
    return buy, hold, sell

def get_earnings_date(t):
    try:
        cal = t.calendar
        if not isinstance(cal, dict): return None
        dates = cal.get("Earnings Date", [])
        for d in (dates if hasattr(dates,"__iter__") else [dates]):
            dt = pd.Timestamp(d).date()
            if dt >= date.today(): return dt
    except: return None

def get_stock(ticker):
    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}
        hist = t.history(period="1y", auto_adjust=True)
        if hist.empty: return None

        curr  = float(hist["Close"].iloc[-1])
        hi52  = float(hist["High"].max())
        lo52  = float(hist["Low"].min())
        from_hi = (hi52-curr)/hi52*100 if hi52 else None

        pct_today = 0.0
        if len(hist)>=2:
            pct_today=(curr-float(hist["Close"].iloc[-2]))/float(hist["Close"].iloc[-2])*100

        ytd = hist[hist.index.year==NOW.year]
        pct_ytd=((curr-float(ytd["Close"].iloc[0]))/float(ytd["Close"].iloc[0])*100
                 if not ytd.empty else 0.0)

        tgt_mean = info.get("targetMeanPrice")
        tgt_high = info.get("targetHighPrice")
        tgt_low  = info.get("targetLowPrice")
        n_analysts = int(info.get("numberOfAnalystOpinions") or 0)
        upside = ((tgt_mean-curr)/curr*100) if tgt_mean else None
        upside_hi = ((tgt_high-curr)/curr*100) if tgt_high else None

        rec_mean = info.get("recommendationMean")
        rec_key  = (info.get("recommendationKey") or "").replace("-"," ").title()
        buy, hold, sell = get_rec_counts(t)
        total = buy+hold+sell
        buy_pct = round(buy/total*100) if total else None

        def clean_pe(v):
            return round(v,1) if v and 0<v<2000 else None

        rev_growth   = info.get("revenueGrowth")
        gross_margin = info.get("grossMargins")
        op_margin    = info.get("operatingMargins")

        news=[]
        try:
            for item in (t.news or [])[:3]:
                if item.get("title"):
                    news.append({"title":item["title"],"url":item.get("link","#")})
        except: pass

        return {
            "ticker":     ticker,
            "name":       info.get("shortName", ticker),
            "sector":     info.get("sector",""),
            "price":      round(curr,2),
            "hi52":       round(hi52,2),
            "lo52":       round(lo52,2),
            "from_hi":    round(from_hi,1) if from_hi else None,
            "pct_today":  round(pct_today,2),
            "pct_ytd":    round(pct_ytd,1),
            "tgt_mean":   round(tgt_mean,2) if tgt_mean else None,
            "tgt_high":   round(tgt_high,2) if tgt_high else None,
            "tgt_low":    round(tgt_low,2) if tgt_low else None,
            "upside":     round(upside,1) if upside else None,
            "upside_hi":  round(upside_hi,1) if upside_hi else None,
            "n_analysts": n_analysts,
            "rec_mean":   round(rec_mean,2) if rec_mean else None,
            "rec_key":    rec_key,
            "buy":buy,"hold":hold,"sell":sell,
            "buy_pct":    buy_pct,
            "rev_growth": round(rev_growth*100,1) if rev_growth else None,
            "gross_margin":round(gross_margin*100,1) if gross_margin else None,
            "op_margin":  round(op_margin*100,1) if op_margin else None,
            "pe_trailing":clean_pe(info.get("trailingPE")),
            "pe_forward": clean_pe(info.get("forwardPE")),
            "peg":        round(info.get("pegRatio"),2) if info.get("pegRatio") and 0<info.get("pegRatio",99)<20 else None,
            "mcap":       info.get("marketCap"),
            "earnings_dt":get_earnings_date(t),
            "rsi":        calc_rsi(hist["Close"]),
            "news":       news,
            "hist":       hist,
        }
    except Exception as e:
        print(f"  [{ticker}] {e}")
        return None

# ── Skill criteria (pure logic, no AI) ────────────────────────────────────
def skill_verdict(s):
    passes, fails, kills = [], [], []

    rg = s.get("rev_growth")
    if rg is not None: (passes if rg>=20 else fails).append(f"Rev growth {rg:+.0f}% (need >20%)")
    else: fails.append("Rev growth: no data")

    bp = s.get("buy_pct")
    if bp is not None: (passes if bp>=80 else fails).append(f"Buy consensus {bp:.0f}% (need ≥80%)")
    else: fails.append("Buy consensus: no data")

    sell = s.get("sell",0)
    if sell>=2: kills.append(f"{sell} sell ratings (hard kill)")
    else: passes.append(f"{sell} sell ratings ✓")

    fh = s.get("from_hi")
    if fh is not None: (passes if 10<=fh<=40 else fails).append(f"{fh:.1f}% below 52wk high (need 10–40%)")
    else: fails.append("52wk high: no data")

    up = s.get("upside")
    if up is not None:
        (passes if up>=15 else fails).append(f"PT upside {up:+.0f}% (need ≥15%)")
        if s.get("tgt_high") and s["price"]>s["tgt_high"]:
            kills.append(f"Above highest PT ${s['tgt_high']} (hard kill)")
    else: fails.append("Analyst PT: no data")

    mc = s.get("mcap")
    if mc: (passes if mc>=2e9 else fails).append(f"Mkt cap {fmt_mcap(mc)} (need >$2B)")

    rsi = s.get("rsi")
    if rsi and rsi>78 and fh and fh<5:
        kills.append(f"RSI {rsi} + near 52wk high (hard kill)")

    if kills: v="PASS"
    elif len(fails)==0: v="BUY"
    elif len(fails)==1: v="WATCH"
    else: v="WAIT"

    return {"verdict":v,"passes":passes,"fails":fails,"kills":kills,
            "score":len(passes)-len(kills)*2}

# ── Market pulse ───────────────────────────────────────────────────────────
def get_market_pulse():
    indices={"S&P 500":"^GSPC","Nasdaq":"^IXIC","VIX":"^VIX","10Y Yield":"^TNX"}
    pulse={}
    for name,sym in indices.items():
        try:
            hist=yf.Ticker(sym).history(period="5d",auto_adjust=True)
            if len(hist)>=2:
                prev=hist["Close"].iloc[-2]; curr=hist["Close"].iloc[-1]
                pulse[name]={"value":float(curr),"pct":float((curr-prev)/prev*100)}
        except: pass
    return pulse

# ── Screener ───────────────────────────────────────────────────────────────
def run_screener(portfolio, limit=10):
    exclude=set(t.upper() for t in portfolio)
    universe=[t for t in UNIVERSE if t not in exclude]
    print(f"  Screening {len(universe)} tickers…")
    results=[]
    for ticker in universe:
        print(f"    {ticker}",end=" ",flush=True)
        s=get_stock(ticker)
        if not s: print("✗"); continue
        v=skill_verdict(s); s["verdict_data"]=v
        if v["verdict"] in ("BUY","WATCH"): results.append(s); print(f"✓ {v['verdict']}")
        else: print(f"· {v['verdict']}")
    results.sort(key=lambda x:({"BUY":0,"WATCH":1}.get(x["verdict_data"]["verdict"],9),
                                -x["verdict_data"]["score"]))
    return results[:limit]

# ── Earnings calendar ──────────────────────────────────────────────────────
def get_earnings_calendar(tickers):
    upcoming=[]; today=date.today(); cutoff=today+timedelta(days=45)
    for ticker in tickers:
        try:
            t=yf.Ticker(ticker); cal=t.calendar
            if not isinstance(cal,dict): continue
            dates=cal.get("Earnings Date",[])
            for d in (dates if hasattr(dates,"__iter__") else [dates]):
                dt=pd.Timestamp(d).date()
                if today<=dt<=cutoff: upcoming.append({"ticker":ticker,"date":dt})
        except: pass
    upcoming.sort(key=lambda x:x["date"])
    # dedupe
    seen=set(); out=[]
    for e in upcoming:
        k=(e["ticker"],e["date"])
        if k not in seen: seen.add(k); out.append(e)
    return out

# ── Charts ─────────────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf=io.BytesIO(); fig.savefig(buf,format="png",dpi=130,bbox_inches="tight",
                                  facecolor=fig.get_facecolor()); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

BG="#0d1117"; SURFACE="#161b22"; BORDER="#30363d"
BLUE="#1f6feb"; BLUE_LT="#58a6ff"; GREEN="#3fb950"; AMBER="#f0883e"
RED="#ff7b72"; TEXT="#c9d1d9"; MUTED="#8b949e"

def chart_range(stocks):
    items=[(s["ticker"],s["lo52"],s["hi52"],s["price"])
           for s in stocks if s.get("lo52") and s.get("hi52")]
    if not items: return ""
    n=len(items); fig,ax=plt.subplots(figsize=(9,max(3,n*0.58)))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for i,(ticker,lo,hi,curr) in enumerate(items):
        span=hi-lo
        ax.barh(i,span,left=lo,height=0.45,color=SURFACE,zorder=2)
        ax.barh(i,curr-lo,left=lo,height=0.45,color=BLUE,alpha=0.75,zorder=3)
        ax.plot(curr,i,"|",color=BLUE_LT,markersize=16,markeredgewidth=2.5,zorder=5)
        ax.text(lo-(span*.01),i,f"${lo:,.0f}",ha="right",va="center",
                fontsize=7.5,color=MUTED)
        ax.text(hi+(span*.01),i,f"${hi:,.0f}",ha="left",va="center",
                fontsize=7.5,color=MUTED)
        ax.text(curr,i+0.30,f"${curr:,.2f}",ha="center",va="bottom",
                fontsize=8,color=BLUE_LT,fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels([t for t,*_ in items],color=TEXT,fontsize=11,fontweight="bold")
    ax.tick_params(axis="x",colors=MUTED,labelsize=8)
    ax.spines[:].set_visible(False)
    ax.set_title("52-week range  ·  ▎ = current price",color=MUTED,fontsize=10,pad=8)
    ax.invert_yaxis(); fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)

def chart_watchlist(watchlist_cfg, prices):
    items=[(ticker,prices.get(ticker),cfg.get("buy_at"),cfg.get("direction","below"))
           for ticker,cfg in watchlist_cfg.items() if prices.get(ticker)]
    if not items: return ""
    n=len(items); fig,ax=plt.subplots(figsize=(8,max(2.5,n*0.65)))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    for i,(ticker,price,target,direction) in enumerate(items):
        if direction=="below":
            pct_away=(price-target)/target*100
            progress=max(0,min(1,target/price))
            color=GREEN if pct_away<=5 else (AMBER if pct_away<=15 else SURFACE)
            label=f"{pct_away:+.1f}% above target ${target:,.0f}"
        else:
            pct_away=(target-price)/price*100; progress=max(0,min(1,price/target))
            color=AMBER; label=f"{pct_away:.1f}% to go  target ${target:,.0f}"
        ax.barh(i,1.0,height=0.5,color=SURFACE,zorder=2)
        ax.barh(i,progress,height=0.5,color=color,alpha=0.8,zorder=3)
        ax.text(0.012,i,ticker,ha="left",va="center",fontsize=11,
                fontweight="bold",color=TEXT,zorder=4)
        ax.text(0.988,i,f"${price:,.2f}  ·  {label}",ha="right",va="center",
                fontsize=8.5,color=MUTED,zorder=4)
    ax.set_yticks([]); ax.set_xticks([]); ax.spines[:].set_visible(False)
    ax.set_xlim(0,1)
    ax.set_title("Watch list — proximity to entry target",color=MUTED,fontsize=10,pad=8)
    ax.invert_yaxis(); fig.tight_layout(pad=1.2)
    return fig_to_b64(fig)

def chart_alloc(portfolio_data):
    palette=[BLUE,GREEN,AMBER,RED,"#a371f7",BLUE_LT,"#56d364","#ffa657",
             "#d2a8ff","#79c0ff","#f85149"]
    labels=[]; values=[]; colors=[]
    for i,s in enumerate(portfolio_data):
        if s.get("price"):
            labels.append(s["ticker"]); values.append(s["price"])
            colors.append(palette[i%len(palette)])
    if not values: return ""
    fig,ax=plt.subplots(figsize=(5,5)); fig.patch.set_facecolor(BG)
    wedges,texts,autos=ax.pie(values,labels=labels,colors=colors,autopct="%1.0f%%",
                               pctdistance=0.82,startangle=140,
                               wedgeprops={"linewidth":2,"edgecolor":BG})
    for t in texts: t.set_color(TEXT); t.set_fontsize(9)
    for a in autos: a.set_color(BG); a.set_fontsize(8); a.set_fontweight("bold")
    ax.add_patch(plt.Circle((0,0),.55,color=BG))
    ax.text(0,0,"Portfolio\nallocation",ha="center",va="center",
            fontsize=9,color=MUTED,linespacing=1.5)
    ax.set_title("Holdings by price weight",color=MUTED,fontsize=10,pad=12)
    fig.tight_layout(); return fig_to_b64(fig)

# ── HTML builder ───────────────────────────────────────────────────────────
def badge(label,bg,fg):
    return (f'<span style="background:{bg};color:{fg};padding:2px 10px;'
            f'border-radius:20px;font-size:11px;font-weight:700">{label}</span>')

VERDICT_BADGE={
    "BUY":  lambda: badge("🟢 BUY","#1a4731",GREEN),
    "WATCH":lambda: badge("🟡 WATCH","#3d2800",AMBER),
    "WAIT": lambda: badge("⏳ WAIT","#21262d",MUTED),
    "PASS": lambda: badge("🔴 PASS","#4a1b1b",RED),
}

def img_tag(b64):
    return (f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;border-radius:8px;margin-bottom:20px">'
            if b64 else "")

def build_html(pulse,port_data,watchlist_cfg,watch_prices,screener,earnings,
               b64_range,b64_watch,b64_alloc):

    # Market pulse strip
    pulse_cells=""
    for name,d in pulse.items():
        is_fear="VIX" in name or "Yield" in name
        good=(d["pct"]<0) if is_fear else (d["pct"]>0)
        color=GREEN if good else RED
        sign="+" if d["pct"]>0 else ""
        pulse_cells+=f"""
        <div style="text-align:center;padding:12px 20px;
                    border-right:1px solid #21262d">
          <div style="font-size:10px;color:{MUTED};text-transform:uppercase;
                      letter-spacing:.07em;margin-bottom:4px">{name}</div>
          <div style="font-size:20px;font-weight:700;color:{TEXT}">{d['value']:,.2f}</div>
          <div style="font-size:13px;color:{color};font-weight:600">
            {sign}{d['pct']:.2f}%</div>
        </div>"""

    # Portfolio table rows
    port_rows=""
    for s in port_data:
        tc=GREEN if s["pct_today"]>=0 else RED
        ts=fmt_pct(s["pct_today"])
        uc=GREEN if (s.get("upside") or 0)>=15 else (AMBER if (s.get("upside") or 0)>0 else RED)
        rsi_c=(RED if (s.get("rsi") or 0)>75 else
               GREEN if (s.get("rsi") or 0)<35 else TEXT)
        earn_s=""
        if s.get("earnings_dt"):
            days=(s["earnings_dt"]-date.today()).days
            if 0<=days<=45:
                earn_s=(f'<span style="background:#0c2d6b;color:{BLUE_LT};'
                        f'padding:1px 7px;border-radius:10px;font-size:10px">'
                        f'Earns {s["earnings_dt"].strftime("%b %d")} ({days}d)</span>')
        port_rows+=f"""<tr>
          <td style="font-weight:700;color:{TEXT}">{s['ticker']}</td>
          <td style="color:{MUTED};font-size:12px">{s['name'][:26]}</td>
          <td style="font-weight:700">{fmt_price(s['price'])}</td>
          <td style="color:{tc};font-weight:600">{ts}</td>
          <td>{fmt_price(s.get('tgt_mean'))}</td>
          <td style="color:{uc};font-weight:600">{fmt_pct(s.get('upside')) if s.get('upside') else '—'}</td>
          <td style="color:{MUTED};font-size:12px">{s.get('buy_pct','—')}{'% buy' if s.get('buy_pct') else ''}</td>
          <td>{fmt_pct(s.get('rev_growth')) if s.get('rev_growth') else '—'}</td>
          <td style="color:{rsi_c}">{s.get('rsi','—')}</td>
          <td style="font-size:12px">{earn_s}</td>
        </tr>"""

    # Watch list cards
    watch_html=""
    for ticker,cfg in watchlist_cfg.items():
        price=watch_prices.get(ticker); target=cfg.get("buy_at")
        if not price: continue
        direction=cfg.get("direction","below")
        if direction=="below":
            pct_away=(price-target)/target*100; verb="above target"
        else:
            pct_away=(target-price)/price*100; verb="to target"
        close=pct_away<=5
        pct_color=GREEN if close else MUTED
        alert=(f'<div style="color:{GREEN};font-size:12px;font-weight:700;margin-top:6px">'
               f'🔔 APPROACHING — consider acting now</div>' if close else "")
        watch_html+=f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};
                    border-radius:8px;padding:14px 16px;margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;align-items:baseline">
            <span style="font-size:17px;font-weight:700;color:{TEXT}">{ticker}</span>
            <span style="color:{pct_color};font-size:13px;font-weight:600">
              {pct_away:.1f}% {verb}</span>
          </div>
          <div style="margin-top:5px;font-size:13px;color:{MUTED}">
            Current <strong style="color:{TEXT}">{fmt_price(price)}</strong>
            &nbsp;·&nbsp; Target <strong style="color:{BLUE_LT}">{fmt_price(target)}</strong>
          </div>
          <div style="margin-top:5px;font-size:12px;color:#6e7681">{cfg.get('note','')}</div>
          {alert}
        </div>"""

    # Screener cards
    screener_html=""
    for s in screener:
        vd=s.get("verdict_data",{}); v=vd.get("verdict","")
        vbadge=(VERDICT_BADGE.get(v,lambda: badge(v,"#21262d",MUTED)))()
        passes_li="".join(f'<li style="color:{GREEN}">✓ {p}</li>'
                          for p in vd.get("passes",[]))
        fails_li="".join(f'<li style="color:{AMBER}">· {f}</li>'
                         for f in vd.get("fails",[]))
        kills_li="".join(f'<li style="color:{RED}">✗ {k}</li>'
                         for k in vd.get("kills",[]))
        news_html="".join(
            f'<div style="font-size:11px;color:#6e7681;margin-top:3px">'
            f'» <a href="{n["url"]}" style="color:{BLUE_LT};text-decoration:none">'
            f'{n["title"][:90]}…</a></div>'
            for n in s.get("news",[])[:2])
        tc=GREEN if s["pct_today"]>=0 else RED
        screener_html+=f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};
                    border-radius:8px;padding:16px;margin-bottom:12px">
          <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
              <span style="font-size:19px;font-weight:700;color:{TEXT}">{s['ticker']}</span>
              <span style="font-size:13px;color:{MUTED};margin-left:8px">{s['name']}</span>
            </div>
            {vbadge}
          </div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);
                      gap:8px;margin:12px 0;font-size:12px">
            <div style="background:{BG};padding:8px;border-radius:6px">
              <div style="color:{MUTED}">Price</div>
              <div style="color:{TEXT};font-weight:700">{fmt_price(s['price'])}</div>
              <div style="color:{tc}">{fmt_pct(s['pct_today'])} today</div>
            </div>
            <div style="background:{BG};padding:8px;border-radius:6px">
              <div style="color:{MUTED}">PT upside</div>
              <div style="color:{GREEN};font-weight:700">{fmt_pct(s.get('upside')) if s.get('upside') else '—'}</div>
              <div style="color:#6e7681">{s.get('n_analysts',0)} analysts</div>
            </div>
            <div style="background:{BG};padding:8px;border-radius:6px">
              <div style="color:{MUTED}">Rev growth</div>
              <div style="color:{GREEN};font-weight:700">{fmt_pct(s.get('rev_growth')) if s.get('rev_growth') else '—'}</div>
              <div style="color:#6e7681">YoY trailing</div>
            </div>
            <div style="background:{BG};padding:8px;border-radius:6px">
              <div style="color:{MUTED}">Below high</div>
              <div style="color:{TEXT};font-weight:700">{fmt_pct(s.get('from_hi')) if s.get('from_hi') else '—'}</div>
              <div style="color:#6e7681">{s.get('buy_pct','—')}{'% buy' if s.get('buy_pct') else ''}</div>
            </div>
          </div>
          <ul style="margin:0 0 8px;padding-left:16px;font-size:12px;line-height:1.9">
            {passes_li}{fails_li}{kills_li}
          </ul>
          {news_html}
        </div>"""

    if not screener_html:
        screener_html=(f'<div style="color:{MUTED};padding:16px">'
                       'No new candidates pass this week.</div>')

    # Earnings rows
    earn_rows=""
    for e in earnings[:20]:
        days=(e["date"]-date.today()).days
        uc=RED if days<=7 else (AMBER if days<=14 else MUTED)
        earn_rows+=f"""<tr>
          <td style="font-weight:700;color:{TEXT}">{e['ticker']}</td>
          <td>{e['date'].strftime('%b %d, %Y')}</td>
          <td style="color:{uc};font-weight:{'700' if days<=7 else '400'}">{days}d</td>
        </tr>"""
    if not earn_rows:
        earn_rows=f'<tr><td colspan="3" style="color:{MUTED}">No earnings in next 45 days</td></tr>'

    css=f"""
    *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         background:{BG};color:{TEXT};line-height:1.5;font-size:14px}}
    .wrap{{max-width:880px;margin:0 auto;padding:0 16px 48px}}
    .sec{{margin-bottom:32px}}
    .sec-title{{font-size:11px;font-weight:600;letter-spacing:.1em;
               text-transform:uppercase;color:{MUTED};
               border-bottom:1px solid #21262d;padding-bottom:8px;margin-bottom:16px}}
    table{{width:100%;border-collapse:collapse;font-size:13px}}
    th{{color:{MUTED};font-weight:600;font-size:11px;text-transform:uppercase;
       letter-spacing:.05em;padding:7px 8px;border-bottom:1px solid #21262d;text-align:left}}
    td{{padding:8px 8px;border-bottom:1px solid {SURFACE};vertical-align:middle}}
    tr:last-child td{{border-bottom:none}}
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Market Brief — {DATE}</title>
<style>{css}</style>
</head>
<body><div class="wrap">

<div style="background:#0c2d6b;border-radius:10px;
            padding:28px 28px 22px;margin:24px 0 28px">
  <div style="font-size:10px;font-weight:700;letter-spacing:.18em;
              text-transform:uppercase;color:rgba(255,255,255,.5);margin-bottom:6px">
    Weekly Market Brief</div>
  <div style="font-size:26px;font-weight:700;color:#fff">{DATE}</div>
  <div style="font-size:12px;color:rgba(255,255,255,.55);margin-top:4px">
    yfinance · no AI APIs · generated {NOW.strftime('%H:%M ET')}</div>
</div>

<div class="sec">
  <div class="sec-title">Market pulse</div>
  <div style="display:flex;background:{SURFACE};border:1px solid {BORDER};
              border-radius:8px;overflow:hidden">{pulse_cells}</div>
</div>

<div class="sec">
  <div class="sec-title">Portfolio — 52-week range</div>
  {img_tag(b64_range)}
</div>

<div class="sec">
  <div class="sec-title">Portfolio snapshot</div>
  <div style="background:{SURFACE};border:1px solid {BORDER};
              border-radius:8px;overflow:hidden">
  <table>
    <thead><tr>
      <th>Ticker</th><th>Name</th><th>Price</th><th>Today</th>
      <th>PT</th><th>Upside</th><th>Consensus</th>
      <th>Rev growth</th><th>RSI</th><th></th>
    </tr></thead>
    <tbody>{port_rows}</tbody>
  </table></div>
</div>

<div class="sec">
  <div class="sec-title">Watch list — entry targets</div>
  {img_tag(b64_watch)}
  {watch_html}
</div>

<div class="sec">
  <div class="sec-title">Screener — new candidates this week</div>
  <div style="font-size:12px;color:#6e7681;margin-bottom:14px">
    {len(UNIVERSE)} tickers screened · growth mode criteria:
    rev &gt;20%, buy ≥80%, 0–1 sells, price 10–40% off 52wk high, PT upside ≥15%
  </div>
  {screener_html}
</div>

<div class="sec">
  <div class="sec-title">Earnings calendar — next 45 days</div>
  <div style="background:{SURFACE};border:1px solid {BORDER};
              border-radius:8px;overflow:hidden">
  <table>
    <thead><tr><th>Ticker</th><th>Date</th><th>Days away</th></tr></thead>
    <tbody>{earn_rows}</tbody>
  </table></div>
</div>

<div class="sec">
  <div class="sec-title">Portfolio allocation</div>
  <div style="max-width:360px;margin:0 auto">{img_tag(b64_alloc)}</div>
</div>

<div style="border-top:1px solid #21262d;padding-top:16px;
            font-size:11px;color:#6e7681;text-align:center">
  Not financial advice · data via yfinance / Yahoo Finance ·
  {NOW.strftime('%Y-%m-%d %H:%M')} ET
</div>
</div></body></html>"""

# ── GitHub Actions summary ─────────────────────────────────────────────────
def write_gha_summary(pulse, port_data, triggered, screener, earnings):
    path=os.environ.get("GITHUB_STEP_SUMMARY")
    if not path: return
    lines=[f"# 📊 Market Brief — {DATE}\n"]

    lines+=["## Market pulse\n","| Index | Value | Change |","|---|---|---|"]
    for name,d in pulse.items():
        s="+" if d["pct"]>0 else ""
        lines.append(f"| {name} | {d['value']:,.2f} | {s}{d['pct']:.2f}% |")
    lines.append("")

    if triggered:
        lines+=["## 🔔 Watch price alerts\n"]
        for t in triggered:
            lines.append(f"- **{t['ticker']}** — {t['reason']}")
        lines.append("")

    buys=[s for s in screener if s.get("verdict_data",{}).get("verdict")=="BUY"]
    if buys:
        lines+=["## 🟢 New screener buys\n"]
        for s in buys:
            lines.append(f"- **{s['ticker']}** — {fmt_price(s['price'])} · "
                         f"PT upside {fmt_pct(s.get('upside'))} · "
                         f"Rev {fmt_pct(s.get('rev_growth'))}")
        lines.append("")

    lines+=["## Portfolio snapshot\n",
            "| Ticker | Price | Today | PT upside | Rev growth |",
            "|---|---|---|---|---|"]
    for s in port_data:
        lines.append(f"| {s['ticker']} | {fmt_price(s['price'])} | "
                     f"{fmt_pct(s['pct_today'])} | "
                     f"{fmt_pct(s.get('upside')) if s.get('upside') else '—'} | "
                     f"{fmt_pct(s.get('rev_growth')) if s.get('rev_growth') else '—'} |")
    lines.append("")

    if earnings:
        lines+=["## Upcoming earnings\n"]
        for e in earnings[:12]:
            days=(e["date"]-date.today()).days
            lines.append(f"- **{e['ticker']}** — {e['date'].strftime('%b %d')} ({days}d)")
        lines.append("")

    lines.append("---\n*Full report saved as `report.html` artifact.*")
    with open(path,"w") as f: f.write("\n".join(lines))

# ── Discord ────────────────────────────────────────────────────────────────
def send_discord(webhook, msg):
    if not webhook: return
    try: requests.post(webhook,json={"content":msg[:2000]},timeout=10)
    except Exception as e: print(f"  Discord: {e}")

def discord_alert(triggered, screener):
    buys=[s for s in screener if s.get("verdict_data",{}).get("verdict")=="BUY"]
    if not triggered and not buys: return ""
    lines=[f"**📊 Market Scanner — {NOW.strftime('%b %d')}**\n"]
    if triggered:
        lines.append("🔔 **Watch price alerts**")
        for t in triggered: lines.append(f"  • **{t['ticker']}** {t['reason']}")
    if buys:
        lines.append("\n🟢 **New screener finds**")
        for s in buys:
            lines.append(f"  • **{s['ticker']}** {fmt_price(s['price'])} · "
                         f"PT upside {fmt_pct(s.get('upside'))}")
    lines.append("\n_Full report in Actions → Artifacts_")
    return "\n".join(lines)

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    cfg=load_config()
    print(f"[scanner] {DATE}")

    port_tickers=cfg.get("portfolio",[])
    watchlist_cfg=cfg.get("watchlist",{})
    move_pct=cfg.get("alert_move_pct",5.0)
    do_screener=(datetime.today().weekday()==cfg.get("weekly_scan_weekday",0) or
                 cfg.get("run_screener_daily",False))
    webhook=os.environ.get("DISCORD_WEBHOOK",cfg.get("discord_webhook",""))

    print("Fetching market pulse…")
    pulse=get_market_pulse()

    print("Fetching portfolio…")
    port_data=[]
    for t in port_tickers:
        print(f"  {t}",end=" ",flush=True)
        s=get_stock(t)
        if s: port_data.append(s); print("✓")
        else: print("✗")

    print("Checking watchlist…")
    watch_prices={}
    for t in watchlist_cfg:
        s=get_stock(t)
        if s: watch_prices[t]=s["price"]; print(f"  {t}: ${s['price']}")

    triggered=[]
    for ticker,cfg_item in watchlist_cfg.items():
        price=watch_prices.get(ticker); target=cfg_item.get("buy_at")
        if price and target:
            direction=cfg_item.get("direction","below")
            if ((direction=="below" and price<=target) or
                (direction=="above" and price>=target)):
                triggered.append({"ticker":ticker,"price":price,"target":target,
                                   "reason":f"${price} hit target ${target}"})

    screener=[]
    if do_screener:
        print("\nRunning screener…")
        screener=run_screener(port_tickers)

    print("\nBuilding earnings calendar…")
    all_earn_tickers=list(set(port_tickers+list(watchlist_cfg.keys())))
    earnings=get_earnings_calendar(all_earn_tickers)
    print(f"  {len(earnings)} events in 45 days")

    print("Generating charts…")
    b64_range=chart_range(port_data)
    b64_watch=chart_watchlist(watchlist_cfg,watch_prices)
    b64_alloc=chart_alloc(port_data)

    print("Building HTML report…")
    html=build_html(pulse,port_data,watchlist_cfg,watch_prices,
                    screener,earnings,b64_range,b64_watch,b64_alloc)

    out=Path("report.html"); out.write_text(html,encoding="utf-8")
    print(f"  → {out} ({len(html)//1024}KB)")

    write_gha_summary(pulse,port_data,triggered,screener,earnings)

    msg=discord_alert(triggered,screener)
    if msg: print("\nSending Discord alert…"); send_discord(webhook,msg)
    else: print("\nNo urgent signals — Discord silent")

    print("\n[done]")

if __name__=="__main__":
    main()
