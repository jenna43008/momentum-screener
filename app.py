import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math
import random

# =============================================================
# SETTINGS (unchanged)
# =============================================================
THREADS               = 20
AUTO_REFRESH_MS       = 60_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 5.0
DEFAULT_MIN_VOLUME    = 100_000
DEFAULT_MIN_BREAKOUT  = 0.0

st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v9")

st.set_page_config(page_title="V9 Momentum + Trend + AI Commentary", layout="wide")
st.title("🚀 V9 — Momentum Screener + Trend + Alignment + AI Commentary (Restored)")
st.caption("Everything returned — Trend + Alignment + Volume + Hybrid AI Insight")

# =============================================================
# SIDEBAR (unchanged UI — only reads volume)
# =============================================================
with st.sidebar:
    st.header("Universe")
    watchlist_text = st.text_area("Watchlist tickers:", "")
    max_universe = st.slider("Max scan size", 50, 600, 200, 50)
    universe_mode = st.radio("Universe mode",
        ["Classic (Alphabetical Slice)", "Randomized Slice", "Live Volume Ranked (slower)"]
    )
    volume_rank_pool = st.slider("Live Volume Rank Pool",100,2000,600,100)

    enable_enrichment = st.checkbox("Float/Short/News (slower)",False)

    st.header("Filters")
    max_price    = st.number_input("Max Price ($)",1.0,1000.0,DEFAULT_MAX_PRICE,1.0)
    min_volume   = st.number_input("Min Volume",10_000,10_000_000,DEFAULT_MIN_VOLUME,10_000)
    min_breakout = st.number_input("Min Score",-50.0,200.0,0.0,1.0)
    min_pm_move  = st.number_input("Min PM%",-50,200,0.0,0.5)
    min_yday_gain= st.number_input("Min Yesterday%",-50,200,0.0,0.5)

    squeeze_only  = st.checkbox("Short Squeeze Only")
    catalyst_only = st.checkbox("Must Have Catalyst")
    vwap_only     = st.checkbox("Above VWAP Only")

    # ---- Order Flow Filter ----
    enable_ofb_filter = st.checkbox("Enable OFB filter",False)
    min_ofb = st.slider("Min Order Flow Bias",0.00,1.00,0.50,0.01)

    # ---- Alerts (unchanged) ----
    st.subheader("🔊 Alerts")
    enable_alerts = st.checkbox("Enable Alerts",True)
    ALERT_SCORE_THRESHOLD = st.slider("Alert Score≥",10,200,30,5)
    ALERT_PM_THRESHOLD    = st.slider("Alert PM≥",1,150,4,1)
    ALERT_VWAP_THRESHOLD  = st.slider("Alert VWAP≥",1,50,2,1)

# =============================================================
# LOAD SYMBOLS
# =============================================================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq=pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",sep="|",skipfooter=1,engine="python")
    other =pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",sep="|",skipfooter=1,engine="python")
    nasdaq["Exchange"]="NASDAQ"
    other = other.rename(columns={"ACT Symbol":"Symbol"})
    df=pd.concat([nasdaq[["Symbol","ETF","Exchange"]],other[["Symbol","ETF","Exchange"]]])
    return df[df.Symbol.str.contains(r"^[A-Z]{1,5}$", na=False)].to_dict("records")

# =============================================================
# BUILD UNIVERSE (V9 modes intact)
# =============================================================
def build_universe(w,max_u,mode,pool):
    wl=w.strip()
    if wl:
        tick=sorted(set(w.upper().replace("\n"," ").replace(","," ").split()))
        return [{"Symbol":t} for t in tick]

    syms=load_symbols()
    if mode=="Randomized Slice":
        random.shuffle(syms);return syms[:max_u]

    if mode=="Live Volume Ranked (slower)":
        ranked=[]
        for s in syms[:pool]:
            try:
                d=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m",prepost=True)
                if not d.empty: ranked.append({**s,"Vol":float(d.Volume.iloc[-1])})
            except:pass
        return sorted(ranked,key=lambda x:x.get("Vol",0),reverse=True)[:max_u]

    return syms[:max_u]

# =============================================================
# AI Commentary
# =============================================================
def ai_comment(row):
    thoughts=[]
    if row["RVOL_10D"] and row["RVOL_10D"]>2: thoughts.append("High demand expansion")
    if row["VWAP%"]  and row["VWAP%"]>0:     thoughts.append("Trading above VWAP — strength")
    if row["FlowBias"] and row["FlowBias"]>0.6: thoughts.append("Buyers in control")
    if row["10D%"] and row["10D%"]>8: thoughts.append("10-day structural uptrend")
    return " | ".join(thoughts) if thoughts else "Neutral — waiting confirmation"

# =============================================================
# SCAN — Trend + Alignment restored
# =============================================================
def scan_one(sym,enrich,ofb,min_ofb):
    try:
        t=sym["Symbol"]
        stock=yf.Ticker(t)
        hist=stock.history(period="10d",interval="1d")
        if hist.empty:return None

        close=hist.Close; vol=hist.Volume
        price=float(close.iloc[-1]); dv=vol.iloc[-1]
        if price>max_price or dv<min_volume:return None

        yday=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>1 else None
        m3=(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>3 else None
        m10=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        # RSI + Trend restored
        rs=(close.diff().clip(lower=0).rolling(7).mean())/((-close.diff().clip(upper=0)).rolling(7).mean())
        rsi=float((100-(100/(1+rs))).iloc[-1])
        ema10=float(close.ewm(span=10,adjust=False).mean().iloc[-1])
        trend="🔥 Breakout" if price>ema10 and rsi>55 else "Neutral"

        rvol=float(dv/vol.mean())

        # Intraday signals
        pm=vwap=o=None
        intra=stock.history(period="1d",interval="2m",prepost=True)
        if len(intra)>=3:
            pm=(intra.Close.iloc[-1]-intra.Close.iloc[-2])/intra.Close.iloc[-2]*100
            typ=(intra.High+intra.Low+intra.Close)/3
            if intra.Volume.sum()>0:
                vwap=(price-((typ*intra.Volume).sum()/intra.Volume.sum()))/price*100

            sign=(intra.Close>intra.Open).astype(int)-(intra.Close<intra.Open).astype(int)
            buy=(intra.Volume*(sign>0)).sum();sell=(intra.Volume*(sign<0)).sum()
            o=buy/(buy+sell) if buy+sell>0 else None

        if ofb and (o is None or o<min_ofb): return None

        score=round(max(pm or 0,0)*1.6+max(yday or 0,0)*0.8+max(m3 or 0,0)*1.2+max(m10 or 0,0)*0.6+
                     ((rsi-55)*0.4 if rsi>55 else 0)+
                     ((rvol-1.2)*2 if rvol>1.2 else 0)+
                     ((min(vwap,6))*1.5 if vwap and vwap>0 else 0)+
                     ((o-0.5)*22 if o else 0),2)
        prob=round((1/(1+math.exp(-score/20)))*100,1)

        # ✳ RESTORED MTF ALIGNMENT EXACTLY
        alignment = (
            "✅ Aligned Bullish" if sum([
                (pm and pm>0),(m3 and m3>0),(m10 and m10>0)
            ])>=2 else "🟡 Mixed"
        )

        return {
            "Symbol":t,"Price":round(price,2),"Volume":int(dv),
            "Score":score,"Prob_Rise%":prob,
            "PM%":round(pm,2) if pm else None,
            "YDay%":round(yday,2) if yday else None,
            "3D%":round(m3,2) if m3 else None,
            "10D%":round(m10,2),
            "RSI7":round(rsi,2),
            "RVOL_10D":round(rvol,2),
            "VWAP%":round(vwap,2) if vwap else None,
            "FlowBias":round(o,2) if o else None,
            "Trend":trend,               # 🔥 restored
            "MTF_Trend":alignment,        # 🔥 restored
            "Spark":close,
            "AI":ai_comment({
                "RVOL_10D":rvol,"VWAP%":vwap,"FlowBias":o,"PM%":pm,"10D%":m10
            })
        }
    except:
        return None

@st.cache_data(ttl=6)
def run_scan(w,mu,en,ofb,m_ofb,mode,pool):
    uni=build_universe(w,mu,mode,pool)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,u,en,ofb,m_ofb) for u in uni]):
            r=f.result()
            if r:out.append(r)
    return pd.DataFrame(out)


