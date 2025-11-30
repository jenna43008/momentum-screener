import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math
import random

# ========================= SETTINGS =========================
THREADS               = 20
AUTO_REFRESH_MS       = 10_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"
DEFAULT_MAX_PRICE     = 50.0
DEFAULT_MIN_VOLUME    = 100_000
DEFAULT_MIN_BREAKOUT  = 0.0

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v8")

# ========================= PAGE SETUP =========================
st.set_page_config(page_title="V8.9 — Volume-Ranked Screener",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.title("🚀 V8.9 — 10-Day Momentum Breakout Screener (Real-Time Volume Ranked)")
st.caption("Now scans highest-volume movers instead of A-tickers 🔥")

# ========================= SIDEBAR CONTROLS =========================
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area(
        "Watchlist (comma/newline separated):", "",
        help="Example: TSLA, AMD, NVDA, META"
    )

    max_universe = st.slider("Max symbols (no watchlist):", 50, 600, 200, 50)

    region_mode = st.radio(
        "Region Filter",
        ["Global (no filter)", "US + Canada Only"],
        index=1
    )

    enable_enrichment = st.checkbox("Float/Short/News Enrichment", value=False)

    st.markdown("---")
    st.header("Filters")

    max_price = st.number_input("Max Price ($)", 1.0, 1000.0, DEFAULT_MAX_PRICE, 1.0)
    min_volume = st.number_input("Min Volume", 10_000, 10_000_000, DEFAULT_MIN_VOLUME, 10_000)
    min_breakout = st.number_input("Min Score", -50.0, 200.0, 0.0, 1.0)

    min_pm_move = st.number_input("Min Premarket %", -50, 200, 0, 0.5)
    min_yday_gain = st.number_input("Min Yesterday %", -50, 200, 0, 0.5)

    squeeze_only  = st.checkbox("Short-Squeeze Only")
    catalyst_only = st.checkbox("Must Have Recent News")
    vwap_only     = st.checkbox("Above VWAP Only")

    min_ofb = st.slider("Min Order Flow Bias", 0.0, 1.0, 0.50, 0.01)

    st.markdown("---")
    st.subheader("🔊 Audio Alerts")

    ALERT_SCORE_THRESHOLD = st.slider("Alert: Score ≥", 10, 200, 30, 5)
    ALERT_PM_THRESHOLD    = st.slider("Alert: Premarket ≥ %", 1, 150, 4)
    ALERT_VWAP_THRESHOLD  = st.slider("Alert: VWAP Dist ≥ %", 1, 50, 2)

    if st.button("🧹 Refresh Now"):
        st.cache_data.clear()
        st.success("Cache cleared — rescanning...")

# ========================= SYMBOL LOADING =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                          sep="|", skipfooter=1, engine="python")
    other  = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                          sep="|", skipfooter=1, engine="python")

    nasdaq["Exchange"]="NASDAQ"
    other["Exchange"]=other["Exchange"].fillna("NYSE/AMEX/ARCA")
    other = other.rename(columns={"ACT Symbol":"Symbol"})

    df = pd.concat([nasdaq[["Symbol","ETF","Exchange"]],
                    other[["Symbol","ETF","Exchange"]]]).dropna(subset=["Symbol"])

    df = df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$", na=False)]
    df["Country"] = "US"
    return df.to_dict("records")


# ========================= 🚀 FIXED — REAL-TIME UNIVERSE =========================
def build_universe(watchlist_text, max_universe):
    wl = watchlist_text.strip()
    if wl:
        lst = wl.replace("\n"," ").replace(","," ").split()
        return [{"Symbol":t.upper(),"Exchange":"WATCH"} for t in lst]

    all_syms = load_symbols()

    # 🔥 Rank by real-time volume instead of alphabetical
    def quick_volume(stockdict):
        try:
            return yf.Ticker(stockdict["Symbol"]).fast_info.get("last_volume",0)
        except:
            return 0

    ranked = sorted(all_syms, key=quick_volume, reverse=True)
    return ranked[:max_universe]     # ← NOW REAL-MARKET SCAN 🔥🔥


# ========================= CORE ENGINE =========================
def scan_one(sym, enrich, region_mode, min_ofb):
    try:
        t = sym["Symbol"]
        stock = yf.Ticker(t)
        hist = stock.history(period="10d",interval="1d")
        if hist.empty or len(hist)<5: return None

        close=hist["Close"]; vol=hist["Volume"]
        price=float(close.iloc[-1]); last_vol=float(vol.iloc[-1])

        if price>max_price or last_vol<min_volume: return None

        # Calculate signals
        yday = ((close.iloc[-1]/close.iloc[-2]-1)*100) if len(close)>=2 else None
        m3   = ((close.iloc[-1]/close.iloc[-4]-1)*100) if len(close)>=4 else None
        m10  = ((close.iloc[-1]/close.iloc[0]-1)*100)

        rsi7 = 100-(100/(1+(close.diff().clip(lower=0).rolling(7).mean()/(-close.diff().clip(upper=0).rolling(7).mean()))))
        rsi7=float(rsi7.iloc[-1])

        ema=close.ewm(span=10,adjust=False).mean().iloc[-1]
        ema_tag="🔥 Breakout" if price>ema and rsi7>55 else "Neutral"

        avg10=vol.mean(); rvol=last_vol/avg10 if avg10>0 else None

        intra=stock.history(period="1d",interval="2m",prepost=True)
        pre=vwap=ofb=None
        if len(intra)>3:
            pre=(intra["Close"].iloc[-1]/intra["Close"].iloc[-2]-1)*100
            tp=(intra["High"]+intra["Low"]+intra["Close"])/3
            vwap=(tp*intra["Volume"]).sum()/intra["Volume"].sum()
            vwap=(price-vwap)/vwap*100 if vwap else None
            s=(intra["Close"]>intra["Open"]).astype(int)-(intra["Close"]<intra["Open"]).astype(int)
            buy=(intra["Volume"]*(s>0)).sum(); sell=(intra["Volume"]*(s<0)).sum()
            ofb=buy/(buy+sell) if buy+sell>0 else None

        # 🔥 exclude weak OFB
        if ofb is None or ofb < min_ofb: return None

        # Score model
        score=0
        if pre: score+=max(pre,0)*1.6
        if yday: score+=max(yday,0)*0.8
        if m3: score+=max(m3,0)*1.2
        if m10: score+=max(m10,0)*0.6
        if rsi7>55: score+=(rsi7-55)*0.4
        if rvol and rvol>1.2: score+=(rvol-1.2)*2
        if vwap and vwap>0: score+=min(vwap,6)*1.5
        if ofb: score+=(ofb-0.5)*22

        prob=1/(1+math.exp(-score/20))*100

        return {"Symbol":t,"Price":round(price,2),"Vol":int(last_vol),
                "Score":round(score,2),"Prob_Rise%":round(prob,1),
                "PM%":round(pre,2) if pre else None,"YDay%":round(yday,2) if yday else None,
                "3D%":round(m3,2) if m3 else None,"10D%":round(m10,2) if m10 else None,
                "RSI7":round(rsi7,2),"RVOL":round(rvol,2) if rvol else None,
                "VWAP%":round(vwap,2) if vwap else None,"FlowBias":round(ofb,2) if ofb else None,
                "Trend":ema_tag,"Spark":close}

    except: return None

@st.cache_data(ttl=6)
def run_scan(w,l,enh,rgn,min_ofb):
    uni=build_universe(w,l)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s,enh,rgn,min_ofb) for s in uni]):
            r=f.result()
            if r: out.append(r)
    return pd.DataFrame(out)
