import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math

# ========================= SETTINGS =========================
THREADS               = 20
AUTO_REFRESH_MS       = 10_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 50
DEFAULT_MIN_VOLUME    = 100_000

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v8_5")

# ========================= PAGE SETUP =========================
st.set_page_config(
    page_title="V8.5 — Live Momentum Screener + Order Flow Meter",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 V8.5 — Live Momentum Screener w/ Order Flow Bias Heat Meter + Alerts")
st.caption("10-Day Model • VWAP + Micro OVB Meter • Audio Alerts • Sparkline Charts • Real-Time Buy/Sell Pressure")

# ========================= SIDEBAR =========================
with st.sidebar:

    st.header("Filters")

    max_price = st.number_input("Max Price ($)", 1, 1000, DEFAULT_MAX_PRICE)
    min_vol   = st.number_input("Min Daily Volume", 10_000, 10_000_000, DEFAULT_MIN_VOLUME)

    min_breakout = st.slider("Min Score", 0, 200, 0)
    min_PM = st.slider("Min Premarket %", 0, 100, 0)
    min_Yday = st.slider("Min Yesterday %", 0, 100, 0)

    squeeze_only = st.checkbox("Short Squeeze Only")
    vwap_only    = st.checkbox("Above VWAP Only")

    st.markdown("---")
    st.subheader("🔊 Audio Alert Conditions")

    ALERT_SCORE = st.slider("Alert if Score ≥", 5, 200, 30)
    ALERT_PM    = st.slider("Alert if PM% ≥", 1, 150, 5)
    ALERT_VWAP  = st.slider("Alert if VWAP% ≥", 1, 30, 2)

    # >>> NEW ALERT FOR ORDER FLOW BIAS <<<
    ALERT_OFB   = st.slider("Alert if OFB ≥", 0.50, 1.00, 0.70, step=0.01)
    st.caption("Higher value = stronger buy pressure required to alert 🚨")

    st.markdown("---")
    if st.button("Manual Refresh"):
        st.cache_data.clear()
        st.experimental_rerun()

# ========================= SYMBOLS =========================
@st.cache_data(ttl=1200)
def load_universe():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", sep="|", skipfooter=1, engine="python")
    other  = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", sep="|", skipfooter=1, engine="python")

    nasdaq["Exchange"] = "NASDAQ"
    other["Exchange"]  = "NYSE/AMEX/ARCA"
    other = other.rename(columns={"ACT Symbol": "Symbol"})

    df = pd.concat([nasdaq[["Symbol","Exchange"]], other[["Symbol","Exchange"]]]).dropna()
    df = df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$")]
    return df.to_dict("records")

# ========================= ORDER FLOW VISUAL =========================
def ofb_graph(value: float):
    """Creates a horizontal bar meter showing buy/sell pressure visually."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[value], y=["OFB"], orientation="h",
        marker=dict(color="green" if value >= 0.55 else "orange" if value >= 0.50 else "red")
    ))
    fig.update_layout(height=40, width=180, xaxis=dict(range=[0,1],visible=False),yaxis=dict(visible=False))
    return fig

# ========================= CORE LOGIC =========================
def scan_one(sym):
    try:
        ticker = sym["Symbol"]
        t = yf.Ticker(ticker)

        hist = t.history(period="10d", interval="1d")
        if hist.empty: return None

        price = hist["Close"].iloc[-1]
        vol   = hist["Volume"].iloc[-1]

        if price > max_price or vol < min_vol: return None

        # === Momentum Windows ===
        yday = (hist["Close"].iloc[-1] - hist["Close"].iloc[-2]) / hist["Close"].iloc[-2] * 100 if len(hist)>2 else None
        m3   = (hist["Close"].iloc[-1] - hist["Close"].iloc[-4]) / hist["Close"].iloc[-4] * 100 if len(hist)>4 else None

        # RSI7
        delta = hist["Close"].diff()
        gain  = delta.clip(lower=0).rolling(7).mean()
        loss  = (-delta.clip(upper=0)).rolling(7).mean()
        rsi   = float((100 - (100/(1+ gain/loss))).iloc[-1])

        # Intraday for OFB + VWAP
        intra = t.history(period="1d", interval="2m", prepost=True)
        if intra.empty: return None

        last = intra["Close"].iloc[-1]
        prev = intra["Close"].iloc[-2]
        PM   = (last-prev)/prev*100 if prev>0 else None

        # VWAP
        typical = (intra["High"]+intra["Low"]+intra["Close"])/3
        VWAP = (typical*intra["Volume"]).sum()/intra["Volume"].sum()
        vdist = (price-VWAP)/VWAP*100 if VWAP>0 else None

        # === ORDER FLOW BIAS (0-1) ===
        direction = (intra["Close"]>intra["Open"]).astype(int) - (intra["Close"]<intra["Open"]).astype(int)
        buy_v  = float((intra["Volume"]*(direction>0)).sum())
        sell_v = float((intra["Volume"]*(direction<0)).sum())
        ofb = buy_v/(buy_v+sell_v) if buy_v+sell_v>0 else None

        # === SCORE (short-term model) ===
        score = 0
        score+= max(PM,0)*1.2 if PM else 0
        score+= max(yday,0)*0.7 if yday else 0
        score+= max(m3,0)*1.1 if m3 else 0
        score+= (rsi-55)*0.4 if rsi>55 else 0
        score+= (vdist)*0.8 if vdist and vdist>0 else 0
        score+= (ofb-0.55)*30 if ofb and ofb>0.55 else 0
        score = round(score,2)

        return {
            "Symbol": ticker,
            "Price": round(price,2),
            "Score": score,
            "PM%": round(PM,2),
            "YDay%": round(yday,2) if yday else None,
            "3D%": round(m3,2) if m3 else None,
            "RSI7": round(rsi,2),
            "VWAP%": round(vdist,2) if vdist else None,
            "OFB": round(ofb,2) if ofb else None,
            "Spark": hist["Close"],
        }

    except:
        return None

# ========================= RUN =========================
with st.spinner("🔄 Scanning market w/ Order Flow Bias meter…"):
    data = load_universe()
    results=[]

    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for r in concurrent.futures.as_completed([ex.submit(scan_one,s) for s in data[:250]]):
            if r.result(): results.append(r.result())

df = pd.DataFrame(results).sort_values("Score", ascending=False)

# ========================= APPLY FILTERS + DISPLAY =========================
if df.empty:
    st.error("No signals found — loosen conditions!")
else:
    df = df[df["Score"]>=min_breakout]
    df = df[df["PM%"]>=min_PM]
    df = df[df["YDay%"]>=min_Yday]

    st.subheader(f"🔥 Live Signals ({len(df)}) — Ranked by Score")

    if "alerted" not in st.session_state:
        st.session_state.alerted=set()

    for _,row in df.iterrows():
        symbol=row["Symbol"]

        ## ====== ALERT CONDITIONS ======
        if symbol not in st.session_state.alerted:

            if row["Score"]>=ALERT_SCORE:
                st.session_state.alerted.add(symbol)
                st.warning(f"🔊 {symbol} — Score {row['Score']}!")

            if row["PM%"]>=ALERT_PM:
                st.session_state.alerted.add(symbol)
                st.error(f"🚨 {symbol} — Premarket Surge +{row['PM%']}%")

            if row["VWAP%"] and row["VWAP%"]>=ALERT_VWAP:
                st.session_state.alerted.add(symbol)
                st.success(f"💥 {symbol} — VWAP Breakout")

            ## === NEW OFB ALERT ===
            if row["OFB"] and row["OFB"]>=ALERT_OFB:
                st.session_state.alerted.add(symbol)
                st.warning(f"🔥 BUY PRESSURE DETECTED — {symbol} OFB {row['OFB']}")

        ## ====== UI LAYOUT ======
        col1,col2,col3,col4=st.columns([2,2,3,3])

        col1.markdown(f"**{symbol}**")
        col1.write(f"💲 Price: {row['Price']}")
        col1.write(f"⚡ Score: {row['Score']}")
        col1.write(f"RSI7: {row['RSI7']}")

        col2.write(f"PM: {row['PM%']}% | Yday: {row['YDay%']}%")
        col2.write(f"3D: {row['3D%']}%")
        col2.write(f"VWAP Dist: {row['VWAP%']}%")

        ## === ORDER FLOW BIAS VISUAL ===
        col3.write(f"OFB: {row['OFB']}")
        col3.plotly_chart(ofb_graph(row["OFB"]), use_container_width=False)

        col4.plotly_chart(go.Figure(data=[go.Scatter(y=row["Spark"], mode="lines")]), use_container_width=True)

        st.divider()

