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

DEFAULT_MAX_PRICE     = 50.0
DEFAULT_MIN_VOLUME    = 100_000

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v8_uscan")

# ========================= UI =========================
st.set_page_config(page_title="V8 Hybrid + US/CA Filter", layout="wide")
st.title("🚀 V8 — US/Canada Filter + Location Column + Orderflow Floor")
st.caption("10-day model • VWAP + OFB • US/Canada toggle • Global optional")

with st.sidebar:
    st.header("SCAN MODE + REGION FILTER")
    watchlist_text = st.text_area("Watchlist Symbols:", "", height=80)

    max_universe = st.slider("Max symbols if not watchlist", 50, 600, 200, 50)

    region_mode = st.radio(
        "Market Universe:",
        ["Global", "US + Canada Only (recommended speed boost)"],
        index=1,  # default ON for US+CA
    )

    enable_enrichment = st.checkbox("Enable Float/Short/News (slower)", False)

    st.markdown("---")
    st.subheader("Filters")

    max_price = st.number_input("Max Price ($)", 1.0, 1000.0, DEFAULT_MAX_PRICE)
    min_volume = st.number_input("Min Volume", 10_000, 10_000_000, DEFAULT_MIN_VOLUME)
    min_breakout = st.slider("Min Score", -20, 200, 0)
    min_pm = st.slider("Min Premarket %", -20, 200, 0)
    min_yday = st.slider("Min Yesterday %", -20, 200, 0)

    vwap_only = st.checkbox("Above VWAP only")
    squeeze_only = st.checkbox("Squeeze Only")
    catalyst_only = st.checkbox("Must have News Catalyst")

    # 🔥 NEW – Filter OFB minimum
    min_ofb = st.slider("Min Order Flow Bias (0–1)", 0.00, 1.00, 0.55, 0.01)

    st.markdown("---")
    st.subheader("Audio Alerts")

    ALERT_SCORE = st.slider("Alert Score ≥", 5, 200, 30)
    ALERT_PM    = st.slider("Alert Premarket ≥", 1, 150, 4)
    ALERT_VWAP  = st.slider("Alert VWAP ≥", 1, 50, 2)
    ALERT_OFB   = st.slider("Alert OFB ≥", 0.5, 1.0, 0.72, 0.01)

    if st.button("Refresh Now"):
        st.cache_data.clear()
        st.success("Cleared — rescanning now.")


# ========================= SYMBOL UNIVERSE =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                         sep="|", skipfooter=1, engine="python")
    other  = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                         sep="|", skipfooter=1, engine="python")

    nasdaq["Exchange"] = "NASDAQ"
    other["Exchange"]  = other["Exchange"].fillna("NYSE/AMEX/ARCA")
    other = other.rename(columns={"ACT Symbol":"Symbol"})

    df = pd.concat([nasdaq[["Symbol","ETF","Exchange"]],other[["Symbol","ETF","Exchange"]]])
    df = df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$",na=False)]
    df["Country"] = "US"   # default assignment

    return df.to_dict("records")

def build_universe():
    wl = watchlist_text.strip()

    # If watchlist exists → skip global scan entirely
    if wl:
        tickers = sorted(set(wl.replace("\n"," ").replace(","," ").split()))
        return [{"Symbol":t.upper(),"Exchange":"WATCH","Country":"WATCH"} for t in tickers]

    universe = load_symbols()

    # ========================= NEW REGION FILTER =========================
    if region_mode == "US + Canada Only (recommended speed boost)":
        # Pull Toronto Stock Exchange listings via Yahoo (TSX-T)
        try:
            tsx = pd.read_csv("https://www.tsx.com/json/company-directory/search?sector=all", on_bad_lines='skip')
        except:
            tsx = pd.DataFrame()

        ca_syms = []
        if not tsx.empty:
            ca_syms = [s.replace(".", "-") for s in tsx.get("symbol","").tolist()]

        # Append Canadian tickers
        for sym in ca_syms[:300]:  # keeps fast
            universe.append({"Symbol":sym,"Exchange":"TSX","ETF":None,"Country":"CA"})

        # Filter out non US/CA from combined list
        universe = [s for s in universe if s.get("Country","US") in ["US","CA"]]

    # Limit universe for performance
    return universe[:max_universe]


# ========================= SCAN ONE SYMBOL =========================
def scan_one(sym):
    try:
        ticker = sym["Symbol"]
        exch   = sym.get("Exchange","?")
        country= sym.get("Country","US")  # new location display

        t = yf.Ticker(ticker)

        hist = t.history(period="10d",interval="1d")
        if hist.empty: return None

        close= hist["Close"]; vol= hist["Volume"]
        price=float(close.iloc[-1]); v=vol.iloc[-1]
        if price>max_price or v<min_volume: return None

        # === Momentum windows
        yday=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if close.iloc[-2]>0 else None
        m3  =(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10 =(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        # RSI7
        delta=close.diff()
        rsi=100-(100/(1+ delta.clip(lower=0).rolling(7).mean()/(-delta.clip(upper=0).rolling(7).mean())))
        rsi7=float(rsi.iloc[-1])

        # EMA10
        ema10=float(close.ewm(span=10,adjust=False).mean().iloc[-1])
        trend="🔥 Breakout" if price>ema10 and rsi7>55 else "Neutral"

        # === Intraday VWAP + OFB
        intra=t.history(period="1d",interval="2m",prepost=True)
        if intra.empty: return None
        
        # Premarket %
        last= intra["Close"].iloc[-1]
        prev =intra["Close"].iloc[-2]
        pm = (last-prev)/prev*100 if prev>0 else None

        typ=(intra["High"]+intra["Low"]+intra["Close"])/3
        if intra["Volume"].sum()>0:
            vwap=float((typ*intra["Volume"]).sum()/intra["Volume"].sum())
            vdist=(price-vwap)/vwap*100
        else:
            vdist=None

        # Order Flow Bias
        df=intra[["Open","Close","Volume"]]
        sign=(df["Close"]>df["Open"]).astype(int)-(df["Close"]<df["Open"]).astype(int)
        b=(df["Volume"]*(sign>0)).sum(); s=(df["Volume"]*(sign<0)).sum()
        ofb=b/(b+s) if (b+s)>0 else None

        # ========= NEW FILTER: Exclude low OFB =========
        if ofb is not None and ofb < min_ofb:
            return None

        # === Enrichment Optional
        catalyst=squeeze=lowfloat=False
        sector=industry="Unknown"
        
        if enable_enrichment:
            info=t.get_info() or {}
            fl=info.get("floatShares"); sh=info.get("shortPercentOfFloat")
            lowfloat=bool(fl and fl<20_000_000)
            squeeze =bool(sh and sh>0.15)
            sector  =info.get("sector","NA"); industry=info.get("industry","NA")

            news=t.get_news()
            if news and "providerPublishTime" in news[0]:
                age=(datetime.now(timezone.utc)-datetime.fromtimestamp(news[0]["providerPublishTime"],tz=timezone.utc)).days
                catalyst = age<=3

        # === Score calculation (10-day model)
        score = (
            max(pm or 0,0)*1.6 +
            max(yday or 0,0)*0.8 +
            max(m3 or 0,0)*1.2 +
            max(m10 or 0,0)*0.6 +
            (rsi7-55)*0.4 if rsi7>55 else 0 +
            ((vdist or 0)*1.5 if vdist>0 else 0) +
            ((ofb-0.5)*22 if ofb else 0) +
            (8 if catalyst else 0) +
            (12 if squeeze else 0)
        )
        score=round(score,2)

        spark=close  # sparkline data

        return {
            "Symbol":ticker,
            "Country":country,         # NEW column
            "Exchange":exch,
            "Price":round(price,2),
            "Score":score,
            "PM%":round(pm,2) if pm else None,
            "YDay%":round(yday,2) if yday else None,
            "3D%":round(m3,2),
            "10D%":round(m10,2),
            "RSI7":round(rsi7,2),
            "VWAP%":round(vdist,2) if vdist else None,
            "FlowBias":round(ofb,2) if ofb else None,
            "Squeeze":squeeze,
            "LowFloat":lowfloat,
            "Catalyst":catalyst,
            "Trend":trend,
            "Spark":spark
        }

    except:
        return None


@st.cache_data(ttl=8)
def run_scan():
    syms=build_universe()  
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s) for s in syms]):
            r=f.result()
            if r: out.append(r)
    return pd.DataFrame(out)


# ========================= MAIN DISPLAY =========================
with st.spinner("Scanning…"):
    df=run_scan()

if df.empty:
    st.error("No results — loosen your filters.")
    st.stop()

# Final UI
df=df[df["Score"]>=min_breakout]
df=df[df["PM%"]>=min_pm]
df=df[df["YDay%"]>=min_yday]

if vwap_only: df=df[df["VWAP%"]>0]
if squeeze_only: df=df[df["Squeeze"]]
if catalyst_only: df=df[df["Catalyst"]]

df = df.sort_values("Score",ascending=False)

st.subheader(f"🔥 Signals — {len(df)} tickers")
for _,row in df.iterrows():
    sym=row["Symbol"]

    c1,c2,c3 = st.columns([2,2,2])
    c1.markdown(f"**{sym}** ({row['Country']})")   # LOCATION VISIBLE HERE
    c1.write(f"Price: {row['Price']}  | Score {row['Score']}")
    c2.write(f"PM% {row['PM%']} | YDay {row['YDay%']}")
    c2.write(f"10D {row['10D%']}  3D {row['3D%']}")
    c3.write(f"VWAP {row['VWAP%']}% | OFB {row['FlowBias']}")
    st.line_chart(row["Spark"])
    st.divider()

st.download_button("📥 Export CSV",df.to_csv(index=False),"v8_uscan_output.csv")


