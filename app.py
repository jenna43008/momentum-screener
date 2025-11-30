# ==================== V8 (Updated – All numeric values rounded to 2 decimals) ====================

import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math

THREADS = 20
AUTO_REFRESH_MS = 10_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL = "2m"
INTRADAY_RANGE = "1d"

DEFAULT_MAX_PRICE = 50.0
DEFAULT_MIN_VOLUME = 100_000
DEFAULT_MIN_BREAKOUT = 0.0

st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v8")

st.set_page_config(page_title="V8 — 10-Day Screener", layout="wide")
st.title("🚀 V8 Momentum Screener (All Values Rounded to 2 Decimals)")

# =================================== SIDEBAR ===================================
with st.sidebar:
    st.header("Screen Settings")

    watchlist_text = st.text_area("Watchlist (optional)", "", height=80)
    max_universe = st.slider("Max Symbols (no watchlist)", 50, 600, 200, 50)

    enable_enrichment = st.checkbox("Enable Float/Short/News Enrichment (slower)", False)

    st.subheader("Filters")
    max_price = st.number_input("Max Price", 1.0, 1000.0, DEFAULT_MAX_PRICE)
    min_volume = st.number_input("Min Volume", 10_000, 10_000_000, DEFAULT_MIN_VOLUME)
    min_breakout = st.number_input("Min Breakout Score", -50.0, 200.0, 0.0)
    min_pm_move = st.number_input("Min Premarket %", -50.0, 200.0, 0.0)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0)

    squeeze_only = st.checkbox("Short Squeeze Only")
    catalyst_only = st.checkbox("Catalyst Required")
    vwap_only = st.checkbox("Must Be Above VWAP")

    st.subheader("🔊 Audio Alerts")
    ALERT_SCORE_THRESHOLD = st.slider("Score ≥", 10, 200, 30)
    ALERT_PM_THRESHOLD = st.slider("Premarket ≥ %", 1, 150, 4)
    ALERT_VWAP_THRESHOLD = st.slider("VWAP Dist ≥ %", 1, 50, 2)

# =================================== LOAD SYMBOLS ===================================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", sep="|", skipfooter=1, engine="python")
    other = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", sep="|", skipfooter=1, engine="python")

    nasdaq["Exchange"] = "NASDAQ"
    other["Exchange"] = other["Exchange"].fillna("NYSE/AMEX/ARCA")
    other = other.rename(columns={"ACT Symbol":"Symbol"})

    df = pd.concat([nasdaq[["Symbol","Exchange"]],other[["Symbol","Exchange"]]])
    return df[df["Symbol"].str.contains("^[A-Z]{1,5}$",regex=True)].to_dict("records")

def build_universe(text,max_u):
    if text.strip():
        wl = {s.upper() for s in text.replace(","," ").split()}
        return [{"Symbol":s,"Exchange":"WATCH"} for s in wl]
    return load_symbols()[:max_u]

# =================================== SCORING ===================================
def score_calc(pm,y3,y10,rsi,rvol,vwap,flow,cat,sq):
    val=0
    if pm: val+=pm*1.6
    if y3: val+=y3*1.2
    if y10: val+=y10*0.6
    if rsi>55: val+=(rsi-55)*0.4
    if rvol>1.2: val+=(rvol-1.2)*2
    if vwap>0: val+=min(vwap,6)*1.5
    if flow: val+=(flow-0.5)*22
    if cat: val+=8
    if sq: val+=12
    return round(val,2)

def probability(score): return round((1/(1+math.exp(-score/20)))*100,2)

# =================================== MAIN SCAN ===================================
def scan_one(sym,enrich):
    try:
        t = sym["Symbol"]
        s = yf.Ticker(t)

        hist = s.history(period="10d",interval="1d")
        if hist.empty: return None

        close=hist["Close"]; vol=hist["Volume"]
        price = round(close.iloc[-1],2)
        last_vol = int(vol.iloc[-1])

        if price>max_price or last_vol<min_volume: return None

        # ROUNDED CALCS
        yday = round((price-close.iloc[-2])/close.iloc[-2]*100,2) if len(close)>=2 else None
        m3   = round((price-close.iloc[-4])/close.iloc[-4]*100,2) if len(close)>=4 else None
        m10  = round((price-close.iloc[0])/close.iloc[0]*100,2)

        delta=close.diff()
        rsi = round((100-(100/(1+delta.clip(lower=0).rolling(7).mean()/
                                 -delta.clip(upper=0).rolling(7).mean()))).iloc[-1],2)

        ema10 = round(close.ewm(span=10).mean().iloc[-1],2)
        rvol = round(last_vol/vol.mean(),2)

        # === Intraday (2m) ===
        pm=vwap=flow=None
        intra=s.history(period="1d",interval="2m",prepost=True)
        if len(intra)>=3:
            pm = round((intra["Close"].iloc[-1]-intra["Close"].iloc[-2])/
                       intra["Close"].iloc[-2]*100,2)

            tvwap=(intra["Volume"]*((intra["High"]+intra["Low"]+intra["Close"])/3)).sum()
            tot=intra["Volume"].sum()
            if tot>0:
                iv=tvwap/tot
                vwap = round((price-iv)/iv*100,2)

            of=intra[["Open","Close","Volume"]]
            sign= (of["Close"]>of["Open"]).astype(int)-(of["Close"]<of["Open"]).astype(int)
            b=(of["Volume"]*(sign>0)).sum(); s=(of["Volume"]*(sign<0)).sum()
            if b+s>0: flow = round(b/(b+s),2)

        # === Enrichment ===
        sq=cat=False; sec=ind="Unknown"; short=None; low_float=False

        if enrich:
            info=s.get_info() or {}
            fl=info.get("floatShares"); sp=info.get("shortPercentOfFloat")
            low_float = fl and fl<20_000_000
            sq = sp and sp>0.15
            short = round(sp*100,2) if sp else None
            sec=info.get("sector","Unknown"); ind=info.get("industry","Unknown")

            try:
                news=s.get_news()
                if news and "providerPublishTime" in news[0]:
                    p=datetime.fromtimestamp(news[0]["providerPublishTime"],tz=timezone.utc)
                    cat=(datetime.now(timezone.utc)-p).days<=3
            except: pass

        score=score_calc(pm,yday,m3,m10,rsi,rvol,vwap,flow,cat,sq)
        prob=probability(score)

        return {
            "Symbol":t,
            "Price":price,
            "Volume":last_vol,
            "PM%":pm,"YDay%":yday,"3D%":m3,"10D%":m10,
            "RSI7":rsi,"RVOL_10D":rvol,"VWAP%":vwap,"FlowBias":flow,
            "Score":score,"Prob_Rise%":prob,
            "LowFloat?":low_float,"Squeeze?":sq,
            "Catalyst":cat,"Sector":sec,"Industry":ind,
            "Spark":close
        }
    except:return None

@st.cache_data(ttl=6)
def run_scan(w,limit,enrich):
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s,enrich) for s in build_universe(w,limit)]):
            if (r:=f.result()) is not None: rows.append(r)
    return pd.DataFrame(rows)

# =================================== DISPLAY ===================================
df=run_scan(watchlist_text,max_universe,enable_enrichment)

if df.empty:
    st.warning("No results found. Try lowering filters.")
    st.stop()

df=df[df["Score"]>=min_breakout]
if min_pm_move: df=df[df["PM%"].fillna(-999)>=min_pm_move]
if min_yday_gain: df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
if squeeze_only: df=df[df["Squeeze?"]]
if catalyst_only: df=df[df["Catalyst"]]
if vwap_only: df=df[df["VWAP%"].fillna(-999)>0]

df=df.sort_values(by=["Score","PM%","RSI7"],ascending=[False,False,False])

st.subheader(f"🔥 {len(df)} Breakout Candidates (Rounded to 2 Decimals)")

for _,r in df.iterrows():
    c1,c2,c3,c4=st.columns([2,3,3,3])

    c1.markdown(f"### {r['Symbol']}")
    c1.write(f"💲 Price: **{r['Price']:.2f}**")
    c1.write(f"📊 Volume: **{r['Volume']:,}**")
    c1.write(f"🔥 Score: **{r['Score']:.2f}**  → Prob Rise: {r['Prob_Rise%']:.2f}%")

    c2.write(f"PM%: {r['PM%']}")
    c2.write(f"YDay: {r['YDay%']} | 3D: {r['3D%']} | 10D: {r['10D%']}")
    c2.write(f"RSI: {r['RSI7']} | RVOL10D: {r['RVOL_10D']}")

    c3.write(f"VWAP Dist: **{r['VWAP%']}%**")
    c3.write(f"Order Flow: {r['FlowBias']}")
    if enable_enrichment:
        c3.write(f"Squeeze={r['Squeeze?']} | LowFloat={r['LowFloat?']}")
        c3.write(f"{r['Sector']} / {r['Industry']}")

    # Sparkline
    spark=go.Figure()
    spark.add_trace(go.Scatter(y=r['Spark'],mode="lines",line=dict(width=2)))
    spark.update_layout(height=60,margin=dict(l=2,r=2,t=2,b=2),xaxis=dict(visible=False),yaxis=dict(visible=False))
    c4.plotly_chart(spark,use_container_width=True)

    st.divider()

st.download_button("📥 Download CSV",df.to_csv(index=False).encode("utf-8"),"V8_2decimal.csv")

st.caption("Research Only — Not Financial Advice")
