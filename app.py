import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math

# ========================= CORE SETTINGS =========================
THREADS               = 20
AUTO_REFRESH_MS       = 12_000      # Refresh roughly every 12 seconds
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_8_8")

# ========================= UI HEADER =========================
st.set_page_config(page_title="V8.8 Hybrid Screener", layout="wide")
st.title("🚀 V8.8 — Hybrid Market Scan + Watchlist + Catalyst/Float Enrichment")
st.caption("10-day model • PM Momentum • VWAP • Order Flow Bias Meter • Short Float Detection • News Catalysts • Scalp-to-swing hybrid")

# ========================= SIDEBAR CONTROL CENTER =========================
with st.sidebar:
    st.subheader("📍 Market Mode")
    
    enable_watchlist_only = st.checkbox("Watchlist-only Mode (skip full market)", value=False)
    watchlist_input = st.text_area("Watchlist Symbols:", placeholder="AAPL,TSLA,NVDA,AMD...", height=80)

    max_universe = st.slider("Max tickers full-scan", 50, 800, 250, step=50)
    
    st.markdown("---")
    st.subheader("📊 Enrichment Options")
    enable_float_short    = st.checkbox("Enable Float/Short Scan", value=False)
    enable_news_catalyst  = st.checkbox("Enable News/Catalyst Scan", value=False)

    st.markdown("---")
    st.subheader("📌 Filters")
    max_price   = st.number_input("Max Price", 1.0, 2000.0, 50.0)
    min_vol     = st.number_input("Min Daily Volume", 10_000, 20_000_000, 100_000, step=10_000)
    min_score   = st.slider("Min Score", 0, 200, 0)
    min_pm      = st.slider("Min Premarket %", 0, 100, 0)
    min_yday    = st.slider("Min Yesterday %", 0, 200, 0)
    squeeze_only= st.checkbox("Short Squeeze Only")
    vwap_only   = st.checkbox("Above VWAP only")

    st.markdown("---")
    st.subheader("🔊 Audio Alert Levels")
    ALERT_SCORE = st.slider("Alert Score ≥", 5, 200, 35)
    ALERT_PM    = st.slider("Alert Premarket ≥", 1, 150, 4)
    ALERT_VWAP  = st.slider("Alert VWAP% ≥", 1, 50, 2)
    ALERT_OFB   = st.slider("Alert OFB ≥", 0.50, 1.00, 0.72, step=0.01)

    if st.button("Manual Refresh"):
        st.cache_data.clear()
        st.experimental_rerun()

# ========================= SYMBOL LOAD =========================
@st.cache_data(ttl=1200)
def load_universe():
    nas = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",sep="|",skipfooter=1,engine="python")
    oth = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",sep="|",skipfooter=1,engine="python")
    
    nas["Exchange"]="NASDAQ"
    oth["Exchange"]=oth["Exchange"].fillna("NYSE/AMEX")
    oth=oth.rename(columns={"ACT Symbol":"Symbol"})
    
    all=pd.concat([nas[["Symbol","Exchange"]],oth[["Symbol","Exchange"]]]).dropna()
    return all[all["Symbol"].str.contains(r"^[A-Z]{1,5}$")]

def build_universe():
    if enable_watchlist_only:
        wl = watchlist_input.replace(",", " ").upper().split()
        tokens = [t.strip() for t in wl if len(t)>=1]
        return pd.DataFrame({"Symbol":tokens,"Exchange":["WATCH"]*len(tokens)})
    df=load_universe()
    return df.head(max_universe)

# ========================= ORDER FLOW GRAPH =========================
def ofb_meter(val):
    color="green" if val>=0.66 else "orange" if val>=0.55 else "red"
    fig=go.Figure(go.Bar(x=[val],orientation="h",marker=dict(color=color)))
    fig.update_layout(height=40,width=160,xaxis=dict(range=[0,1],visible=False),yaxis=dict(visible=False),margin=dict(l=1,r=1,t=1,b=1))
    return fig

# ========================= MOMENTUM SCORING =========================
def score_v8(pm,yday,m3,rsi,vwap,ofb,rvol):
    s=0
    s+=max(pm,0)*1.4 if pm else 0
    s+=max(yday,0)*0.8 if yday else 0
    s+=max(m3,0)*1.0 if m3 else 0
    s+=(rsi-55)*0.35 if rsi>55 else 0
    s+=vwap*0.7 if vwap and vwap>0 else 0
    s+=(ofb-0.55)*24 if ofb and ofb>0.55 else 0
    s+=min(rvol-1,2)*2 if rvol and rvol>1 else 0
    return round(s,2)

# ========================= SCAN ONE =========================
def scan(sym):
    try:
        tk=sym["Symbol"]
        t=yf.Ticker(tk)

        hist=t.history(period="10d",interval="1d")
        if hist.empty: return None

        price=float(hist["Close"].iloc[-1])
        vol=float(hist["Volume"].iloc[-1])
        if price>max_price or vol<min_vol: return None

        # short momentum windows
        if len(hist)>=2: yday=(price-hist["Close"].iloc[-2])/hist["Close"].iloc[-2]*100
        else: yday=None
        if len(hist)>=4: m3=(price-hist["Close"].iloc[-4])/hist["Close"].iloc[-4]*100
        else: m3=None

        # RSI7
        d=hist["Close"].diff()
        rsi=float((100-(100/(1+ d.clip(lower=0).rolling(7).mean()/(-d.clip(upper=0).rolling(7).mean())))).iloc[-1])

        # Intraday for PM+VWAP+OFB
        intra=t.history(period="1d",interval="2m",prepost=True)
        if intra.empty: return None

        lc=float(intra["Close"].iloc[-1])
        pc=float(intra["Close"].iloc[-2])
        PM=(lc-pc)/pc*100 if pc>0 else None

        typ=(intra["High"]+intra["Low"]+intra["Close"])/3
        VWAP=(typ*intra["Volume"]).sum()/intra["Volume"].sum()
        Vdist=(price-VWAP)/VWAP*100 if VWAP>0 else None

        dir=(intra["Close"]>intra["Open"]).astype(int)-(intra["Close"]<intra["Open"]).astype(int)
        b=float((intra["Volume"]*(dir>0)).sum())
        s=float((intra["Volume"]*(dir<0)).sum())
        OFB=b/(b+s) if b+s>0 else None

        # 10-day relative volume
        RVOL=vol/hist["Volume"].mean()

        # Optional float/short
        lowfloat=squeeze=catalyst=False
        short_pct=float_shares=None

        if enable_float_short or enable_news_catalyst:
            info=t.get_info() or {}

        if enable_float_short:
            float_shares=info.get("floatShares")
            short_pct=info.get("shortPercentOfFloat")
            lowfloat=bool(float_shares and float_shares<20_000_000)
            squeeze=bool(short_pct and short_pct>0.14)

        if enable_news_catalyst:
            news=t.get_news()
            if news and "providerPublishTime" in news[0]:
                age=(datetime.now(timezone.utc)-datetime.fromtimestamp(news[0]["providerPublishTime"],tz=timezone.utc)).days
                catalyst=age<=3

        SCORE=score_v8(PM,yday,m3,rsi,Vdist,OFB,RVOL)

        return {
            "Symbol":tk,"Price":round(price,2),"Score":SCORE,
            "PM%":round(PM,2) if PM else None,"YDay%":round(yday,2) if yday else None,"3D%":round(m3,2) if m3 else None,
            "RSI7":round(rsi,2),"VWAP%":round(Vdist,2) if Vdist else None,"OFB":round(OFB,2) if OFB else None,
            "RVOL":round(RVOL,2),
            "Squeeze":squeeze,"LowFloat":lowfloat,"Catalyst":catalyst,
            "Spark":hist["Close"],
        }
    except:
        return None

# ========================= RUN SCAN =========================
@st.cache_data(ttl=8)
def run():
    univ=build_universe().to_dict("records")
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan,s) for s in univ]):
            if f.result(): out.append(f.result())
    return pd.DataFrame(out)

# ========================= MAIN DISPLAY =========================
with st.spinner("Scanning market…"):
    df=run()

if df.empty:
    st.error("No signals — loosen filters.")
    st.stop()

# apply filters
df=df[df["Score"]>=min_score]
df=df[df["PM%"]>=min_pm]
df=df[df["YDay%"]>=min_yday]
if squeeze_only: df=df[df["Squeeze"]]
if vwap_only: df=df[df["VWAP%"]>0]

df=df.sort_values("Score",ascending=False)
st.subheader(f"🔥 Signals {len(df)} — ranked by Score")

if "alerted" not in st.session_state: st.session_state.alerted=set()

for _,r in df.iterrows():
    sym=r["Symbol"]

    # ===== Alerts =====
    if sym not in st.session_state.alerted:
        if r["Score"]>=ALERT_SCORE:
            st.session_state.alerted.add(sym); st.warning(f"🔥 {sym} Score {r['Score']}")
        if r["PM%"]>=ALERT_PM:
            st.session_state.alerted.add(sym); st.error(f"🚨 Premarket {sym} +{r['PM%']}%")
        if r["VWAP%"] and r["VWAP%"]>=ALERT_VWAP:
            st.session_state.alerted.add(sym); st.success(f"💥 VWAP Breakout {sym}")
        if r["OFB"] and r["OFB"]>=ALERT_OFB:
            st.session_state.alerted.add(sym); st.info(f"⚡ Strong Buying {sym} OFB {r['OFB']}")

    c1,c2,c3,c4=st.columns([2,2,3,3])

    c1.markdown(f"**{sym}**")
    c1.write(f"💲 {r['Price']} | Score: {r['Score']}")
    c1.write(f"PM: {r ['PM%']}% • YDay {r['YDay%']}% • 3D {r['3D%']}%")

    c2.write(f"RSI7: {r['RSI7']}")
    c2.write(f"VWAP%: {r['VWAP%']}")
    c2.write(f"RVOL: {r['RVOL']}x")

    # 🔥 Order Flow Bias Meter Visual
    c3.write(f"OFB: {r['OFB']}")
    c3.plotly_chart(ofb_meter(r["OFB"]),use_container_width=False)

    # Optional enrichment tags
    if enable_float_short:     c3.write(f"Squeeze:{r['Squeeze']} • LowFloat:{r['LowFloat']}")
    if enable_news_catalyst:   c3.write(f"Catalyst:{r['Catalyst']}")

    c4.plotly_chart(go.Figure(go.Scatter(y=r["Spark"],mode="lines")),use_container_width=True)
    st.divider()

st.download_button("📥 Export to CSV", df.to_csv(index=False), "v8_8_output.csv")

