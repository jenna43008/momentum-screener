#########################################
#   V8 with Added Volume Signals ONLY  #
#   No UI changes removed or modified #
#########################################

import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math

# ========================= SETTINGS =========================
THREADS=20
AUTO_REFRESH_MS=10_000
HISTORY_LOOKBACK_DAYS=10
INTRADAY_INTERVAL="2m"
INTRADAY_RANGE="1d"

DEFAULT_MAX_PRICE=50.0
DEFAULT_MIN_VOLUME=100_000
DEFAULT_MIN_BREAKOUT=0.0

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS,key="refresh_v8")

# ========================= PAGE SETUP =========================
st.set_page_config(page_title="V8 – 10-Day Momentum Screener",layout="wide")
st.title("🚀 V8 — 10-Day Momentum Breakout Screener (Faster + Watchlist Mode)")
st.caption("Short-window model • EMA10 • RSI(7) • 3D & 10D momentum • RVOL + Order Flow • Audio alerts")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Universe")
    watchlist_text=st.text_area("Watchlist tickers:",value="",height=80)
    max_universe=st.slider("Max symbols to scan",50,600,200,50)
    enable_enrichment=st.checkbox("Include float/short + news",value=False)

    st.markdown("---")
    st.header("Filters")
    max_price=st.number_input("Max Price ($)",1.0,1000.0,DEFAULT_MAX_PRICE,1.0)
    min_volume=st.number_input("Min Daily Volume",10_000,10_000_000,DEFAULT_MIN_VOLUME,10_000)
    min_breakout=st.number_input("Min Breakout Score",-50.0,200.0,0.0,1.0)
    min_pm_move=st.number_input("Min Premarket %",-50.0,200.0,0.0,0.5)
    min_yday_gain=st.number_input("Min Yesterday %",-50.0,200.0,0.0,0.5)
    squeeze_only=st.checkbox("Short-Squeeze Only")
    catalyst_only=st.checkbox("Must Have News/Earnings")
    vwap_only=st.checkbox("Above VWAP Only")

    st.markdown("---")
    st.subheader("🔊 Audio Alert Thresholds")
    ALERT_SCORE_THRESHOLD=st.slider("Score ≥",10,200,30)
    ALERT_PM_THRESHOLD=st.slider("Premarket % ≥",1,150,4)
    ALERT_VWAP_THRESHOLD=st.slider("VWAP Dist % ≥",1,50,2)

# ========================= SYMBOL LOAD =========================
@st.cache_data(ttl=900)
def load_symbols():
    nas=pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",sep="|",skipfooter=1,engine="python")
    oth=pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",sep="|",skipfooter=1,engine="python")
    nas["Exchange"]="NASDAQ"
    oth["Exchange"]=oth["Exchange"].fillna("NYSE/AMEX/ARCA")
    oth=oth.rename(columns={"ACT Symbol":"Symbol"})
    df=pd.concat([nas[["Symbol","ETF","Exchange"]],oth[["Symbol","ETF","Exchange"]]])
    df=df[df["Symbol"].str.contains("^[A-Z]{1,5}$",regex=True)]
    return df.to_dict("records")

def build_universe(w,max_u):
    wl=w.strip()
    if wl:
        raw=wl.replace("\n"," ").replace(","," ").split()
        ticks=sorted(set(x.upper() for x in raw))
        return [{"Symbol":t,"Exchange":"WATCH"} for t in ticks]
    return load_symbols()[:max_u]

# ========================= NEW VOLUME SIGNALS ADDED =========================
def compute_volume_signals(volume_series):
    cur=int(volume_series.iloc[-1])
    avg10=int(volume_series.mean())
    vr=round(cur/avg10,2) if avg10>0 else None
    tr="🔺 Rising Vol" if cur>volume_series.iloc[-2] else "🔻 Cooling Vol"
    return cur,avg10,vr,tr

# ========================= SCAN ONE =========================
def scan_one(sym,enrich):
    try:
        t=sym["Symbol"]
        stock=yf.Ticker(t)
        hist=stock.history(period="10d",interval="1d")

        if hist.empty:return None
        close=hist["Close"]; vol=hist["Volume"]

        price=float(close.iloc[-1])
        if price>max_price or vol.iloc[-1]<min_volume: return None

        # Momentum returns
        y=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>2 else None
        m3=(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>4 else None
        m10=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        # RSI7
        delta=close.diff()
        rsi=100-(100/(1+(delta.clip(lower=0).rolling(7).mean()/(-delta.clip(upper=0).rolling(7).mean()))))
        rsi=float(rsi.iloc[-1])

        # Volume Signals (added)
        CurVol,Vol10Avg,Vol_Ratio,Vol_Trend=compute_volume_signals(vol)

        # RVOL existing
        rvol=round(CurVol/Vol10Avg,2)

        # Intraday
        intra=stock.history(period="1d",interval="2m",prepost=True)
        pm=None;vwap=None;flow=None
        if not intra.empty:
            c=intra["Close"]; o=intra["Open"]; v=intra["Volume"]
            pm=(c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100
            tp=(intra["High"]+intra["Low"]+intra["Close"])/3
            vw=(tp*v).sum()/v.sum(); vwap=(price-vw)/vw*100

            sign=(c>o).astype(int)-(c<o).astype(int)
            buy=(v*(sign>0)).sum(); sell=(v*(sign<0)).sum()
            flow=round(buy/(buy+sell),2) if (buy+sell)>0 else None

        score=0
        for val,weight in [(pm,1.6),(y,0.8),(m3,1.2),(m10,0.6)]:
            if val: score+=max(val,0)*weight
        if rsi>55:score+=(rsi-55)*0.4
        if rvol>1.2:score+=(rvol-1.2)*2
        if vwap and vwap>0:score+=min(vwap,6)*1.5
        if flow:score+=(flow-0.5)*22

        return {
            "Symbol":t,"Price":round(price,2),
            "CurVol":CurVol,"Vol10Avg":Vol10Avg,"Vol_Ratio":Vol_Ratio,"VolTrend":Vol_Trend,
            "Score":round(score,2),"PM%":round(pm,2) if pm else None,
            "YDay%":round(y,2) if y else None,"3D%":round(m3,2),"10D%":round(m10,2),
            "RSI7":round(rsi,2),"RVOL_10D":rvol,"VWAP%":vwap,"FlowBias":flow,
            "Spark":close
        }
    except:return None

# ========================= RUN SCAN =========================
with st.spinner("Scanning…"):
    univ=build_universe(watchlist_text,max_universe)
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        df=pd.DataFrame([f.result() for f in concurrent.futures.as_completed(
            [ex.submit(scan_one,i,enable_enrichment) for i in univ]
        ) if f.result()])

if df.empty:st.error("No results.");st.stop()

df=df[df.Score>=min_breakout].sort_values(["Score","PM%"],ascending=False)

# ========================= DISPLAY =========================
st.subheader(f"🔥 Momentum Signals — {len(df)}")

for _,r in df.iterrows():
    c1,c2,c3=st.columns([2,3,3])
    c1.markdown(f"### {r.Symbol}  💲{r.Price}")
    c1.write(f"Volume: {r.CurVol:,} (Avg10: {r.Vol10Avg:,})")
    c1.write(f"RVOL: {r.Vol_Ratio}x  |  Trend: {r.VolTrend}")
    c1.write(f"Score {r.Score} | PM {r['PM%']}%")

    c2.write(f"3D {r['3D%']}% | 10D {r['10D%']}% | RSI7 {r.RSI7}")
    c2.write(f"VWAP {r['VWAP%']}% | Flow {r.FlowBias}")

    c3.plotly_chart(go.Figure(data=[go.Scatter(y=r.Spark.values,mode='lines')]),
                    use_container_width=True)

st.download_button("📥 Export CSV",df.to_csv(index=False),"vol_signals.csv")
