import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math

# ========================= SETTINGS =========================
THREADS               = 15
AUTO_REFRESH_MS       = 10_000       # auto-refresh every 10 seconds
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 50.0
DEFAULT_MIN_VOLUME    = 100_000
DEFAULT_MIN_BREAKOUT  = 0.0

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh")

# ========================= PAGE UI =========================
st.set_page_config(page_title="V9 — Live Volume Ranked")
st.title("🚀 V9 — Real-Time Volume Ranked Momentum Screener")
st.caption("Prioritized by live intraday volume • Real-time movers • OFB • EMA • RSI • Short-squeeze logic")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area("Watchlist tickers (optional):", "")

    max_universe = st.slider("Max Symbols to Scan", 50, 600, 200, step=50)

    region_mode = st.radio("Region", ["Global", "US + Canada Only"], index=1)

    enable_enrichment = st.checkbox("Float/Short/News Enrichment (slower)", False)

    # 🔊 NEW: alert toggle
    enable_alerts = st.checkbox("Enable Alerts", True)

    st.subheader("Filters")
    max_price = st.number_input("Max Price", 1.0, 1000.0, DEFAULT_MAX_PRICE, 1.0)
    min_volume = st.number_input("Min Daily Volume", 10000, 10_000_000, DEFAULT_MIN_VOLUME, step=10000)
    min_breakout = st.number_input("Min Breakout Score", -50.0, 200.0, 0.0, step=1.0)
    min_pm_move = st.number_input("Min Premarket %", -50.0, 200.0, 0.0, step=0.5)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0, step=0.5)

    squeeze_only = st.checkbox("Short-Squeeze Only")
    catalyst_only = st.checkbox("News Required")
    vwap_only = st.checkbox("Above VWAP Only")

    min_ofb = st.slider("Min Order Flow Bias (0-1)", 0.00, 1.00, 0.50, step=0.01)

    # 🔊 Alert Thresholds
    st.markdown("### 🔊 Alert Triggers")
    ALERT_SCORE_THRESHOLD = st.slider("Score ≥", 10, 200, 30)
    ALERT_PM_THRESHOLD    = st.slider("Premarket % ≥", 1, 150, 4)
    ALERT_VWAP_THRESHOLD  = st.slider("VWAP Dist % ≥", 1, 50, 2)

    if st.button("🧹 Refresh Now"):
        st.cache_data.clear()
        st.success("Universe + Cache Reset — Reloading...")

# ========================= SYMBOL SOURCING =========================
@st.cache_data(ttl=600)
def load_symbols():
    nas = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", sep="|", skipfooter=1, engine="python")
    oth = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", sep="|", skipfooter=1, engine="python")

    nas["Exchange"]="NASDAQ"
    oth["Exchange"]=oth["Exchange"].fillna("NYSE/AMEX/ARCA")
    oth = oth.rename(columns={"ACT Symbol":"Symbol"})

    df=pd.concat([nas[["Symbol","Exchange"]],oth[["Symbol","Exchange"]]]).dropna()
    df=df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$",na=False)]
    df["Country"]="US"
    return df.to_dict("records")

# 🔥 LIVE VOLUME-RANKED UNIVERSE — MAIN CHANGE
def build_universe():
    wl = watchlist_text.strip()
    if wl:
        tick=[t.upper() for t in wl.replace(","," ").replace("\n"," ").split() if t.strip()]
        return [{"Symbol":t,"Exchange":"WATCH","Country":"Unknown"} for t in tick]

    symbols = load_symbols()

    ranked=[] # pull intraday vol for sort ranking
    for sym in symbols:
        try:
            d=yf.Ticker(sym["Symbol"]).history(period="1d", interval="2m", prepost=True)
            if not d.empty:
                ranked.append({**sym,"LiveVol":float(d["Volume"].iloc[-1])})
        except: pass

    # Highest volume → first
    return sorted(ranked, key=lambda x:x["LiveVol"], reverse=True)[:max_universe]

# ========================= SCORING + INDICATORS =========================
def score_model(pm,y,m3,m10,rsi,rvol,c,sq,vwap,flow):
    s=0
    if pm:  s+=pm*1.6
    if y:   s+=y*0.8
    if m3:  s+=m3*1.2
    if m10: s+=m10*0.6
    if rsi and rsi>55: s+=(rsi-55)*0.4
    if rvol and rvol>1.2: s+=(rvol-1.2)*2
    if vwap and vwap>0: s+=min(vwap,6)*1.5
    if flow: s+=(flow-0.5)*22
    if c: s+=8
    if sq: s+=12
    return round(s,2)

def spark(series):
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=series.values,mode="lines",line=dict(width=2)))
    fig.update_layout(height=60,width=160,margin=dict(l=2,r=2,t=2,b=2),xaxis=dict(visible=False),yaxis=dict(visible=False))
    return fig

# ========================= SCAN CORE =========================
def scan_one(sym):
    try:
        ticker=sym["Symbol"]
        t=yf.Ticker(ticker)

        hist=t.history(period="10d")
        if hist.empty: return None

        close=hist["Close"]; volume=hist["Volume"]
        price=float(close.iloc[-1]); vol_last=float(volume.iloc[-1])
        if price>max_price or vol_last<min_volume: return None

        # returns
        y=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>=2 else None
        m3=(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        rsi=100-(100/(1+close.diff().clip(lower=0).rolling(7).mean()/(-close.diff().clip(upper=0).rolling(7).mean())))
        rsi=float(rsi.iloc[-1])

        avg10=volume.mean(); rvol=vol_last/avg10 if avg10>0 else None

        intra=t.history(period="1d",interval="2m",prepost=True)
        if intra.empty: return None
        c=intra["Close"]; o=intra["Open"]; v=intra["Volume"]

        pm=(c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100 if c.iloc[-2]>0 else None
        tp=(intra["High"]+intra["Low"]+intra["Close"])/3
        vwap=(price-(tp*v).sum()/v.sum())/( (tp*v).sum()/v.sum())*100 if v.sum()>0 else None

        sign=(c>o).astype(int)-(c<o).astype(int)
        buy,v_sell=(v*(sign>0)).sum(),(v*(sign<0)).sum()
        flow=buy/(buy+v_sell) if (buy+v_sell)>0 else None

        if flow is None or flow < min_ofb: return None

        short=None; sector=""; industry=""; squeeze=False; catalyst=False
        if enable_enrichment:
            info=t.get_info() or {}
            sf=info.get("floatShares",0)
            short=info.get("shortPercentOfFloat")
            squeeze= short and short>0.15
            sector=info.get("sector",""); industry=info.get("industry","")

        score=score_model(pm,y,m3,m10,rsi,rvol,catalyst,squeeze,vwap,flow)

        return {
            "Symbol":ticker,"Price":round(price,2),"PM%":round(pm,2) if pm else None,
            "YDay%":round(y,2) if y else None,"3D%":round(m3,2) if m3 else None,
            "10D%":round(m10,2),"RSI7":round(rsi,2),
            "RVOL_10D":round(rvol,2) if rvol else None,
            "VWAP%":round(vwap,2) if vwap else None,"FlowBias":round(flow,2),
            "Short%Float":round(short*100,2) if short else None,
            "Squeeze?":squeeze,"Sector":sector,"Industry":industry,
            "Score":score,"Spark":close,
        }
    except: return None

# ========================= RUN SCAN =========================
with st.spinner("Scanning real-time volume movers…"):
    universe=build_universe()
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        results=[f.result() for f in concurrent.futures.as_completed([ex.submit(scan_one,s) for s in universe]) if f.result()]

df=pd.DataFrame(results)

if df.empty: st.error("No results — loosen filters"); st.stop()

df=df[df.Score>=min_breakout]
df=df.sort_values(["Score","PM%","FlowBias"],ascending=[False,False,False])

# ========================= DISPLAY + ALERTS =========================
st.subheader(f"🔥 Live Volume Ranked — {len(df)} tickers")

for _,r in df.iterrows():
    sym=r.Symbol

    if enable_alerts and "alerted" not in st.session_state:
        st.session_state.alerted=set()

    if enable_alerts and sym not in st.session_state.alerted:
        if r.Score>=ALERT_SCORE_THRESHOLD: st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} Score {r.Score}")
        elif r["PM%"] and r["PM%"]>=ALERT_PM_THRESHOLD: st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} PM% {r['PM%']}")
        elif r["VWAP%"] and r["VWAP%"]>=ALERT_VWAP_THRESHOLD: st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} VWAP {r['VWAP%']}")

    c1,c2,c3=st.columns([2,3,3])
    c1.markdown(f"### {sym}  💲{r.Price}")
    c1.write(f"🔥 Score {r.Score}")
    c1.write(f"PM {r['PM%']}%  |  Yday {r['YDay%']}%")
    c1.write(f"FlowBias {r.FlowBias}  |  RVOL {r.RVOL_10D}x")

    c2.write(f"3D {r['3D%']}% | 10D {r['10D%']}% | RSI7 {r.RSI7}")
    c2.write(f"VWAP Dist {r['VWAP%']}%  | Squeeze {r['Squeeze?']}")
    c2.write(f"Sector: {r.Sector}  | Industry: {r.Industry}")

    c3.plotly_chart(spark(r.Spark))

    st.divider()

st.download_button("📥 Download Results CSV",df.to_csv(index=False),"v9_live_volume_ranked.csv")
