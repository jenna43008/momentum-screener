import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
import random
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math

# ========================= SETTINGS =========================
THREADS               = 15
AUTO_REFRESH_MS       = 10_000       # auto-refresh every 10 sec
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 50.0
DEFAULT_MIN_VOLUME    = 100_000
DEFAULT_MIN_BREAKOUT  = 0.0

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v9")

# ========================= UI =========================
st.set_page_config(page_title="V9 — Live Volume Ranked Screener", layout="wide")
st.title("🚀 V9 — Real-Time Volume Ranked Momentum Screener")
st.caption("Now fully randomized — no more alphabetical scans • Live volume priority • Fast rotation")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area("Watchlist (optional)", "")

    max_universe = st.slider("Max Symbols to Scan", 50, 600, 200, step=50)
    region_mode = st.radio("Region", ["Global", "US + Canada Only"], index=1)

    enable_enrichment = st.checkbox("Float/Short Interest + News Enrichment (slower)", False)
    enable_alerts     = st.checkbox("Enable Audio Alerts", True)

    st.subheader("Filters")
    max_price = st.number_input("Max Price ($)", 1.0, 1000.0, DEFAULT_MAX_PRICE, step=1.0)
    min_volume = st.number_input("Min Daily Volume", 10_000, 10_000_000, DEFAULT_MIN_VOLUME, step=10_000)
    min_breakout = st.number_input("Min Breakout Score", -50.0, 200.0, 0.0, step=1.0)
    min_pm_move = st.number_input("Min Premarket %", -50.0, 200.0, 0.0, step=0.5)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0, step=0.5)

    squeeze_only  = st.checkbox("Short-Squeeze Only")
    catalyst_only = st.checkbox("Must Have News Catalyst")
    vwap_only     = st.checkbox("Above VWAP Only")

    min_ofb = st.slider("Min Order Flow Bias (0–1)", 0.0, 1.0, 0.50, step=0.01)

    st.markdown("### 🔊 Alert Triggers")
    ALERT_SCORE_THRESHOLD = st.slider("Alert Score ≥", 10, 200, 30)
    ALERT_PM_THRESHOLD    = st.slider("Alert Premarket % ≥", 1, 150, 4)
    ALERT_VWAP_THRESHOLD  = st.slider("Alert VWAP Dist % ≥", 1, 50, 2)

    if st.button("🧹 Reset / Refresh"):
        st.cache_data.clear()
        st.success("Cache cleared — fresh market sweep running now.")

# ========================= SYMBOL SOURCING =========================
@st.cache_data(ttl=600)
def load_symbols():
    nas = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                      sep="|", skipfooter=1, engine="python")
    oth = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                      sep="|", skipfooter=1, engine="python")

    nas["Exchange"] = "NASDAQ"
    oth["Exchange"] = oth["Exchange"].fillna("NYSE/AMEX/ARCA")
    oth = oth.rename(columns={"ACT Symbol":"Symbol"})

    df = pd.concat([nas[["Symbol","Exchange"]], oth[["Symbol","Exchange"]]]).dropna()
    df = df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$", na=False)]
    df["Country"] = "US"
    return df.to_dict("records")

# ========================= RANDOMIZED UNIVERSE FIX =========================
def build_universe():
    wl = watchlist_text.strip()

    # Watchlist mode overrides universe selection
    if wl:
        tick = [t.upper() for t in wl.replace(","," ").replace("\n"," ").split() if t.strip()]
        random.shuffle(tick)   # optional shuffle
        return [{"Symbol": t, "Exchange": "WATCH", "Country": "Unknown"} for t in tick]

    # Full US universe randomized (no alphabetical bias)
    symbols = load_symbols()
    random.shuffle(symbols)  # 🔥 key fix
    return symbols[:max_universe]

# ========================= SCORING =========================
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
    fig.update_layout(height=60,width=160,margin=dict(l=2,r=2,t=2,b=2),
                      xaxis=dict(visible=False),yaxis=dict(visible=False))
    return fig

# ========================= MAIN SCANNER =========================
def scan_one(sym):
    try:
        ticker=sym["Symbol"]
        t=yf.Ticker(ticker)

        hist=t.history(period="10d")
        if hist.empty: return None

        close=hist["Close"]; vol=hist["Volume"]
        price=float(close.iloc[-1]); vol_last=float(vol.iloc[-1])
        if price>max_price or vol_last<min_volume: return None

        y=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>=2 else None
        m3=(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        rsi=100-(100/(1+close.diff().clip(lower=0).rolling(7).mean()/
                         (-close.diff().clip(upper=0).rolling(7).mean())))
        rsi=float(rsi.iloc[-1])

        avg10=vol.mean(); rvol=vol_last/avg10 if avg10>0 else None

        intra=t.history(period="1d",interval="2m",prepost=True)
        if intra.empty: return None
        c=intra["Close"]; o=intra["Open"]; v=intra["Volume"]

        pm=(c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100 if c.iloc[-2]>0 else None
        tp=(intra["High"]+intra["Low"]+intra["Close"])/3
        vwap=(price-(tp*v).sum()/v.sum())/((tp*v).sum()/v.sum())*100 if v.sum()>0 else None

        sign=(c>o).astype(int)-(c<o).astype(int)
        buy=(v*(sign>0)).sum(); sell=(v*(sign<0)).sum()
        flow=buy/(buy+sell) if (buy+sell)>0 else None
        if flow is None or flow<min_ofb: return None

        short=None; sector=""; industry=""; squeeze=False; catalyst=False
        if enable_enrichment:
            info=t.get_info() or {}
            float_shares=info.get("floatShares",0)
            short=info.get("shortPercentOfFloat")

            squeeze = short and short>0.15
            sector=info.get("sector",""); industry=info.get("industry","")

        score=score_model(pm,y,m3,m10,rsi,rvol,catalyst,squeeze,vwap,flow)

        return {
            "Symbol":ticker,"Price":round(price,2),"PM%":round(pm,2) if pm else None,
            "YDay%":round(y,2) if y else None,"3D%":round(m3,2) if m3 else None,
            "10D%":round(m10,2),"RSI7":round(rsi,2),"RVOL_10D":round(rvol,2) if rvol else None,
            "VWAP%":round(vwap,2) if vwap else None,"FlowBias":round(flow,2),
            "Short%Float":round(short*100,2) if short else None,"Squeeze?":squeeze,
            "Sector":sector,"Industry":industry,"Score":score,"Spark":close,
        }
    except:
        return None

# ========================= EXECUTE SCAN =========================
with st.spinner("Scanning real-time randomized universe…"):
    universe=build_universe()
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        results=[f.result() for f in concurrent.futures.as_completed([ex.submit(scan_one,s) for s in universe]) if f.result()]

df=pd.DataFrame(results)

if df.empty:
    st.error("No symbols met criteria — relax filters.")
    st.stop()

df=df[df.Score>=min_breakout]
df=df.sort_values(["Score","PM%","FlowBias"],ascending=[False,False,False])

# ========================= DISPLAY =========================
st.subheader(f"🔥 Live Volume Ranked (Randomized) — {len(df)} symbols")

if "alerted" not in st.session_state:
    st.session_state.alerted=set()

for _,r in df.iterrows():
    sym=r.Symbol

    if enable_alerts and sym not in st.session_state.alerted:
        if r.Score>=ALERT_SCORE_THRESHOLD:
            st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} | Score {r.Score}")
        elif r["PM%"] and r["PM%"]>=ALERT_PM_THRESHOLD:
            st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} | PM {r['PM%']}%")
        elif r["VWAP%"] and r["VWAP%"]>=ALERT_VWAP_THRESHOLD:
            st.session_state.alerted.add(sym); st.warning(f"🔔 {sym} | VWAP {r['VWAP%']}%")

    c1,c2,c3 = st.columns([2,3,3])
    c1.markdown(f"### **{sym}**  💲{r.Price}")
    c1.write(f"🔥 Score {r.Score}")
    c1.write(f"Premarket {r['PM%']}% | Yesterday {r['YDay%']}%")
    c1.write(f"FlowBias {r.FlowBias} | RVOL {r.RVOL_10D}x")

    c2.write(f"3D {r['3D%']}% | 10D {r['10D%']}% | RSI7 {r.RSI7}")
    c2.write(f"VWAP {r['VWAP%']}% | Squeeze {r['Squeeze?']}")
    c2.write(f"Sector: {r.Sector} | Industry: {r.Industry}")

    c3.plotly_chart(spark(r.Spark))

    st.divider()

# ========================= EXPORT =========================
st.download_button("📥 Download Results CSV",
                   df.to_csv(index=False),
                   "V9_Randomized_Live_Screener.csv")

st.caption("For research & education only — not financial advice.")


