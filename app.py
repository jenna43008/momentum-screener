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
AUTO_REFRESH_MS       = 60_000        # refresh every 60s
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 5.0
DEFAULT_MIN_VOLUME    = 100_000
DEFAULT_MIN_BREAKOUT  = 0.0

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v8")

# ========================= PAGE =========================
st.set_page_config(page_title="V8 Momentum Screener", layout="wide")
st.title("🚀 V8 — 10-Day Momentum Breakout Screener")
st.caption("EMA10 • RSI7 • 3D/10D momentum • RVOL10 • VWAP & Order Flow • Alerts + Watchlist")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Universe")
    watchlist_text = st.text_area("Watchlist (optional)", "", height=80)

    max_universe = st.slider("Max symbols to scan", 50, 600, 200, step=50)

    enable_enrichment = st.checkbox("Include float/short/news (slower)", False)

    st.markdown("---")
    st.header("Filters")
    max_price     = st.number_input("Max Price ($)", 1.0, 1000.0, DEFAULT_MAX_PRICE, 1.0)
    min_volume    = st.number_input("Min Daily Volume", 10_000, 10_000_000, DEFAULT_MIN_VOLUME, 10_000)
    min_breakout  = st.number_input("Min Breakout Score", -50.0, 200.0, 0.0, 1.0)
    min_pm_move   = st.number_input("Min Premarket %", -50.0, 200.0, 0.0, 0.5)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0, 0.5)

    squeeze_only  = st.checkbox("Short-Squeeze Only")
    catalyst_only = st.checkbox("Must Have News/Earnings")
    vwap_only     = st.checkbox("Above VWAP Only")

    st.markdown("---")
    st.subheader("🔊 AUDIO ALERT SETTINGS")

    # 🚨 NEW: ALERT ON/OFF SWITCH
    ENABLE_ALERTS = st.checkbox("Enable Alerts", True)  # <--- THIS IS THE NEW TOGGLE

    ALERT_SCORE_THRESHOLD = st.slider("Alert on Score ≥", 10, 200, 30, 5)
    ALERT_PM_THRESHOLD    = st.slider("Alert on Premarket % ≥", 1, 150, 4, 1)
    ALERT_VWAP_THRESHOLD  = st.slider("Alert on VWAP% ≥", 1, 50, 2, 1)

    st.markdown("---")
    if st.button("🧹 RESET / CLEAR CACHE"):
        st.cache_data.clear()
        st.success("Cache cleared — rescanning...")

# ========================= SYMBOL LOAD =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                         sep="|", skipfooter=1, engine="python")
    other  = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                         sep="|", skipfooter=1, engine="python")

    nasdaq["Exchange"]="NASDAQ"
    other["Exchange"]=other["Exchange"].fillna("NYSE/AMEX/ARCA")
    other = other.rename(columns={"ACT Symbol":"Symbol"})

    df = pd.concat([nasdaq[["Symbol","Exchange"]],
                    other [["Symbol","Exchange"]]]).dropna()

    df = df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$", na=False)]
    return df.to_dict("records")

def build_universe(watchlist_text,max_universe):
    wl = watchlist_text.strip()
    if wl:
        raw = wl.replace("\n"," ").replace(","," ").split()
        tick = sorted(set(x.upper() for x in raw if x.strip()))
        return [{"Symbol":t,"Exchange":"WATCH"} for t in tick]

    return load_symbols()[:max_universe]

# ========================= SCORING =========================
def short_window_score(pm,y,m3,m10,rsi,rvol,c,sq,vwap,flow):
    s=0
    if pm: s+=pm*1.6
    if y: s+=y*0.8
    if m3:s+=m3*1.2
    if m10:s+=m10*0.6
    if rsi and rsi>55: s+=(rsi-55)*0.4
    if rvol and rvol>1.2: s+=(rvol-1.2)*2
    if vwap and vwap>0: s+=min(vwap,6)*1.5
    if flow: s+=(flow-0.5)*22
    if c: s+=8
    if sq: s+=12
    return round(s,2)

def breakout_probability(score):
    try: return round((1/(1+math.exp(-score/20)))*100,1)
    except: return None

def multi_timeframe_label(pm,m3,m10):
    b = sum([(pm and pm>0),(m3 and m3>0),(m10 and m10>0)])
    return ["🔻 Not Aligned","🟡 Mixed","🟢 Bullish","✅ Full Alignment"][b]

# ========================= SCAN CORE =========================
def scan_one(sym,enable_enrichment):
    try:
        t = yf.Ticker(sym["Symbol"])
        hist = t.history(period="10d")
        if hist.empty: return None

        close=hist["Close"]; vol=hist["Volume"]
        price=float(close.iloc[-1]); vol_last=float(vol.iloc[-1])
        if price>max_price or vol_last<min_volume: return None

        y   = (close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>=2 else None
        m3  = (close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10 = (close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        rsi = 100-(100/(1+(close.diff().clip(lower=0).rolling(7).mean()/
                           (-close.diff().clip(upper=0).rolling(7).mean()))))
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

        if flow is None: return None

        squeeze=False; catalyst=False; sector="Unknown"; industry="Unknown"
        if enable_enrichment:
            info=t.get_info() or {}
            short=info.get("shortPercentOfFloat")
            sector=info.get("sector","Unknown")
            industry=info.get("industry","Unknown")
            squeeze = short and short>0.15

        score=short_window_score(pm,y,m3,m10,rsi,rvol,catalyst,squeeze,vwap,flow)
        prob = breakout_probability(score)

        return {
            "Symbol":sym["Symbol"],"Exchange":sym.get("Exchange",""),
            "Price":round(price,2),"Score":score,"Prob_Rise%":prob,
            "PM%":pm,"YDay%":y,"3D%":m3,"10D%":m10,"RSI7":round(rsi,2),
            "RVOL_10D":round(rvol,2) if rvol else None,
            "VWAP%":round(vwap,2) if vwap else None,
            "FlowBias":round(flow,2),
            "Spark":close,
        }
    except:
        return None

@st.cache_data(ttl=6)
def run_scan(w,max_u,en):
    universe=build_universe(w,max_u)
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        results=[f.result() for f in concurrent.futures.as_completed([ex.submit(scan_one,s,en) for s in universe]) if f.result()]
    return pd.DataFrame(results) if results else pd.DataFrame()

# ========================= AUDIO STATE =========================
if "alerted" not in st.session_state:
    st.session_state.alerted=set()

def trigger_alert(sym,msg):
    if not ENABLE_ALERTS: return   # <-- ALERT KILL SWITCH HERE
    st.session_state.alerted.add(sym)
    st.warning(f"🔔 {sym}: {msg}")
    st.markdown("""
    <audio autoplay>
        <source src="https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg" type="audio/ogg">
    </audio>""", unsafe_allow_html=True)

# ========================= RUN =========================
with st.spinner("Scanning market..."):
    df=run_scan(watchlist_text,max_universe,enable_enrichment)

if df.empty: st.error("No results — widen filters."); st.stop()

df=df[df.Score>=min_breakout]
if min_pm_move>0: df=df[df["PM%"].fillna(-999)>=min_pm_move]
if min_yday_gain>0: df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
if squeeze_only: df=df[df["FlowBias"]>0.6]
df=df.sort_values(["Score","PM%","RSI7"],ascending=[False,False,False])

st.subheader(f"🔥 Momentum Board — {len(df)}")

for _,row in df.iterrows():
    sym=row.Symbol

    if ENABLE_ALERTS and sym not in st.session_state.alerted:
        if row.Score>=ALERT_SCORE_THRESHOLD: trigger_alert(sym,f"Score {row.Score}")
        elif row["PM%"] and row["PM%"]>=ALERT_PM_THRESHOLD: trigger_alert(sym,f"PM {row['PM%']}%")
        elif row["VWAP%"] and row["VWAP%"]>=ALERT_VWAP_THRESHOLD: trigger_alert(sym,f"VWAP {row['VWAP%']}%")

    c1,c2,c3,c4=st.columns([2,3,3,3])
    c1.write(f"### {sym}")
    c1.write(f"Price {row.Price} | Score **{row.Score}** | Rise Prob {row['Prob_Rise%']}%")

    c2.write(f"PM {row['PM%']}% | Yday {row['YDay%']}%")
    c2.write(f"3D {row['3D%']}% | 10D {row['10D%']}% | RSI7 {row.RSI7}")

    c3.write(f"VWAP {row['VWAP%']}% | FlowBias {row.FlowBias}")
    c3.write(f"RVOL {row.RVOL_10D}x")

    c4.plotly_chart(sparkline(row["Spark"]), use_container_width=False)

    st.divider()

st.download_button("📥 Export CSV", df.to_csv(index=False), "v8_screen.csv")

st.caption("Educational only — not financial advice.")

