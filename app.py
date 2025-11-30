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
AUTO_REFRESH_MS       = 10_000       # auto-refresh every 10 seconds
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 50.0
DEFAULT_MIN_VOLUME    = 100_000
DEFAULT_MIN_BREAKOUT  = 0.0

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v8")

# ========================= PAGE SETUP =========================
st.set_page_config(
    page_title="V8 – 10-Day Momentum Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 V8 — 10-Day Momentum Breakout Screener (Faster + Watchlist Mode)")
st.caption("Short-window model • EMA10 • RSI(7) • 3D & 10D momentum • 10D RVOL • VWAP + order flow • Audio alerts")

# ========================= SIDEBAR CONTROLS =========================
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area("Watchlist tickers (comma/space/newline separated):", "", height=80)

    max_universe = st.slider("Max symbols to scan when no watchlist", 50, 600, 200, 50)

    region_mode = st.radio("Region Filter",
        ["Global (no country filter)", "US + Canada Only"], index=1)

    enable_enrichment = st.checkbox("Include float/short + news (slower, more data)", False)

    # 🔥 NEW — USER WANTED ALERTS TOGGLE
    enable_alerts = st.checkbox("🔕 Enable Audio Alerts", True)

    st.markdown("---")
    st.header("Filters")

    max_price     = st.number_input("Max Price ($)", 1.0, 1000.0, DEFAULT_MAX_PRICE, 1.0)
    min_volume    = st.number_input("Min Daily Volume", 10000, 10_000_000, DEFAULT_MIN_VOLUME, 10000)
    min_breakout  = st.number_input("Min Breakout Score", -50.0, 200.0, 0.0, 1.0)
    min_pm_move   = st.number_input("Min Premarket %", -50.0, 200.0, 0.0, 0.5)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0, 0.5)

    squeeze_only  = st.checkbox("Short-Squeeze Only")
    catalyst_only = st.checkbox("Must Have News/Earnings")
    vwap_only     = st.checkbox("Above VWAP Only (VWAP% > 0)")

    min_ofb = st.slider("Min Order Flow Bias (0–1, buyer control)", 0.00, 1.00, 0.50, 0.01)

    st.markdown("---")
    st.subheader("🔊 Alert Thresholds")

    ALERT_SCORE_THRESHOLD = st.slider("Alert when Score ≥", 10, 200, 30, 5)
    ALERT_PM_THRESHOLD    = st.slider("Alert when Premarket % ≥", 1, 150, 4, 1)
    ALERT_VWAP_THRESHOLD  = st.slider("Alert when VWAP Dist % ≥", 1, 50, 2, 1)

    if st.button("🧹 Refresh Now"):
        st.cache_data.clear()
        st.success("Cache cleared — new scan running now.")


# ========================= SYMBOLS =========================
@st.cache_data(ttl=900)
def load_symbols():
    nas = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", sep="|", skipfooter=1, engine="python")
    oth = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", sep="|", skipfooter=1, engine="python")

    nas["Exchange"]="NASDAQ"
    oth["Exchange"]=oth["Exchange"].fillna("NYSE/AMEX/ARCA")
    oth = oth.rename(columns={"ACT Symbol":"Symbol"})

    df=pd.concat([nas[["Symbol","ETF","Exchange"]], oth[["Symbol","ETF","Exchange"]]])
    df=df[df.Symbol.str.contains(r"^[A-Z]{1,5}$",na=False)]
    df["Country"]="US"
    return df.to_dict("records")


def build_universe(wlist,max_u):
    wl=wlist.strip()
    if wl:
        tick=sorted(set(s.upper() for s in wl.replace(","," ").replace("\n"," ").split()))
        return [{"Symbol":t,"Exchange":"WATCH","Country":"Unknown"} for t in tick]
    return load_symbols()[:max_u]


# ========================= SCORING =========================
def short_window_score(pm,y3,y10,m3,m10,rsi,rvol,cat,sq,vwap,flow):
    s=0
    if pm:  s+=pm*1.6
    if y3:  s+=y3*0.8
    if m3:  s+=m3*1.2
    if m10: s+=m10*0.6
    if rsi and rsi>55: s+=(rsi-55)*0.4
    if rvol and rvol>1.2: s+=(rvol-1.2)*2.0
    if vwap and vwap>0: s+=min(vwap,6)*1.5
    if flow: s+=(flow-0.5)*22
    if cat: s+=8
    if sq:  s+=12
    return round(s,2)

def breakout_probability(x):
    try: return round(100/(1+math.exp(-x/20)),1)
    except: return None


# ========================= SCAN =========================
def scan_one(sym, enrich, region, minflow):
    try:
        t=sym["Symbol"]
        stock=yf.Ticker(t)

        hist=stock.history(period="10d")
        if hist.empty or len(hist)<5: return None
        close=hist.Close; volume=hist.Volume

        price=float(close.iloc[-1]); v=volume.iloc[-1]
        if price>max_price or v<min_volume: return None

        # returns
        y3=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>=2 else None
        m3=(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        delta=close.diff()
        gain=delta.clip(lower=0).rolling(7).mean()
        loss=(-delta.clip(upper=0)).rolling(7).mean()
        rsi=100-(100/(1+gain/loss)); rsi=float(rsi.iloc[-1])

        rvol=v/volume.mean() if volume.mean()>0 else None

        intra=stock.history(period=INTRADAY_RANGE,interval=INTRADAY_INTERVAL,prepost=True)
        if intra.empty: return None
        c=intra.Close; o=intra.Open; iv=intra.Volume

        pm=(c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100 if c.iloc[-2]>0 else None
        tp=(intra.High+intra.Low+intra.Close)/3
        vwap=((tp*iv).sum()/iv.sum()) if iv.sum()>0 else None
        vwap_dist=(price-vwap)/vwap*100 if vwap else None

        sign=(c>o).astype(int)-(c<o).astype(int)
        buy=(iv*(sign>0)).sum(); sell=(iv*(sign<0)).sum()
        flow=buy/(buy+sell) if (buy+sell)>0 else None
        if flow is None or flow<minflow: return None

        squeeze=False; catalyst=False; sector="Unknown"; industry="Unknown"; sp=None
        if enrich:
            info=stock.get_info() or {}
            fs=info.get("floatShares"); sp=info.get("shortPercentOfFloat")
            sector=info.get("sector","Unknown"); industry=info.get("industry","Unknown")
            squeeze=bool(sp and sp>0.15)
            try:
                news=stock.get_news()
                if news and "providerPublishTime" in news[0]:
                    pub=datetime.fromtimestamp(news[0]["providerPublishTime"],tz=timezone.utc)
                    catalyst=(datetime.now(timezone.utc)-pub).days<=3
            except: pass

        score=short_window_score(pm,y3,m3,m10,rsi,rvol,catalyst,squeeze,vwap_dist,flow)

        return {
            "Symbol":t,"Price":round(price,2),
            "Score":score,"Prob_Rise%":breakout_probability(score),
            "PM%":round(pm,2) if pm else None, "YDay%":round(y3,2) if y3 else None,
            "3D%":round(m3,2) if m3 else None, "10D%":round(m10,2),
            "RSI7":round(rsi,2),"RVOL_10D":round(rvol,2) if rvol else None,
            "VWAP%":round(vwap_dist,2) if vwap_dist else None,
            "FlowBias":round(flow,2),
            "Squeeze?":squeeze,"ShortFloat%":round(sp*100,2) if sp else None,
            "Sector":sector,"Industry":industry,"Spark":close
        }
    except: return None


@st.cache_data(ttl=6)
def run_scan(wl,maxu,enrich,region,minflow):
    universe=build_universe(wl,maxu)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s,enrich,region,minflow) for s in universe]):
            r=f.result()
            if r: out.append(r)
    return pd.DataFrame(out) if out else pd.DataFrame()

# ========================= ALERT MEMORY =========================
if "alerted" not in st.session_state:
    st.session_state.alerted=set()

if "alert_log" not in st.session_state:
    st.session_state.alert_log=[]

# ========================= LIVE ALERT BANNER =========================
if enable_alerts and st.session_state.alert_log:
    st.markdown("### 🔥 Recent Alerts")
    st.info(" | ".join(st.session_state.alert_log[-8:]))

# ========================= ALERT FUNCTION =========================
def trigger_audio_alert(symbol,reason):
    if enable_alerts:
        st.session_state.alerted.add(symbol)
        st.session_state.alert_log.append(f"{symbol}: {reason}")
        st.warning(f"🔔 {symbol}: {reason}")
        st.markdown("""
            <audio autoplay>
            <source src="https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg" type="audio/ogg">
            </audio>
        """, unsafe_allow_html=True)

# ========================= MAIN =========================
with st.spinner("Scanning markets…"):
    df=run_scan(watchlist_text,max_universe,enable_enrichment,region_mode,min_ofb)

if df.empty:
    st.error("No results — loosen filters"); st.stop()

df=df[df.Score>=min_breakout]
if min_pm_move: df=df[df["PM%"].fillna(-999)>=min_pm_move]
if min_yday_gain: df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
if squeeze_only: df=df[df["Squeeze?"]]
if catalyst_only: df=df[df["Squeeze?"]]  # optional — tied to catalyst
if vwap_only: df=df[df["VWAP%"].fillna(-999)>0]

df=df.sort_values(["Score","PM%","RSI7"],ascending=[False,False,False])

st.subheader(f"🔥 Momentum Board — {len(df)} symbols")

# ========================= ROW DISPLAY + ALERTS =========================
for _,r in df.iterrows():
    sym=r.Symbol

    if enable_alerts and sym not in st.session_state.alerted:
        if r.Score>=ALERT_SCORE_THRESHOLD: trigger_audio_alert(sym,f"Score {r.Score}")
        elif r["PM%"] and r["PM%"]>=ALERT_PM_THRESHOLD: trigger_audio_alert(sym,f"PM {r['PM%']}%")
        elif r["VWAP%"] and r["VWAP%"]>=ALERT_VWAP_THRESHOLD: trigger_audio_alert(sym,f"VWAP {r['VWAP%']}%")

    c1,c2,c3=st.columns([2,3,3])
    c1.markdown(f"### {sym} 💲{r.Price}")
    c1.write(f"🔥 Score {r.Score}  |  Prob {r.Prob_Rise%}%")
    c1.write(f"PM {r['PM%']}%  |  3D {r['3D%']}%  |  10D {r['10D%']}%")

    c2.write(f"RSI7 {r.RSI7}  | RVOL {r.RVOL_10D}x | FlowBias {r.FlowBias}")
    c2.write(f"VWAP {r['VWAP%']}% | Squeeze {r['Squeeze?']}")
    c2.write(f"Sector {r.Sector} | Industry {r.Industry}")

    c3.plotly_chart(go.Figure(data=[go.Scatter(y=r.Spark.values,mode="lines")]),
                    use_container_width=True)
    st.divider()

st.download_button("📥 Download CSV",df.to_csv(index=False),"screener.csv")
