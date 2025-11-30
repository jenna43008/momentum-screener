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
DEFAULT_MIN_BREAKOUT  = 0.0

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v89")

# ========================= PAGE SETUP =========================
st.set_page_config(
    page_title="V8.9 — Live Volume Ranked Momentum Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 V8.9 — 10-Day Momentum Breakout Screener")
st.caption("Real-time volume universe • US/Canada toggle • OFB filter • Faster scanning")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.subheader("Universe Selection")

    watchlist_text = st.text_area(
        "Watchlist (comma/newline separated):",
        height=60,
        placeholder="AAPL, TSLA, NVDA ..."
    )

    max_universe = st.slider(
        "Max tickers to scan when no watchlist",
        min_value=50, max_value=800, value=300, step=50
    )

    region_mode = st.radio(
        "Market Region:",
        ["Global", "US + Canada Only"],
        index=1
    )

    enable_enrichment = st.checkbox(
        "Enable Float/Short/News (slower)",
        value=False
    )

    st.divider()
    st.subheader("Filters")

    max_price = st.number_input("Max Price ($)",
                                min_value=1.0, max_value=2000.0,
                                value=DEFAULT_MAX_PRICE, step=1.0)

    min_volume = st.number_input("Min Daily Volume",
                                min_value=10_000, max_value=20_000_000,
                                value=DEFAULT_MIN_VOLUME, step=10_000)

    min_breakout = st.number_input("Min Breakout Score",
                                   min_value=-50.0, max_value=200.0,
                                   value=0.0, step=1.0)

    min_pm_move = st.number_input("Min Premarket %", -50.0, 200.0, 0.0, 0.5)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0, 0.5)

    squeeze_only  = st.checkbox("Short Squeeze Only")
    catalyst_only = st.checkbox("News Required")
    vwap_only     = st.checkbox("Must be above VWAP")

    min_ofb = st.slider("Min Order Flow Bias (0-1)",
                        min_value=0.00, max_value=1.00,
                        value=0.50, step=0.01)

    st.divider()
    st.subheader("🔊 Alerts")

    ALERT_SCORE_THRESHOLD = st.slider("Alert when Score ≥", 10, 200, 30, 5)
    ALERT_PM_THRESHOLD    = st.slider("Alert when PM % ≥", 1, 150, 4)
    ALERT_VWAP_THRESHOLD  = st.slider("Alert when VWAP% ≥", 1, 50, 2)

    if st.button("Refresh Now"):
        st.cache_data.clear()
        st.success("Cache cleared — rescanning")

# ========================= LOAD SYMBOLS =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                         sep="|", skipfooter=1, engine="python")
    other  = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                         sep="|", skipfooter=1, engine="python")

    nasdaq["Exchange"] = "NASDAQ"
    other["Exchange"]  = other["Exchange"].fillna("NYSE/AMEX/ARCA")
    other = other.rename(columns={"ACT Symbol": "Symbol"})

    df = pd.concat([nasdaq[["Symbol","Exchange"]],
                    other[["Symbol","Exchange"]]])

    df = df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$", na=False)]
    df["Country"] = "US"
    return df.to_dict("records")

# ========================= FIXED REAL-TIME TICKER UNIVERSE =========================
def build_universe(watchlist_text, max_universe):
    wl = watchlist_text.strip()

    if wl:
        return [{"Symbol":s.upper(),"Exchange":"WATCH"} for s in wl.replace(","," ").split()]

    tickers = load_symbols()

    def get_volume(x):
        try: return yf.Ticker(x["Symbol"]).fast_info.get("last_volume",0)
        except: return 0

    ranked = sorted(tickers, key=get_volume, reverse=True)
    return ranked[:max_universe]       # 🔥 Real market leaderboard feed

# ========================= SCAN ENGINE =========================
def scan_one(sym, enrich, region_mode, min_ofb):
    try:
        t = sym["Symbol"]; ticker = yf.Ticker(t)

        hist = ticker.history(period="10d", interval="1d")
        if hist.empty: return None

        price = float(hist.Close.iloc[-1])
        vol   = float(hist.Volume.iloc[-1])

        if price > max_price or vol < min_volume: return None

        # Momentum windows
        close = hist.Close
        y  = ((close[-1]/close[-2]-1)*100) if len(close)>=2 else None
        m3 = ((close[-1]/close[-4]-1)*100) if len(close)>=4 else None
        m10= ((close[-1]/close[0]-1)*100)

        # RSI7
        delta = close.diff()
        rsi7 = 100-(100/(1+(delta.clip(lower=0).rolling(7).mean() /
                           -delta.clip(upper=0).rolling(7).mean())))
        rsi7 = float(rsi7.iloc[-1])

        ema10 = close.ewm(span=10).mean().iloc[-1]
        trend = "🔥 Breakout" if price>ema10 and rsi7>55 else "Neutral"

        rvol = vol/hist.Volume.mean()

        # Intraday microstructure
        intra=ticker.history(period="1d",interval="2m",prepost=True)
        pm=vwap=ofb=None
        if len(intra)>3:
            pm=(intra.Close[-1]/intra.Close[-2]-1)*100
            tp=(intra.High+intra.Low+intra.Close)/3
            vwap=(tp*intra.Volume).sum()/intra.Volume.sum()
            vwap=(price-vwap)/vwap*100
            diff=(intra.Close>intra.Open).astype(int)-(intra.Close<intra.Open).astype(int)
            buy=(intra.Volume*(diff>0)).sum(); sell=(intra.Volume*(diff<0)).sum()
            ofb=buy/(buy+sell)

        if ofb is None or ofb < min_ofb:
            return None

        # Score
        score=0
        for v,w in [(pm,1.6),(y,0.8),(m3,1.2),(m10,0.6)]: score+=max(v or 0,0)*w
        if rsi7>55: score+=(rsi7-55)*0.4
        if rvol>1.2: score+=(rvol-1.2)*2
        if vwap>0: score+=min(vwap,6)*1.5
        score+=((ofb-0.5)*22)

        prob=1/(1+math.exp(-score/20))*100

        return {
            "Symbol":t,"Price":round(price,2),"Volume":int(vol),
            "Score":round(score,2),"Prob_Rise%":round(prob,1),
            "PM%":round(pm,2) if pm else None,"YDay%":round(y,2) if y else None,
            "3D%":round(m3,2) if m3 else None,"10D%":round(m10,2),
            "RSI7":round(rsi7,2),"RVOL":round(rvol,2),
            "VWAP%":round(vwap,2) if vwap else None,"OFB":round(ofb,2),
            "Trend":trend,"Spark":close
        }

    except:
        return None


@st.cache_data(ttl=6)
def run_scan(w,l,e,r,min_ofb):
    pool=build_universe(w,l)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed(
            [ex.submit(scan_one,s,e,r,min_ofb) for s in pool]):
            r=f.result()
            if r: out.append(r)
    return pd.DataFrame(out)

# ========================= DISPLAY =========================
with st.spinner("Scanning Live Market..."):
    df=run_scan(watchlist_text,max_universe,enable_enrichment,region_mode,min_ofb)

if df.empty:
    st.warning("No qualifying stocks — relax filter or check market session.")
else:
    df=df[df["Score"]>=min_breakout]

    if min_pm_move: df=df[df["PM%"].fillna(-999)>=min_pm_move]
    if min_yday_gain: df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
    if vwap_only: df=df[df["VWAP%"].fillna(-999)>0]

    df=df.sort_values(["Score","PM%","RSI7"],ascending=[False,False,False])

    st.subheader(f"🔥 {len(df)} Live Momentum Candidates")

    for _,row in df.iterrows():
        sym=row.Symbol
        c1,c2,c3,c4=st.columns([2,2,3,3])

        c1.markdown(f"**{sym}**")
        c1.write(f"💲 {row.Price}")
        c1.write(f"📈 {row.Score}  |  🔮 {row['Prob_Rise%']}%")
        c1.write(f"OFB: {row.OFB}  |  Trend: {row.Trend}")

        c2.write(f"PM% {row['PM%']}")
        c2.write(f"YDay {row['YDay%']}%")
        c2.write(f"3D {row['3D%']}% | 10D {row['10D%']}%")
        c2.write(f"RSI7 {row.RSI7} | RVOL {row.RVOL}")

        c3.write(f"VWAP {row['VWAP%']}%")
        c3.write(f"Volume {row.Volume:,}")

        c4.plotly_chart(
            go.Figure(go.Scatter(y=row.Spark.values,mode="lines")),
            use_container_width=True
        )
        st.divider()

    st.download_button("📥 Export CSV", df.to_csv(index=False),
                       "screener_results.csv","text/csv")

st.caption("Not financial advice — for analysis only.")

