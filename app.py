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
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v8")

# ========================= PAGE SETUP =========================
st.set_page_config(
    page_title="V8 – 10-Day Momentum Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 V8 — 10-Day Momentum Breakout Screener (Faster + Watchlist Mode)")
st.caption("Short-window model • EMA10 • RSI(7) • 3D & 10D momentum • 10D RVOL • VWAP + Order Flow • Audio Alerts")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area("Watchlist tickers:", value="", height=80)

    max_universe = st.slider("Max Symbols Scanned (when no watchlist)", 50, 600, 200, 50)

    enable_enrichment = st.checkbox("📌 Include Float/Short + News (slower)", value=False)

    st.markdown("---")
    st.header("Filters")

    max_price = st.number_input("Max Price ($)", 1.0, 1000.0, DEFAULT_MAX_PRICE)
    min_volume = st.number_input("Min Daily Volume", 10_000, 10_000_000, DEFAULT_MIN_VOLUME)
    min_breakout = st.number_input("Min Breakout Score", -50.0, 200.0, 0.0)
    min_pm_move = st.number_input("Min Premarket %", -50.0, 200.0, 0.0)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0)

    squeeze_only = st.checkbox("Short-Squeeze Only")
    catalyst_only = st.checkbox("Must Have News/Earnings")
    vwap_only = st.checkbox("Above VWAP Only")

    st.markdown("---")
    st.subheader("🔊 Alerts")

    ALERT_SCORE_THRESHOLD = st.slider("Score Alert ≥", 10, 200, 30)
    ALERT_PM_THRESHOLD    = st.slider("Premarket % Alert ≥", 1, 150, 4)
    ALERT_VWAP_THRESHOLD  = st.slider("VWAP Distance % ≥", 1, 50, 2)

    st.markdown("---")
    if st.button("♻ Refresh Now"):
        st.cache_data.clear()
        st.success("Cache cleared — new scan running...")

# ========================= LOAD SYMBOLS =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                         sep="|", skipfooter=1, engine="python")
    other  = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                         sep="|", skipfooter=1, engine="python")

    nasdaq["Exchange"] = "NASDAQ"
    other["Exchange"] = other["Exchange"].fillna("NYSE/AMEX/ARCA")
    other = other.rename(columns={"ACT Symbol": "Symbol"})

    df = pd.concat([nasdaq[["Symbol","ETF","Exchange"]], other[["Symbol","ETF","Exchange"]]])
    df = df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$", na=False)]
    return df.to_dict("records")

def build_universe(text, max_universe):
    if text.strip():
        wl = text.replace("\n"," ").replace(","," ").split()
        return [{"Symbol":s.upper(),"Exchange":"WATCH"} for s in wl]
    return load_symbols()[:max_universe]

# ========================= SCORING =========================
def short_window_score(pm,yday,m3,m10,rsi7,rvol10,catalyst,squeeze,vwap,flow):
    score=0
    if pm: score+=pm*1.6
    if yday: score+=yday*0.8
    if m3: score+=m3*1.2
    if m10: score+=m10*0.6
    if rsi7>55: score+=(rsi7-55)*0.4
    if rvol10>1.2: score+=(rvol10-1.2)*2
    if vwap>0: score+=min(vwap,6)*1.5
    if flow: score+=(flow-0.5)*22
    if catalyst: score+=8
    if squeeze: score+=12
    return round(score,2)

def breakout_probability(score):
    return round((1/(1+math.exp(-score/20)))*100,1)

def multi_timeframe_label(pm,m3,m10):
    b=sum([(pm>0 if pm else 0),(m3>0 if m3 else 0),(m10>0 if m10 else 0)])
    return ("✅ Bullish" if b==3 else "🟢 Leaning" if b==2
            else "🟡 Mixed" if b==1 else "🔻 Weak")

# ========================= SCAN ONE =========================
def scan_one(sym, enrich):
    try:
        t=sym["Symbol"]
        s=yf.Ticker(t)

        hist=s.history(period="10d",interval="1d")
        if hist.empty or len(hist)<5: return None

        close=hist["Close"]; vol=hist["Volume"]
        price=float(close.iloc[-1])
        vol_last=float(vol.iloc[-1])   # ⭐ NEW current volume extracted

        if price>max_price or vol_last<min_volume: return None

        yday=(price-close.iloc[-2])/close.iloc[-2]*100 if len(close)>=2 else None
        m3=(price-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10=(price-close.iloc[0])/close.iloc[0]*100

        delta=close.diff()
        rsi7=float((100-(100/(1+(delta.clip(lower=0).rolling(7).mean()/
                                     (-delta.clip(upper=0).rolling(7).mean()))))).iloc[-1])

        ema10=float(close.ewm(span=10).mean().iloc[-1])

        avg10=vol.mean(); rvol10=vol_last/avg10 if avg10 else None

        intra=s.history(period="1d",interval="2m",prepost=True)
        pm=vwap=flow=None

        if not intra.empty and len(intra)>=3:
            pm=(intra["Close"].iloc[-1]-intra["Close"].iloc[-2])/intra["Close"].iloc[-2]*100
            tp=(intra["High"]+intra["Low"]+intra["Close"])/3
            vwap=(tp*intra["Volume"]).sum()/intra["Volume"].sum()
            vwap=(price-vwap)/vwap*100 if vwap else None
            of=intra[["Open","Close","Volume"]].dropna()
            if not of.empty:
                sign=(of["Close"]>of["Open"]).astype(int)-(of["Close"]<of["Open"]).astype(int)
                buy=(of["Volume"]*(sign>0)).sum(); sell=(of["Volume"]*(sign<0)).sum()
                flow=buy/(buy+sell) if buy+sell>0 else None

        squeeze=low=catalyst=False; sec="Unknown"; ind="Unknown"; short=None
        if enrich:
            info=s.get_info() or {}
            fl=info.get("floatShares"); sh=info.get("shortPercentOfFloat")
            sec=info.get("sector","Unknown"); ind=info.get("industry","Unknown")
            low=fl and fl<20_000_000; squeeze=sh and sh>0.15; short=sh*100 if sh else None
            try:
                n=s.get_news()
                if n and "providerPublishTime" in n[0]:
                    p=datetime.fromtimestamp(n[0]["providerPublishTime"],tz=timezone.utc)
                    catalyst=(datetime.now(timezone.utc)-p).days<=3
            except: pass

        score=short_window_score(pm,yday,m3,m10,rsi7,rvol10,catalyst,squeeze,vwap,flow)
        prob=breakout_probability(score)

        return {
            "Symbol":t,"Exchange":sym.get("Exchange","US"),
            "Price":round(price,2),
            "Volume":int(vol_last),      # ⭐ NEW VOLUME ADDED HERE
            "Score":score,"Prob_Rise%":prob,
            "PM%":pm,"YDay%":yday,"3D%":m3,"10D%":m10,
            "RSI7":rsi7,"RVOL_10D":rvol10,"VWAP%":vwap,"FlowBias":flow,
            "Squeeze?":squeeze,"LowFloat?":low,"Sector":sec,"Industry":ind,
            "Catalyst":catalyst,"MTF_Trend":multi_timeframe_label(pm,m3,m10),
            "Spark":close
        }

    except: return None

@st.cache_data(ttl=6)
def run_scan(text,maxn,enrich):
    u=build_universe(text,maxn)
    res=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s,enrich) for s in u]):
            r=f.result()
            if r: res.append(r)
    return pd.DataFrame(res)

# ========================= UI DISPLAY =========================
with st.spinner("Scanning..."):
    df=run_scan(watchlist_text,max_universe,enable_enrichment)

if df.empty:
    st.error("No results found.")
else:
    df=df[df["Score"]>=min_breakout]
    if min_pm_move!=0: df=df[df["PM%"].fillna(-999)>=min_pm_move]
    if min_yday_gain!=0: df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
    if squeeze_only: df=df[df["Squeeze?"]]
    if catalyst_only: df=df[df["Catalyst"]]
    if vwap_only: df=df[df["VWAP%"].fillna(-999)>0]

    df=df.sort_values(by=["Score","PM%","RSI7"],ascending=[False,False,False])

    st.subheader(f"🔥 10-Day Momentum — {len(df)} tickers")

    for _,row in df.iterrows():
        sym=row["Symbol"]
        c1,c2,c3,c4=st.columns([2,3,3,3])

        c1.markdown(f"**{sym}** ({row['Exchange']})")
        c1.write(f"💲 Price: {row['Price']}")
        c1.write(f"📊 Volume: {row['Volume']:,}")  # ⭐ NEW — NOW VISIBLE IN UI
        c1.write(f"🔥 Score: {row['Score']}")
        c1.write(f"📈 Prob Rise: {row['Prob_Rise%']}%")
        c1.write(row["MTF_Trend"])

        c2.write(f"PM%: {row['PM%']}")
        c2.write(f"YDay: {row['YDay%']}")
        c2.write(f"3D%: {row['3D%']} | 10D%: {row['10D%']}")
        c2.write(f"RSI7: {row['RSI7']} | RVOL10: {row['RVOL_10D']}")

        c3.write(f"VWAP Dist: {row['VWAP%']}")
        c3.write(f"Flow Bias: {row['FlowBias']}")
        if enable_enrichment:
            c3.write(f"Squeeze: {row['Squeeze?']} | Low Float: {row['LowFloat?']}")
            c3.write(f"{row['Sector']} / {row['Industry']}")

        c4.plotly_chart(go.Figure(data=[go.Scatter(y=row["Spark"].values)]), use_container_width=True)

    st.download_button(
        "📥 Download CSV",
        data=df.to_csv(index=False),
        file_name="v8_screener.csv"
    )

st.caption("For research + education only.")
