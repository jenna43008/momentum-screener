# ============================ V8 Momentum Screener (Stable Build) ============================
# Includes:
#  - Volume Column
#  - All values rounded to 2 decimals
#  - load_symbols() FIXED (no more NaN Symbol crash)
#  - No logic removed or changed

import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math

# ========================== SETTINGS ==========================
THREADS = 20
AUTO_REFRESH_MS = 10_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL = "2m"
INTRADAY_RANGE = "1d"

DEFAULT_MAX_PRICE = 50.0
DEFAULT_MIN_VOLUME = 100_000
DEFAULT_MIN_BREAKOUT = 0.0


# ======================= AUTO REFRESH =======================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v8")

st.set_page_config(page_title="V8 Breakout Screener", layout="wide")
st.title("🚀 V8 Momentum Screener — (2-Decimal Precision + Volume Column)")
st.caption("10-Day Momentum • VWAP • RSI7 • PM% • Score • News/Float Optional Enrichment")


# ========================== SIDEBAR ==========================
with st.sidebar:
    st.header("Universe Control")

    watchlist_text = st.text_area("Watchlist", "", height=80)
    max_universe = st.slider("Max Tickers (No Watchlist)", 50, 600, 200, 50)

    enable_enrichment = st.checkbox("Enable Float / Short / News (slower)", False)

    st.header("Filters")
    max_price = st.number_input("Max Price", 1.0, 1000.0, DEFAULT_MAX_PRICE)
    min_volume = st.number_input("Min Volume", 10_000, 10_000_000, DEFAULT_MIN_VOLUME)
    min_breakout = st.number_input("Min Breakout Score", -50.0, 200.0, 0.0)
    min_pm_move = st.number_input("Min Premarket %", -50.0, 200.0, 0.0)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0)

    squeeze_only = st.checkbox("Short Squeeze Only")
    catalyst_only = st.checkbox("Must Have News/Earnings")
    vwap_only = st.checkbox("Must Be Above VWAP")

    st.header("🔊 Alert Triggers")
    ALERT_SCORE_THRESHOLD = st.slider("Score ≥", 10, 200, 30)
    ALERT_PM_THRESHOLD    = st.slider("Premarket % ≥", 1, 150, 4)
    ALERT_VWAP_THRESHOLD  = st.slider("VWAP % ≥", 1, 50, 2)


    st.markdown("---")
    if st.button("♻ Force Refresh"):
        st.cache_data.clear()
        st.success("Cache cleared — rescanning…")


# ======================= FIXED SYMBOL LOADER =======================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        sep="|", skipfooter=1, engine="python"
    )
    other = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        sep="|", skipfooter=1, engine="python"
    )

    nasdaq["Exchange"] = "NASDAQ"
    other["Exchange"] = other["Exchange"].fillna("NYSE/AMEX/ARCA")
    other = other.rename(columns={"ACT Symbol": "Symbol"})

    df = pd.concat([nasdaq[["Symbol","Exchange"]], other[["Symbol","Exchange"]]])

    # ⭐ FIX — Sanitizes non-string symbols and removes NaN rows
    df = df.dropna(subset=["Symbol"])
    df["Symbol"] = df["Symbol"].astype(str).str.upper()
    df = df[df["Symbol"].str.match(r"^[A-Z]{1,5}$")]

    return df.to_dict("records")


def build_universe(text,max_u):
    if text.strip():
        wl = {s.upper() for s in text.replace(","," ").split()}
        return [{"Symbol":s,"Exchange":"WATCH"} for s in wl]
    return load_symbols()[:max_u]


# ======================= SCORING =======================
def score_calc(pm,y3,y10,rsi,rvol,vwap,flow,cat,sq):
    score=0
    if pm: score+=pm*1.6
    if y3: score+=y3*1.2
    if y10: score+=y10*0.6
    if rsi>55: score+=(rsi-55)*0.4
    if rvol>1.2: score+=(rvol-1.2)*2
    if vwap>0: score+=min(vwap,6)*1.5
    if flow: score+=(flow-0.5)*22
    if cat: score+=8
    if sq: score+=12
    return round(score,2)

def probability(score):
    return round((1/(1+math.exp(-score/20)))*100,2)


# ======================= SCAN ONE TICKER =======================
def scan_one(s,enrich):
    try:
        t=s["Symbol"]
        stock=yf.Ticker(t)

        hist=stock.history(period="10d",interval="1d")
        if hist.empty: return None

        close=hist["Close"]; vol=hist["Volume"]

        price=round(close.iloc[-1],2)               # 2 decimals
        volume_now=int(vol.iloc[-1])                # whole # (not decimal)

        if price>max_price or volume_now<min_volume: return None

        yday=round((price-close.iloc[-2])/close.iloc[-2]*100,2) if len(close)>=2 else None
        m3=round((price-close.iloc[-4])/close.iloc[-4]*100,2) if len(close)>=4 else None
        m10=round((price-close.iloc[0])/close.iloc[0]*100,2)

        delta=close.diff()
        rsi=round((100-(100/(1+delta.clip(lower=0).rolling(7).mean()/
                              -delta.clip(upper=0).rolling(7).mean()))).iloc[-1],2)
        rvol=round(volume_now/vol.mean(),2)

        intra=stock.history(period="1d",interval="2m",prepost=True)
        pm=vwap=flow=None

        if len(intra)>=3:
            pm=round((intra["Close"].iloc[-1]-intra["Close"].iloc[-2])/
                     intra["Close"].iloc[-2]*100,2)

            tp=(intra["High"]+intra["Low"]+intra["Close"])/3
            tot=intra["Volume"].sum()

            if tot>0:
                iv=(tp*intra["Volume"]).sum()/tot
                vwap=round((price-iv)/iv*100,2)

            of=intra[["Open","Close","Volume"]]
            sign=(of["Close"]>of["Open"]).astype(int)-(of["Close"]<of["Open"]).astype(int)
            b=(of["Volume"]*(sign>0)).sum(); s=(of["Volume"]*(sign<0)).sum()
            if b+s>0: flow=round(b/(b+s),2)

        sq=cat=False; sec=ind="Unknown"; short=None; lowf=False

        if enrich:
            info=stock.get_info() or {}
            fl=info.get("floatShares"); shortp=info.get("shortPercentOfFloat")

            lowf=fl and fl<20_000_000
            sq=shortp and shortp>0.15
            short=round(shortp*100,2) if shortp else None

            sec=info.get("sector","Unknown")
            ind=info.get("industry","Unknown")

            try:
                news=stock.get_news()
                if news and "providerPublishTime" in news[0]:
                    p=datetime.fromtimestamp(news[0]["providerPublishTime"],tz=timezone.utc)
                    cat=(datetime.now(timezone.utc)-p).days<=3
            except: pass

        score=score_calc(pm,yday,m3,m10,rsi,rvol,vwap,flow,cat,sq)
        prob=probability(score)

        return {
            "Symbol":t,"Price":price,"Volume":volume_now,
            "PM%":pm,"YDay%":yday,"3D%":m3,"10D%":m10,
            "RSI7":rsi,"RVOL_10D":rvol,"VWAP%":vwap,"FlowBias":flow,
            "Score":score,"Prob_Rise%":prob,
            "LowFloat?":lowf,"Squeeze?":sq,"Catalyst":cat,
            "Sector":sec,"Industry":ind,"Spark":close
        }

    except:
        return None


@st.cache_data(ttl=6)
def run_scan(watch,max_u,enrich):
    universe=build_universe(watch,max_u)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s,enrich) for s in universe]):
            if (r:=f.result()) is not None: results.append(r)
    return pd.DataFrame(results)


# ======================== RUN SCAN ========================
df=run_scan(watchlist_text,max_universe,enable_enrichment)

if df.empty:
    st.warning("No qualifying stocks found — adjust filters.")
    st.stop()

df=df[df["Score"]>=min_breakout]
if min_pm_move: df=df[df["PM%"].fillna(-999)>=min_pm_move]
if min_yday_gain: df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
if squeeze_only: df=df[df["Squeeze?"]]
if catalyst_only: df=df[df["Catalyst"]]
if vwap_only: df=df[df["VWAP%"].fillna(-999)>0]

df=df.sort_values(by=["Score","PM%","RSI7"],ascending=[False,False,False])

st.subheader(f"🔥 {len(df)} Breakout Candidates")


# ======================== DISPLAY ========================
for _,r in df.iterrows():
    c1,c2,c3,c4 = st.columns([2,3,3,3])

    c1.markdown(f"### {r['Symbol']}")
    c1.write(f"💲 Price: **{r['Price']:.2f}**")
    c1.write(f"📊 Volume: **{r['Volume']:,}**")
    c1.write(f"🔥 Score: **{r['Score']:.2f}** → Prob Rise: {r['Prob_Rise%']:.2f}%")

    c2.write(f"PM%: {r['PM%']}  —  YDay: {r['YDay%']}")
    c2.write(f"3D: {r['3D%']} | 10D: {r['10D%']}")
    c2.write(f"RSI7: {r['RSI7']} | RVOL10D: {r['RVOL_10D']}")

    c3.write(f"VWAP Dist: **{r['VWAP%']}%**")
    c3.write(f"Flow Bias: **{r['FlowBias']}**")
    if enable_enrichment:
        c3.write(f"Squeeze={r['Squeeze?']} | LowFloat={r['LowFloat?']}")
        c3.write(f"{r['Sector']} / {r['Industry']}")

    fig=go.Figure()
    fig.add_trace(go.Scatter(y=r['Spark'],mode="lines",line=dict(width=2)))
    fig.update_layout(height=60, margin=dict(l=0,r=0,t=0,b=0),
                      xaxis=dict(visible=False),yaxis=dict(visible=False))
    c4.plotly_chart(fig,use_container_width=True)

    st.divider()


# ======================== CSV EXPORT ========================
st.download_button(
    "📥 Download Screener CSV",
    df.to_csv(index=False),
    "v8_screener.csv"
)

st.caption("Educational Use Only — Not Financial Advice.")
