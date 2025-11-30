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
st.caption("Short-window model • EMA10 • RSI(7) • 3D & 10D momentum • 10D RVOL • VWAP + order flow • Watchlist mode • Audio alerts")

# ========================= SIDEBAR CONTROLS =========================
with st.sidebar:
    st.header("Universe")
    watchlist_text = st.text_area("Watchlist tickers (comma/space/newline separated):", "", height=80)
    max_universe = st.slider("Max symbols to scan when no watchlist", 50, 600, 200, 50)
    enable_enrichment = st.checkbox("Include float/short + news (slower, more data)", False)

    st.markdown("---")
    st.header("Filters")
    max_price = st.number_input("Max Price ($)", 1.0, 1000.0, DEFAULT_MAX_PRICE)
    min_volume = st.number_input("Min Daily Volume", 10000, 10000000, DEFAULT_MIN_VOLUME, 10000)
    min_breakout = st.number_input("Min Breakout Score", -50.0, 200.0, 0.0)
    min_pm_move = st.number_input("Min Premarket %", -50.0, 200.0, 0.0)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0)

    squeeze_only = st.checkbox("Short-Squeeze Only")
    catalyst_only = st.checkbox("Must Have News/Earnings")
    vwap_only = st.checkbox("Above VWAP Only (VWAP% > 0)")

    st.markdown("---")
    st.subheader("🔊 Audio Alert Thresholds")
    ALERT_SCORE_THRESHOLD = st.slider("Alert when Score ≥", 10, 200, 30, 5)
    ALERT_PM_THRESHOLD    = st.slider("Alert when Premarket % ≥", 1, 150, 4, 1)
    ALERT_VWAP_THRESHOLD  = st.slider("Alert when VWAP Dist % ≥", 1, 50, 2, 1)

    # ==================== NEW — ALERT DISABLE =====================
    alerts_enabled = st.checkbox("Disable Audio Alerts?", value=False)

    # ==================== NEW — MIN FLOW FILTER ==================
    use_flow_filter = st.checkbox("Enable Minimum Flow Bias Filter", value=False)
    min_flow_bias = st.slider("Min Flow % (0–1)", 0.00, 1.00, 0.50, 0.01) if use_flow_filter else 0.00

    st.markdown("---")
    if st.button("🧹 Refresh Now"):
        st.cache_data.clear()
        st.success("Cache cleared — fresh scan will run now.")


# ========================= SYMBOL LOAD =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", sep="|", skipfooter=1, engine="python")
    other  = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", sep="|", skipfooter=1, engine="python")

    nasdaq["Exchange"]="NASDAQ"
    other["Exchange"]=other["Exchange"].fillna("NYSE/AMEX/ARCA")
    other = other.rename(columns={"ACT Symbol":"Symbol"})
    df=pd.concat([nasdaq[["Symbol","ETF","Exchange"]],other[["Symbol","ETF","Exchange"]]]).dropna(subset=["Symbol"])
    df=df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$",na=False)]
    return df.to_dict("records")


def build_universe(w,maxu):
    wl=w.strip()
    if wl:
        raw=wl.replace("\n"," ").replace(","," ").split()
        ticks=sorted(set(s.upper() for s in raw if s.strip()))
        return [{"Symbol":t,"Exchange":"WATCH"} for t in ticks]
    return load_symbols()[:maxu]


# ========================= SCAN =========================
@st.cache_data(ttl=6)
def run_scan(w,maxu,enrich):
    uni=build_universe(w,maxu)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS)as ex:
        futures=[ex.submit(scan_one,sym,enrich)for sym in uni]
        for f in concurrent.futures.as_completed(futures):
            r=f.result()
            if r: out.append(r)
    if not out: return pd.DataFrame()
    return pd.DataFrame(out)


def scan_one(sym,enrich):
    try:
        tk=yf.Ticker(sym["Symbol"])
        hist=tk.history(period=f"{HISTORY_LOOKBACK_DAYS}d",interval="1d")
        if hist.empty or len(hist)<5: return None
        close=hist["Close"]; volume=hist["Volume"]
        price=float(close.iloc[-1]); vol_last=float(volume.iloc[-1])

        if price>max_price or vol_last<min_volume: return None
        y=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>=2 else None
        m3=(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100
        rsi=100-(100/(1+(close.diff().clip(lower=0).rolling(7).mean()/(-close.diff().clip(upper=0).rolling(7).mean()))))
        rsi=float(rsi.iloc[-1])
        avg10=volume.mean(); rvol=vol_last/avg10 if avg10>0 else None

        intra=tk.history(period=INTRADAY_RANGE,interval=INTRADAY_INTERVAL,prepost=True)
        if intra.empty: return None

        ic=intra["Close"]; io=intra["Open"]; iv=intra["Volume"]
        pm=(ic.iloc[-1]-ic.iloc[-2])/ic.iloc[-2]*100 if ic.iloc[-2]>0 else None
        tp=(intra["High"]+intra["Low"]+intra["Close"])/3; tot=iv.sum()
        vwap=(price-(tp*iv).sum()/tot)/((tp*iv).sum()/tot)*100 if tot>0 else None

        sign=(ic>io).astype(int)-(ic<io).astype(int)
        buy=(iv*(sign>0)).sum(); sell=(iv*(sign<0)).sum()
        flow_bias=buy/(buy+sell) if (buy+sell)>0 else None

        # ================= NEW: FLOW FILTER APPLIED =================
        if use_flow_filter and (flow_bias is None or flow_bias < min_flow_bias):
            return None

        catalyst=False; squeeze=False; sec=""; ind=""; short=None
        if enrich:
            try:
                info=tk.get_info() or {}
                short=info.get("shortPercentOfFloat")
                squeeze=short and short>0.15
                sec=info.get("sector",""); ind=info.get("industry","")
                news=tk.get_news()
                if news and"providerPublishTime"in news[0]:
                    pub=datetime.fromtimestamp(news[0]["providerPublishTime"],tz=timezone.utc)
                    catalyst=(datetime.now(timezone.utc)-pub).days<=3
            except: pass

        score=short_window_score(pm,y,m3,m10,rsi,rvol,catalyst,squeeze,vwap,flow_bias)
        prob=breakout_probability(score)

        return {
            "Symbol":sym["Symbol"],"Exchange":sym.get("Exchange",""),
            "Price":round(price,2),"Score":score,"Prob_Rise%":prob,
            "PM%":round(pm,2)if pm else None,"YDay%":round(y,2)if y else None,
            "3D%":round(m3,2)if m3 else None,"10D%":round(m10,2),
            "RSI7":round(rsi,2),"RVOL_10D":round(rvol,2)if rvol else None,
            "VWAP%":round(vwap,2)if vwap else None,"FlowBias":round(flow_bias,2)if flow_bias else None,
            "Squeeze?":squeeze,"LowFloat?":None,"Catalyst":catalyst,
            "Sector":sec,"Industry":ind,"MTF_Trend":multi_timeframe_label(pm,m3,m10),
            "Spark":close
        }
    except:
        return None


# ========================= RUN =========================
with st.spinner("Scanning (10-day momentum)…"):
    df=run_scan(watchlist_text,max_universe,enable_enrichment)

if df.empty:
    st.error("No results. Adjust filters.")
else:
    df=df[df.Score>=min_breakout]
    if min_pm_move: df=df[df["PM%"].fillna(-999)>=min_pm_move]
    if min_yday_gain: df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
    if squeeze_only: df=df[df["Squeeze?"]]
    if catalyst_only: df=df[df["Catalyst"]]
    if vwap_only: df=df[df["VWAP%"].fillna(-999)>0]
    df=df.sort_values(["Score","PM%","RSI7"],ascending=[False,False,False])

    st.subheader(f"🔥 {len(df)} Momentum Signals Found")

    for _,row in df.iterrows():
        sym=row["Symbol"]

        # ================= NEW ALERT MUTE =================
        if not alerts_enabled:
            pass
        else:
            if sym not in st.session_state.alerted:
                if row["Score"]>=ALERT_SCORE_THRESHOLD:
                    trigger_audio_alert(sym,f"Score {row['Score']}")
                elif row["PM%"] and row["PM%"]>=ALERT_PM_THRESHOLD:
                    trigger_audio_alert(sym,f"Premarket {row['PM%']}%")
                elif row["VWAP%"] and row["VWAP%"]>=ALERT_VWAP_THRESHOLD:
                    trigger_audio_alert(sym,f"VWAP {row['VWAP%']}%")

        c1,c2,c3,c4=st.columns([2,3,3,3])
        c1.markdown(f"**{sym}** ({row['Exchange']})")
        c1.write(f"💲 Price: {row['Price']} — Score {row['Score']}")
        c1.write(f"Prob {row['Prob_Rise%']}%")
        c1.write(row["MTF_Trend"]); c1.write(row["EMA10 Trend"])

        c2.write(f"PM% {row['PM%']} | YDay {row['YDay%']}")
        c2.write(f"3D {row['3D%']} | 10D {row['10D%']} | RSI {row['RSI7']}")
        c2.write(f"RVOL {row['RVOL_10D']}x")

        c3.write(f"VWAP {row['VWAP%']}% | Flow {row['FlowBias']}")
        if enable_enrichment: c3.write(f"Squeeze {row['Squeeze?']}  | Sec: {row['Sector']} / {row['Industry']}")

        c4.plotly_chart(go.Figure(data=[go.Scatter(y=row["Spark"].values,mode="lines")],
            layout=go.Layout(height=60,margin=dict(l=3,r=3,t=3,b=3))))

        st.divider()

    st.download_button("📥 Download CSV",df.to_csv(index=False),"screen.csv")

st.caption("For research use only. Not financial advice.")

