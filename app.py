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
    page_title="🚀 V8 Momentum Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 V8 — 10-Day Momentum Breakout Screener")
st.caption(
    "RSI7 • EMA10 • 3D/10D returns • Premarket • VWAP • Order flow • Watchlist • Optional Alerts"
)

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Universe")
    watchlist_text = st.text_area("Watchlist tickers:", "", height=80)
    max_universe = st.slider("Max symbols when no watchlist", 50, 600, 200, 50)

    region_mode = st.radio("Region Filter", ["Global (all)", "US + Canada Only"], index=1)
    enable_enrichment = st.checkbox("Float/Short + News (slower)", False)

    st.markdown("---")
    st.header("Filters")
    max_price = st.number_input("Max Price $", 1.0, 1000.0, DEFAULT_MAX_PRICE)
    min_volume = st.number_input("Min Daily Volume", 10000, 10000000, DEFAULT_MIN_VOLUME, 10000)
    min_breakout = st.number_input("Min Score", -50.0, 200.0, 0.0)
    min_pm_move = st.number_input("Min Premarket %", -50.0, 200.0, 0.0)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0)

    squeeze_only = st.checkbox("Short-Squeeze Only")
    catalyst_only = st.checkbox("Must Have News")
    vwap_only = st.checkbox("Above VWAP Only")

    # =============== ORDER FLOW FILTER ===============
    min_ofb = st.slider("Min OFB (Order Flow Bias)", 0.00, 1.00, 0.50, 0.01)

    # =============== ALERT SYSTEM TOGGLE ===============
    st.markdown("---")
    enable_alerts = st.checkbox("🔔 Enable Alerts", True)
    st.subheader("Alert Levels (active only if enabled)")

    ALERT_SCORE_THRESHOLD = st.slider("Trigger when Score ≥", 10, 200, 30)
    ALERT_PM_THRESHOLD = st.slider("Trigger when Premarket % ≥", 1, 150, 4)
    ALERT_VWAP_THRESHOLD = st.slider("Trigger when VWAP Dist ≥", 1, 50, 2)

    st.markdown("---")
    if st.button("🧹 Refresh Now"):
        st.cache_data.clear()
        st.success("Cache cleared — refreshing")


# ========================= SYMBOL LOAD =========================
@st.cache_data(ttl=900)
def load_symbols():
    nas = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", sep="|", skipfooter=1, engine="python")
    oth = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", sep="|", skipfooter=1, engine="python")

    nas["Exchange"]="NASDAQ"
    oth["Exchange"]=oth["Exchange"].fillna("NYSE/AMEX/ARCA")
    oth = oth.rename(columns={"ACT Symbol":"Symbol"})

    df = pd.concat([nas[["Symbol","ETF","Exchange"]], oth[["Symbol","ETF","Exchange"]]]).dropna(subset=["Symbol"])
    df = df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$", na=False)]
    df["Country"]="US"
    return df.to_dict("records")


def build_universe(watchlist_text,max_universe):
    wl = watchlist_text.strip()
    if wl:
        raw = wl.replace("\n"," ").replace(","," ").split()
        ticks = sorted(set(t.upper() for t in raw if t.strip()))
        return [{"Symbol":t,"Exchange":"WATCH","Country":"Unknown"} for t in ticks]
    return load_symbols()[:max_universe]


# ========================= SCORING =========================
def short_window_score(pm,y,m3,m10,rsi,rvol,c,sq,vwap,flow):
    s=0
    if pm:   s+=pm*1.6
    if y:    s+=y*0.8
    if m3:   s+=m3*1.2
    if m10:  s+=m10*0.6
    if rsi and rsi>55: s+=(rsi-55)*0.4
    if rvol and rvol>1.2: s+=(rvol-1.2)*2.0
    if vwap and vwap>0: s+=min(vwap,6)*1.5
    if flow: s+=(flow-0.5)*22
    if c: s+=8
    if sq: s+=12
    return round(s,2)

def breakout_probability(score):
    try: return round((1/(1+math.exp(-score/20)))*100,1)
    except: return None


# ========================= AUDIO + POPUP =========================
if "alerted" not in st.session_state: st.session_state.alerted=set()
if "alert_banner" not in st.session_state: st.session_state.alert_banner=[]

def trigger_alert(sym,msg):
    st.session_state.alerted.add(sym)
    st.session_state.alert_banner.append(f"🔔 {sym} — {msg}")

    st.markdown("""
        <audio autoplay>
            <source src="https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg" type="audio/ogg">
        </audio>
    """,unsafe_allow_html=True)


# ========================= SCAN =========================
@st.cache_data(ttl=6)
def run_scan(w,m,enrich,region,min_ofb):
    uni=build_universe(w,m)
    out=[]

    def scan(sym):
        try:
            tk=yf.Ticker(sym["Symbol"])
            h=tk.history(period=f"{HISTORY_LOOKBACK_DAYS}d")
            if h.empty or len(h)<5: return None

            c=h["Close"]; v=h["Volume"]
            price=float(c.iloc[-1]); vol=float(v.iloc[-1])
            if price>max_price or vol<min_volume: return None

            y=((c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100) if len(c)>=2 else None
            m3=((c.iloc[-1]-c.iloc[-4])/c.iloc[-4]*100) if len(c)>=4 else None
            m10=((c.iloc[-1]-c.iloc[0])/c.iloc[0]*100)

            rsi=100-(100/(1+(c.diff().clip(lower=0).rolling(7).mean()/(-c.diff().clip(upper=0).rolling(7).mean()))))
            rsi=float(rsi.iloc[-1])

            avg=v.mean(); rvol=vol/avg if avg>0 else None

            intra=tk.history(period=INTRADAY_RANGE,interval=INTRADAY_INTERVAL,prepost=True)
            if intra.empty: return None

            ic=intra["Close"]; io=intra["Open"]; iv=intra["Volume"]
            pm=((ic.iloc[-1]-ic.iloc[-2])/ic.iloc[-2]*100) if ic.iloc[-2]>0 else None

            tp=(intra["High"]+intra["Low"]+intra["Close"])/3; tot=iv.sum()
            vwap=((price-(tp*iv).sum()/tot)/((tp*iv).sum()/tot)*100) if tot>0 else None

            sign=(ic>io).astype(int)-(ic<io).astype(int)
            buy=(iv*(sign>0)).sum(); sell=(iv*(sign<0)).sum()
            flow=buy/(buy+sell) if (buy+sell)>0 else None

            if flow is None or flow<min_ofb: return None

            sq=False; cata=False; sec=""; ind=""; short=None
            if enrich:
                info=tk.get_info() or {}
                short=info.get("shortPercentOfFloat")
                sq=short and short>0.15
                sec=info.get("sector",""); ind=info.get("industry","")
                try:
                    news=tk.get_news()
                    if news and"providerPublishTime"in news[0]:
                        pub=datetime.fromtimestamp(news[0]["providerPublishTime"],tz=timezone.utc)
                        cata=(datetime.now(timezone.utc)-pub).days<=3
                except: pass

            score=short_window_score(pm,y,m3,m10,rsi,rvol,cata,sq,vwap,flow)
            prob=breakout_probability(score)

            return {
                "Symbol":sym["Symbol"],"Price":round(price,2),"Score":score,
                "Prob_Rise%":prob,
                "PM%":round(pm,2)if pm else None,
                "YDay%":round(y,2)if y else None,
                "3D%":round(m3,2)if m3 else None,
                "10D%":round(m10,2),
                "RSI7":round(rsi,2),
                "RVOL_10D":round(rvol,2)if rvol else None,
                "VWAP%":round(vwap,2)if vwap else None,
                "FlowBias":round(flow,2)if flow else None,
                "Squeeze?":sq,"Catalyst":cata,"Sector":sec,"Industry":ind,
                "Spark":c
            }
        except: return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS)as ex:
        for r in concurrent.futures.as_completed([ex.submit(scan,s)for s in uni]):
            if r.result(): out.append(r.result())

    return pd.DataFrame(out)


# ========================= EXECUTE =========================
with st.spinner("Scanning..."):
    df=run_scan(watchlist_text,max_universe,enable_enrichment,region_mode,min_ofb)

if df.empty:
    st.error("No matches — adjust filters.")
    st.stop()

df=df[df.Score>=min_breakout]
if min_pm_move: df=df[df["PM%"].fillna(-999)>=min_pm_move]
if min_yday_gain: df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
if squeeze_only: df=df[df["Squeeze?"]]
if catalyst_only: df=df[df["Catalyst"]]
if vwap_only: df=df[df["VWAP%"].fillna(-999)>0]

df=df.sort_values(["Score","PM%","RSI7"],ascending=[False,False,False])

# ========================= ALERT BANNER =========================
if enable_alerts and st.session_state.alert_banner:
    st.markdown("### 🔥 **Live Alerts**")
    for a in st.session_state.alert_banner[-5:]:
        st.warning(a)
    st.markdown("---")

# ========================= DISPLAY =========================
st.subheader(f"📈 Momentum Breakouts — {len(df)} signals")

for _,r in df.iterrows():
    sym=r["Symbol"]

    # ALERT CHECK
    if enable_alerts and sym not in st.session_state.alerted:
        if r["Score"]>=ALERT_SCORE_THRESHOLD:
            trigger_alert(sym,f"Score {r['Score']}")
        elif r["PM%"] and r["PM%"]>=ALERT_PM_THRESHOLD:
            trigger_alert(sym,f"Premarket {r['PM%']}%")
        elif r["VWAP%"] and r["VWAP%"]>=ALERT_VWAP_THRESHOLD:
            trigger_alert(sym,f"VWAP {r['VWAP%']}%")

    c1,c2,c3,c4=st.columns([2,3,3,3])
    c1.markdown(f"### **{sym}**")
    c1.write(f"💲 {r['Price']}")
    c1.write(f"🔥 Score {r['Score']}  |  Prob {r['Prob_Rise%']}%")   # <-- Fixed here

    c2.write(f"PM {r['PM%']}% | Y {r['YDay%']}%")
    c2.write(f"3D {r['3D%']}% | 10D {r['10D%']}% | RSI {r['RSI7']}")

    c3.write(f"RVOL {r['RVOL_10D']}x | VWAP {r['VWAP%']}% | Flow {r['FlowBias']}")

    c4.plotly_chart(go.Figure(
        data=[go.Scatter(y=r["Spark"].values,mode="lines")],
        layout=go.Layout(height=60,margin=dict(l=2,r=2,t=2,b=2))
    ))

    st.divider()


st.download_button("📥 Download CSV", df.to_csv(index=False), "V8_output.csv")
st.caption("Not financial advice.")
