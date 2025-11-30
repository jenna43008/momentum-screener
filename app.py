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
st.caption(
    "Short-window model • EMA10 • RSI(7) • 3D & 10D momentum • 10D RVOL • "
    "VWAP + order flow • Watchlist mode • Audio alerts"
)

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area(
        "Watchlist tickers (comma/space/newline separated):",
        value="",
        height=80,
        help="Example: AAPL, TSLA, NVDA, AMD",
    )

    max_universe = st.slider(
        "Max symbols to scan when no watchlist",
        min_value=50, max_value=600, value=200, step=50
    )

    enable_enrichment = st.checkbox("Include float/short + news (slower)", value=False)

    st.markdown("---")
    st.header("Filters")

    max_price    = st.number_input("Max Price ($)",     1.0, 1000.0, DEFAULT_MAX_PRICE, 1.0)
    min_volume   = st.number_input("Min Daily Volume", 10_000, 10_000_000, DEFAULT_MIN_VOLUME, 10_000)
    min_breakout = st.number_input("Min Breakout Score",-50.0,200.0,0.0,1.0)
    min_pm_move  = st.number_input("Min Premarket %",   -50.0,200.0,0.0,0.5)
    min_yday_gain= st.number_input("Min Yesterday %",   -50.0,200.0,0.0,0.5)

    squeeze_only  = st.checkbox("Short-Squeeze Only")
    catalyst_only = st.checkbox("Must Have News/Earnings")
    vwap_only     = st.checkbox("Above VWAP Only (VWAP% > 0)")

    # 🔥 NEW: Minimum Flow Filter (added without removing UI controls)
    min_flow_bias = st.slider("Min Flow Bias (0–1)",0.00,1.00,0.50,0.01)

    st.markdown("---")
    st.subheader("🔊 Audio Alert Thresholds")

    ALERT_SCORE_THRESHOLD = st.slider("Alert when Score ≥",10,200,30,5)
    ALERT_PM_THRESHOLD    = st.slider("Alert when Premarket % ≥",1,150,4,1)
    ALERT_VWAP_THRESHOLD  = st.slider("Alert when VWAP Dist % ≥",1,50,2,1)

    # NEW — Enable/Disable Alerts
    alert_enabled = st.checkbox("Enable Alerts", value=True)

    if st.button("🧹 Refresh Now"):
        st.cache_data.clear()
        st.success("Cache cleared — rescanning")

# ========================= SYMBOL LOAD (FIXED) =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                         sep="|",skipfooter=1,engine="python")
    other  = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                         sep="|",skipfooter=1,engine="python")

    nasdaq["Exchange"]="NASDAQ"
    other["Exchange"] = other["Exchange"].fillna("NYSE/AMEX/ARCA")
    other = other.rename(columns={"ACT Symbol":"Symbol"})

    df = pd.concat([nasdaq[["Symbol","ETF","Exchange"]],
                    other [["Symbol","ETF","Exchange"]]])

    df["Symbol"]=df["Symbol"].fillna("")                      # <-- FIX
    df=df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$",na=False)]# <-- SAFE SYMBOL FILTER

    return df.to_dict("records")

def build_universe(wl,max_u):
    wl=wl.strip()
    if wl:
        raw = wl.replace("\n"," ").replace(","," ").split()
        tick = sorted(set(t.upper() for t in raw if t.strip()))
        return [{"Symbol":t,"Exchange":"WATCH"} for t in tick]

    return load_symbols()[:max_u]

# ========================= SCORE MODEL =========================
def short_window_score(pm,y,m3,m10,rsi,rvol,cat,sq,vwap,flow):
    s=0
    if pm:    s+=pm*1.6
    if y:     s+=y*0.8
    if m3:    s+=m3*1.2
    if m10:   s+=m10*0.6
    if rsi>55:s+=(rsi-55)*0.4
    if rvol>1.2:s+=(rvol-1.2)*2
    if vwap>0:s+=min(vwap,6)*1.5
    if flow:  s+=(flow-0.5)*22
    if cat:   s+=8
    if sq:    s+=12
    return round(s,2)

def breakout_probability(score):
    try: return round((1/(1+math.exp(-score/20))) *100,1)
    except:return None

def multi_label(pm,m3,m10):
    pos=sum([
        pm is not None and pm>0,
        m3 is not None and m3>0,
        m10 is not None and m10>0])
    return ["🔻 No Align","🟡 Mixed","🟢 Bullish","✅ Perfect"][pos]

# ========================= SCAN ONE + 🔥 VOLUME ADDED =========================
def scan_one(s, enrich=False):
    try:
        t=yf.Ticker(s["Symbol"])
        hist=t.history(period="10d")

        if len(hist)<5: return None
        close=hist["Close"]; vol=hist["Volume"]
        price=float(close.iloc[-1]); vol_last=float(vol.iloc[-1])
        if price>max_price or vol_last<min_volume: return None

        # returns
        y=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>=2 else None
        m3=(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        # RSI
        rs=close.diff()
        gain=rs.clip(lower=0).rolling(7).mean()
        loss=(-rs.clip(upper=0)).rolling(7).mean()
        rsi=float((100-(100/(1+gain/loss))).iloc[-1])

        # 🔥 NEW VOLUME SIGNALS
        vol_avg10=float(vol.mean())
        rvol10=round(vol_last/vol_avg10,2) if vol_avg10>0 else None

        # intraday vol + flow + vwap
        intra=t.history(period="1d",interval="2m",prepost=True)
        if intra.empty: return None
        c=intra["Close"]; o=intra["Open"]; v=intra["Volume"]

        pm=(c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100 if c.iloc[-2]>0 else None

        tp=(intra["High"]+intra["Low"]+intra["Close"])/3
        vwap=(price-(tp*v).sum()/v.sum())/( (tp*v).sum()/v.sum())*100 if v.sum()>0 else None

        sign=(c>o).astype(int)-(c<o).astype(int)
        buy=(v*(sign>0)).sum(); sell=(v*(sign<0)).sum()
        flow=float(buy/(buy+sell)) if (buy+sell)>0 else None

        # 🔥 apply flow restriction (added only, not altering UI)
        if flow is None or flow < min_flow_bias:
            return None

        # enrichment
        squeeze=False; catalyst=False; sector="";industry=""
        if enrich:
            info=t.get_info() or {}
            short=info.get("shortPercentOfFloat")
            squeeze=short and short>0.15

        score=short_window_score(pm,y,m3,m10,rsi,rvol10,catalyst,squeeze,vwap,flow)
        prob=breakout_probability(score)

        return{
            "Symbol":s["Symbol"],
            "Price":round(price,2),
            "CurVol":int(vol_last),                   # NEW
            "Vol10Avg":int(vol_avg10),                # NEW
            "RVOL":rvol10,                            # NEW
            "Score":score,"Prob_Rise%":prob,
            "PM%":round(pm,2) if pm else None,
            "YDay%":round(y,2) if y else None,
            "3D%":round(m3,2) if m3 else None,
            "10D%":round(m10,2),
            "RSI7":round(rsi,2),
            "VWAP%":round(vwap,2) if vwap else None,
            "FlowBias":round(flow,2) if flow else None,
            "Squeeze?":squeeze,"Spark":close,
        }
    except:return None

@st.cache_data(ttl=6)
def run_scan(wl,max_u,enrich):
    u=build_universe(wl,max_u)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s,enrich) for s in u]):
            r=f.result()
            if r:out.append(r)
    return pd.DataFrame(out)

# ========================= DISPLAY =========================
with st.spinner("Scanning…"):
    df= run_scan(watchlist_text,max_universe,enable_enrichment)

if df.empty:
    st.error("No matches — loosen filters"); st.stop()

df=df[df.Score>=min_breakout]
df=df[(df["PM%"].fillna(-999)>=min_pm_move)]
df=df[(df["YDay%"].fillna(-999)>=min_yday_gain)]
if squeeze_only: df=df[df["Squeeze?"]]
if catalyst_only:df=df[df["Catalyst"] if "Catalyst" in df else False]
if vwap_only:    df=df[df["VWAP%"].fillna(-999)>0]

df=df.sort_values(["Score","PM%","RSI7"],ascending=[False,False,False])
st.subheader(f"🔥 10-Day Momentum Board — {len(df)} symbols")

# ============= ALERT SYSTEM (unchanged UI, now toggleable) =============
if "alerted"not in st.session_state:st.session_state.alerted=set()

def trigger(sym,reason):
    if not alert_enabled:return
    if sym in st.session_state.alerted:return
    st.session_state.alerted.add(sym)
    st.warning(f"🔔 {sym} ALERT — {reason}")

for _,r in df.iterrows():
    sym=r.Symbol
    if alert_enabled:
        if r.Score>=ALERT_SCORE_THRESHOLD:trigger(sym,f"Score {r.Score}")
        elif r["PM%"] and r["PM%"]>=ALERT_PM_THRESHOLD:trigger(sym,f"PM {r['PM%']}%")
        elif r["VWAP%"] and r["VWAP%"]>=ALERT_VWAP_THRESHOLD:trigger(sym,f"VWAP {r['VWAP%']}%")

    c1,c2,c3,c4=st.columns([2,3,3,3])
    c1.write(f"**{sym}**  💲{r.Price}")
    c1.write(f"Score: {r.Score} | Prob {r['Prob_Rise%']}%")
    c1.write(f"Flow {r.FlowBias} | RVOL {r.RVOL}x") # NEW VOLUME DISPLAY
    c1.write(f"Vol: {r.CurVol:,} vs Avg {r.Vol10Avg:,}") # NEW

    c2.write(f"PM% {r['PM%']} | Yday {r['YDay%']}%")
    c2.write(f"3D {r['3D%']}% | 10D {r['10D%']}% | RSI {r['RSI7']}")

    c3.write(f"VWAP {r['VWAP%']}% | Squeeze {r['Squeeze?']}")

    fig=go.Figure()
    fig.add_trace(go.Scatter(y=r.Spark.values,mode="lines"))
    fig.update_layout(height=60,width=170,margin=dict(l=3,r=3,t=3,b=3),
                      xaxis=dict(visible=False),yaxis=dict(visible=False))
    c4.plotly_chart(fig,use_container_width=False)

    st.divider()

st.download_button("📥 Download CSV",df.to_csv(index=False),"momentum_v9_volume.csv")
st.caption("For research + education. Not financial advice.")


