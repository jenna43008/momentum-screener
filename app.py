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
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v9")

# ========================= PAGE =========================
st.set_page_config(page_title="V9 – 10-Day Momentum Screener", layout="wide")
st.title("🚀 V9 — 10-Day Volume + Momentum Breakout Screener")
st.caption("EMA10 🔥 Trend • Multi-Timeframe Alignment • RVOL • VWAP • Flow Bias • Alerts")

# ========================= SIDEBAR =========================
with st.sidebar:

    st.header("Universe")
    watchlist_text = st.text_area("Watchlist (optional):", "", height=80)
    max_universe = st.slider("Max Symbols", 50, 600, 200, 50)
    enable_enrichment = st.checkbox("Float/Short + News (slower)")

    st.markdown("---")
    st.header("Filters")

    max_price    = st.number_input("Max Price", 1.0,1000.0,DEFAULT_MAX_PRICE)
    min_volume   = st.number_input("Min Volume",10_000,10_000_000,DEFAULT_MIN_VOLUME,10_000)
    min_breakout = st.number_input("Min Score",-50.0,200.0,0.0,1.0)
    min_pm_move  = st.number_input("Min Premarket %",-50.0,200.0,0.0,0.5)
    min_yday_gain= st.number_input("Min Yesterday %",-50.0,200.0,0.0,0.5)

    squeeze_only  = st.checkbox("Short Squeeze Only")
    catalyst_only = st.checkbox("Must Have News")
    vwap_only     = st.checkbox("Above VWAP Only")

    # 🔥 ADD but not modify — minimum flow bias filter
    min_flow_bias = st.slider("Min Flow Bias (0–1)",0.00,1.00,0.50,0.01)

    st.markdown("---")
    st.subheader("🔊 Alerts")

    ALERT_SCORE_THRESHOLD = st.slider("Score ≥",10,200,30)
    ALERT_PM_THRESHOLD    = st.slider("Premarket ≥%",1,150,4)
    ALERT_VWAP_THRESHOLD  = st.slider("VWAP ≥%",1,50,2)

    # 🔥 ADDED — disable alerts toggle
    alert_enabled = st.checkbox("Enable Alerts",True)

    if st.button("🧹 Reset + Refresh"):
        st.cache_data.clear()
        st.success("Cache cleared")

# ========================= UNIVERSE =========================
@st.cache_data(ttl=900)
def load_symbols():
    nas=pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                    sep="|",skipfooter=1,engine="python")
    oth=pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                    sep="|",skipfooter=1,engine="python")
    nas["Exchange"]="NASDAQ"
    oth["Exchange"]=oth["Exchange"].fillna("NYSE/AMEX/ARCA")
    oth=oth.rename(columns={"ACT Symbol":"Symbol"})
    df=pd.concat([nas[["Symbol","Exchange"]],oth[["Symbol","Exchange"]]])
    df["Symbol"]=df["Symbol"].fillna("")
    df=df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$",na=False)]
    return df.to_dict("records")

def build_universe(wl,max_u):
    if wl.strip():
        tick={s.upper() for s in wl.replace(","," ").replace("\n"," ").split()}
        return [{"Symbol":t,"Exchange":"WATCH"} for t in sorted(tick)]
    return load_symbols()[:max_u]

# ========================= MODEL =========================
def short_window_score(pm,y,m3,m10,rsi,rvol,cat,sq,vwap,flow):
    s=0
    if pm:  s+=pm*1.6
    if y:   s+=y*0.8
    if m3:  s+=m3*1.2
    if m10: s+=m10*0.6
    if rsi>55: s+=(rsi-55)*0.4
    if rvol>1.2: s+=(rvol-1.2)*2
    if vwap>0:   s+=min(vwap,6)*1.5
    if flow:     s+=(flow-0.5)*22
    if cat:      s+=8
    if sq:       s+=12
    return round(s,2)

def multi_timeframe_label(pm,m3,m10):
    count=sum([(pm>0 if pm else False),(m3>0 if m3 else False),(m10>0 if m10 else False)])
    if count==3: return "✅ Full Alignment"
    if count==2: return "🟢 Strong Bullish Lean"
    if count==1: return "🟡 Mixed"
    return "🔻 Weak / No Trend"

def scan_one(sym,enrich):
    try:
        t=yf.Ticker(sym["Symbol"])
        hist=t.history(period="10d")
        if len(hist)<4: return None

        close=hist["Close"]; vol=hist["Volume"]
        price=float(close.iloc[-1]); v_last=float(vol.iloc[-1])
        if price>max_price or v_last<min_volume: return None
        y=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>2 else None
        m3=(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>4 else None
        m10=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100
        delta=close.diff(); rsi=float((100-(100/(1+(delta.clip(lower=0).rolling(7).mean()/(-delta.clip(upper=0).rolling(7).mean()))))).iloc[-1])
        avg10=float(vol.mean()); rvol=round(v_last/avg10,2) if avg10>0 else None
        ema10=float(close.ewm(span=10).mean().iloc[-1])
        ema_trend="🔥 Breakout" if price>ema10 and rsi>55 else "Neutral"

        intra=t.history(period="1d",interval="2m",prepost=True)
        if intra.empty:return None
        c=intra["Close"]; o=intra["Open"]; v=intra["Volume"]
        pm=(c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100 if c.iloc[-2]>0 else None
        tp=(intra["High"]+intra["Low"]+intra["Close"])/3
        vwap_val=(tp*v).sum()/v.sum() if v.sum()>0 else None
        vwap=(price-vwap_val)/vwap_val*100 if vwap_val else None
        sign=(c>o).astype(int)-(c<o).astype(int)
        buy=(v*(sign>0)).sum(); sell=(v*(sign<0)).sum()
        flow=float(buy/(buy+sell)) if (buy+sell)>0 else None

        if flow is None or flow<min_flow_bias: return None

        squeeze=False; cat=False
        if enrich:
            info=t.get_info() or {}
            squeeze=(info.get("shortPercentOfFloat") or 0)>0.15

        score=short_window_score(pm,y,m3,m10,rsi,rvol,cat,squeeze,vwap,flow)
        trend=multi_timeframe_label(pm,m3,m10)

        return{
            "Symbol":sym["Symbol"],"Price":round(price,2),
            "CurVol":int(v_last),"AvgVol10":int(avg10),"RVOL":rvol,
            "Score":score,"MTF_Trend":trend,"EMA10 Trend":ema_trend,
            "PM%":pm,"YDay%":y,"3D%":m3,"10D%":m10,"RSI7":rsi,
            "VWAP%":vwap,"FlowBias":flow,"Squeeze?":squeeze,"Spark":close,
        }
    except:return None

@st.cache_data(ttl=6)
def run_scan(wl,max_u,enrich):
    u=build_universe(wl,max_u)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s,enrich)for s in u]):
            if f.result(): out.append(f.result())
    return pd.DataFrame(out)

# ========================= MAIN DISPLAY =========================
with st.spinner("Scanning…"):
    df=run_scan(watchlist_text,max_universe,enable_enrichment)

if df.empty: st.error("No results — loosen filters"); st.stop()

df=df[df.Score>=min_breakout]
df=df[df["PM%"].fillna(-999)>=min_pm_move]
df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
if squeeze_only:  df=df[df["Squeeze?"]]
if catalyst_only: df=df[df.get("Catalyst",False)]
if vwap_only:     df=df[df["VWAP%"].fillna(-999)>0]

df=df.sort_values(["Score","PM%","RSI7"],ascending=[False,False,False])

st.subheader(f"🔥 {len(df)} Breakout Candidates")

if "alerted" not in st.session_state: st.session_state.alerted=set()

def alert(sym,msg):
    if alert_enabled and sym not in st.session_state.alerted:
        st.session_state.alerted.add(sym)
        st.warning(f"🔔 {sym}: {msg}")

for _,r in df.iterrows():

    sym=r.Symbol
    if alert_enabled:
        if r.Score>=ALERT_SCORE_THRESHOLD: alert(sym,f"Score {r.Score}")
        elif r["PM%"] and r["PM%"]>=ALERT_PM_THRESHOLD: alert(sym,f"PM {r['PM%']}%")
        elif r["VWAP%"] and r["VWAP%"]>=ALERT_VWAP_THRESHOLD: alert(sym,f"VWAP {r['VWAP%']}%")

    c1,c2,c3,c4=st.columns([2,3,3,3])

    c1.markdown(f"### {sym}  💲{r.Price}")
    c1.write(f"🔥 Score **{r.Score}** | Trend: **{r['EMA10 Trend']}**")
    c1.write(f"{r['MTF_Trend']}")

    c2.write(f"PM {r['PM%']}% | Y {r['YDay%']}%")
    c2.write(f"3D {r['3D%']}% | 10D {r['10D%']}%")
    c2.write(f"RSI7 {r['RSI7']}")

    c3.write(f"VWAP {r['VWAP%']}% | Flow {r['FlowBias']}")
    c3.write(f"🚀 Vol Now {r.CurVol:,} vs Avg {r.AvgVol10:,} (RVOL {r.RVOL}x)")
    c3.write(f"Squeeze {r['Squeeze?']}")

    fig=go.Figure(data=[go.Scatter(y=r.Spark.values,mode='lines')])
    fig.update_layout(height=90,margin=dict(l=5,r=5,t=5,b=5),xaxis=dict(visible=False),yaxis=dict(visible=False))
    c4.plotly_chart(fig,use_container_width=True)

    st.divider()

st.download_button("📥 CSV Export",df.to_csv(index=False),"V9_TrendVolume.csv")


