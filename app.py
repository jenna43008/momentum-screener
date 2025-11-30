import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math
import random

# ========================= SETTINGS =========================
THREADS               = 20
AUTO_REFRESH_MS       = 60_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 5.0
DEFAULT_MIN_VOLUME    = 100_000
DEFAULT_MIN_BREAKOUT  = 0.0

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v9")

# ========================= PAGE SETUP =========================
st.set_page_config(
    page_title="V9 – 10-Day Momentum Screener (Hybrid Volume/Randomized)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 V9 — 10-Day Momentum Breakout Screener (Hybrid Speed + Volume + Randomized)")
st.caption(
    "Short-window model • EMA10 • RSI(7) • 3D & 10D momentum • 10D RVOL • "
    "VWAP + order flow • Watchlist mode • Alerts • V9 Universe Modes"
)

# ========================= SIDEBAR CONTROLS =========================
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area("Watchlist tickers:", "", height=80)

    max_universe = st.slider("Max symbols to scan",50,600,200,50)

    st.markdown("---")
    st.subheader("V9 Universe Mode")
    universe_mode = st.radio("Universe Construction",
        ["Classic (Alphabetical Slice)", "Randomized Slice", "Live Volume Ranked (slower)"],
        index=0)

    volume_rank_pool = st.slider("Volume-Rank Pool Size",100,2000,600,100)

    enable_enrichment = st.checkbox("Include float/short/news",value=False)

    st.markdown("---")
    st.header("Filters")

    max_price       = st.number_input("Max Price",1.0,1000.0,DEFAULT_MAX_PRICE)
    min_volume      = st.number_input("Min Volume",10_000,10_000_000,DEFAULT_MIN_VOLUME,10_000)
    min_breakout    = st.number_input("Min Score",-50.0,200.0,0.0,1.0)
    min_pm_move     = st.number_input("Min Premarket %",-50.0,200.0,0.0,0.5)
    min_yday_gain   = st.number_input("Min Yesterday %",-50.0,200.0,0.0,0.5)

    squeeze_only    = st.checkbox("Short Squeeze Only")
    catalyst_only   = st.checkbox("Must Have News")
    vwap_only       = st.checkbox("Above VWAP Only")

    st.markdown("---")
    st.subheader("Order Flow Filter (optional)")
    enable_ofb_filter = st.checkbox("Use Order Flow Bias Filter",value=False)
    min_ofb = st.slider("Min Flow Bias (0-1)",0.00,1.00,0.50,0.01)

    st.markdown("---")
    st.subheader("🔊 Alerts")
    enable_alerts = st.checkbox("Enable Alerts",True)

    ALERT_SCORE_THRESHOLD = st.slider("Score ≥",10,200,30)
    ALERT_PM_THRESHOLD    = st.slider("Premarket ≥%",1,150,4)
    ALERT_VWAP_THRESHOLD  = st.slider("VWAP ≥%",1,50,2)

    if st.button("🧹 Refresh Now"):
        st.cache_data.clear()
        st.success("Cache cleared.")

# ========================= LOAD SYMBOLS =========================
@st.cache_data(ttl=900)
def load_symbols():
    nas = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                      sep="|",skipfooter=1,engine="python")
    oth = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                      sep="|",skipfooter=1,engine="python")
    nas["Exchange"]="NASDAQ"
    oth["Exchange"]=oth["Exchange"].fillna("NYSE/AMEX/ARCA")
    oth=oth.rename(columns={"ACT Symbol":"Symbol"})
    df=pd.concat([nas[["Symbol","Exchange"]],oth[["Symbol","Exchange"]]])
    df=df[df.Symbol.str.contains("^[A-Z]{1,5}$",na=False)]
    return df.to_dict("records")

def build_universe(watchlist,max_u,mode,vol_pool):
    wl=watchlist.strip()
    if wl:
        t={s.upper() for s in wl.replace(","," ").replace("\n"," ").split()}
        return [{"Symbol":x,"Exchange":"WATCH"} for x in sorted(t)]

    syms=load_symbols()

    if mode=="Randomized Slice":
        base=syms[:]
        random.shuffle(base)
        return base[:max_u]

    if mode=="Live Volume Ranked (slower)":
        ranked=[]
        for s in syms[:vol_pool]:
            try:
                d=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m",prepost=True)
                if not d.empty:
                    ranked.append({**s,"LiveVol":float(d["Volume"].iloc[-1])})
            except: pass
        if not ranked: return syms[:max_u]
        return sorted(ranked,key=lambda x:x.get("LiveVol",0),reverse=True)[:max_u]

    return syms[:max_u]

# ========================= MODEL =========================
def short_window_score(pm,y,m3,m10,rsi,rvol,cat,sq,vwap,flow):
    s=0
    if pm:  s+=pm*1.6
    if y:   s+=y*0.8
    if m3:  s+=m3*1.2
    if m10: s+=m10*0.6
    if rsi>55: s+=(rsi-55)*0.4
    if rvol and rvol>1.2: s+=(rvol-1.2)*2
    if vwap and vwap>0:   s+=min(vwap,6)*1.5
    if flow: s+=(flow-0.5)*22
    if cat:  s+=8
    if sq:   s+=12
    return round(s,2)

def breakout_probability(score):
    try: return round((1/(1+math.exp(-score/20)))*100,1)
    except: return None

def multi_timeframe_label(pm,m3,m10):
    b=sum([(pm>0 if pm else False),(m3>0 if m3 else False),(m10>0 if m10 else False)])
    return ["🔻 Not Aligned","🟡 Mixed","🟢 Leaning Bullish","✅ Full Alignment"][b]

# ========================= SCAN ONE =========================
def scan_one(sym,enrich,ofb,min_ofb):
    try:
        t=yf.Ticker(sym["Symbol"])
        hist=t.history(period="10d")
        if len(hist)<5:return None

        c=hist["Close"]; v=hist["Volume"]
        price=float(c.iloc[-1]); vol_last=float(v.iloc[-1])
        if price>max_price or vol_last<min_volume:return None

        y=(c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100 if len(c)>2 else None
        m3=(c.iloc[-1]-c.iloc[-4])/c.iloc[-4]*100 if len(c)>4 else None
        m10=(c.iloc[-1]-c.iloc[0])/c.iloc[0]*100

        avg10=float(v.mean()); rvol=vol_last/avg10 if avg10>0 else None
        rsi=float((100-(100/(1+(c.diff().clip(lower=0).rolling(7).mean()/(-c.diff().clip(upper=0).rolling(7).mean()))))).iloc[-1])
        ema10=float(c.ewm(span=10,adjust=False).mean().iloc[-1])
        ema_trend="🔥 Breakout" if price>ema10 and rsi>55 else "Neutral"

        intra=t.history(period="1d",interval="2m",prepost=True)
        pm=vwap=flow=None
        if len(intra)>3:
            cl,op,iv=intra["Close"],intra["Open"],intra["Volume"]
            if cl.iloc[-2]>0:pm=(cl.iloc[-1]-cl.iloc[-2])/cl.iloc[-2]*100
            tp=(intra["High"]+intra["Low"]+intra["Close"])/3
            if iv.sum()>0:vwap=(price-(tp*iv).sum()/iv.sum())/((tp*iv).sum()/iv.sum())*100
            sign=(cl>op).astype(int)-(cl<op).astype(int)
            b=(iv*(sign>0)).sum(); s=(iv*(sign<0)).sum()
            flow=float(b/(b+s)) if (b+s)>0 else None

        if ofb and (flow is None or flow<min_ofb):return None

        sq=ct=False
        if enrich:
            info=t.get_info() or {}
            sq=(info.get("shortPercentOfFloat") or 0)>0.15

        score=short_window_score(pm,y,m3,m10,rsi,rvol,ct,sq,vwap,flow)
        trend=multi_timeframe_label(pm,m3,m10)

        return{
            "Symbol":sym["Symbol"],"Exchange":sym.get("Exchange",""),
            "Price":round(price,2),
            "CurVol":int(vol_last),              # 👈 **ADDED**
            "AvgVol10":int(avg10),               # 👈 **ADDED**
            "Score":score,"Prob_Rise%":breakout_probability(score),
            "PM%":pm,"YDay%":y,"3D%":m3,"10D%":m10,"RSI7":rsi,
            "EMA10 Trend":ema_trend,"RVOL_10D":rvol,"VWAP%":vwap,"FlowBias":flow,
            "Squeeze?":sq,"Catalyst":ct,"MTF_Trend":trend,"Spark":c,
        }
    except:return None

# ========================= RUN SCAN =========================
@st.cache_data(ttl=6)
def run_scan(w,m,e,ofb,min_ofb,mode,pool):
    u=build_universe(w,m,mode,pool)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s,e,ofb,min_ofb)for s in u]):
            if f.result():out.append(f.result())
    return pd.DataFrame(out)

# ========================= UI OUTPUT =========================
with st.spinner("Scanning…"):
    df=run_scan(watchlist_text,max_universe,enable_enrichment,enable_ofb_filter,min_ofb,universe_mode,volume_rank_pool)

if df.empty: st.error("No results"); st.stop()

df=df[df.Score>=min_breakout]
df=df[df["PM%"].fillna(-999)>=min_pm_move]
df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
if squeeze_only:df=df[df["Squeeze?"]]
if catalyst_only:df=df[df["Catalyst"]]
if vwap_only:df=df[df["VWAP%"].fillna(-999)>0]

df=df.sort_values(["Score","PM%","RSI7"],ascending=[False,False,False])

st.subheader(f"🔥 Results: {len(df)} symbols")

for_,r in df.iterrows():

    c1,c2,c3,c4=st.columns([2,3,3,3])

    c1.markdown(f"**{r.Symbol}** ({r.Exchange})")
    c1.write(f"💲 {r.Price} | 🔥 {r.Score}  |  Prob {r['Prob_Rise%']}%")
    c1.write(r["MTF_Trend"]); c1.write(f"Trend: {r['EMA10 Trend']}")

    # 👇 Volume added here (no layout changed)
    c2.write(f"📊 Volume: {r.CurVol:,}  |  10D Avg: {r.AvgVol10:,}")   # ← ADDED
    c2.write(f"PM {r['PM%']}%  |  YDay {r['YDay%']}%")
    c2.write(f"3D {r['3D%']}%  |  10D {r['10D%']}%  |  RSI {r['RSI7']}")

    c3.write(f"VWAP {r['VWAP%']}%  |  Flow {r['FlowBias']}")
    c3.write(f"RVOL {r['RVOL_10D']}x")

    fig=go.Figure(data=[go.Scatter(y=r.Spark.values,mode="lines")])
    fig.update_layout(height=90,margin=dict(l=5,r=5,t=5,b=5),xaxis={"visible":False},yaxis={"visible":False})
    c4.plotly_chart(fig,use_container_width=True)

    st.divider()

st.download_button(
    "📥 Download CSV",
    df.to_csv(index=False),
    "v9_momentum_with_volume.csv"
)





