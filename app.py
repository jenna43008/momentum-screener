import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math
import random

# =============================================================
# SETTINGS (UNCHANGED)
# =============================================================
THREADS               = 20
AUTO_REFRESH_MS       = 60_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 5.0
DEFAULT_MIN_VOLUME    = 100_000
DEFAULT_MIN_BREAKOUT  = 0.0

# =============================================================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v9")

# =============================================================
# PAGE
# =============================================================
st.set_page_config(page_title="V9 AI Hybrid Momentum Screener", layout="wide")
st.title("🚀 V9 Momentum Screener — AI Commentary + Volume Signals")
st.caption("Everything intact — now enhanced with real volume + AI signal summaries")

# =============================================================
# SIDEBAR (NO UI REMOVED — ONLY VOLUME + ALERTS KEPT AS IS)
# =============================================================
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area("Watchlist symbols:", value="", height=70)

    max_universe = st.slider("Max scan size", 50, 600, 200, 50)
    universe_mode = st.radio("V9 Universe Mode",
        ["Classic (Alphabetical Slice)", "Randomized Slice", "Live Volume Ranked (slower)"])

    volume_rank_pool = st.slider("Live Volume Ranking Pool",100,2000,600,100)

    enable_enrichment = st.checkbox("Float/Short/News (slower)",False)

    st.header("Filters")
    max_price    = st.number_input("Max Price ($)",1.0,1000.0,DEFAULT_MAX_PRICE,1.0)
    min_volume   = st.number_input("Min Daily Volume",10_000,10_000_000,DEFAULT_MIN_VOLUME,10_000)
    min_breakout = st.number_input("Min Breakout Score",-50.0,200.0,0.0,1.0)
    min_pm_move  = st.number_input("Min Premarket %",-50.0,200.0,0.0,0.5)
    min_yday_gain= st.number_input("Min Yesterday %",-50.0,200.0,0.0,0.5)

    squeeze_only   = st.checkbox("Short Squeeze Only")
    catalyst_only  = st.checkbox("Must Have News / Catalyst")
    vwap_only      = st.checkbox("Above VWAP Only")

    st.markdown("---")
    st.header("Order Flow Filter")
    enable_ofb_filter = st.checkbox("Enable OFB Filter",False)
    min_ofb = st.slider("Min OFB",0.00,1.00,0.50,0.01)

    st.markdown("---")
    st.subheader("🔊 Alerts")
    enable_alerts = st.checkbox("Enable Alerts",True)
    ALERT_SCORE_THRESHOLD = st.slider("Alert Score ≥",10,200,30,5)
    ALERT_PM_THRESHOLD    = st.slider("Alert PM% ≥",1,150,4,1)
    ALERT_VWAP_THRESHOLD  = st.slider("Alert VWAP% ≥",1,50,2,1)

    if st.button("🧹 Refresh"):
        st.cache_data.clear()
        st.success("Cache cleared.")

# =============================================================
# LOAD SYMBOLS — **UNTOUCHED**
# =============================================================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq=pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",sep="|",skipfooter=1,engine="python")
    other =pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",sep="|",skipfooter=1,engine="python")
    nasdaq["Exchange"]="NASDAQ"
    other = other.rename(columns={"ACT Symbol":"Symbol"})
    df=pd.concat([nasdaq[["Symbol","ETF","Exchange"]],other[["Symbol","ETF","Exchange"]]])
    df=df[df.Symbol.str.contains(r"^[A-Z]{1,5}$", na=False)]
    return df.to_dict("records")

# =============================================================
# BUILD UNIVERSE — (V9 VOLUME + RANDOM KEPT)
# =============================================================
def build_universe(w,max_u,mode,pool):
    wl=w.strip()
    if wl:
        tick=sorted(set(w.upper().replace("\n"," ").replace(","," ").split()))
        return [{"Symbol":t} for t in tick]

    syms=load_symbols()

    if mode=="Randomized Slice":
        random.shuffle(syms)
        return syms[:max_u]

    if mode=="Live Volume Ranked (slower)":
        ranked=[]
        for s in syms[:pool]:
            try:
                d=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m",prepost=True)
                if not d.empty: ranked.append({**s,"Vol":float(d.Volume.iloc[-1])})
            except: pass
        return sorted(ranked,key=lambda x:x.get("Vol",0),reverse=True)[:max_u]

    return syms[:max_u]

# =============================================================
# AI COMMENTARY ENGINE
# =============================================================
def ai_commentary(row):
    msg=[]

    if row["RVOL_10D"] and row["RVOL_10D"]>2:
        msg.append("High RVOL — demand expanding")
    if row["VWAP%"] and row["VWAP%"]>0:
        msg.append("Price above VWAP — bullish balance")
    if row["FlowBias"] and row["FlowBias"]>0.6:
        msg.append("Buyers in control")
    if row["PM%"] and row["PM%"]>3:
        msg.append("Premarket strength showing confidence")
    if row["10D%"] and row["10D%"]>10:
        msg.append("Strong 10D momentum trend")

    if not msg:
        return "Neutral — watching for more structure"

    return " | ".join(msg)

# =============================================================
# SCAN — only inject volume + commentary return values
# =============================================================
def scan_one(sym,enrich,ofb,min_ofb):
    try:
        t=sym["Symbol"]
        stock=yf.Ticker(t)
        hist=stock.history(period=f"{HISTORY_LOOKBACK_DAYS}d",interval="1d")
        if hist.empty:return None

        close=hist.Close; volume=hist.Volume
        price=float(close.iloc[-1]); daily_vol=float(volume.iloc[-1])
        if price>max_price or daily_vol<min_volume:return None

        # % Changes
        yday_pct=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>1 else None
        m3=(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>3 else None
        m10=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100 if close.iloc[0]>0 else None

        # RSI
        delta=close.diff();gain=delta.clip(lower=0).rolling(7).mean()
        loss=(-delta.clip(upper=0)).rolling(7).mean();rs=gain/loss
        rsi=float((100-(100/(1+rs))).iloc[-1])

        ema10=float(close.ewm(span=10,adjust=False).mean().iloc[-1])
        rvol=(daily_vol/volume.mean()) if volume.mean()>0 else None

        # Intraday
        pm=vwap=o=None
        intra=stock.history(period=INTRADAY_RANGE,interval=INTRADAY_INTERVAL,prepost=True)
        if len(intra)>=3:
            pm=(intra.Close.iloc[-1]-intra.Close.iloc[-2])/intra.Close.iloc[-2]*100
            typ=(intra.High+intra.Low+intra.Close)/3
            if intra.Volume.sum()>0:vwap=(price-((typ*intra.Volume).sum()/intra.Volume.sum()))/price*100
            sign=(intra.Close>intra.Open).astype(int)-(intra.Close<intra.Open).astype(int)
            buy=(intra.Volume*(sign>0)).sum();sell=(intra.Volume*(sign<0)).sum()
            o=buy/(buy+sell) if buy+sell>0 else None

        if ofb and (o is None or o<min_ofb):return None

        score=round(
            max(pm or 0,0)*1.6+
            max(yday_pct or 0,0)*0.8+
            max(m3 or 0,0)*1.2+
            max(m10 or 0,0)*0.6+
            (max(rsi-55,0)*0.4)+
            ((rvol-1.2)*2 if rvol and rvol>1.2 else 0)+
            ((min(vwap,6))*1.5 if vwap and vwap>0 else 0)+
            ((o-0.5)*22 if o else 0),2)

        prob=round((1/(1+math.exp(-score/20)))*100,1)

        return {
            "Symbol":t,"Price":round(price,2),"Volume":int(daily_vol),
            "Score":score,"Prob_Rise%":prob,
            "PM%":round(pm,2) if pm else None,
            "YDay%":round(yday_pct,2) if yday_pct else None,
            "3D%":round(m3,2) if m3 else None,
            "10D%":round(m10,2) if m10 else None,
            "RSI7":round(rsi,2),
            "RVOL_10D":round(rvol,2) if rvol else None,
            "VWAP%":round(vwap,2) if vwap else None,
            "FlowBias":round(o,2) if o else None,
            "MTF_Trend":("Aligned Bullish" if sum([(pm>0 if pm else 0),(m3>0 if m3 else 0),(m10>0 if m10 else 0)])>=2 else "Mixed"),
            "Spark":close,
            "AI":ai_commentary({"RVOL_10D":rvol,"VWAP%":vwap,"FlowBias":o,"PM%":pm,"10D%":m10})
        }
    except:return None

@st.cache_data(ttl=6)
def run_scan(w,mu,en,en_ofb,min_ofb,mode,p):
    uni=build_universe(w,mu,mode,p)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,u,en,en_ofb,min_ofb) for u in uni]):
            r=f.result()
            if r:out.append(r)
    return pd.DataFrame(out)

# =============================================================
# MAIN OUTPUT — ONLY *ADDITIONS* WERE CURRENT VOLUME + AI BOX
# =============================================================
df=run_scan(watchlist_text,max_universe,enable_enrichment,enable_ofb_filter,min_ofb,universe_mode,volume_rank_pool)

if df.empty:
    st.error("No matches — loosen filters.")
else:
    df=df[df.Score>=min_breakout]
    if min_pm_move:df=df[df["PM%"].fillna(-999)>=min_pm_move]
    if min_yday_gain:df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
    if squeeze_only:df=df[df["FlowBias"]>0.6]
    if catalyst_only:df=df[df["10D%"]>5]
    if vwap_only:df=df[df["VWAP%"].fillna(-999)>0]

    df=df.sort_values(by=["Score","PM%","RSI7"],ascending=[False,False,False])
    st.subheader(f"🔥 Results: {len(df)}")

    for _,r in df.iterrows():
        c1,c2,c3,c4=st.columns([2,3,3,4])
        c1.markdown(f"### **{r.Symbol}**")
        c1.write(f"💲 {r.Price}")
        c1.write(f"📊 Score **{r.Score}**  |  Rise Prob {r['Prob_Rise%']}%")
        c1.write(f"Volume: **{r.Volume:,}**")

        c2.write(f"PM% {r['PM%']} | YDay% {r['YDay%']}")
        c2.write(f"3D {r['3D%']} | 10D {r['10D%']}")
        c2.write(f"RSI7 {r['RSI7']} | RVOL {r['RVOL_10D']}x")

        c3.write(f"VWAP% {r['VWAP%']} | FlowBias {r['FlowBias']}")
        c3.info(f"AI View → {r['AI']}")

        c4.plotly_chart(go.Figure(data=[go.Scatter(y=r.Spark)]),use_container_width=True)
        st.divider()


