# ADDITION INSIDE scan_one() RETURN DICTIONARY:

return {
    "Symbol": ticker,
    "Exchange": exchange,
    "Price": round(price, 2),
    "CurVol": int(vol_last),                # ← NEW
    "Score": score,
    "Prob_Rise%": prob_rise,
    "PM%": round(premarket_pct, 2) if premarket_pct is not None else None,
    "YDay%": round(yday_pct, 2) if yday_pct is not None else None,
    "3D%": round(m3, 2) if m3 is not None else None,
    "10D%": round(m10, 2) if m10 is not None else None,
    "RSI7": round(rsi7, 2),
    "EMA10 Trend": ema_trend,
    "RVOL_10D": round(rvol10, 2) if rvol10 is not None else None,
    "VWAP%": round(vwap_dist, 2) if vwap_dist is not None else None,
    "FlowBias": round(order_flow_bias, 2) if order_flow_bias is not None else None,
    "Squeeze?": squeeze,
    "LowFloat?": low_float,
    "Catalyst": catalyst,
    "Sector": sector,
    "Industry": industry,
    "MTF_Trend": mtf_label,
    "Spark": spark_series,
}

# ———————————————————————————————————————————————————————
# UI SECTION — ADD THIS ONE LINE UNDER PRICE
# inside for _, row in df.iterrows(): under c1 — DO NOT MOVE ANYTHING

c1.write(f"💲 Price: {row['Price']}")
c1.write(f"📊 Vol: {row['CurVol']:,}")      # ← NEW (formatted with commas)
c1.write(f"🔥 Score: **{row['Score']}**")
c1.write(f"📈 Prob_Rise: {row['Prob_Rise%']}%")

# ———————————————————————————————————————————————————————
# CSV EXPORT — ADD CurVol to column list

csv_cols = [
    "Symbol", "Exchange", "Price", "CurVol",  # ← Added here
    "Score", "Prob_Rise%",
    "PM%", "YDay%", "3D%", "10D%", "RSI7", "EMA10 Trend",
    "RVOL_10D", "VWAP%", "FlowBias", "Squeeze?", "LowFloat?",
    "Sector", "Industry", "Catalyst", "MTF_Trend",
]


