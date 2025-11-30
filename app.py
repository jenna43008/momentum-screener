@st.cache_data(ttl=900)
def load_symbols():
    """Load US symbols (NASDAQ + otherlisted) safely with NaN protection."""
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

    # FIX — prevent crash from NA rows & keep original UI intact
    df = pd.concat([nasdaq[["Symbol","ETF","Exchange"]],
                    other[["Symbol","ETF","Exchange"]]])

    df["Symbol"] = df["Symbol"].fillna("")   # <-- Allows str.contains without error
    df = df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$", na=False)]

    return df.to_dict("records")


st.download_button("📥 Download CSV",df.to_csv(index=False),"momentum_v9_volume.csv")
st.caption("For research + education. Not financial advice.")


