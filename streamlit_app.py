import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt

st.set_page_config("Intraday Option Volume-Price Analytics", layout="wide")
st.title("📊 Intraday Option Price–Volume Analyzer")

# ---------------- Sidebar ----------------
st.sidebar.header("Config")
rolling_n = st.sidebar.number_input("Rolling window (bars)", 3, 60, 5)
spread_cutoff = st.sidebar.slider("Max bid–ask spread %", 0.0, 1.0, 0.2)
st.sidebar.markdown("Upload **incremental Option‑Chain CSVs** (5‑min snapshots) 👇")

files = st.file_uploader("Drop CSV files", type=["csv"], accept_multiple_files=True)
if not files:
    st.info("⬅️ Upload CSV files to begin analysis.")
    st.stop()

# ---------------- Load & Combine ----------------
frames = []
for f in files:
    name = f.name.replace(".csv", "")
    try:
        ts = datetime.strptime(name.split("_")[-2] + "_" + name.split("_")[-1], "%d%m%Y_%H%M%S")
    except Exception:
        ts = datetime.now()
    df = pd.read_csv(f)
    df["timestamp"] = ts
    frames.append(df)

raw = pd.concat(frames).sort_values(["timestamp", "CE_strikePrice"]).reset_index(drop=True)

# ---------------- Cleaning ----------------
@st.cache_data
def clean_data(df, cutoff=0.2):
    df = df.copy()
    for side in ["CE", "PE"]:
        df[f"{side}_mid"] = (df[f"{side}_buyPrice1"] + df[f"{side}_sellPrice1"]) / 2
        df[f"{side}_spread_pct"] = abs(df[f"{side}_sellPrice1"] - df[f"{side}_buyPrice1"]) / df[f"{side}_mid"].replace(0, np.nan)
    df = df[(df["CE_spread_pct"] < cutoff) & (df["PE_spread_pct"] < cutoff)]
    return df

df = clean_data(raw, spread_cutoff)
st.success(f"✅ Loaded {len(files)} snapshots, {len(df)} rows after cleaning.")

# ---------------- Feature Engineering ----------------
@st.cache_data
def compute_features(df, rolling_n):
    feat = df.copy()
    # true interval deltas (since volumes cumulative)
    for s in ["CE", "PE"]:
        feat[f"{s}_vol_delta"] = feat.groupby("CE_strikePrice")[f"{s}_totalTradedVolume"].diff().fillna(0)
        feat[f"{s}_OI_delta"] = feat.groupby("CE_strikePrice")[f"{s}_openInterest"].diff().fillna(0)
        feat[f"{s}_price_delta"] = feat.groupby("CE_strikePrice")[f"{s}_lastPrice"].diff().fillna(0)

    agg = feat.groupby("timestamp").agg({
        "CE_lastPrice": "mean",
        "PE_lastPrice": "mean",
        "CE_openInterest": "sum",
        "PE_openInterest": "sum",
        "CE_vol_delta": "sum",
        "PE_vol_delta": "sum",
        "CE_OI_delta": "sum",
        "PE_OI_delta": "sum"
    }).rename(columns={
        "CE_lastPrice": "CE_Price",
        "PE_lastPrice": "PE_Price"
    })

    # volume totals
    agg["Total_Volume"] = agg["CE_vol_delta"] + agg["PE_vol_delta"]
    agg["Volume_Spike"] = agg["Total_Volume"] / agg["Total_Volume"].rolling(rolling_n).mean()
    agg["Vol_Imbalance"] = (agg["CE_vol_delta"] - agg["PE_vol_delta"]) / (agg["Total_Volume"].replace(0, np.nan))

    # price deltas
    agg["ΔPrice_CE"] = agg["CE_Price"].diff()
    agg["ΔPrice_PE"] = agg["PE_Price"].diff()

    # VWAP proxy
    agg["VWAP"] = (
        (agg["CE_Price"] * agg["CE_vol_delta"] + agg["PE_Price"] * agg["PE_vol_delta"])
        / agg["Total_Volume"].replace(0, np.nan)
    ).fillna(method="ffill")
    agg["ΔVWAP"] = agg["VWAP"].diff()

    # core correlations
    agg["Corr_PriceVol"] = (
        agg["ΔPrice_CE"].rolling(rolling_n, min_periods=3)
        .corr(agg["CE_vol_delta"])
    )
    agg["Absorption_Index"] = np.abs(agg["CE_OI_delta"]) / (np.abs(agg["CE_vol_delta"]) + 1)
    agg["Volume_Momentum"] = (agg["Total_Volume"] / agg["Total_Volume"].shift(rolling_n)) - 1
    agg["Pressure_Score"] = (
        np.sign(agg["ΔPrice_CE"].fillna(0)) * agg["CE_vol_delta"].fillna(0)
    ).rolling(rolling_n, min_periods=3).sum()

    agg.fillna(0, inplace=True)
    return agg

agg = compute_features(df, rolling_n)

# ---------------- Regime Detection ----------------
def detect_regime(row):
    if row["ΔPrice_CE"] * row["CE_OI_delta"] > 0 and row["Volume_Spike"] > 1:
        return "trend"
    if abs(row["ΔPrice_CE"]) < 0.05 and abs(row["CE_OI_delta"]) < 1000:
        return "range"
    if abs(row["ΔPrice_CE"]) > 0.2 and row["Volume_Spike"] > 1.5:
        return "breakout"
    if row["ΔPrice_CE"] > 0 and row["CE_OI_delta"] < 0:
        return "exhaustion"
    return "quiet"

agg["Regime"] = agg.apply(detect_regime, axis=1)

# ---------------- Dashboard ----------------
st.subheader("🧭 Latest Snapshot")
latest = agg.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Vol Imbalance", f"{latest['Vol_Imbalance']:.2f}")
col2.metric("ΔVWAP", f"{latest['ΔVWAP']:.2f}")
col3.metric("Volume Spike", f"{latest['Volume_Spike']:.2f}")
col4.metric("Detected Regime", latest["Regime"])

# ---------------- Charts ----------------
st.subheader("📈 Volume vs Price Insights")

chart_df = agg.reset_index()
line_vol = alt.Chart(chart_df).mark_line(color="#F39C12").encode(
    x="timestamp:T", y=alt.Y("Vol_Imbalance:Q", title="Vol Imbalance")
)
line_vwap = alt.Chart(chart_df).mark_line(color="#2ECC71").encode(
    x="timestamp:T", y=alt.Y("ΔVWAP:Q", title="ΔVWAP")
)
st.altair_chart(alt.layer(line_vol, line_vwap).resolve_scale(y="independent"), use_container_width=True)

st.subheader("📋 Rule-Based Summary Table")
st.dataframe(agg.tail(30), use_container_width=True)

st.caption("© 2024 — Built for educational market analytics. No trading advice.")
