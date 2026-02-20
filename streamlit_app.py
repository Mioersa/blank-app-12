import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt

# ---------------- Basic Setup ----------------
st.set_page_config("Intraday Option Volume-Price Analyzer", layout="wide")
st.title("📊 Intraday Option Price–Volume Analyzer")

# ---------- Sidebar ----------
st.sidebar.header("Settings")
rolling_n = st.sidebar.number_input("Rolling window (bars)", 3, 60, 5)
spread_cutoff = st.sidebar.slider("Max bid–ask spread %", 0.0, 1.0, 0.2)
st.sidebar.markdown("Upload **incremental 5‑min Option‑Chain CSVs** 👇")

uploads = st.file_uploader("Drop CSV files", type=["csv"], accept_multiple_files=True)
if not uploads:
    st.info("⬅️ Upload CSVs to begin analysis.")
    st.stop()

# ---------- Load ----------
frames = []
for f in uploads:
    name = f.name.replace(".csv", "")
    try:
        ts = datetime.strptime(name.split("_")[-2] + "_" + name.split("_")[-1], "%d%m%Y_%H%M%S")
    except Exception:
        ts = datetime.now()
    df = pd.read_csv(f)
    df["timestamp"] = ts
    frames.append(df)

raw = pd.concat(frames).sort_values(["timestamp", "CE_strikePrice"]).reset_index(drop=True)

# ---------- Clean ----------
@st.cache_data
def clean_data(df, cutoff):
    df = df.copy()
    for s in ["CE", "PE"]:
        df[f"{s}_mid"] = (df[f"{s}_buyPrice1"] + df[f"{s}_sellPrice1"]) / 2
        df[f"{s}_spread_pct"] = abs(df[f"{s}_sellPrice1"] - df[f"{s}_buyPrice1"]) / df[f"{s}_mid"].replace(0, np.nan)
    df = df[(df["CE_spread_pct"] < cutoff) & (df["PE_spread_pct"] < cutoff)]
    return df

df = clean_data(raw, spread_cutoff)
st.success(f"✅ Loaded {len(uploads)} files, {len(df)} rows after cleaning.")

# ---------- Compute Features ----------
@st.cache_data
def compute_features(df, rolling_n):
    feat = df.copy()
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
        "PE_OI_delta": "sum",
    }).rename(columns={"CE_lastPrice": "CE_Price", "PE_lastPrice": "PE_Price"})

    agg["Total_Volume"] = agg["CE_vol_delta"] + agg["PE_vol_delta"]
    agg["Vol_Imbalance"] = (agg["CE_vol_delta"] - agg["PE_vol_delta"]) / (agg["Total_Volume"].replace(0, np.nan))
    agg["ΔPrice_CE"] = agg["CE_Price"].diff()
    agg["ΔPrice_PE"] = agg["PE_Price"].diff()

    agg["VWAP"] = (
        (agg["CE_Price"] * agg["CE_vol_delta"] + agg["PE_Price"] * agg["PE_vol_delta"])
        / agg["Total_Volume"].replace(0, np.nan)
    ).fillna(method="ffill")
    agg["ΔVWAP"] = agg["VWAP"].diff()

    agg["Volume_Spike"] = agg["Total_Volume"] / agg["Total_Volume"].rolling(rolling_n).mean()
    agg["Pressure_Score"] = (
        np.sign(agg["ΔPrice_CE"].fillna(0)) * agg["CE_vol_delta"].fillna(0)
    ).rolling(rolling_n, min_periods=3).sum()

    # simplified signal strength (-1,0,+1)
    def regime_strength(r):
        score = 0
        if r["Vol_Imbalance"] > 0.3 and r["ΔVWAP"] > 0:
            score += 1
        if r["Vol_Imbalance"] < -0.3 and r["ΔVWAP"] < 0:
            score -= 1
        if r["Pressure_Score"] > 0:
            score += 1
        if r["Pressure_Score"] < 0:
            score -= 1
        return np.clip(score, -1, 1)

    agg["Signal_Strength"] = agg.apply(regime_strength, axis=1)
    agg.fillna(0, inplace=True)
    return agg

agg = compute_features(df, rolling_n)
chart_df = agg.reset_index()

# ---------- Summary Table ----------
st.subheader("📋 Summary Table — All Uploaded Files")
st.dataframe(chart_df, use_container_width=True)

# ---------- Separate Charts ----------
st.subheader("📈 Volume Imbalance Trend")
vol_chart = (
    alt.Chart(chart_df)
    .mark_bar(color="#FFA500", opacity=0.8)
    .encode(
        x="timestamp:T",
        y=alt.Y("Vol_Imbalance:Q", title="Volume Imbalance (CE–PE)"),
        tooltip=["timestamp:T", "Vol_Imbalance"]
    )
)
st.altair_chart(vol_chart, use_container_width=True)

st.subheader("📈 ΔVWAP Trend")
vwap_chart = (
    alt.Chart(chart_df)
    .mark_bar(color="#00CC66", opacity=0.8)
    .encode(
        x="timestamp:T",
        y=alt.Y("ΔVWAP:Q", title="ΔVWAP (Price Drift)"),
        tooltip=["timestamp:T", "ΔVWAP"]
    )
)
st.altair_chart(vwap_chart, use_container_width=True)

# ---------- Signal Strength Bar (Fixed Altair v6) ----------
st.subheader("💡 Signal Strength per Snapshot (-1 = Bearish, 0 = Neutral, +1 = Bullish)")
sig_chart = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(
        x="timestamp:T",
        y=alt.Y("Signal_Strength:Q", title="Signal Strength"),
        color=alt.Color(
            "Signal_Strength:Q",
            scale=alt.Scale(domain=[-1, 0, 1], range=["#DB2828", "#AAAAAA", "#21BA45"]),
            legend=alt.Legend(title="Signal")
        ),
        tooltip=["timestamp:T", "Signal_Strength"]
    )
)
st.altair_chart(sig_chart, use_container_width=True)

# ---------- Download ----------
csv = chart_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Metrics CSV", csv, "option_volume_metrics.csv", "text/csv")

st.caption("© 2024 — Experimental analytics tool. Not trading advice.")
