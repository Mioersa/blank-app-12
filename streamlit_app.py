import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# ---------------- CONFIG ----------------
st.set_page_config("Deep Intraday Option Analyzer", layout="wide")
st.title("📊 Deep Intraday Option Analytics – Volume • Regime • Forecasts")

# ---------- SIDEBAR ----------
st.sidebar.header("Settings")
rolling_n = st.sidebar.number_input("Rolling window (bars)", 3, 60, 5)
spread_cutoff = st.sidebar.slider("Max bid–ask spread %", 0.0, 1.0, 0.2)
st.sidebar.markdown("Upload **incremental 5‑min Option‑Chain CSVs** 👇")
uploads = st.file_uploader("", type="csv", accept_multiple_files=True)
if not uploads:
    st.info("⬅️ Upload CSVs to begin analysis.")
    st.stop()

# ---------- LOAD ----------
frames = []
for f in uploads:
    try:
        name = f.name.replace(".csv", "")
        ts = datetime.strptime(
            name.split("_")[-2] + "_" + name.split("_")[-1], "%d%m%Y_%H%M%S"
        )
    except Exception:
        ts = datetime.now()
    df = pd.read_csv(f)
    df["timestamp"] = ts
    frames.append(df)

raw = pd.concat(frames).sort_values(["timestamp", "CE_strikePrice"]).reset_index(drop=True)

# ---------- CLEAN ----------
@st.cache_data
def clean(df, cutoff):
    df = df.copy()
    for s in ["CE", "PE"]:
        df[f"{s}_mid"] = (df[f"{s}_buyPrice1"] + df[f"{s}_sellPrice1"]) / 2
        df[f"{s}_spread_pct"] = abs(
            df[f"{s}_sellPrice1"] - df[f"{s}_buyPrice1"]
        ) / df[f"{s}_mid"].replace(0, np.nan)
    df = df[(df["CE_spread_pct"] < cutoff) & (df["PE_spread_pct"] < cutoff)]
    return df


df = clean(raw, spread_cutoff)
st.success(f"✅ Loaded {len(uploads)} files, {len(df)} rows after cleaning.")

# ---------- FEATURES ----------
@st.cache_data
def build_features(df, rolling_n):
    feat = df.copy()
    for s in ["CE", "PE"]:
        feat[f"{s}_vol_delta"] = feat.groupby("CE_strikePrice")[
            f"{s}_totalTradedVolume"
        ].diff().fillna(0)
        feat[f"{s}_OI_delta"] = feat.groupby("CE_strikePrice")[
            f"{s}_openInterest"
        ].diff().fillna(0)
        feat[f"{s}_price_delta"] = feat.groupby("CE_strikePrice")[
            f"{s}_lastPrice"
        ].diff().fillna(0)

    agg = feat.groupby("timestamp").agg(
        CE_price=("CE_lastPrice", "mean"),
        PE_price=("PE_lastPrice", "mean"),
        CE_vol=("CE_vol_delta", "sum"),
        PE_vol=("PE_vol_delta", "sum"),
        CE_OI_delta=("CE_OI_delta", "sum"),
        PE_OI_delta=("PE_OI_delta", "sum"),
    )

    agg["Total_Volume"] = agg["CE_vol"] + agg["PE_vol"]
    agg["Vol_Imbalance"] = (agg["CE_vol"] - agg["PE_vol"]) / (
        agg["Total_Volume"].replace(0, np.nan)
    )
    agg["ΔVWAP"] = (
        (agg["CE_price"] * agg["CE_vol"] + agg["PE_price"] * agg["PE_vol"])
        / agg["Total_Volume"].replace(0, np.nan)
    ).diff()

    agg["Vol_Spike"] = agg["Total_Volume"] / agg["Total_Volume"].rolling(rolling_n).mean()
    agg["Vol_Momentum"] = agg["Total_Volume"] / agg["Total_Volume"].shift(rolling_n) - 1
    agg["Pressure_Score"] = (
        np.sign(agg["CE_price"].diff().fillna(0)) * agg["CE_vol"].fillna(0)
    ).rolling(rolling_n, min_periods=1).sum()
    agg["Absorption_Index"] = np.abs(agg["CE_OI_delta"]) / (np.abs(agg["CE_vol"]) + 1)
    agg.fillna(0, inplace=True)
    return agg


feat = build_features(df, rolling_n)

# ---------- REGIME DISCOVERY ----------
chart_df = feat.reset_index()
kmod = KMeans(n_clusters=3, random_state=0).fit(
    chart_df[["Vol_Spike", "Vol_Imbalance", "Pressure_Score"]]
)
chart_df["Regime_Cluster"] = kmod.labels_

# sentiment mapping by mean imbalance
sent_map = {}
for cl in chart_df["Regime_Cluster"].unique():
    val = chart_df.loc[chart_df["Regime_Cluster"] == cl, "Vol_Imbalance"].mean()
    if val > 0.1:
        sent_map[cl] = "Bullish"
    elif val < -0.1:
        sent_map[cl] = "Bearish"
    else:
        sent_map[cl] = "Neutral"
chart_df["Sentiment"] = chart_df["Regime_Cluster"].map(sent_map)

# ---------- PREDICTIVE SIGNALS ----------
@st.cache_data
def predictive_models(feat):
    df = feat.copy()
    df["ΔCE_price_next"] = df["CE_price"].diff().shift(-1).fillna(0)
    df["ΔPE_price_next"] = df["PE_price"].diff().shift(-1).fillna(0)
    X = (
        df[["Vol_Spike", "Vol_Imbalance", "Pressure_Score", "Absorption_Index"]]
        .shift(1)
        .fillna(0)
    )
    ridge_ce = Ridge().fit(X, df["ΔCE_price_next"])
    ridge_pe = Ridge().fit(X, df["ΔPE_price_next"])
    df["Pred_ΔCE"] = ridge_ce.predict(X)
    df["Pred_ΔPE"] = ridge_pe.predict(X)
    return df


feat_pred = predictive_models(feat)
chart_df = feat_pred.reset_index()
st.subheader("📋 Summary Table – All Uploaded Files")
st.dataframe(chart_df, use_container_width=True)

# ---------- PLOTS ----------
st.subheader("📈 Volume Imbalance Over Time")
imb_chart = (
    alt.Chart(chart_df)
    .mark_line(color="#FFA500")
    .encode(x="timestamp:T", y="Vol_Imbalance:Q")
)
st.altair_chart(imb_chart, use_container_width=True)

st.subheader("📈 ΔVWAP Over Time")
vwap_chart = (
    alt.Chart(chart_df)
    .mark_line(color="#00CC66")
    .encode(x="timestamp:T", y="ΔVWAP:Q")
)
st.altair_chart(vwap_chart, use_container_width=True)

st.subheader("🌀 Regime Discovery (with Sentiment)")
reg_plot = (
    alt.Chart(chart_df)
    .mark_circle(size=70)
    .encode(
        x="Vol_Spike:Q",
        y="Pressure_Score:Q",
        color=alt.Color("Sentiment:N",
                        scale=alt.Scale(domain=["Bearish","Neutral","Bullish"],
                                        range=["#DB2828","#AAAAAA","#21BA45"])),
        tooltip=["timestamp:T","Vol_Spike","Pressure_Score","Sentiment"]
    )
)
st.altair_chart(reg_plot, use_container_width=True)
st.markdown(
    f"**Regime Sentiments:** " + 
    ", ".join([f"Cluster {c} → {lab}" for c, lab in sent_map.items()])
)

st.subheader("🔮 Predicted ΔCE and ΔPE Price Movements")
pred_ce = alt.Chart(chart_df).mark_line(color="#2980B9").encode(x="timestamp:T", y="Pred_ΔCE:Q")
pred_pe = alt.Chart(chart_df).mark_line(color="#E67E22").encode(x="timestamp:T", y="Pred_ΔPE:Q")
st.altair_chart(alt.layer(pred_ce, pred_pe).resolve_scale(y="independent"), use_container_width=True)

csv = chart_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Metrics CSV", csv, "option_deep_metrics.csv", "text/csv")

st.caption("© 2024 — Experimental analytics demo. Not trading advice.")
