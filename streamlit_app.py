import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from datetime import datetime

st.set_page_config("Deep Intraday Futures Analyzer", layout="wide")
st.title("📈 Deep Intraday Futures Analytics – Predictive & Regime Insights")

# ---------- SIDEBAR ----------
st.sidebar.header("Settings")
rolling_n = st.sidebar.number_input("Rolling window (bars)", 3, 60, 5)
uploads = st.file_uploader("Upload 5‑min Futures CSVs", type="csv", accept_multiple_files=True)
if not uploads:
    st.stop()

# ---------- LOAD ----------
frames = []
for f in uploads:
    try:
        nm = f.name.replace(".csv", "")
        ts = datetime.strptime(nm.split("_")[-2] + "_" + nm.split("_")[-1], "%d%m%Y_%H%M%S")
    except Exception:
        ts = datetime.now()
    df = pd.read_csv(f)
    df["timestamp"] = ts
    frames.append(df)

raw = pd.concat(frames).sort_values(["timestamp"]).reset_index(drop=True)
st.success(f"✅ Loaded {len(uploads)} files | {len(raw)} rows total.")

# ---------- FEATURE ENGINEERING ----------
@st.cache_data
def build_features(df, n):
    data = df.copy()
    data["Δprice"] = data["closePrice"] - data["openPrice"]
    data["ΔOI"] = data["openInterest"].diff()
    data["LogRet"] = np.log(data["closePrice"]).diff()
    data["Volatility"] = (data["highPrice"] - data["lowPrice"]) / data["openPrice"]
    data["Mom_3bar"] = data["LogRet"].rolling(3).sum()
    data["Vol_Spike"] = data["volume"] / data["volume"].rolling(10).mean()
    data["OI_Pressure"] = data["ΔOI"] * np.sign(data["Δprice"])

    # Targets
    data["next_ret"] = data["LogRet"].shift(-1)
    data["next_vol"] = data["Volatility"].shift(-1)
    data["next_turnover"] = data["totalTurnover"].shift(-1)
    data.fillna(0, inplace=True)
    return data

feat = build_features(raw, rolling_n)

# ---------- REGIME DISCOVERY ----------
chart_df = feat.copy()
kmod = KMeans(n_clusters=3, random_state=0).fit(
    chart_df[["Volatility", "ΔOI", "LogRet"]]
)
chart_df["Regime_Cluster"] = kmod.labels_

# label regimes by vol/oi correlation
sent_map = {}
for c in sorted(chart_df["Regime_Cluster"].unique()):
    sub = chart_df.loc[chart_df["Regime_Cluster"] == c]
    corr = sub["Volatility"].corr(sub["ΔOI"])
    if corr > 0.2:
        sent_map[c] = "Trending"
    elif corr < -0.2:
        sent_map[c] = "Choppy"
    else:
        sent_map[c] = "Neutral"
chart_df["Regime"] = chart_df["Regime_Cluster"].map(sent_map)

# ---------- PREDICTIONS ----------
@st.cache_data
def predictive_models(df):
    X = df[["Mom_3bar","Vol_Spike","OI_Pressure","LogRet","Volatility"]].fillna(0)

    models = {}
    for target in ["next_ret","next_vol","next_turnover"]:
        y = df[target].fillna(0)
        m = Ridge().fit(X, y)
        df[f"Pred_{target}"] = m.predict(X)
        models[target] = m
    return df, models

pred_df, _ = predictive_models(chart_df)
chart_df = pred_df.copy()

# ---------- VISUALS ----------
st.subheader("📋 Summary Table")
st.dataframe(chart_df[[
    "timestamp","closePrice","Δprice","ΔOI","Volatility","Regime",
    "Pred_next_ret","Pred_next_vol","Pred_next_turnover"
]], use_container_width=True)

# --- return prediction plot ---
st.subheader("🔮 Predicted vs Actual – Next‑Bar Return")
ret_chart = (
    alt.Chart(chart_df)
    .mark_line(color="#1f77b4")
    .encode(x="timestamp:T", y="Pred_next_ret:Q")
)
act_ret = alt.Chart(chart_df).mark_line(color="#ff7f0e", strokeDash=[4,2]).encode(
    x="timestamp:T", y="next_ret:Q"
)
st.altair_chart(alt.layer(ret_chart, act_ret).resolve_scale(y="independent"), use_container_width=True)

# --- volatility forecast plot ---
st.subheader("🌪 Volatility Forecast")
vol_chart = (
    alt.Chart(chart_df)
    .mark_line(color="#2ca02c")
    .encode(x="timestamp:T", y="Pred_next_vol:Q")
)
act_vol = alt.Chart(chart_df).mark_line(color="#16a085", strokeDash=[4,2]).encode(
    x="timestamp:T", y="next_vol:Q"
)
st.altair_chart(alt.layer(vol_chart, act_vol).resolve_scale(y="independent"), use_container_width=True)

# --- regime scatter ---
st.subheader("🌀 Regime Tagging")
reg_chart = (
    alt.Chart(chart_df)
    .mark_circle(size=70, opacity=0.8)
    .encode(
        x=alt.X("Volatility:Q"),
        y=alt.Y("ΔOI:Q"),
        color=alt.Color(
            "Regime:N",
            scale=alt.Scale(domain=["Trending","Choppy","Neutral"],
                            range=["#21ba45","#db2828","#aaaaaa"]),
        ),
        tooltip=["timestamp:T", "Volatility", "ΔOI", "Regime"],
    )
)
st.altair_chart(reg_chart, use_container_width=True)
st.markdown(" ".join([f"**Cluster {c} → {lab}**" for c, lab in sent_map.items()]))

# ---------- DOWNLOAD ----------
csv = chart_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Metrics CSV", csv, "futures_metrics.csv", "text/csv")

st.caption("© 2024 · Educational analytics demo · no trading advice.")
