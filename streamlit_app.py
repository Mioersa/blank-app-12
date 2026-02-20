import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# --- Streamlit Config ---
st.set_page_config("Intraday Option Deep Analytics", layout="wide")
st.title("📊 Deep Intraday Option Analytics + Predictive Signals")

# --- User Settings ---
rolling_n = st.sidebar.number_input("Rolling window (bars)", 3, 60, 5)
spread_cutoff = st.sidebar.slider("Max bid–ask spread %", 0.0, 1.0, 0.2)
st.sidebar.markdown("Upload **incremental 5-min Option‑Chain CSVs** 👇")
uploads = st.file_uploader("", type="csv", accept_multiple_files=True)
if not uploads:
    st.info("Upload CSVs to begin.")
    st.stop()

# --- Load & Combine ---
frames = []
for f in uploads:
    ts = datetime.now()
    try:
        name = f.name.replace(".csv", "")
        ts = datetime.strptime(name.split("_")[-2] + "_" + name.split("_")[-1], "%d%m%Y_%H%M%S")
    except:
        pass
    df = pd.read_csv(f)
    df["timestamp"] = ts
    frames.append(df)
raw = pd.concat(frames).sort_values(["timestamp", "CE_strikePrice"]).reset_index(drop=True)

# --- Clean Data ---
@st.cache_data
def clean(df, cutoff):
    df = df.copy()
    for s in ["CE", "PE"]:
        df[f"{s}_mid"] = (df[f"{s}_buyPrice1"] + df[f"{s}_sellPrice1"]) / 2
        df[f"{s}_spread"] = abs(df[f"{s}_sellPrice1"] - df[f"{s}_buyPrice1"])/df[f"{s}_mid"].replace(0,np.nan)
    return df[(df["CE_spread"]<cutoff)&(df["PE_spread"]<cutoff)]
df = clean(raw, spread_cutoff)
st.success(f"Loaded {len(uploads)} files → {len(df)} rows after cleaning.")

# --- Feature Engineering ---
@st.cache_data
def build_features(df, rolling_n):
    data = df.copy()
    for s in ["CE","PE"]:
        data[f"{s}_vol"] = data.groupby("CE_strikePrice")[f"{s}_totalTradedVolume"].diff().fillna(0)
        data[f"{s}_oi_delta"] = data.groupby("CE_strikePrice")[f"{s}_openInterest"].diff().fillna(0)
        data[f"{s}_price_delta"] = data.groupby("CE_strikePrice")[f"{s}_lastPrice"].diff().fillna(0)

    agg = data.groupby("timestamp").agg(
        CE_price=("CE_lastPrice","mean"),
        PE_price=("PE_lastPrice","mean"),
        CE_vol=("CE_vol","sum"),
        PE_vol=("PE_vol","sum"),
        CE_oi_delta=("CE_oi_delta","sum"),
    )
    agg["total_vol"] = agg["CE_vol"] + agg["PE_vol"]
    agg["vol_imbalance"] = (agg["CE_vol"] - agg["PE_vol"]) / agg["total_vol"].replace(0,np.nan)
    agg["Δvwap"] = ((agg["CE_price"]*agg["CE_vol"] + agg["PE_price"]*agg["PE_vol"])/agg["total_vol"].replace(0,np.nan)).diff()
    agg["vol_spike"] = agg["total_vol"]/agg["total_vol"].rolling(rolling_n).mean()
    agg["vol_momentum"] = agg["total_vol"]/agg["total_vol"].shift(rolling_n) - 1
    agg["pressure_score"] = (np.sign(agg["CE_price"].diff().fillna(0)) * agg["CE_vol"]).rolling(rolling_n,min_periods=1).sum()
    agg["absorption_idx"] = np.abs(agg["CE_oi_delta"]) / (agg["CE_vol"].abs() + 1)
    agg.fillna(0,inplace=True)
    return agg

feat = build_features(df, rolling_n)
chart_df = feat.reset_index()

# --- Regime Discovery (k-means clusters as regimes) ---
from sklearn.cluster import KMeans
reg = KMeans(n_clusters=3, random_state=0).fit(chart_df[["vol_spike","vol_imbalance","pressure_score"]])
chart_df["Regime_Cluster"] = reg.labels_.astype(str)

# --- Predictive Signal: Predict next ΔCE_price using Ridge ---
feat["ΔCE_price"] = feat["CE_price"].diff().shift(-1).fillna(0)
X = feat[["vol_spike","vol_imbalance","pressure_score","absorption_idx"]].shift(1).fillna(0)
y = feat["ΔCE_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = Ridge().fit(X_train, y_train)
feat["Predicted_ΔCE"] = model.predict(X)
chart_df = feat.reset_index()

# --- Display Metrics & Charts ---
st.subheader("Summary Table")
st.dataframe(chart_df, use_container_width=True)

st.subheader("Volume Imbalance")
st.altair_chart(alt.Chart(chart_df).mark_line(color="#FFA500").encode(x="timestamp:T", y="vol_imbalance:Q"), use_container_width=True)

st.subheader("ΔVWAP Trend")
st.altair_chart(alt.Chart(chart_df).mark_line(color="#00CC66").encode(x="timestamp:T", y="Δvwap:Q"), use_container_width=True)

st.subheader("Regime Clusters")
st.altair_chart(
    alt.Chart(chart_df).mark_point(filled=True, size=60).encode(
        x="vol_spike:Q", y="pressure_score:Q", color="Regime_Cluster:N"
    ), use_container_width=True
)

st.subheader("Predicted ΔCE Price")
st.altair_chart(
    alt.Chart(chart_df).mark_line(color="#0000FF").encode(x="timestamp:T", y="Predicted_ΔCE:Q"), use_container_width=True
)

st.caption("© Deploy responsibly. No trading advice.")
