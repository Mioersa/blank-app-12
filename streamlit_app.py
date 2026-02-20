import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt

# ======================================
st.set_page_config("Rule‑Based Intraday Option Signals", layout="wide")
st.title("📊 Rule‑Based Intraday Option Signal System (Advanced Price‑Volume Edition)")

# ---- SIDEBAR ----
rolling_n = st.sidebar.number_input("Rolling window (bars)", 3, 60, 5)
spread_cutoff = st.sidebar.slider("Max bid‑ask spread %", 0.0, 1.0, 0.2)
basis = st.sidebar.radio("Top‑strike ranking basis", ["Open Interest", "Volume"])
num_strikes = st.sidebar.number_input("Top strikes by basis", 1, 30, 6)
st.sidebar.markdown("Upload **Option‑Chain CSV files** 👇")
uploaded = st.file_uploader("Drop CSV files (multiple allowed)",
                             type=["csv"], accept_multiple_files=True)
if not uploaded:
    st.info("⬅️ Upload CSVs to start.")
    st.stop()

# ---- LOAD ----
frames=[]
for f in uploaded:
    try:
        base=f.name.replace(".csv","")
        ts=datetime.strptime(base.split("_")[-2]+"_"+base.split("_")[-1],"%d%m%Y_%H%M%S")
    except Exception:
        ts=datetime.now()
    df=pd.read_csv(f)
    df["timestamp"]=ts
    frames.append(df)
raw=pd.concat(frames,ignore_index=True).sort_values("timestamp")
st.success(f"✅ Loaded {len(uploaded)} file(s), {len(raw)} rows.")

# ---- CLEAN ----
def clean_data(df,cuto=0.2):
    df=df.copy()
    df["timestamp"]=pd.to_datetime(df["timestamp"],errors="coerce")
    req=["CE_buyPrice1","CE_sellPrice1","PE_buyPrice1","PE_sellPrice1"]
    avail=[c for c in req if c in df.columns]
    df=df[(df[avail]>0).all(axis=1)]
    df["mid_CE"]=(df["CE_buyPrice1"]+df["CE_sellPrice1"])/2
    df["mid_PE"]=(df["PE_buyPrice1"]+df["PE_sellPrice1"])/2
    df["mid_CE"].replace(0,np.nan,inplace=True)
    df["spread_pct"]=abs(df["CE_sellPrice1"]-df["CE_buyPrice1"])/df["mid_CE"]
    df=df[df["spread_pct"]<cuto]
    if "CE_expiryDate" in df.columns:
        df["CE_expiryDate"]=pd.to_datetime(df["CE_expiryDate"],errors="coerce")
        df["days_to_expiry"]=(df["CE_expiryDate"]-df["timestamp"]).dt.days
    else:
        df["days_to_expiry"]=1
    df["days_to_expiry"].fillna(1,inplace=True)
    df["θ_adj_CE"]=df["CE_lastPrice"]/np.sqrt(df["days_to_expiry"].clip(lower=1))
    df["θ_adj_PE"]=df["PE_lastPrice"]/np.sqrt(df["days_to_expiry"].clip(lower=1))
    return df

df=clean_data(raw,spread_cutoff)

# ---- UTILS ----
def rolling_corr(a,b,window=10,minp=3):
    arr=np.full(len(a),np.nan)
    for i in range(window,len(a)):
        xa,xb=a[i-window:i], b[i-window:i]
        if np.std(xa)>1e-8 and np.std(xb)>1e-8:
            arr[i]=np.corrcoef(xa,xb)[0,1]
    return pd.Series(arr).fillna(method="bfill").fillna(0)

# ---- FEATURES ----
def compute_features(df,rolling_n=5,top_n=6,basis="Open Interest"):
    df=df.copy().sort_values("timestamp")
    df["CE_vol_delta"]=df.groupby("CE_strikePrice")["CE_totalTradedVolume"].diff().fillna(0)
    df["PE_vol_delta"]=df.groupby("CE_strikePrice")["PE_totalTradedVolume"].diff().fillna(0)
    df["total_vol"]=df["CE_vol_delta"]+df["PE_vol_delta"]
    df["total_OI"]=df["CE_openInterest"]+df["PE_openInterest"]
    metric="total_OI" if basis.startswith("Open") else "total_vol"
    mean_strike=df.groupby("CE_strikePrice")[metric].mean()
    top_strikes=mean_strike.nlargest(top_n)
    covered_pct=round(100*top_strikes.sum()/mean_strike.sum(),2)
    df=df[df["CE_strikePrice"].isin(top_strikes.index)]

    agg=df.groupby("timestamp").agg({
        "CE_lastPrice":"mean","PE_lastPrice":"mean",
        "CE_openInterest":"sum","PE_openInterest":"sum",
        "CE_changeinOpenInterest":"sum","PE_changeinOpenInterest":"sum",
        "CE_vol_delta":"sum","PE_vol_delta":"sum",
        "CE_impliedVolatility":"mean","PE_impliedVolatility":"mean"
    })
    # Existing derived fields
    agg["ΔPrice_CE"]=agg["CE_lastPrice"].diff()
    agg["ΔOI_CE"]=agg["CE_changeinOpenInterest"].diff()
    agg["ΔPrice_PE"]=agg["PE_lastPrice"].diff()
    agg["ΔOI_PE"]=agg["PE_changeinOpenInterest"].diff()
    agg["IV_skew"]=agg["CE_impliedVolatility"]-agg["PE_impliedVolatility"]
    agg["ΔIV"]=agg["IV_skew"].diff()
    agg["PCR_OI"]=agg["PE_openInterest"]/agg["CE_openInterest"].replace(0,np.nan)
    agg["ΔPCR"]=agg["PCR_OI"].diff()
    total_vol=agg["CE_vol_delta"]+agg["PE_vol_delta"]
    agg["Volume_spike"]=total_vol/total_vol.rolling(rolling_n).mean()

    # --- Advanced price-volume metrics ---
    agg["VWAP"]=(
        (agg["CE_lastPrice"]*agg["CE_vol_delta"] + agg["PE_lastPrice"]*agg["PE_vol_delta"])
        /(agg["CE_vol_delta"]+agg["PE_vol_delta"]).replace(0,np.nan)
    ).fillna(method="ffill")
    agg["ΔVWAP"]=agg["VWAP"].diff()
    agg["Vol_imbalance"]=(agg["CE_vol_delta"]-agg["PE_vol_delta"])/(agg["CE_vol_delta"]+agg["PE_vol_delta"]).replace(0,np.nan)
    agg["Absorption_idx"]=agg["CE_vol_delta"].abs()/(agg["ΔOI_CE"].abs()+1)
    agg["Cum_tick_flow"]=np.cumsum(np.sign(agg["ΔPrice_CE"].fillna(0))*agg["CE_vol_delta"])
    agg["Corr_PriceVol"]=rolling_corr(agg["ΔPrice_CE"].values,agg["CE_vol_delta"].values,window=rolling_n)
    agg["Corr_IVVol"]=rolling_corr(agg["ΔIV"].values,agg["Volume_spike"].values,window=rolling_n)

    agg.fillna(0,inplace=True)
    return agg,covered_pct

df_feat,covered_pct=compute_features(df,rolling_n,num_strikes,basis)
st.caption(f"**Top {num_strikes} strikes** cover ≈ {covered_pct}% of total {basis.lower()}.")

# ---- REGIME LOGIC ----
def detect_regime(row):
    reg,bias="quiet","neutral"
    if row["ΔPrice_CE"]*row["ΔOI_CE"]>0 and row["Volume_spike"]>1: reg="trend"
    elif abs(row["ΔPrice_CE"])<0.05 and abs(row["ΔOI_CE"])<1000: reg="range"
    elif abs(row["ΔPrice_CE"])>0.2 and row["Volume_spike"]>1.5 and row["ΔIV"]>0: reg="breakout"
    elif row["ΔPrice_CE"]>0 and row["ΔOI_CE"]<0 and row["ΔIV"]<0: reg="exhaustion"
    if row["PCR_OI"]<0.8: bias="bullish"
    elif row["PCR_OI"]>1.2: bias="bearish"
    return reg,bias

def generate_signal(row):
    if row["regime"]=="trend" and row["bias"]=="bullish": return "BUY_CALL"
    if row["regime"]=="trend" and row["bias"]=="bearish": return "BUY_PUT"
    if row["regime"]=="range": return "SELL_STRANGLE"
    if row["regime"]=="breakout": return "MOMENTUM_TRADE"
    if row["regime"]=="exhaustion": return "EXIT_POSITION"
    return "HOLD"

df_feat[["regime","bias"]]=df_feat.apply(detect_regime,axis=1,result_type="expand")
df_feat["signal"]=df_feat.apply(generate_signal,axis=1)
df_feat["signal_numeric"]=df_feat["signal"].map({
    "BUY_CALL":1,"BUY_PUT":1,"MOMENTUM_TRADE":1,
    "SELL_STRANGLE":0,"HOLD":0,"EXIT_POSITION":-1
}).fillna(0)

# ---- HUMAN READABLE SUPPLEMENT ----
def signal_summary(r):
    text=[]
    if r["Vol_imbalance"]>0.3: text.append("🔼 Call‑side volume dominating → upward flow pressure.")
    elif r["Vol_imbalance"]<-0.3: text.append("🔽 Put‑side volume dominating → downside pressure.")
    else: text.append("🟧 Volume balanced between CE & PE.")
    if r["ΔVWAP"]>0: text.append("VWAP rising → buyers lifting offer prices.")
    elif r["ΔVWAP"]<0: text.append("VWAP falling → sellers in control.")
    if r["Corr_PriceVol"]>0.5: text.append("Strong price‑volume positive corr → conviction buying.")
    elif r["Corr_PriceVol"]<-0.5: text.append("Negative price‑volume corr → distribution / absorption.")
    if r["Absorption_idx"]<1: text.append("Absorption – OI not rising with volume → possible trap/reversal.")
    if r["Corr_IVVol"]>0.4: text.append("IV rising with volume → speculative interest building.")
    elif r["Corr_IVVol"]<-0.4: text.append("IV falling as volume grows → hedges unwinding.")

    # Conclusive directional statement
    concl="CE/PE prices likely steady."
    if r["Vol_imbalance"]>0.3 and r["ΔVWAP"]>0: concl="📈 CE prices could rise, PE could soften."
    elif r["Vol_imbalance"]<-0.3 and r["ΔVWAP"]<0: concl="📉 PE prices could rise, CE could weaken."
    text.append(concl)
    return "\n".join(text)

df_feat["Implied_Signal_Text"]=df_feat.apply(signal_summary,axis=1)

# ---- DISPLAY ----
lat=df_feat.iloc[-1]
c1,c2,c3,c4=st.columns(4)
c1.metric("Current PCR (OI)",round(float(lat["PCR_OI"]),2))
c2.metric("# Trend Bars",int((df_feat["regime"]=="trend").sum()))
c3.metric("Latest Signal",lat["signal"])
c4.metric("Rows Processed",len(df_feat))

def pcr_text(p):
    if p<0.7:return"🐂 Bullish – calls lead"
    if 0.7<=p<=1.2:return"🟧 Neutral structure"
    return"🐻 Bearish – puts build"
st.caption(f"**PCR Interpretation:** {pcr_text(lat['PCR_OI'])}")

st.subheader("🧾 Recent Signals + Implied Metrics")
st.dataframe(
    df_feat.tail(10)[["signal","bias","Vol_imbalance","ΔVWAP","Corr_PriceVol","Absorption_idx","Corr_IVVol","Implied_Signal_Text"]],
    use_container_width=True
)

st.subheader("🌀 Signal / Bias Timeline")
sig_chart=alt.Chart(df_feat.reset_index()).mark_circle(size=80).encode(
    x="timestamp:T",
    y=alt.Y("signal_numeric:Q",scale=alt.Scale(domain=[-1.2,1.2]),
            title="Signal (‑1 = Sell, 0 = Hold, +1 = Buy)"),
    color="bias:N",tooltip=["timestamp","signal","bias","regime"])
st.altair_chart(sig_chart,use_container_width=True)

# --- Extra visual for imbalance ---
st.subheader("⚖️ Volume Imbalance & VWAP Trend")
imb_chart=alt.LayerChart(df_feat.reset_index())
imbalance=alt.Chart(df_feat.reset_index()).mark_line(color="orange").encode(
    x="timestamp:T",y="Vol_imbalance:Q")
vwap=alt.Chart(df_feat.reset_index()).mark_line(color="green").encode(
    x="timestamp:T",y="ΔVWAP:Q")
st.altair_chart(imbalance+vwap,use_container_width=True)
