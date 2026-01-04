import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.graph_objects as go
from collections import deque
from datetime import datetime
import joblib
import json

# =====================================================
# CONFIG
# =====================================================
API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = "data/raw/creditcard.csv"

STREAM_INTERVAL = 2  # seconds
MAX_POINTS = 200

REVIEW_THRESHOLD = 0.90
BLOCK_THRESHOLD = 0.99999993

# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(
    page_title="Fraud Intelligence Console",
    layout="wide"
)

# =====================================================
# CSS (SAFE & CLEAN)
# =====================================================
st.markdown("""
<style>
body { background-color: #0E1117; }
.metric-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 16px;
    border-radius: 12px;
    text-align: center;
}
.metric-title { font-size: 13px; color: #AAAAAA; }
.metric-value { font-size: 28px; font-weight: 600; color: #00E5FF; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL ARTIFACTS
# =====================================================
model = joblib.load("model/fraud_logistic_model.pkl")
scaler = joblib.load("model/scaler.pkl")

with open("model/feature_list.json") as f:
    FEATURE_LIST = json.load(f)

coefficients = model.coef_[0]

# =====================================================
# FEATURE ENGINEERING (MATCHES TRAINING)
# =====================================================
def prepare_features(row: pd.Series) -> pd.Series:
    features = row.copy()

    # Hour from Time (seconds)
    features["Hour"] = int((features["Time"] // 3600) % 24)

    # Scaled Amount
    features["Scaled_Amount"] = scaler.transform(
        [[features["Amount"]]]
    )[0][0]

    # Drop unused raw columns
    features = features.drop(
        ["Time", "Amount", "Class"],
        errors="ignore"
    )

    return features

# =====================================================
# SESSION STATE
# =====================================================
if "probs" not in st.session_state:
    st.session_state.probs = deque(maxlen=MAX_POINTS)
    st.session_state.decisions = deque(maxlen=MAX_POINTS)
    st.session_state.times = deque(maxlen=MAX_POINTS)
    st.session_state.amounts = deque(maxlen=MAX_POINTS)
    st.session_state.row_idx = 0
    st.session_state.streaming = False

# =====================================================
# HEADER
# =====================================================
st.markdown("## ◉ Fraud Intelligence Core")
st.markdown(
    f"**Decision Policy:** ┆ REVIEW ≥ {REVIEW_THRESHOLD} ┆ BLOCK ≥ {BLOCK_THRESHOLD}"
)

# =====================================================
# KPI CARDS
# =====================================================
k1, k2, k3, k4 = st.columns(4)

k1.markdown(
    f"<div class='metric-box'><div class='metric-title'>Transactions</div>"
    f"<div class='metric-value'>{len(st.session_state.probs)}</div></div>",
    unsafe_allow_html=True
)

k2.markdown(
    f"<div class='metric-box'><div class='metric-title'>BLOCK</div>"
    f"<div class='metric-value'>{st.session_state.decisions.count('BLOCK')}</div></div>",
    unsafe_allow_html=True
)

k3.markdown(
    f"<div class='metric-box'><div class='metric-title'>REVIEW</div>"
    f"<div class='metric-value'>{st.session_state.decisions.count('REVIEW')}</div></div>",
    unsafe_allow_html=True
)

avg_risk = np.mean(st.session_state.probs) if st.session_state.probs else 0
k4.markdown(
    f"<div class='metric-box'><div class='metric-title'>Avg Risk</div>"
    f"<div class='metric-value'>{avg_risk:.3f}</div></div>",
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# LIVE RISK STREAM
# =====================================================
st.markdown("### 🔥 Live Risk Stream")

fig = go.Figure()
color_map = {"ALLOW": "#00C853", "REVIEW": "#FFB020", "BLOCK": "#FF4D4F"}

for decision in ["ALLOW", "REVIEW", "BLOCK"]:
    idx = [i for i, d in enumerate(st.session_state.decisions) if d == decision]
    fig.add_trace(go.Scatter(
        x=[list(st.session_state.times)[i] for i in idx],
        y=[list(st.session_state.probs)[i] for i in idx],
        mode="markers",
        name=decision,
        marker=dict(
            size=[max(6, list(st.session_state.amounts)[i] / 20) for i in idx],
            color=color_map[decision],
            opacity=0.85
        )
    ))

fig.add_hline(y=REVIEW_THRESHOLD, line_dash="dot", line_color="#FFB020")
fig.add_hline(y=BLOCK_THRESHOLD, line_dash="dot", line_color="#FF4D4F")

fig.update_layout(
    height=420,
    xaxis_title="Time",
    yaxis_title="Fraud Probability",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Live fraud probability stream from real transactions")

# =====================================================
# STREAM CONTROLS
# =====================================================
df = pd.read_csv(DATA_PATH)

c1, c2 = st.columns(2)
if c1.button("▶ Start Live Stream"):
    st.session_state.streaming = True
if c2.button("⏸ Pause Stream"):
    st.session_state.streaming = False

# =====================================================
# STREAMING LOGIC (BULLETPROOF)
# =====================================================
if st.session_state.streaming and st.session_state.row_idx < len(df):
    row = df.iloc[st.session_state.row_idx]

    engineered = prepare_features(row)

    missing = set(FEATURE_LIST) - set(engineered.index)
    if missing:
        st.error(f"Missing features after engineering: {missing}")
        st.session_state.streaming = False
        st.stop()

    features = engineered[FEATURE_LIST].to_dict()
    st.session_state.last_features = features


    response = requests.post(API_URL, json={"features": features})

    if response.status_code != 200:
        st.error(f"API Error {response.status_code}: {response.text}")
        st.session_state.streaming = False
        st.stop()

    try:
        result = response.json()
    except Exception:
        st.error(f"Invalid API response: {response.text}")
        st.session_state.streaming = False
        st.stop()

    st.session_state.probs.append(result["fraud_probability"])
    st.session_state.decisions.append(result["decision"])
    st.session_state.times.append(datetime.now().strftime("%H:%M:%S"))
    st.session_state.amounts.append(abs(features["Scaled_Amount"]) * 100)

    st.session_state.row_idx += 1
    time.sleep(STREAM_INTERVAL)
    st.experimental_rerun()

# =====================================================
# EXPLANATION PANEL
# =====================================================
st.divider()
st.markdown("### ◉ Latest Transaction Explanation")

if "last_features" in st.session_state:
    contrib = (
        pd.Series(coefficients, index=FEATURE_LIST) *
        pd.Series(st.session_state.last_features)
    ).sort_values(key=abs, ascending=False)

    st.bar_chart(contrib.head(8))
    st.caption("Top contributing features to the latest fraud score")

else:
    st.info("No transaction processed yet.")


# =====================================================
# REVIEW QUEUE
# =====================================================
st.divider()
st.markdown("### ⧗ Review Queue (Human-in-the-Loop)")

review_df = pd.DataFrame({
    "Time": st.session_state.times,
    "Risk": st.session_state.probs,
    "Decision": st.session_state.decisions
})

review_df = review_df[review_df["Decision"] == "REVIEW"].sort_values(
    "Risk", ascending=False
)

st.dataframe(review_df.tail(10), use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("🟢 System Stable | ◉ Model v1.0 | Real-Time Fraud Intelligence Console")
