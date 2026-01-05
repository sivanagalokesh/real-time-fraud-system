import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================
REFRESH_INTERVAL = 5      # seconds
MAX_POINTS = 200

REVIEW_THRESHOLD = 0.90
BLOCK_THRESHOLD = 0.99999993

# =====================================================
# PATHS (CLOUD SAFE)
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[1]
LOG_PATH = BASE_DIR / "monitoring" / "transaction_logs.csv"

# =====================================================
# PAGE SETUP
# =====================================================
st.set_page_config(
    page_title="Fraud Intelligence Console",
    layout="wide"
)

# =====================================================
# CSS (Premium Dark UI)
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
.explain-box {
    background: rgba(255,255,255,0.04);
    border-left: 4px solid #00E5FF;
    padding: 14px;
    border-radius: 6px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("## 🧠 Fraud Intelligence Core")
st.markdown(
    f"**Decision Policy:** ┆ REVIEW ≥ {REVIEW_THRESHOLD} ┆ BLOCK ≥ {BLOCK_THRESHOLD}"
)

# =====================================================
# LOAD LOG DATA
# =====================================================
if not LOG_PATH.exists():
    st.warning("⏳ Waiting for live transactions from API...")
    st.stop()

df = pd.read_csv(LOG_PATH)

if df.empty:
    st.info("No transactions logged yet.")
    st.stop()

df = df.tail(MAX_POINTS).reset_index(drop=True)

# =====================================================
# KPI CARDS
# =====================================================
k1, k2, k3, k4 = st.columns(4)

k1.markdown(
    f"<div class='metric-box'><div class='metric-title'>Transactions</div>"
    f"<div class='metric-value'>{len(df)}</div></div>",
    unsafe_allow_html=True
)

k2.markdown(
    f"<div class='metric-box'><div class='metric-title'>BLOCK</div>"
    f"<div class='metric-value'>{(df.decision == 'BLOCK').sum()}</div></div>",
    unsafe_allow_html=True
)

k3.markdown(
    f"<div class='metric-box'><div class='metric-title'>REVIEW</div>"
    f"<div class='metric-value'>{(df.decision == 'REVIEW').sum()}</div></div>",
    unsafe_allow_html=True
)

k4.markdown(
    f"<div class='metric-box'><div class='metric-title'>Avg Risk</div>"
    f"<div class='metric-value'>{df.fraud_probability.mean():.3f}</div></div>",
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# LIVE RISK STREAM
# =====================================================
st.markdown("### 🔥 Live Fraud Risk Stream")

df["tx_idx"] = range(len(df))

color_map = {
    "ALLOW": "#00C853",
    "REVIEW": "#FFB020",
    "BLOCK": "#FF4D4F"
}

fig = go.Figure()

for decision, color in color_map.items():
    subset = df[df.decision == decision]
    fig.add_trace(go.Scatter(
        x=subset["tx_idx"],
        y=subset["fraud_probability"],
        mode="markers",
        name=decision,
        marker=dict(color=color, size=8, opacity=0.85)
    ))

fig.add_hline(y=REVIEW_THRESHOLD, line_dash="dot", line_color="#FFB020")
fig.add_hline(y=BLOCK_THRESHOLD, line_dash="dot", line_color="#FF4D4F")

fig.update_layout(
    height=420,
    template="plotly_dark",
    xaxis_title="Transaction Index (Live)",
    yaxis_title="Fraud Probability"
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Real-time fraud probability stream from API decisions")

# =====================================================
# VISUAL EXPLANATION PANEL (RISK-LEVEL)
# =====================================================
st.divider()
st.markdown("### 🧠 Risk Decision Explanation")

latest = df.iloc[-1]

if latest.fraud_probability >= BLOCK_THRESHOLD:
    explanation = "🚨 **High-confidence fraud detected.** Probability exceeds automatic block threshold."
elif latest.fraud_probability >= REVIEW_THRESHOLD:
    explanation = "⚠️ **Uncertain transaction.** Routed to human review for verification."
else:
    explanation = "✅ **Low-risk transaction.** Allowed automatically."

st.markdown(f"""
<div class="explain-box">
<b>Latest Risk Score:</b> {latest.fraud_probability:.4f}<br>
<b>Decision:</b> {latest.decision}<br><br>
{explanation}
</div>
""", unsafe_allow_html=True)

# =====================================================
# RISK DISTRIBUTION (CONTEXTUAL EXPLANATION)
# =====================================================
st.markdown("### 📊 Risk Score Distribution (Recent)")

hist_fig = go.Figure()
hist_fig.add_trace(go.Histogram(
    x=df.fraud_probability,
    nbinsx=40,
    marker_color="#00E5FF",
    opacity=0.85
))

hist_fig.add_vline(x=REVIEW_THRESHOLD, line_dash="dot", line_color="#FFB020")
hist_fig.add_vline(x=BLOCK_THRESHOLD, line_dash="dot", line_color="#FF4D4F")

hist_fig.update_layout(
    height=300,
    template="plotly_dark",
    xaxis_title="Fraud Probability",
    yaxis_title="Transaction Count"
)

st.plotly_chart(hist_fig, use_container_width=True)
st.caption("Shows where most transactions fall relative to decision thresholds")

# =====================================================
# REVIEW QUEUE
# =====================================================
st.divider()
st.markdown("### ⧗ Review Queue (Human-in-the-Loop)")

review_df = df[df.decision == "REVIEW"].sort_values(
    "fraud_probability", ascending=False
)

st.dataframe(review_df.tail(10), use_container_width=True)

# =====================================================
# AUTO REFRESH (STREAMLIT-CLOUD SAFE)
# =====================================================
time.sleep(REFRESH_INTERVAL)
st.rerun()
