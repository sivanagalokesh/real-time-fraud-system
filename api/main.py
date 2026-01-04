from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import pandas as pd
import joblib
import json
import csv
from pathlib import Path
from datetime import datetime

# =====================================================
# APP INITIALIZATION
# =====================================================
app = FastAPI(
    title="Real-Time Fraud Risk Scoring API",
    description="Production-ready fraud scoring with ALLOW / REVIEW / BLOCK decisions",
    version="1.0"
)

# =====================================================
# PATH RESOLUTION (PRODUCTION SAFE)
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "model" / "fraud_logistic_model.pkl"
FEATURE_PATH = BASE_DIR / "model" / "feature_list.json"
LOG_FILE = BASE_DIR / "monitoring" / "transaction_logs.csv"

LOG_FILE.parent.mkdir(exist_ok=True)

# =====================================================
# LOAD MODEL & FEATURES
# =====================================================
model = joblib.load(MODEL_PATH)

with open(FEATURE_PATH, "r") as f:
    FEATURE_LIST = json.load(f)

# =====================================================
# DECISION THRESHOLDS (FINALIZED)
# =====================================================
REVIEW_THRESHOLD = 0.90
BLOCK_THRESHOLD = 0.993

# =====================================================
# REQUEST / RESPONSE SCHEMAS
# =====================================================
class TransactionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    fraud_probability: float
    decision: str

# =====================================================
# LOGGING FUNCTION (PERSISTENT)
# =====================================================
def log_transaction(probability: float, decision: str):
    write_header = not LOG_FILE.exists()

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "fraud_probability", "decision"])
        writer.writerow([
            datetime.now().isoformat(),
            round(float(probability), 6),
            decision
        ])

# =====================================================
# PREPROCESS INPUT
# =====================================================
def preprocess_input(features: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame([features])

    missing = set(FEATURE_LIST) - set(df.columns)
    extra = set(df.columns) - set(FEATURE_LIST)

    if missing:
        raise ValueError(f"Missing features: {missing}")
    if extra:
        raise ValueError(f"Unexpected features: {extra}")

    return df[FEATURE_LIST]

# =====================================================
# DECISION LOGIC
# =====================================================
def make_decision(probability: float) -> str:
    if probability >= BLOCK_THRESHOLD:
        return "BLOCK"
    elif probability >= REVIEW_THRESHOLD:
        return "REVIEW"
    else:
        return "ALLOW"

# =====================================================
# PREDICTION ENDPOINT
# =====================================================
@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: TransactionRequest):
    try:
        X = preprocess_input(transaction.features)

        fraud_probability = model.predict_proba(X)[0, 1]
        decision = make_decision(fraud_probability)

        # 🔴 PERSIST LOG
        log_transaction(fraud_probability, decision)

        return {
            "fraud_probability": round(float(fraud_probability), 6),
            "decision": decision
        }

    except Exception as e:
        return {"error": str(e)}

# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/health")
def health():
    return {
        "status": "OK",
        "model_loaded": True,
        "review_threshold": REVIEW_THRESHOLD,
        "block_threshold": BLOCK_THRESHOLD
    }
