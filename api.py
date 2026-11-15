import os
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from pipeline import (
    load_pipeline,
    preprocess_for_inference,
    compute_unified_risk_score,
    auto_detect_and_map_columns,
    train_fresh_model_on_any_dataset,
    engineer_features
)

app = FastAPI(
    title="FinShield Anomaly Detection & Fraud Intelligence API",
    description="Production REST API for FinShield: Autonomous Financial Anomaly Surveillance & Dual-Model Fraud Intelligence.",
    version="2.0.0"
)

# Load pipeline at startup
iso_model, lof_model, scaler, metadata, explainer = load_pipeline()

class TransactionPayload(BaseModel):
    transaction_id: Optional[str] = Field("TX99999", description="Unique transaction ID")
    account_id: Optional[str] = Field("AC00001", description="Customer account ID")
    transaction_amount: float = Field(..., gt=0, description="Monetary amount of transaction in USD")
    account_balance: float = Field(..., ge=0, description="Post-transaction account balance")
    customer_age: Optional[int] = Field(35, ge=18, le=100)
    customer_occupation: Optional[str] = Field("Engineer", description="Doctor, Student, Engineer, Retired, etc.")
    transaction_type: Optional[str] = Field("Debit", description="Debit or Credit")
    channel: Optional[str] = Field("Online", description="Online, ATM, or Branch")
    location: Optional[str] = Field("Houston", description="Transaction city")
    transaction_seconds_duration: Optional[int] = Field(65, ge=1)
    login_attempts: Optional[int] = Field(1, ge=1, le=10)
    transaction_hour: Optional[int] = Field(14, ge=0, le=23)
    time_since_previous: Optional[float] = Field(120.0, ge=0.0)
    device_changed: Optional[bool] = Field(False)
    ip_changed: Optional[bool] = Field(False)
    day_of_week: Optional[str] = Field("Monday")

class PredictionResponse(BaseModel):
    transaction_id: str
    verdict: str
    risk_level: str
    risk_score_pct: float
    isolation_forest_anomaly: bool
    lof_anomaly: bool
    isolation_forest_score: float
    risk_factors: List[str]

@app.get("/")
def root():
    return {
        "brand": "FinShield",
        "service": "FinShield Autonomous Financial Anomaly Surveillance API",
        "tagline": "Autonomous Financial Anomaly Surveillance & Dual-Model Behavioral Fraud Intelligence",
        "status": "online",
        "documentation": "/docs"
    }

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "isolation_forest": iso_model is not None,
            "local_outlier_factor": lof_model is not None,
            "scaler": scaler is not None
        },
        "features_count": len(metadata["feature_columns"]),
        "iqr_upper_limit": metadata["upper_limit"]
    }

@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_transaction(tx: TransactionPayload):
    try:
        is_weekend = tx.day_of_week in ["Saturday", "Sunday"]
        amt_balance_ratio = tx.transaction_amount / (tx.account_balance if tx.account_balance > 0 else 0.01)
        high_val_flag = tx.transaction_amount > metadata['upper_limit']
        
        row_dict = {
            'transaction_amount': tx.transaction_amount,
            'customer_age': tx.customer_age,
            'transaction_seconds_duration': tx.transaction_seconds_duration,
            'login_attempts': tx.login_attempts,
            'account_balance': tx.account_balance,
            'transaction_hour': tx.transaction_hour,
            'time_since_previous': tx.time_since_previous,
            'amount_balance_ratio': amt_balance_ratio,
            'transaction_type': tx.transaction_type,
            'location': tx.location,
            'channel': tx.channel,
            'customer_occupation': tx.customer_occupation,
            'device_changed': tx.device_changed,
            'ip_changed': tx.ip_changed,
            'is_weekend': is_weekend,
            'high_value_transaction': high_val_flag,
            'day_of_week': tx.day_of_week
        }
        
        df_in = pd.DataFrame([row_dict])
        X_single = preprocess_for_inference(df_in, scaler, metadata)
        
        iso_pred = iso_model.predict(X_single)[0]
        lof_pred = lof_model.predict(X_single)[0]
        iso_score = float(iso_model.decision_function(X_single)[0])
        lof_score = float(lof_model.score_samples(X_single)[0])
        
        risk_score = compute_unified_risk_score(iso_score, lof_score)
        
        iso_flag = bool(iso_pred == -1)
        lof_flag = bool(lof_pred == -1)
        
        if iso_flag and lof_flag:
            verdict = "CRITICAL_ANOMALY"
            risk_level = "CRITICAL"
        elif iso_flag or lof_flag:
            verdict = "ELEVATED_RISK"
            risk_level = "HIGH" if risk_score > 65 else "MEDIUM"
        else:
            verdict = "NORMAL"
            risk_level = "LOW"
            
        factors = []
        if high_val_flag:
            factors.append(f"Amount (${tx.transaction_amount:,.2f}) exceeds dataset IQR threshold (${metadata['upper_limit']:,.2f}).")
        if amt_balance_ratio > 0.5:
            factors.append(f"High amount-to-balance ratio ({amt_balance_ratio*100:.1f}%).")
        if tx.login_attempts >= 3:
            factors.append(f"{tx.login_attempts} failed login attempts prior to transaction.")
        if tx.device_changed:
            factors.append("Transaction originates from an unrecognized device.")
        if tx.ip_changed:
            factors.append("Transaction originates from an anomalous IP address.")
        if tx.time_since_previous < 2.0:
            factors.append(f"High velocity: Only {tx.time_since_previous:.1f} minutes since last transaction.")
            
        return PredictionResponse(
            transaction_id=tx.transaction_id or "TX_UNKNOWN",
            verdict=verdict,
            risk_level=risk_level,
            risk_score_pct=risk_score,
            isolation_forest_anomaly=iso_flag,
            lof_anomaly=lof_flag,
            isolation_forest_score=iso_score,
            risk_factors=factors
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/batch-upload")
async def batch_upload_csv(file: UploadFile = File(...), train_fresh: bool = False, contamination: float = 0.05):
    try:
        df = pd.read_csv(file.file)
        if train_fresh:
            res_df, features = train_fresh_model_on_any_dataset(df, contamination=contamination)
            return {
                "records_processed": len(res_df),
                "mode": "fresh_dataset_training",
                "features_used": features,
                "anomalies_detected": int((res_df['Audit_Verdict'] != 'NORMAL').sum()),
                "sample_results": res_df.head(10).to_dict(orient="records")
            }
        else:
            mapping = auto_detect_and_map_columns(df)
            df_mapped = df.rename(columns=mapping)
            enriched = engineer_features(df_mapped, upper_limit=metadata['upper_limit'])
            X_batch = preprocess_for_inference(enriched, scaler, metadata)
            
            iso_preds = iso_model.predict(X_batch)
            lof_preds = lof_model.predict(X_batch)
            iso_scores = iso_model.decision_function(X_batch)
            lof_scores = lof_model.score_samples(X_batch)
            
            risk_scores = [
                compute_unified_risk_score(s, l) for s, l in zip(iso_scores, lof_scores)
            ]
            
            enriched['Isolation_Forest'] = np.where(iso_preds == -1, 'Anomaly', 'Normal')
            enriched['LOF'] = np.where(lof_preds == -1, 'Anomaly', 'Normal')
            enriched['Risk_Score_Pct'] = risk_scores
            
            conditions = [
                (enriched['Isolation_Forest'] == 'Anomaly') & (enriched['LOF'] == 'Anomaly'),
                (enriched['Isolation_Forest'] == 'Anomaly') | (enriched['LOF'] == 'Anomaly')
            ]
            enriched['Audit_Verdict'] = np.select(conditions, ['CRITICAL (Both)', 'WARNING (Single Model)'], default='NORMAL')
            
            return {
                "records_processed": len(enriched),
                "mode": "pretrained_pipeline_inference",
                "columns_mapped": mapping,
                "anomalies_detected": int((enriched['Audit_Verdict'] != 'NORMAL').sum()),
                "critical_count": int((enriched['Audit_Verdict'] == 'CRITICAL (Both)').sum()),
                "warning_count": int((enriched['Audit_Verdict'] == 'WARNING (Single Model)').sum()),
                "sample_results": enriched[['Audit_Verdict', 'Risk_Score_Pct', 'Isolation_Forest', 'LOF']].head(10).to_dict(orient="records")
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
