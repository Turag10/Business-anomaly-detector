import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bank_transactions_data_2.csv")

# City coordinates for threat map visualization
CITY_COORDINATES = {
    'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Chicago': (41.8781, -87.6298),
    'Houston': (29.7604, -95.3698),
    'Phoenix': (33.4484, -112.0740),
    'Philadelphia': (39.9526, -75.1652),
    'San Antonio': (29.4241, -98.4936),
    'San Diego': (32.7157, -117.1611),
    'Dallas': (32.7767, -96.7970),
    'San Jose': (37.3382, -121.8863),
    'Austin': (30.2672, -97.7431),
    'Jacksonville': (30.3322, -81.6557),
    'Fort Worth': (32.7555, -97.3308),
    'Columbus': (39.9612, -82.9988),
    'Charlotte': (35.2271, -80.8431),
    'San Francisco': (37.7749, -122.4194),
    'Indianapolis': (39.7684, -86.1581),
    'Seattle': (47.6062, -122.3321),
    'Denver': (39.7392, -104.9903),
    'Washington': (38.9072, -77.0369),
    'Boston': (42.3601, -71.0589),
    'El Paso': (31.7619, -106.4850),
    'Nashville': (36.1627, -86.7816),
    'Detroit': (42.3314, -83.0458),
    'Oklahoma City': (35.4676, -97.5164),
    'Portland': (45.5152, -122.6784),
    'Las Vegas': (36.1699, -115.1398),
    'Memphis': (35.1495, -90.0490),
    'Louisville': (38.2527, -85.7585),
    'Baltimore': (39.2904, -76.6122),
    'Milwaukee': (43.0389, -87.9065),
    'Albuquerque': (35.0844, -106.6504),
    'Tucson': (32.2226, -110.9747),
    'Fresno': (36.7468, -119.7726),
    'Mesa': (33.4152, -111.8315),
    'Sacramento': (38.5816, -121.4944),
    'Atlanta': (33.7490, -84.3880),
    'Kansas City': (39.0997, -94.5786),
    'Colorado Springs': (38.8339, -104.8214),
    'Omaha': (41.2565, -95.9345),
    'Raleigh': (35.7796, -78.6382),
    'Miami': (25.7617, -80.1918),
    'Virginia Beach': (36.8529, -75.9780)
}

COLUMN_RENAME = {
    'TransactionID': 'transaction_id',
    'AccountID': 'account_id',
    'TransactionAmount': 'transaction_amount',
    'TransactionDate': 'transaction_date',
    'TransactionType': 'transaction_type',
    'Location': 'location',
    'DeviceID': 'device_id',
    'IP Address': 'ip_address',
    'MerchantID': 'merchant_id',
    'Channel': 'channel',
    'CustomerAge': 'customer_age',
    'CustomerOccupation': 'customer_occupation',
    'TransactionDuration': 'transaction_seconds_duration',
    'LoginAttempts': 'login_attempts',
    'AccountBalance': 'account_balance',
    'PreviousTransactionDate': 'previous_transaction_date',
}

NUM_COLS = [
    'transaction_amount',
    'customer_age',
    'transaction_seconds_duration',
    'login_attempts',
    'account_balance',
    'transaction_hour',
    'time_since_previous',
    'amount_balance_ratio'
]

ID_COLS = [
    'transaction_id',
    'account_id',
    'device_id',
    'ip_address',
    'merchant_id',
    'transaction_date',
    'previous_transaction_date'
]

def compute_unified_risk_score(iso_score, lof_score=None):
    """
    Normalizes model anomaly decision values to a unified Risk Score (0 to 100%).
    iso_score: decision_function (typically -0.3 to +0.2; negative = anomaly)
    lof_score: score_samples (typically -2.0 to -1.0; more negative = anomaly)
    """
    # Scale iso_score so that < 0 becomes high risk (> 50%)
    iso_risk = np.clip((0.10 - iso_score) / 0.25 * 100, 0, 100)
    
    if lof_score is not None:
        lof_risk = np.clip((-lof_score - 1.0) / 1.2 * 100, 0, 100)
        composite = 0.55 * iso_risk + 0.45 * lof_risk
    else:
        composite = iso_risk
        
    return float(np.round(np.clip(composite, 0, 100), 1))

def engineer_features(df, upper_limit=None):
    """
    Applies behavioral feature engineering matching the baseline research.
    """
    df = df.copy()
    
    # Rename if original Kaggle casing is present
    for orig, ren in COLUMN_RENAME.items():
        if orig in df.columns and ren not in df.columns:
            df.rename(columns={orig: ren}, inplace=True)
            
    # Datetime conversions
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    else:
        df['transaction_date'] = pd.Timestamp.now()

    if 'previous_transaction_date' in df.columns:
        df['previous_transaction_date'] = pd.to_datetime(df['previous_transaction_date'], errors='coerce')
    else:
        df['previous_transaction_date'] = df['transaction_date'] - pd.Timedelta(hours=2)
    
    # Sort for account-level behavioral tracking
    if 'account_id' in df.columns and 'transaction_date' in df.columns:
        df = df.sort_values(['account_id', 'transaction_date']).reset_index(drop=True)
        
        # Device changes
        if 'device_id' in df.columns:
            df['device_changed'] = df.groupby('account_id')['device_id'].transform(lambda x: x.ne(x.shift()))
            first_idx = df.groupby('account_id').head(1).index
            df.loc[first_idx, 'device_changed'] = False
        else:
            df['device_changed'] = False

        # IP changes
        if 'ip_address' in df.columns:
            df['ip_changed'] = df.groupby('account_id')['ip_address'].transform(lambda x: x != x.shift())
            first_idx = df.groupby('account_id').head(1).index
            df.loc[first_idx, 'ip_changed'] = False
        else:
            df['ip_changed'] = False
    else:
        if 'device_changed' not in df.columns:
            df['device_changed'] = False
        if 'ip_changed' not in df.columns:
            df['ip_changed'] = False
            
    # Temporal features
    df['transaction_hour'] = df['transaction_date'].dt.hour.fillna(12)
    df['day_of_week'] = df['transaction_date'].dt.day_name().fillna('Monday')
    df['is_weekend'] = (df['transaction_date'].dt.day_of_week >= 5).fillna(False)
    
    # Time since previous (minutes)
    df['time_since_previous'] = (
        (df['transaction_date'] - df['previous_transaction_date']).dt.total_seconds() / 60
    ).fillna(60.0)
    
    # Financial ratio
    if 'account_balance' not in df.columns:
        df['account_balance'] = 5000.0
    if 'transaction_amount' not in df.columns:
        df['transaction_amount'] = 100.0

    df['amount_balance_ratio'] = df['transaction_amount'] / (df['account_balance'].replace(0, 0.01))
    
    # High-value transaction flag based on IQR upper limit
    if upper_limit is not None:
        df['high_value_transaction'] = df['transaction_amount'] > upper_limit
    else:
        q1 = df['transaction_amount'].quantile(0.25)
        q3 = df['transaction_amount'].quantile(0.75)
        iqr = q3 - q1
        computed_limit = q3 + 1.5 * iqr
        df['high_value_transaction'] = df['transaction_amount'] > computed_limit

    # Add coordinates if location exists
    if 'location' in df.columns:
        coords = df['location'].map(CITY_COORDINATES)
        df['latitude'] = coords.apply(lambda c: c[0] if isinstance(c, tuple) else np.nan)
        df['longitude'] = coords.apply(lambda c: c[1] if isinstance(c, tuple) else np.nan)

    return df

def train_and_save_pipeline(csv_path=DATA_PATH, contamination=0.05):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    df_raw = pd.read_csv(csv_path)
    q1 = df_raw['TransactionAmount'].quantile(0.25)
    q3 = df_raw['TransactionAmount'].quantile(0.75)
    iqr = q3 - q1
    upper_limit = float(q3 + 1.5 * iqr)
    
    df_c = engineer_features(df_raw, upper_limit=upper_limit)
    
    existing_id_cols = [c for c in ID_COLS + ['latitude', 'longitude'] if c in df_c.columns]
    df_model = df_c.drop(columns=existing_id_cols, errors='ignore')
    
    cat_cols = df_model.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
    
    scaler = RobustScaler()
    df_model[NUM_COLS] = scaler.fit_transform(df_model[NUM_COLS])
    
    feature_columns = df_model.columns.tolist()
    X = df_model.copy()
    
    # Train Isolation Forest
    iso = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    iso.fit(X)
    
    # Train LOF with novelty=True for out-of-sample inference
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        novelty=True
    )
    lof.fit(X)
    
    # Pre-initialize SHAP Explainer
    explainer = shap.TreeExplainer(iso)
    joblib.dump(explainer, os.path.join(ARTIFACTS_DIR, "shap_explainer.joblib"))
    
    joblib.dump(iso, os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib"))
    joblib.dump(lof, os.path.join(ARTIFACTS_DIR, "lof.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    
    metadata = {
        "feature_columns": feature_columns,
        "num_cols": NUM_COLS,
        "upper_limit": upper_limit,
        "contamination": contamination,
        "categorical_columns": cat_cols,
        "unique_locations": sorted([loc for loc in df_raw['Location'].dropna().unique().tolist()]),
        "unique_occupations": sorted([occ for occ in df_raw['CustomerOccupation'].dropna().unique().tolist()]),
        "unique_channels": sorted([ch for ch in df_raw['Channel'].dropna().unique().tolist()]),
        "unique_types": sorted([tp for tp in df_raw['TransactionType'].dropna().unique().tolist()])
    }
    
    with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Pipeline successfully trained and saved to {ARTIFACTS_DIR}!")
    return df_c, iso, lof, scaler, metadata

def load_pipeline():
    iso = joblib.load(os.path.join(ARTIFACTS_DIR, "isolation_forest.joblib"))
    lof = joblib.load(os.path.join(ARTIFACTS_DIR, "lof.joblib"))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    explainer_path = os.path.join(ARTIFACTS_DIR, "shap_explainer.joblib")
    explainer = joblib.load(explainer_path) if os.path.exists(explainer_path) else shap.TreeExplainer(iso)
    
    with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return iso, lof, scaler, metadata, explainer

def preprocess_for_inference(df_features, scaler, metadata):
    """
    Prepares a feature engineered dataframe to exactly match the trained feature space.
    """
    df = df_features.copy()
    drop_candidates = ID_COLS + ['latitude', 'longitude', 'iso_anomaly', 'lof_anomaly', 'iso_score', 'risk_category']
    existing_id_cols = [c for c in drop_candidates if c in df.columns]
    df_model = df.drop(columns=existing_id_cols, errors='ignore')
    
    cat_cols = df_model.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
    
    # Ensure all numerical columns exist
    num_cols = metadata["num_cols"]
    for c in num_cols:
        if c not in df_model.columns:
            df_model[c] = 0.0
            
    df_model[num_cols] = scaler.transform(df_model[num_cols])
    
    # Align columns to feature_columns
    feature_cols = metadata["feature_columns"]
    for c in feature_cols:
        if c not in df_model.columns:
            df_model[c] = 0
            
    df_aligned = df_model[feature_cols].copy().fillna(0)
    return df_aligned

def explain_transaction_shap(explainer, X_single, feature_columns, top_n=6):
    """
    Calculates SHAP contributions for a single transaction.
    Returns DataFrame of top positive (pushing towards anomaly) and negative features.
    """
    shap_vals = explainer.shap_values(X_single)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
        
    vals = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
    
    df_shap = pd.DataFrame({
        'Feature': feature_columns,
        'Impact': vals,
        'AbsImpact': np.abs(vals)
    }).sort_values('AbsImpact', ascending=False)
    
    return df_shap.head(top_n)

def auto_detect_and_map_columns(df):
    """
    Smart fuzzy detection for ANY custom transaction dataset uploaded by the user.
    """
    cols = [str(c).lower().strip() for c in df.columns]
    orig_cols = list(df.columns)
    mapping = {}
    
    amount_keywords = ['amount', 'trans_amount', 'transactionamount', 'amt', 'value', 'price', 'total', 'volume']
    balance_keywords = ['balance', 'accountbalance', 'acc_balance', 'cur_bal', 'post_balance']
    date_keywords = ['date', 'transactiondate', 'timestamp', 'time', 'tx_date', 'datetime']
    account_keywords = ['account', 'accountid', 'user_id', 'customer_id', 'acc_no', 'client_id']
    channel_keywords = ['channel', 'channel_type', 'medium', 'method', 'payment_method']
    login_keywords = ['login', 'loginattempts', 'attempts', 'failed_logins']
    age_keywords = ['age', 'customerage', 'client_age']
    occ_keywords = ['occupation', 'customeroccupation', 'profession', 'job']
    loc_keywords = ['location', 'city', 'state', 'country', 'geo']

    for orig, c in zip(orig_cols, cols):
        if any(k == c or k in c for k in amount_keywords) and 'transaction_amount' not in mapping.values():
            mapping[orig] = 'transaction_amount'
        elif any(k == c or k in c for k in balance_keywords) and 'account_balance' not in mapping.values():
            mapping[orig] = 'account_balance'
        elif any(k == c or k in c for k in date_keywords) and 'transaction_date' not in mapping.values():
            mapping[orig] = 'transaction_date'
        elif any(k == c or k in c for k in account_keywords) and 'account_id' not in mapping.values():
            mapping[orig] = 'account_id'
        elif any(k == c or k in c for k in channel_keywords) and 'channel' not in mapping.values():
            mapping[orig] = 'channel'
        elif any(k == c or k in c for k in login_keywords) and 'login_attempts' not in mapping.values():
            mapping[orig] = 'login_attempts'
        elif any(k == c or k in c for k in age_keywords) and 'customer_age' not in mapping.values():
            mapping[orig] = 'customer_age'
        elif any(k == c or k in c for k in occ_keywords) and 'customer_occupation' not in mapping.values():
            mapping[orig] = 'customer_occupation'
        elif any(k == c or k in c for k in loc_keywords) and 'location' not in mapping.values():
            mapping[orig] = 'location'
            
    return mapping

def train_fresh_model_on_any_dataset(df_custom, contamination=0.05):
    """
    Fits fresh Isolation Forest and LOF models directly on ANY custom dataset.
    """
    df_clean = df_custom.copy()
    # Find all numeric columns
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    # Encode categoricals if present
    if cat_cols:
        df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
    else:
        df_encoded = df_clean.copy()
        
    num_features = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_encoded[num_features].fillna(0))
    
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    iso_preds = iso.fit_predict(X_scaled)
    iso_scores = iso.decision_function(X_scaled)
    
    lof = LocalOutlierFactor(n_neighbors=min(20, max(5, len(df_clean)-1)), contamination=contamination)
    lof_preds = lof.fit_predict(X_scaled)
    lof_scores = lof.negative_outlier_factor_
    
    df_result = df_custom.copy()
    df_result['Isolation_Forest'] = np.where(iso_preds == -1, 'Anomaly', 'Normal')
    df_result['LOF'] = np.where(lof_preds == -1, 'Anomaly', 'Normal')
    df_result['Anomaly_Score'] = iso_scores
    
    # Calculate Risk Score (0 - 100%)
    risk_scores = [
        compute_unified_risk_score(s, l) for s, l in zip(iso_scores, lof_scores)
    ]
    df_result['Risk_Score_Pct'] = risk_scores
    
    conditions = [
        (df_result['Isolation_Forest'] == 'Anomaly') & (df_result['LOF'] == 'Anomaly'),
        (df_result['Isolation_Forest'] == 'Anomaly') | (df_result['LOF'] == 'Anomaly')
    ]
    df_result['Audit_Verdict'] = np.select(conditions, ['CRITICAL (Both)', 'WARNING (Single Model)'], default='NORMAL')
    
    return df_result, num_features

if __name__ == "__main__":
    train_and_save_pipeline()
