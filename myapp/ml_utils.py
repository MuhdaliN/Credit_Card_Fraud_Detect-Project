# myapp/ml_utils.py
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
from django.conf import settings

# Global model variables
model = None
scaler = None
ML_MODELS_LOADED = False

def load_models():
    """Load ML models"""
    global model, scaler, ML_MODELS_LOADED
    
    try:
        model_dir = os.path.join(settings.BASE_DIR, "myapp", "ml")
        model_path = os.path.join(model_dir, "model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            ML_MODELS_LOADED = True
            print("ML models loaded successfully")
            return True
    except Exception as e:
        print(f"Error loading ML models: {e}")
    
    return False

# Preprocess function (same as before)
def preprocess_bank_statement(df):
    """
    Preprocess bank statement CSV data
    Returns dataframe with additional features
    """
    # Make a copy
    df_processed = df.copy()
    
    # Standardize column names
    df_processed.columns = [col.strip().replace(' ', '_').replace('.', '') for col in df_processed.columns]
    
    # Handle date column
    date_column = None
    for col in df_processed.columns:
        if 'date' in col.lower() or 'Date' in col:
            date_column = col
            break
    
    if date_column:
        try:
            df_processed['Date'] = pd.to_datetime(df_processed[date_column], errors='coerce')
        except:
            df_processed['Date'] = pd.to_datetime('today')
    else:
        df_processed['Date'] = pd.to_datetime('today')
    
    # Extract time features
    df_processed['Hour'] = df_processed['Date'].dt.hour
    df_processed['DayOfWeek'] = df_processed['Date'].dt.dayofweek
    df_processed['DayOfMonth'] = df_processed['Date'].dt.day
    df_processed['Month'] = df_processed['Date'].dt.month
    
    # Handle amount column
    amount_column = None
    for col in df_processed.columns:
        if 'amount' in col.lower() or 'Amount' in col or 'amt' in col.lower():
            amount_column = col
            break
    
    if amount_column:
        # Clean amount column - remove commas, currency symbols
        df_processed['Amount'] = df_processed[amount_column].astype(str).str.replace(',', '')
        df_processed['Amount'] = df_processed['Amount'].str.replace('₹', '').str.replace('$', '').str.replace('£', '')
        df_processed['Amount'] = pd.to_numeric(df_processed['Amount'], errors='coerce').fillna(0)
    else:
        df_processed['Amount'] = 0
    
    # Handle transaction details/description
    details_column = None
    for col in df_processed.columns:
        if 'detail' in col.lower() or 'desc' in col.lower() or 'particular' in col.lower():
            details_column = col
            break
    
    if details_column:
        df_processed['Details'] = df_processed[details_column].astype(str).fillna('')
    else:
        df_processed['Details'] = ''
    
    # Balance column
    balance_column = None
    for col in df_processed.columns:
        if 'balance' in col.lower() or 'Balance' in col:
            balance_column = col
            break
    
    if balance_column:
        df_processed['Balance'] = pd.to_numeric(df_processed[balance_column].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    else:
        df_processed['Balance'] = 0
    
    # Transaction type (credit/debit)
    df_processed['Transaction_Type'] = 'DEBIT'
    if 'Transaction_Type' in df_processed.columns:
        pass
    elif 'Type' in df_processed.columns:
        df_processed['Transaction_Type'] = df_processed['Type']
    else:
        # Determine from amount sign or details
        if amount_column:
            df_processed['Transaction_Type'] = df_processed['Amount'].apply(
                lambda x: 'CREDIT' if x > 0 else 'DEBIT'
            )
    
    # Apply heuristic risk scoring - THIS LINE IS CRITICAL!
    df_processed = apply_heuristic_risk_scoring(df_processed)
    
    return df_processed

# Heuristic risk scoring function (same as before)
def apply_heuristic_risk_scoring(df):
    """Apply heuristic rules to calculate risk scores"""
    
    # Initialize risk score
    df['Heuristic_Risk_Score'] = 0
    
    # Rule 1: Large transactions
    if 'Amount' in df.columns and df['Amount'].std() > 0:
        amount_zscore = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
        df['Heuristic_Risk_Score'] += np.where(amount_zscore > 3, 25, 0)
        df['Heuristic_Risk_Score'] += np.where(amount_zscore > 2, 15, 0)
    
    # Rule 2: Unusual hours (midnight to 5 AM)
    if 'Hour' in df.columns:
        df['Heuristic_Risk_Score'] += df['Hour'].apply(
            lambda x: 20 if 0 <= x <= 5 else 0
        )
    
    # Rule 3: Weekend transactions
    if 'DayOfWeek' in df.columns:
        df['Heuristic_Risk_Score'] += df['DayOfWeek'].apply(
            lambda x: 10 if x >= 5 else 0  # 5=Saturday, 6=Sunday
        )
    
    # Rule 4: Rapid transactions (if we have timestamps)
    if 'Date' in df.columns:
        df_sorted = df.sort_values('Date')
        if len(df_sorted) > 1:
            time_diffs = df_sorted['Date'].diff().dt.total_seconds() / 60  # minutes
            df_sorted['Heuristic_Risk_Score'] = df_sorted.get('Heuristic_Risk_Score', 0) + np.where(time_diffs < 5, 15, 0)
            df = df_sorted
    
    # Rule 5: Common fraud keywords in description
    if 'Details' in df.columns:
        fraud_keywords = [
            'transfer', 'online', 'payment', 'pos', 'atm', 'withdrawal',
            'cash', 'wallet', 'upi', 'netbanking', 'international'
        ]
        
        def check_keywords(text):
            score = 0
            text_lower = str(text).lower()
            for keyword in fraud_keywords:
                if keyword in text_lower:
                    score += 5
            return min(score, 20)
        
        df['Heuristic_Risk_Score'] += df['Details'].apply(check_keywords)
    
    # Rule 6: Balance dropping significantly
    if 'Balance' in df.columns and len(df) > 10:
        balance_changes = df['Balance'].pct_change().abs() * 100
        df['Heuristic_Risk_Score'] += np.where(balance_changes > 50, 15, 0)
    
    # Cap score at 100
    df['Heuristic_Risk_Score'] = df['Heuristic_Risk_Score'].clip(0, 100)
    
    # Add fraud flag - MAKE SURE THIS LINE EXISTS!
    df['Heuristic_Fraud_Flag'] = df['Heuristic_Risk_Score'].apply(
        lambda x: 'HIGH RISK' if x >= 70 else 'MEDIUM RISK' if x >= 40 else 'LOW RISK'
    )
    
    return df
# Extract features function (same as before)
def extract_ml_features(df):
    """Extract features for ML prediction from processed dataframe"""
    try:
        if df is None or len(df) == 0:
            print("Empty dataframe passed to extract_ml_features")
            return {}
        
        features = {}
        
        # Basic transaction features
        if 'Amount' in df.columns:
            features['avg_amount'] = float(df['Amount'].mean())
            features['std_amount'] = float(df['Amount'].std())
            features['total_transactions'] = len(df)
            features['large_transactions'] = int((df['Amount'] > 5000).sum())
            features['total_debit_amount'] = float(df[df['Transaction_Type'] == 'DEBIT']['Amount'].abs().sum() if 'Transaction_Type' in df.columns else df['Amount'].abs().sum())
            features['total_credit_amount'] = float(df[df['Transaction_Type'] == 'CREDIT']['Amount'].abs().sum() if 'Transaction_Type' in df.columns else 0)
            features['pct_large_txns'] = float((features['large_transactions'] / features['total_transactions'] * 100) if features['total_transactions'] > 0 else 0)
        
        # Risk-related features
        if 'Heuristic_Risk_Score' in df.columns:
            features['avg_risk_score'] = float(df['Heuristic_Risk_Score'].mean())
            features['high_risk_count'] = int((df['Heuristic_Risk_Score'] >= 70).sum())
            features['pct_high_risk'] = float((features['high_risk_count'] / features['total_transactions'] * 100) if features['total_transactions'] > 0 else 0)
        
        # Time-based features
        if 'Hour' in df.columns:
            features['night_transactions'] = int(((df['Hour'] >= 0) & (df['Hour'] <= 5)).sum())
            features['unusual_hours_count'] = int(((df['Hour'] < 9) | (df['Hour'] > 18)).sum())
            features['pct_night_txns'] = float((features['night_transactions'] / features['total_transactions'] * 100) if features['total_transactions'] > 0 else 0)
            features['pct_unusual_hours'] = float((features['unusual_hours_count'] / features['total_transactions'] * 100) if features['total_transactions'] > 0 else 0)
        
        if 'DayOfWeek' in df.columns:
            features['weekend_transactions'] = int((df['DayOfWeek'] >= 5).sum())
            features['pct_weekend_txns'] = float((features['weekend_transactions'] / features['total_transactions'] * 100) if features['total_transactions'] > 0 else 0)
        
        # Transaction type counts
        if 'Transaction_Type' in df.columns:
            features['debit_count'] = int((df['Transaction_Type'] == 'DEBIT').sum())
            features['credit_count'] = int((df['Transaction_Type'] == 'CREDIT').sum())
        else:
            features['debit_count'] = features['total_transactions']
            features['credit_count'] = 0
        
        # Time between transactions
        if 'Date' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('Date')
            time_diffs = df_sorted['Date'].diff().dt.total_seconds() / 60  # minutes
            features['avg_time_between_txns'] = float(time_diffs.mean())
            features['rapid_txns_count'] = int((time_diffs < 5).sum())
            features['pct_rapid_txns'] = float((features['rapid_txns_count'] / features['total_transactions'] * 100) if features['total_transactions'] > 0 else 0)
        else:
            features['avg_time_between_txns'] = 60.0
            features['rapid_txns_count'] = 0
            features['pct_rapid_txns'] = 0.0
        
        # Fill any missing features with default values
        expected_features = [
            'avg_amount', 'std_amount', 'total_transactions', 'debit_count', 
            'credit_count', 'avg_risk_score', 'high_risk_count', 'night_transactions', 
            'weekend_transactions', 'large_transactions', 'total_debit_amount', 
            'total_credit_amount', 'pct_high_risk', 'pct_night_txns', 'pct_weekend_txns',
            'pct_large_txns', 'avg_time_between_txns', 'rapid_txns_count',
            'pct_rapid_txns', 'unusual_hours_count', 'pct_unusual_hours'
        ]
        
        for feat in expected_features:
            if feat not in features:
                features[feat] = 0.0
        
        print(f"Extracted {len(features)} features for ML prediction")
        return features
        
    except Exception as e:
        print(f"Error in extract_ml_features: {e}")
        import traceback
        traceback.print_exc()
        return {}

# Updated predict_fraud_ml function
def predict_fraud_ml(features):
    """Predict fraud probability using ML model"""
    
    global model, scaler, ML_MODELS_LOADED
    
    # Try to load models if not loaded
    if not ML_MODELS_LOADED:
        load_models()
    
    # If ML models still not loaded, use heuristic fallback
    if not ML_MODELS_LOADED or model is None or scaler is None:
        print("ML models not available, using heuristic prediction")
        return heuristic_fraud_prediction(features)
    
    try:
        # Define expected feature names (must match training)
        expected_features = [
            'avg_amount', 'std_amount', 'total_transactions',
            'debit_count', 'credit_count', 'avg_risk_score',
            'high_risk_count', 'night_transactions', 'weekend_transactions',
            'large_transactions', 'total_debit_amount', 'total_credit_amount',
            'pct_high_risk', 'pct_night_txns', 'pct_weekend_txns',
            'pct_large_txns', 'avg_time_between_txns', 'rapid_txns_count',
            'pct_rapid_txns', 'unusual_hours_count', 'pct_unusual_hours'
        ]
        
        # Prepare feature vector
        feature_vector = []
        for feat in expected_features:
            if feat in features:
                feature_vector.append(float(features[feat]))
            else:
                feature_vector.append(0.0)
        
        # Scale features
        feature_vector_scaled = scaler.transform([feature_vector])
        
        # Predict probability
        fraud_probability = model.predict_proba(feature_vector_scaled)[0][1] * 100
        
        print(f"ML Prediction: {fraud_probability:.2f}% fraud probability")
        return float(fraud_probability)
        
    except Exception as e:
        print(f"ML prediction error: {e}")
        # Fallback to heuristic
        return heuristic_fraud_prediction(features)

def heuristic_fraud_prediction(features):
    """Heuristic fallback when ML model is not available"""
    
    # Start with base probability
    probability = 10.0  # Base 10% chance
    
    # Adjust based on features
    if 'pct_high_risk' in features and features['pct_high_risk'] > 0:
        probability += min(features['pct_high_risk'] * 0.5, 30)
    
    if 'pct_rapid_txns' in features and features['pct_rapid_txns'] > 20:
        probability += 20
    
    if 'pct_unusual_hours' in features and features['pct_unusual_hours'] > 30:
        probability += 15
    
    if 'avg_amount' in features and features['avg_amount'] > 5000:
        probability += 10
    
    if 'night_transactions' in features and features['night_transactions'] > 3:
        probability += 15
    
    # Cap at 95%
    return min(probability, 95.0)