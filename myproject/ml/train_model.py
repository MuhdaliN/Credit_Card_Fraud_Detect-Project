# myproject/ml/train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

def generate_synthetic_data(num_samples=1000):
    """Generate synthetic bank transaction data for training"""
    
    np.random.seed(42)
    
    data = []
    
    for i in range(num_samples):
        # Generate features similar to extract_ml_features output
        sample = {
            'avg_amount': np.random.uniform(100, 10000),
            'std_amount': np.random.uniform(10, 5000),
            'total_transactions': np.random.randint(10, 500),
            'debit_count': np.random.randint(5, 250),
            'credit_count': np.random.randint(5, 250),
            'avg_risk_score': np.random.uniform(0, 100),
            'high_risk_count': np.random.randint(0, 50),
            'night_transactions': np.random.randint(0, 30),
            'weekend_transactions': np.random.randint(0, 100),
            'large_transactions': np.random.randint(0, 50),
            'total_debit_amount': np.random.uniform(1000, 100000),
            'total_credit_amount': np.random.uniform(1000, 100000),
            'pct_high_risk': np.random.uniform(0, 50),
            'pct_night_txns': np.random.uniform(0, 30),
            'pct_weekend_txns': np.random.uniform(0, 50),
            'pct_large_txns': np.random.uniform(0, 30),
            'avg_time_between_txns': np.random.uniform(1, 240),
            'rapid_txns_count': np.random.randint(0, 20),
            'pct_rapid_txns': np.random.uniform(0, 40),
            'unusual_hours_count': np.random.randint(0, 100),
            'pct_unusual_hours': np.random.uniform(0, 60),
            'is_fraud': 0  # Default to not fraud
        }
        
        # Generate fraud labels based on heuristic rules
        fraud_score = 0
        if sample['pct_high_risk'] > 20:
            fraud_score += 30
        if sample['pct_rapid_txns'] > 25:
            fraud_score += 25
        if sample['pct_unusual_hours'] > 40:
            fraud_score += 20
        if sample['night_transactions'] > 10:
            fraud_score += 15
        if sample['avg_amount'] > 5000:
            fraud_score += 10
        
        # Label as fraud if score > 50
        sample['is_fraud'] = 1 if fraud_score > 50 else 0
        
        # Add some noise
        if np.random.random() < 0.05:  # 5% random fraud
            sample['is_fraud'] = 1
        elif np.random.random() < 0.05:  # 5% random non-fraud
            sample['is_fraud'] = 0
        
        data.append(sample)
    
    return pd.DataFrame(data)

def train_model():
    """Train and save the ML model"""
    
    print("Generating synthetic training data...")
    df = generate_synthetic_data(num_samples=2000)
    
    # Features and target
    feature_columns = [
        'avg_amount', 'std_amount', 'total_transactions',
        'debit_count', 'credit_count', 'avg_risk_score',
        'high_risk_count', 'night_transactions', 'weekend_transactions',
        'large_transactions', 'total_debit_amount', 'total_credit_amount',
        'pct_high_risk', 'pct_night_txns', 'pct_weekend_txns',
        'pct_large_txns', 'avg_time_between_txns', 'rapid_txns_count',
        'pct_rapid_txns', 'unusual_hours_count', 'pct_unusual_hours'
    ]
    
    X = df[feature_columns]
    y = df['is_fraud']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Fraud cases: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model and scaler
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(model_dir, "model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    feature_names_path = os.path.join(model_dir, "feature_names.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_columns, feature_names_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Feature names saved to: {feature_names_path}")
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(10, len(feature_columns))):
        print(f"{i+1}. {feature_columns[indices[i]]}: {importances[indices[i]]:.4f}")
    
    return model, scaler

if __name__ == "__main__":
    train_model()