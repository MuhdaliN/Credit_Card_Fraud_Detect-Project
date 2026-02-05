# myapp/ml/train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_data(num_samples=2000):
    """Generate synthetic bank transaction data for training"""
    
    np.random.seed(42)
    
    print("Generating synthetic training data...")
    
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
        
        # High risk transactions percentage
        if sample['pct_high_risk'] > 20:
            fraud_score += 30
            
        # Rapid transactions
        if sample['pct_rapid_txns'] > 25:
            fraud_score += 25
            
        # Unusual hours
        if sample['pct_unusual_hours'] > 40:
            fraud_score += 20
            
        # Night transactions
        if sample['night_transactions'] > 10:
            fraud_score += 15
            
        # Large average amount
        if sample['avg_amount'] > 5000:
            fraud_score += 10
            
        # High standard deviation (inconsistent spending)
        if sample['std_amount'] > 3000:
            fraud_score += 10
        
        # Label as fraud if score > 50
        sample['is_fraud'] = 1 if fraud_score > 50 else 0
        
        # Add some noise
        if np.random.random() < 0.03:  # 3% random fraud
            sample['is_fraud'] = 1
        elif np.random.random() < 0.02:  # 2% random non-fraud
            sample['is_fraud'] = 0
        
        data.append(sample)
    
    df = pd.DataFrame(data)
    
    # Balance the dataset a bit
    fraud_count = df['is_fraud'].sum()
    non_fraud_count = len(df) - fraud_count
    
    print(f"Generated {len(df)} samples")
    print(f"Fraud cases: {fraud_count} ({fraud_count/len(df)*100:.1f}%)")
    print(f"Non-fraud cases: {non_fraud_count} ({non_fraud_count/len(df)*100:.1f}%)")
    
    return df

def train_model():
    """Train and save the ML model"""
    
    print("\n" + "="*50)
    print("FRAUD DETECTION ML MODEL TRAINING")
    print("="*50 + "\n")
    
    # Generate training data
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
    print(f"Features used: {len(feature_columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives: {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives: {cm[1,1]}")
    
    # Save model and scaler
    print("\n" + "="*50)
    print("SAVING MODEL FILES")
    print("="*50)
    
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(model_dir, "model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Feature importance
    print("\n" + "="*50)
    print("TOP 10 FEATURE IMPORTANCES")
    print("="*50)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nRank | Feature | Importance")
    print("-" * 40)
    for i in range(min(10, len(feature_columns))):
        print(f"{i+1:4d} | {feature_columns[indices[i]]:25s} | {importances[indices[i]]:.4f}")
    
    # Test prediction on sample data
    print("\n" + "="*50)
    print("SAMPLE PREDICTION TEST")
    print("="*50)
    
    # Create a sample feature vector
    sample_features = {
        'avg_amount': 7500,
        'std_amount': 4000,
        'total_transactions': 150,
        'debit_count': 120,
        'credit_count': 30,
        'avg_risk_score': 65,
        'high_risk_count': 25,
        'night_transactions': 8,
        'weekend_transactions': 40,
        'large_transactions': 20,
        'total_debit_amount': 85000,
        'total_credit_amount': 15000,
        'pct_high_risk': 16.7,
        'pct_night_txns': 5.3,
        'pct_weekend_txns': 26.7,
        'pct_large_txns': 13.3,
        'avg_time_between_txns': 45,
        'rapid_txns_count': 12,
        'pct_rapid_txns': 8.0,
        'unusual_hours_count': 65,
        'pct_unusual_hours': 43.3
    }
    
    # Prepare feature vector
    feature_vector = []
    for feat in feature_columns:
        feature_vector.append(float(sample_features.get(feat, 0)))
    
    # Scale and predict
    feature_vector_scaled = scaler.transform([feature_vector])
    prediction = model.predict(feature_vector_scaled)[0]
    probability = model.predict_proba(feature_vector_scaled)[0][1]
    
    print(f"\nSample prediction:")
    print(f"  Features: High transaction volume, unusual hours, rapid transactions")
    print(f"  Prediction: {'FRAUD' if prediction == 1 else 'NOT FRAUD'}")
    print(f"  Fraud Probability: {probability*100:.2f}%")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. Restart your Django server")
    print("2. Check dashboard shows 'ML Models Loaded'")
    print("3. Upload a CSV file and click 'Analyze CSV'")
    
    return model, scaler, feature_columns

if __name__ == "__main__":
    train_model()