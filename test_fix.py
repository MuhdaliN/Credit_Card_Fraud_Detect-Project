# test_fix.py
import sys
import os
sys.path.append('.')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')

import django
django.setup()

import pandas as pd
from myapp.ml_utils import preprocess_bank_statement, extract_ml_features

# Test with sample data
test_data = {
    'Date': ['2024-01-01 10:00', '2024-01-01 14:00', '2024-01-02 02:00'],
    'Amount': [1000, 50000, 100],
    'Details': ['ATM', 'Online Purchase', 'Grocery'],
    'Balance': [10000, 50000, 49900]
}

df = pd.DataFrame(test_data)
print("Testing preprocessing...")
df_processed = preprocess_bank_statement(df)
print(f"Processed columns: {list(df_processed.columns)}")

print("\nTesting feature extraction...")
features = extract_ml_features(df_processed)
print(f"Features extracted: {len(features)}")
print(f"Sample features: {dict(list(features.items())[:5])}")