#!/usr/bin/env python
"""
Test prediction with the trained model.
"""

import joblib
import pandas as pd
import numpy as np

# Load model
model_data = joblib.load("models/credit_default_model.joblib")
model = model_data['model']
num_imputer = model_data['num_imputer']
num_scaler = model_data['num_scaler']
cat_encoder = model_data['cat_encoder']
numeric_features = model_data['numeric_features']
categorical_features = model_data['categorical_features']

# Sample data (one customer)
sample_data = {
    "limit_bal": 20000,
    "sex": 2,
    "education": 2,
    "marriage": 1,
    "age": 35,
    "pay_0": -1,
    "bill_amt1": 5000,
    "pay_amt1": 2000,
    "bill_amt2": 4500,
    "bill_amt3": 4000,
    "bill_amt4": 3500,
    "bill_amt5": 3000,
    "bill_amt6": 2500,
    "pay_amt2": 1800,
    "pay_amt3": 1600,
    "pay_amt4": 1400,
    "pay_amt5": 1200,
    "pay_amt6": 1000,
    "pay_2": -1,
    "pay_3": -1,
    "pay_4": -1,
    "pay_5": -1,
    "pay_6": -1
}

# Convert to DataFrame
df = pd.DataFrame([sample_data])

# Add derived features if needed
if 'avg_bill_amt' not in df.columns and all(f'bill_amt{i}' in df.columns for i in range(1, 7)):
    bill_cols = [f'bill_amt{i}' for i in range(1, 7)]
    df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
    df['total_bill_amt'] = df[bill_cols].sum(axis=1)

if 'avg_pay_amt' not in df.columns and all(f'pay_amt{i}' in df.columns for i in range(1, 7)):
    pay_cols = [f'pay_amt{i}' for i in range(1, 7)]
    df['avg_pay_amt'] = df[pay_cols].mean(axis=1)
    df['total_pay_amt'] = df[pay_cols].sum(axis=1)

if 'utilization_ratio' not in df.columns and 'limit_bal' in df.columns and 'total_bill_amt' in df.columns:
    df['utilization_ratio'] = df['total_bill_amt'] / df['limit_bal']
    df['utilization_ratio'] = df['utilization_ratio'].replace([np.inf, -np.inf], 0)

# Prepare features
X_num = df[numeric_features].copy() if all(f in df.columns for f in numeric_features) else pd.DataFrame()
X_cat = df[categorical_features].copy() if all(f in df.columns for f in categorical_features) else pd.DataFrame()

# Process numeric features
if len(X_num) > 0:
    X_num_imputed = num_imputer.transform(X_num)
    X_num_scaled = num_scaler.transform(X_num_imputed)
else:
    X_num_scaled = np.array([]).reshape(0, 0)

# Process categorical features
if len(X_cat) > 0:
    # Convert to string for categorical features that are numeric
    for col in categorical_features:
        if col in X_cat.columns and X_cat[col].dtype in ['int64', 'float64']:
            X_cat[col] = X_cat[col].astype(str)
    
    X_cat_encoded = cat_encoder.transform(X_cat)
else:
    X_cat_encoded = np.array([]).reshape(0, 0)

# Combine features
if X_num_scaled.shape[1] > 0 and X_cat_encoded.shape[1] > 0:
    X_processed = np.hstack([X_num_scaled, X_cat_encoded])
elif X_num_scaled.shape[1] > 0:
    X_processed = X_num_scaled
elif X_cat_encoded.shape[1] > 0:
    X_processed = X_cat_encoded
else:
    raise ValueError("No features available for prediction")

# Make prediction
prediction = model.predict(X_processed)[0]
probability = model.predict_proba(X_processed)[0][1]

print(f"Prediction: {'Default' if prediction == 1 else 'No Default'}")
print(f"Probability of default: {probability:.4f}")
print(f"Risk level: {'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.3 else 'LOW'}")
