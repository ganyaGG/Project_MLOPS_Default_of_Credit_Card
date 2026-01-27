from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Optional
import json
import sys

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# Initialize FastAPI app
app = FastAPI(
    title="Credit Default Prediction API",
    description="API for predicting credit card default probability",
    version="1.0.0",
)


# Define input schema
class ClientData(BaseModel):
    limit_bal: float = Field(..., ge=0, description="Credit limit")
    sex: int = Field(..., ge=1, le=2, description="Gender (1=male, 2=female)")
    education: int = Field(..., ge=1, le=4, description="Education level")
    marriage: int = Field(..., ge=1, le=3, description="Marital status")
    age: int = Field(..., ge=18, le=100, description="Age")
    pay_0: int = Field(..., ge=-2, le=8, description="Repayment status in September")
    bill_amt1: float = Field(..., description="Bill amount in September")
    pay_amt1: float = Field(..., ge=0, description="Payment amount in September")

    # Optional features with defaults
    bill_amt2: Optional[float] = Field(0, description="Bill amount in August")
    bill_amt3: Optional[float] = Field(0, description="Bill amount in July")
    bill_amt4: Optional[float] = Field(0, description="Bill amount in June")
    bill_amt5: Optional[float] = Field(0, description="Bill amount in May")
    bill_amt6: Optional[float] = Field(0, description="Bill amount in April")
    pay_amt2: Optional[float] = Field(0, description="Payment amount in August")
    pay_amt3: Optional[float] = Field(0, description="Payment amount in July")
    pay_amt4: Optional[float] = Field(0, description="Payment amount in June")
    pay_amt5: Optional[float] = Field(0, description="Payment amount in May")
    pay_amt6: Optional[float] = Field(0, description="Payment amount in April")
    pay_2: Optional[int] = Field(-2, description="Repayment status in August")
    pay_3: Optional[int] = Field(-2, description="Repayment status in July")
    pay_4: Optional[int] = Field(-2, description="Repayment status in June")
    pay_5: Optional[int] = Field(-2, description="Repayment status in May")
    pay_6: Optional[int] = Field(-2, description="Repayment status in April")

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "pay_6": -1,
            }
        }
    }


class PredictionResponse(BaseModel):
    default_prediction: int = Field(
        ..., description="Predicted class (0=no default, 1=default)"
    )
    default_probability: float = Field(
        ..., ge=0, le=1, description="Probability of default"
    )
    risk_level: str = Field(..., description="Risk level categorization")


# Load model at startup
try:
    model_data = joblib.load("models/credit_default_model.joblib")
    model = model_data["model"]
    num_imputer = model_data["num_imputer"]
    num_scaler = model_data["num_scaler"]
    cat_encoder = model_data["cat_encoder"]
    numeric_features = model_data["numeric_features"]
    categorical_features = model_data["categorical_features"]

    print("✅ Model loaded successfully")
    print(f"   Numeric features: {len(numeric_features)}")
    print(f"   Categorical features: {len(categorical_features)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please train the model first: python src/models/simple_train_fixed.py")
    model = None
    model_data = None
    numeric_features = []
    categorical_features = []


def calculate_derived_features(data: dict) -> dict:
    """Calculate derived features"""
    data = data.copy()

    # Calculate aggregate features
    bill_cols = [f"bill_amt{i}" for i in range(1, 7)]
    pay_cols = [f"pay_amt{i}" for i in range(1, 7)]

    # Convert to lists
    bill_values = [data.get(col, 0) for col in bill_cols]
    pay_values = [data.get(col, 0) for col in pay_cols]

    # Calculate derived features
    data["avg_bill_amt"] = float(np.mean(bill_values)) if bill_values else 0.0
    data["total_bill_amt"] = float(np.sum(bill_values)) if bill_values else 0.0
    data["avg_pay_amt"] = float(np.mean(pay_values)) if pay_values else 0.0
    data["total_pay_amt"] = float(np.sum(pay_values)) if pay_values else 0.0

    # Calculate utilization ratio
    if data["limit_bal"] > 0:
        data["utilization_ratio"] = float(data["total_bill_amt"] / data["limit_bal"])
    else:
        data["utilization_ratio"] = 0.0

    return data


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Credit Default Prediction API",
        "status": "active",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "model not loaded",
        "model_loaded": model is not None,
        "timestamp": pd.Timestamp.now().isoformat(),
        "features": {
            "numeric_count": len(numeric_features),
            "categorical_count": len(categorical_features),
        },
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: ClientData):
    """Predict default probability for a single client"""
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train the model first."
        )

    try:
        # Convert input to dict and calculate derived features
        input_data = data.dict()
        input_data = calculate_derived_features(input_data)

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Prepare numeric features
        X_num = pd.DataFrame()
        if numeric_features:
            available_num_features = [f for f in numeric_features if f in df.columns]
            if available_num_features:
                X_num = df[available_num_features].copy()

        # Prepare categorical features
        X_cat = pd.DataFrame()
        if categorical_features:
            available_cat_features = [
                f for f in categorical_features if f in df.columns
            ]
            if available_cat_features:
                X_cat = df[available_cat_features].copy()

        # Process numeric features
        if not X_num.empty:
            X_num_imputed = num_imputer.transform(X_num)
            X_num_scaled = num_scaler.transform(X_num_imputed)
        else:
            X_num_scaled = np.array([]).reshape(1, 0)

        # Process categorical features
        if not X_cat.empty:
            # Convert to string for categorical features that are numeric
            for col in X_cat.columns:
                if X_cat[col].dtype in ["int64", "float64"]:
                    X_cat[col] = X_cat[col].astype(str)

            X_cat_encoded = cat_encoder.transform(X_cat)
        else:
            X_cat_encoded = np.array([]).reshape(1, 0)

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

        # Determine risk level
        if probability < 0.3:
            risk_level = "LOW"
        elif probability < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return PredictionResponse(
            default_prediction=int(prediction),
            default_probability=float(probability),
            risk_level=risk_level,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model_info")
def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": type(model).__name__,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_count": len(numeric_features) + len(categorical_features),
        "model_params": model.get_params() if hasattr(model, "get_params") else {},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
