#!/usr/bin/env python
"""
Simple fixed training script that works.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Now import from pipeline
try:
    from src.models.pipeline import create_model_pipeline, get_default_features
except ImportError:
    # Fallback if running as script
    from pipeline import create_model_pipeline, get_default_features


def load_data():
    """Load processed data"""
    try:
        train_df = pd.read_csv('data/processed/train.csv')
        test_df = pd.read_csv('data/processed/test.csv')
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Check if target column exists
        if 'default_payment_next_month' not in train_df.columns:
            # Try to find target column
            target_cols = [col for col in train_df.columns if 'default' in col.lower()]
            if target_cols:
                target_col = target_cols[0]
                print(f"Using target column: {target_col}")
                
                # Rename for consistency
                train_df = train_df.rename(columns={target_col: 'default_payment_next_month'})
                test_df = test_df.rename(columns={target_col: 'default_payment_next_month'})
            else:
                raise ValueError("Target column not found in data")
        
        # Separate features and target
        X_train = train_df.drop('default_payment_next_month', axis=1)
        y_train = train_df['default_payment_next_month']
        X_test = test_df.drop('default_payment_next_month', axis=1)
        y_test = test_df['default_payment_next_month']
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please run data preparation first: python src/data/make_dataset.py")
        sys.exit(1)


def create_simple_preprocessor(numeric_features, categorical_features):
    """Create a simple preprocessor that works"""
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Numeric transformer - simple scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer - simple encoding
    # For categorical features that are actually numeric, we need to convert to string first
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def train_model():
    """Train a simple model"""
    
    print("=" * 60)
    print("SIMPLE MODEL TRAINING (FIXED VERSION)")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Get feature lists
    numeric_features, categorical_features = get_default_features()
    
    # Ensure all features exist in the data
    numeric_features = [f for f in numeric_features if f in X_train.columns]
    categorical_features = [f for f in categorical_features if f in X_train.columns]
    
    print(f"\nUsing {len(numeric_features)} numeric features:")
    print(f"  {numeric_features[:5]}..." if len(numeric_features) > 5 else f"  {numeric_features}")
    
    print(f"\nUsing {len(categorical_features)} categorical features:")
    print(f"  {categorical_features[:5]}..." if len(categorical_features) > 5 else f"  {categorical_features}")
    
    # Check data types
    print(f"\nChecking data types...")
    for col in categorical_features:
        if col in X_train.columns:
            print(f"  {col}: {X_train[col].dtype}, unique values: {sorted(X_train[col].dropna().unique()[:5])}...")
    
    # Create a simple model without complex pipeline
    print("\n" + "=" * 60)
    print("Training Logistic Regression...")
    print("=" * 60)
    
    try:
        # Manual preprocessing
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        
        # Handle numeric features
        X_train_num = X_train[numeric_features].copy()
        X_test_num = X_test[numeric_features].copy()
        
        # Impute and scale numeric features
        num_imputer = SimpleImputer(strategy='median')
        X_train_num_imputed = num_imputer.fit_transform(X_train_num)
        X_test_num_imputed = num_imputer.transform(X_test_num)
        
        num_scaler = StandardScaler()
        X_train_num_scaled = num_scaler.fit_transform(X_train_num_imputed)
        X_test_num_scaled = num_scaler.transform(X_test_num_imputed)
        
        # Handle categorical features
        X_train_cat = X_train[categorical_features].copy()
        X_test_cat = X_test[categorical_features].copy()
        
        # Convert to string for categorical features that are numeric
        for col in categorical_features:
            if X_train_cat[col].dtype in ['int64', 'float64']:
                X_train_cat[col] = X_train_cat[col].astype(str)
                X_test_cat[col] = X_test_cat[col].astype(str)
        
        # One-hot encode categorical features
        cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_cat_encoded = cat_encoder.fit_transform(X_train_cat)
        X_test_cat_encoded = cat_encoder.transform(X_test_cat)
        
        # Combine features
        X_train_processed = np.hstack([X_train_num_scaled, X_train_cat_encoded])
        X_test_processed = np.hstack([X_test_num_scaled, X_test_cat_encoded])
        
        print(f"Processed training data shape: {X_train_processed.shape}")
        print(f"Processed test data shape: {X_test_processed.shape}")
        
        # Train Logistic Regression
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        
        print("\nTraining model...")
        model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_test, y_pred)
        }
        
        # Print metrics
        print(f"\n📊 Model Performance:")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"   F1-Score: {metrics['f1']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        
        # Create visualizations
        os.makedirs("reports/figures", exist_ok=True)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend(loc="lower right")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Default', 'Default'],
                    yticklabels=['No Default', 'Default'],
                    ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_ylabel('True Label')
        axes[0, 1].set_xlabel('Predicted Label')
        
        # 3. Feature importance (coefficients)
        # Get feature names
        feature_names = numeric_features.copy()
        
        # Add categorical feature names
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
        all_feature_names = list(feature_names) + list(cat_feature_names)
        
        # Get top coefficients
        coefficients = model.coef_[0]
        top_n = min(20, len(coefficients))
        if top_n > 0:
            # Get indices of top coefficients by absolute value
            idx = np.argsort(np.abs(coefficients))[-top_n:]
            
            axes[1, 0].barh(range(top_n), coefficients[idx])
            axes[1, 0].set_yticks(range(top_n))
            axes[1, 0].set_yticklabels([all_feature_names[i] for i in idx])
            axes[1, 0].set_xlabel('Coefficient Value')
            axes[1, 0].set_title(f'Top {top_n} Feature Coefficients')
        
        # 4. Probability distribution
        axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.5, label='No Default', color='blue')
        axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.5, label='Default', color='red')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Probability Distribution by Class')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = "reports/figures/model_performance_fixed.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📈 Performance plot saved to: {plot_path}")
        
        # Save model and preprocessors
        os.makedirs("models", exist_ok=True)
        
        # Create a dictionary with all components
        model_data = {
            'model': model,
            'num_imputer': num_imputer,
            'num_scaler': num_scaler,
            'cat_encoder': cat_encoder,
            'numeric_features': numeric_features,
            'categorical_features': categorical_features
        }
        
        model_path = "models/simple_credit_model_fixed.joblib"
        joblib.dump(model_data, model_path)
        
        # Also save as default model
        default_path = "models/credit_default_model.joblib"
        joblib.dump(model_data, default_path)
        
        # Save metrics
        metrics_data = {
            "model_type": "LogisticRegression",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "features_used": {
                "numeric": numeric_features,
                "categorical": categorical_features
            }
        }
        
        os.makedirs("metrics", exist_ok=True)
        metrics_file = "metrics/simple_model_metrics_fixed.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"💾 Model saved to: {model_path}")
        print(f"💾 Default model saved to: {default_path}")
        print(f"📊 Metrics saved to: {metrics_file}")
        
        # Create a simple prediction function for testing
        create_test_prediction_script(model_data, numeric_features, categorical_features)
        
        return model_data, metrics
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_test_prediction_script(model_data, numeric_features, categorical_features):
    """Create a simple script to test predictions"""
    
    script_content = '''#!/usr/bin/env python
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
'''

    script_path = "test_prediction.py"
    with open(script_path, "w") as f:
        f.write(script_content)
    
    print(f"📝 Test prediction script created: {script_path}")
    print(f"   Run it with: python {script_path}")


def main():
    """Main function"""
    
    # Train model
    model_data, metrics = train_model()
    
    if model_data is not None:
        print(f"\n{'='*60}")
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print('='*60)
        
        print("\nNext steps:")
        print("1. Test prediction: python test_prediction.py")
        print("2. Test the API: uvicorn src.api.app:app --reload")
        print("3. View metrics in: metrics/simple_model_metrics_fixed.json")
        print("4. View plots in: reports/figures/model_performance_fixed.png")
    else:
        print(f"\n{'='*60}")
        print("❌ TRAINING FAILED")
        print('='*60)


if __name__ == "__main__":
    main()