from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np


def create_preprocessor(numeric_features, categorical_features):
    """Create preprocessor for the pipeline"""
    
    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformer
    # Для категориальных признаков, которые являются числовыми, используем median
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def create_model_pipeline(model_type='LogisticRegression', preprocessor=None, 
                         numeric_features=None, categorical_features=None, **kwargs):
    """Create complete ML pipeline"""
    
    if preprocessor is None:
        if numeric_features is None or categorical_features is None:
            raise ValueError("Either provide preprocessor or both feature lists")
        preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # Select model
    if model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(**kwargs)
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(**kwargs)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(**kwargs, max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline


def get_default_features():
    """Get default feature lists"""
    
    # Basic features from the dataset
    numeric_features = [
        'limit_bal', 'age', 
        'bill_amt1', 'bill_amt2', 'bill_amt3',
        'bill_amt4', 'bill_amt5', 'bill_amt6',
        'pay_amt1', 'pay_amt2', 'pay_amt3',
        'pay_amt4', 'pay_amt5', 'pay_amt6'
    ]
    
    categorical_features = [
        'sex', 'education', 'marriage',
        'pay_0', 'pay_2', 'pay_3',
        'pay_4', 'pay_5', 'pay_6'
    ]
    
    # Add derived features if they exist
    derived_features = [
        'avg_bill_amt', 'total_bill_amt',
        'avg_pay_amt', 'total_pay_amt',
        'utilization_ratio'
    ]
    
    # We'll check which features actually exist when loading data
    return numeric_features, categorical_features


def get_feature_lists_from_data(X):
    """Dynamically get feature lists from data"""
    
    numeric_features = []
    categorical_features = []
    
    for column in X.columns:
        # Check if column is numeric
        if X[column].dtype in ['int64', 'float64', 'int32', 'float32']:
            # Check if it's actually categorical with few unique values
            unique_count = X[column].nunique()
            if unique_count <= 10 and column in ['sex', 'education', 'marriage', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']:
                categorical_features.append(column)
            else:
                numeric_features.append(column)
        else:
            categorical_features.append(column)
    
    return numeric_features, categorical_features


if __name__ == "__main__":
    # Test the pipeline
    print("Testing pipeline creation...")
    
    # Example usage
    num_features = ['limit_bal', 'age']
    cat_features = ['sex', 'education']
    
    pipeline = create_model_pipeline(
        model_type='LogisticRegression',
        numeric_features=num_features,
        categorical_features=cat_features
    )
    
    print(f"Pipeline created successfully!")
    print(f"Pipeline steps: {pipeline.named_steps}")