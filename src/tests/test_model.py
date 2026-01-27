import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.pipeline import create_model_pipeline, get_default_features


def test_pipeline_creation():
    """Test pipeline creation"""
    numeric_features, categorical_features = get_default_features()

    # Test GradientBoosting pipeline
    pipeline_gb = create_model_pipeline(
        model_type="GradientBoosting",
        numeric_features=numeric_features[:5],
        categorical_features=categorical_features[:3],
    )

    assert pipeline_gb is not None
    assert hasattr(pipeline_gb, "fit")
    assert hasattr(pipeline_gb, "predict")

    # Test RandomForest pipeline
    pipeline_rf = create_model_pipeline(
        model_type="RandomForest",
        numeric_features=numeric_features[:5],
        categorical_features=categorical_features[:3],
    )

    assert pipeline_rf is not None

    # Test LogisticRegression pipeline
    pipeline_lr = create_model_pipeline(
        model_type="LogisticRegression",
        numeric_features=numeric_features[:5],
        categorical_features=categorical_features[:3],
    )

    assert pipeline_lr is not None

    # Test invalid model type
    with pytest.raises(ValueError):
        create_model_pipeline(
            model_type="InvalidModel",
            numeric_features=numeric_features[:5],
            categorical_features=categorical_features[:3],
        )


def test_pipeline_fit_predict():
    """Test pipeline fitting and prediction"""
    # Create synthetic data
    n_samples = 100
    X = pd.DataFrame(
        {
            "limit_bal": np.random.uniform(10000, 50000, n_samples),
            "age": np.random.randint(20, 60, n_samples),
            "bill_amt1": np.random.uniform(0, 10000, n_samples),
            "sex": np.random.choice([1, 2], n_samples),
            "education": np.random.choice([1, 2, 3, 4], n_samples),
            "marriage": np.random.choice([1, 2, 3], n_samples),
            "pay_0": np.random.choice([-2, -1, 0, 1, 2], n_samples),
        }
    )

    y = np.random.choice([0, 1], n_samples)

    # Create and fit pipeline
    numeric_features = ["limit_bal", "age", "bill_amt1"]
    categorical_features = ["sex", "education", "marriage", "pay_0"]

    pipeline = create_model_pipeline(
        model_type="LogisticRegression",
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    # Fit pipeline
    pipeline.fit(X, y)

    # Make predictions
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    assert len(predictions) == n_samples
    assert probabilities.shape == (n_samples, 2)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
