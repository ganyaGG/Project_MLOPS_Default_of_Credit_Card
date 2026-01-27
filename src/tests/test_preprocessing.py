import pandas as pd
import numpy as np
import pytest
from src.data.make_dataset import clean_data


def test_clean_data():
    """Test data cleaning function"""
    # Create sample data
    data = {
        'LIMIT_BAL': [10000, 20000, 30000],
        'SEX': [1, 2, 1],
        'EDUCATION': [1, 2, 0],  # 0 should be converted to 4
        'MARRIAGE': [1, 2, 0],   # 0 should be converted to 3
        'AGE': [25, 35, 45],
        'PAY_0': [-1, 0, 1],
        'BILL_AMT1': [1000, 2000, 3000],
        'PAY_AMT1': [500, 1000, 1500],
        'default.payment.next.month': [0, 1, 0]
    }
    
    df = pd.DataFrame(data)
    df_clean = clean_data(df)
    
    # Test column names
    assert 'limit_bal' in df_clean.columns
    assert 'default_payment_next_month' in df_clean.columns
    
    # Test education mapping
    assert df_clean['education'].iloc[2] == 4
    
    # Test marriage mapping
    assert df_clean['marriage'].iloc[2] == 3
    
    # Test derived features
    assert 'avg_bill_amt' in df_clean.columns
    assert 'utilization_ratio' in df_clean.columns


def test_data_validation():
    """Test data validation"""
    # This would test Great Expectations validation
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])