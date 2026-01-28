# src/data/preprocessing.py
import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the credit card default dataset.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # 1. Normalize column names
    df_clean.columns = [col.lower().replace(".", "_") for col in df_clean.columns]
    
    # 2. Clean education column (1,2,3,4 are valid, 0,5,6 should be mapped to 4)
    if "education" in df_clean.columns:
        df_clean["education"] = df_clean["education"].apply(
            lambda x: 4 if x in [0, 5, 6] else x
        )
    
    # 3. Clean marriage column (1,2,3 are valid, 0 should be mapped to 3)
    if "marriage" in df_clean.columns:
        df_clean["marriage"] = df_clean["marriage"].apply(
            lambda x: 3 if x == 0 else x
        )
    
    # 4. Clean pay columns (PAY_0 to PAY_6)
    pay_columns = [col for col in df_clean.columns if col.startswith("pay_")]
    for col in pay_columns:
        # Assuming -2, -1, 0 are valid, 1-9 are months delayed
        # You can add specific logic if needed
        pass
    
    return df_clean


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train and test sets.
    """
    from sklearn.model_selection import train_test_split
    
    X = df.drop("default_payment_next_month", axis=1)
    y = df["default_payment_next_month"]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)