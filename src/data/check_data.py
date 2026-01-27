#!/usr/bin/env python
"""
Simple data checker script.
Run this to quickly check your data before training.
"""

import pandas as pd
import numpy as np
import os
import sys


def check_data_quality(data_path: str):
    """Check basic data quality"""

    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False

    print(f"\n📊 Checking data: {data_path}")
    print("=" * 50)

    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Successfully loaded data")
        print(f"   Shape: {df.shape}")
        print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False

    # Check columns
    print(f"\n📋 Columns ({len(df.columns)} total):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}: {df[col].dtype}")

    # Check for missing values
    print(f"\n🔍 Missing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ No missing values")
    else:
        print("   ⚠️  Missing values found:")
        for col, count in missing[missing > 0].items():
            percentage = count / len(df) * 100
            print(f"      {col}: {count} ({percentage:.1f}%)")

    # Check target variable
    if "default_payment_next_month" in df.columns:
        print(f"\n🎯 Target variable distribution:")
        target_counts = df["default_payment_next_month"].value_counts()
        for val, count in target_counts.items():
            percentage = count / len(df) * 100
            print(f"   {val}: {count} ({percentage:.1f}%)")

        # Check for class imbalance
        if len(target_counts) == 2:
            ratio = target_counts[1] / target_counts[0]
            if ratio < 0.2 or ratio > 5:
                print(f"   ⚠️  Significant class imbalance (ratio: {ratio:.2f})")
    else:
        print(f"\n⚠️  Target variable 'default_payment_next_month' not found")

    # Check numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n📈 Numeric columns statistics (first 5):")
        for col in numeric_cols[:5]:
            print(f"   {col}:")
            print(f"      Min: {df[col].min():.2f}")
            print(f"      Max: {df[col].max():.2f}")
            print(f"      Mean: {df[col].mean():.2f}")
            print(f"      Std: {df[col].std():.2f}")

        # Check for infinite values
        inf_cols = []
        for col in numeric_cols:
            if df[col].replace([np.inf, -np.inf], np.nan).isna().any():
                inf_cols.append(col)

        if inf_cols:
            print(f"   ⚠️  Infinite values found in: {inf_cols}")
        else:
            print(f"   ✅ No infinite values in numeric columns")

    # Check categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        print(f"\n📊 Categorical columns (first 3):")
        for col in categorical_cols[:3]:
            unique_values = df[col].nunique()
            print(f"   {col}: {unique_values} unique values")
            if unique_values <= 10:
                value_counts = df[col].value_counts()
                for val, count in value_counts.items():
                    print(f"      '{val}': {count}")

    # Check duplicates
    duplicate_rows = df.duplicated().sum()
    if duplicate_rows > 0:
        print(
            f"\n⚠️  Found {duplicate_rows} duplicate rows ({duplicate_rows/len(df)*100:.1f}%)"
        )
    else:
        print(f"\n✅ No duplicate rows")

    return True


def main():
    """Main function"""

    print("🧪 DATA QUALITY CHECKER")
    print("=" * 50)

    # Check if data files exist
    data_files = [
        ("data/raw/UCI_Credit_Card.csv", "Raw Data"),
        ("data/processed/train.csv", "Training Data"),
        ("data/processed/test.csv", "Test Data"),
    ]

    all_good = True

    for data_path, data_name in data_files:
        print(f"\n🔎 Checking {data_name}...")
        if os.path.exists(data_path):
            success = check_data_quality(data_path)
            if not success:
                all_good = False
        else:
            print(f"   ⚠️  File not found: {data_path}")
            all_good = False

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print("=" * 50)

    if all_good:
        print("✅ All data checks passed!")
        print("You can proceed with training: python src/models/train.py")
    else:
        print("❌ Some issues found with data.")
        print("Please fix the issues before proceeding.")

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
