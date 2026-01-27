import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any


class DataValidator:
    """Simple data validator without external dependencies"""

    def __init__(self):
        self.rules = self._get_default_rules()

    def _get_default_rules(self) -> Dict:
        """Get default validation rules for credit data"""
        return {
            "column_exists": [
                "limit_bal",
                "sex",
                "education",
                "marriage",
                "age",
                "default_payment_next_month",
            ],
            "not_null": [
                "limit_bal",
                "sex",
                "education",
                "marriage",
                "age",
                "default_payment_next_month",
            ],
            "numeric_ranges": {
                "limit_bal": {"min": 0, "max": 1000000},
                "age": {"min": 18, "max": 100},
                "bill_amt1": {"min": 0, "max": 1000000},
                "pay_amt1": {"min": 0, "max": 1000000},
            },
            "categorical_values": {
                "sex": [1, 2],
                "education": [1, 2, 3, 4],
                "marriage": [1, 2, 3],
                "default_payment_next_month": [0, 1],
            },
            "payment_status_ranges": {
                "pay_0": {"min": -2, "max": 8},
                "pay_2": {"min": -2, "max": 8},
                "pay_3": {"min": -2, "max": 8},
                "pay_4": {"min": -2, "max": 8},
                "pay_5": {"min": -2, "max": 8},
                "pay_6": {"min": -2, "max": 8},
            },
        }

    def validate_column_exists(
        self, df: pd.DataFrame, column_name: str
    ) -> Tuple[bool, str]:
        """Check if column exists in dataframe"""
        if column_name in df.columns:
            return True, f"Column '{column_name}' exists"
        else:
            return False, f"Column '{column_name}' not found"

    def validate_not_null(self, df: pd.DataFrame, column_name: str) -> Tuple[bool, str]:
        """Check for null values"""
        if column_name not in df.columns:
            return False, f"Column '{column_name}' not found for null check"

        null_count = df[column_name].isnull().sum()
        if null_count == 0:
            return True, f"Column '{column_name}': No null values"
        else:
            return False, f"Column '{column_name}': {null_count} null values found"

    def validate_numeric_range(
        self, df: pd.DataFrame, column_name: str, min_val: float, max_val: float
    ) -> Tuple[bool, str]:
        """Check if numeric values are within range"""
        if column_name not in df.columns:
            return False, f"Column '{column_name}' not found for range check"

        # Handle infinite values
        df_col = df[column_name].replace([np.inf, -np.inf], np.nan).dropna()

        out_of_range = df_col[(df_col < min_val) | (df_col > max_val)].shape[0]
        if out_of_range == 0:
            return (
                True,
                f"Column '{column_name}': All values in range [{min_val}, {max_val}]",
            )
        else:
            return False, f"Column '{column_name}': {out_of_range} values out of range"

    def validate_categorical_values(
        self, df: pd.DataFrame, column_name: str, allowed_values: List
    ) -> Tuple[bool, str]:
        """Check if categorical values are valid"""
        if column_name not in df.columns:
            return False, f"Column '{column_name}' not found for categorical check"

        invalid_values = df[~df[column_name].isin(allowed_values)].shape[0]
        if invalid_values == 0:
            return (
                True,
                f"Column '{column_name}': All values in allowed set {allowed_values}",
            )
        else:
            # Get actual invalid values
            actual_invalid = df[~df[column_name].isin(allowed_values)][
                column_name
            ].unique()[:5]
            return (
                False,
                f"Column '{column_name}': {invalid_values} invalid values. Examples: {actual_invalid.tolist()}",
            )

    def validate_data(self, data_path: str) -> Dict[str, Any]:
        """Main validation function"""

        print(f"\n{'='*60}")
        print(f"Data Validation: {os.path.basename(data_path)}")
        print("=" * 60)

        # Read data
        df = pd.read_csv(data_path)

        print(f"Data shape: {df.shape}")
        print(
            f"Columns ({len(df.columns)}): {list(df.columns)[:10]}..."
            if len(df.columns) > 10
            else f"Columns: {list(df.columns)}"
        )

        # Store validation results
        results = {
            "data_file": data_path,
            "timestamp": pd.Timestamp.now().isoformat(),
            "data_shape": df.shape,
            "column_count": len(df.columns),
            "validations": [],
            "summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "all_passed": True,
            },
        }

        # 1. Check column existence
        print("\n1. Checking column existence:")
        for col in self.rules["column_exists"]:
            passed, message = self.validate_column_exists(df, col)
            self._log_result(results, "column_exists", col, passed, message)
            print(f"   {'✅' if passed else '❌'} {message}")

        # 2. Check for null values
        print("\n2. Checking for null values:")
        for col in self.rules["not_null"]:
            passed, message = self.validate_not_null(df, col)
            self._log_result(results, "not_null", col, passed, message)
            print(f"   {'✅' if passed else '❌'} {message}")

        # 3. Check numeric ranges
        print("\n3. Checking numeric ranges:")
        for col, range_dict in self.rules["numeric_ranges"].items():
            passed, message = self.validate_numeric_range(
                df, col, range_dict["min"], range_dict["max"]
            )
            self._log_result(results, "numeric_range", col, passed, message)
            print(f"   {'✅' if passed else '❌'} {message}")

        # 4. Check categorical values
        print("\n4. Checking categorical values:")
        for col, allowed_values in self.rules["categorical_values"].items():
            passed, message = self.validate_categorical_values(df, col, allowed_values)
            self._log_result(results, "categorical", col, passed, message)
            print(f"   {'✅' if passed else '❌'} {message}")

        # 5. Check payment status ranges
        print("\n5. Checking payment status ranges:")
        for col, range_dict in self.rules["payment_status_ranges"].items():
            if col in df.columns:
                passed, message = self.validate_numeric_range(
                    df, col, range_dict["min"], range_dict["max"]
                )
                self._log_result(results, "payment_status", col, passed, message)
                print(f"   {'✅' if passed else '❌'} {message}")

        # 6. Additional checks
        print("\n6. Additional checks:")

        # Check for infinite values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_check_passed = True
        for col in numeric_cols:
            if df[col].replace([np.inf, -np.inf], np.nan).isna().any():
                inf_check_passed = False
                message = f"Column '{col}': Contains infinite values"
                self._log_result(results, "infinite_values", col, False, message)
                print(f"   ❌ {message}")

        if inf_check_passed:
            message = "No infinite values found in numeric columns"
            self._log_result(results, "infinite_values", "all_numeric", True, message)
            print(f"   ✅ {message}")

        # Check data distribution for target variable
        if "default_payment_next_month" in df.columns:
            target_dist = df["default_payment_next_month"].value_counts(normalize=True)
            message = f"Target distribution: 0={target_dist.get(0, 0):.1%}, 1={target_dist.get(1, 0):.1%}"
            self._log_result(
                results,
                "target_distribution",
                "default_payment_next_month",
                True,
                message,
            )
            print(f"   📊 {message}")

        # Update summary
        total = results["summary"]["total_checks"]
        passed = results["summary"]["passed_checks"]
        failed = results["summary"]["failed_checks"]

        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY:")
        print("=" * 60)
        print(f"Total checks: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")

        if failed == 0:
            print(f"\n✅ ALL VALIDATIONS PASSED!")
            results["summary"]["all_passed"] = True
        else:
            print(f"\n❌ SOME VALIDATIONS FAILED!")
            results["summary"]["all_passed"] = False

        # Save results
        self._save_results(results, data_path)

        return results

    def _log_result(
        self, results: Dict, check_type: str, column: str, passed: bool, message: str
    ):
        """Log validation result"""
        result = {
            "check_type": check_type,
            "column": column,
            "passed": passed,
            "message": message,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        results["validations"].append(result)
        results["summary"]["total_checks"] += 1

        if passed:
            results["summary"]["passed_checks"] += 1
        else:
            results["summary"]["failed_checks"] += 1
            results["summary"]["all_passed"] = False

    def _save_results(self, results: Dict, data_path: str):
        """Save validation results to file"""

        # Create directories if needed
        os.makedirs("reports/validation", exist_ok=True)
        os.makedirs("data/expectations", exist_ok=True)

        # Save detailed results
        filename = os.path.basename(data_path).replace(".csv", "")
        results_file = f"reports/validation/{filename}_validation_results.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Save summary
        summary_file = f"data/expectations/{filename}_validation_summary.txt"

        with open(summary_file, "w") as f:
            f.write(f"Validation Summary: {filename}\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Data Shape: {results['data_shape']}\n")
            f.write(f"Total Checks: {results['summary']['total_checks']}\n")
            f.write(f"Passed: {results['summary']['passed_checks']}\n")
            f.write(f"Failed: {results['summary']['failed_checks']}\n")
            f.write(
                f"Status: {'PASS' if results['summary']['all_passed'] else 'FAIL'}\n"
            )

            if not results["summary"]["all_passed"]:
                f.write("\nFailed Checks:\n")
                for validation in results["validations"]:
                    if not validation["passed"]:
                        f.write(
                            f"  - {validation['column']}: {validation['message']}\n"
                        )

        print(f"\nResults saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")

        # Also save to validation_status.txt
        with open("data/validation_status.txt", "a") as f:
            status = "PASS" if results["summary"]["all_passed"] else "FAIL"
            f.write(f"{filename}: {status}\n")


def main():
    """Main function"""

    # Create validator
    validator = DataValidator()

    # Define data paths
    data_paths = [
        ("data/processed/train.csv", "Training Data"),
        ("data/processed/test.csv", "Test Data"),
    ]

    all_passed = True

    for data_path, data_name in data_paths:
        if os.path.exists(data_path):
            print(f"\n{'='*60}")
            print(f"Validating {data_name}")
            print("=" * 60)

            results = validator.validate_data(data_path)

            if not results["summary"]["all_passed"]:
                all_passed = False

            print(
                f"\n{data_name} validation: {'PASS' if results['summary']['all_passed'] else 'FAIL'}"
            )
        else:
            print(f"⚠️  {data_name} not found at: {data_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL VALIDATION SUMMARY")
    print("=" * 60)

    if all_passed:
        print("✅ ALL DATA VALIDATIONS PASSED!")
    else:
        print("❌ SOME DATA VALIDATIONS FAILED!")
        print("Check the reports in reports/validation/ for details.")


if __name__ == "__main__":
    main()
