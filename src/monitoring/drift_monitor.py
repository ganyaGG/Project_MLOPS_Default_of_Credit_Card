import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os
from datetime import datetime


class DriftMonitor:
    def __init__(
        self,
        api_url="http://localhost:8000",
        train_data_path="data/processed/train.csv",
    ):
        self.api_url = api_url
        self.train_data = pd.read_csv(train_data_path)

        # Store training data statistics
        self.train_stats = self._calculate_statistics(self.train_data)

    def _calculate_statistics(self, df):
        """Calculate statistical summaries"""
        stats = {"numerical": {}, "categorical": {}}

        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                stats["numerical"][col] = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "percentiles": {
                        "25": df[col].quantile(0.25),
                        "50": df[col].quantile(0.50),
                        "75": df[col].quantile(0.75),
                    },
                }
            elif df[col].dtype == "object" or df[col].nunique() < 10:
                stats["categorical"][col] = (
                    df[col].value_counts(normalize=True).to_dict()
                )

        return stats

    def calculate_psi(self, expected, actual, buckets=10):
        """Calculate Population Stability Index"""
        # Create buckets based on expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)

        # Handle edge cases
        if len(breakpoints) < 2:
            return 0

        # Calculate expected and actual distributions
        expected_hist, _ = np.histogram(expected, bins=breakpoints)
        actual_hist, _ = np.histogram(actual, bins=breakpoints)

        # Convert to percentages
        expected_perc = expected_hist / len(expected)
        actual_perc = actual_hist / len(actual)

        # Calculate PSI
        psi = 0
        for i in range(len(expected_perc)):
            if expected_perc[i] == 0:
                expected_perc[i] = 0.0001
            if actual_perc[i] == 0:
                actual_perc[i] = 0.0001

            psi += (actual_perc[i] - expected_perc[i]) * np.log(
                actual_perc[i] / expected_perc[i]
            )

        return psi

    def calculate_ks_statistic(self, expected, actual):
        """Calculate Kolmogorov-Smirnov statistic"""
        if len(expected) > 0 and len(actual) > 0:
            statistic, _ = ks_2samp(expected, actual)
            return statistic
        return 0

    def monitor_drift(self, new_data, sample_size=1000):
        """Monitor drift between training and new data"""

        # Sample new data if too large
        if len(new_data) > sample_size:
            new_data = new_data.sample(sample_size, random_state=42)

        # Calculate new data statistics
        new_stats = self._calculate_statistics(new_data)

        drift_report = {
            "timestamp": datetime.now().isoformat(),
            "sample_size": len(new_data),
            "feature_drift": {},
            "overall_drift_score": 0,
            "alerts": [],
        }

        # Check numerical features
        for feature in self.train_stats["numerical"]:
            if feature in new_stats["numerical"]:
                # Calculate PSI
                train_vals = self.train_data[feature].dropna()
                new_vals = new_data[feature].dropna()

                if len(train_vals) > 0 and len(new_vals) > 0:
                    psi = self.calculate_psi(train_vals, new_vals)
                    ks = self.calculate_ks_statistic(train_vals, new_vals)

                    drift_report["feature_drift"][feature] = {
                        "psi": float(psi),
                        "ks_statistic": float(ks),
                        "train_mean": float(
                            self.train_stats["numerical"][feature]["mean"]
                        ),
                        "new_mean": float(new_stats["numerical"][feature]["mean"]),
                        "mean_diff": float(
                            abs(
                                self.train_stats["numerical"][feature]["mean"]
                                - new_stats["numerical"][feature]["mean"]
                            )
                        ),
                    }

                    # Check for drift alerts
                    if psi > 0.2:
                        drift_report["alerts"].append(
                            {
                                "feature": feature,
                                "type": "PSI_ALERT",
                                "message": f"High PSI ({psi:.3f}) detected for {feature}",
                                "severity": "HIGH",
                            }
                        )
                    elif psi > 0.1:
                        drift_report["alerts"].append(
                            {
                                "feature": feature,
                                "type": "PSI_WARNING",
                                "message": f"Moderate PSI ({psi:.3f}) detected for {feature}",
                                "severity": "MEDIUM",
                            }
                        )

        # Check categorical features
        for feature in self.train_stats["categorical"]:
            if feature in new_stats["categorical"]:
                train_dist = self.train_stats["categorical"][feature]
                new_dist = new_stats["categorical"][feature]

                # Calculate Chi-square like statistic
                chi2_stat = 0
                for category in set(list(train_dist.keys()) + list(new_dist.keys())):
                    expected = train_dist.get(category, 0.0001)
                    observed = new_dist.get(category, 0.0001)

                    if expected > 0 and observed > 0:
                        chi2_stat += (observed - expected) ** 2 / expected

                drift_report["feature_drift"][feature] = {
                    "chi2_like_statistic": float(chi2_stat),
                    "train_distribution": train_dist,
                    "new_distribution": new_dist,
                }

                if chi2_stat > 0.1:
                    drift_report["alerts"].append(
                        {
                            "feature": feature,
                            "type": "CATEGORICAL_DRIFT",
                            "message": f"Distribution drift detected for {feature}",
                            "severity": "MEDIUM",
                        }
                    )

        # Calculate overall drift score
        if drift_report["feature_drift"]:
            psi_scores = [
                v.get("psi", 0)
                for v in drift_report["feature_drift"].values()
                if "psi" in v
            ]
            if psi_scores:
                drift_report["overall_drift_score"] = float(np.mean(psi_scores))

        return drift_report

    def monitor_predictions(self, new_data, n_samples=100):
        """Monitor prediction drift"""
        try:
            # Sample data for prediction
            if len(new_data) > n_samples:
                sample_data = new_data.sample(n_samples, random_state=42)
            else:
                sample_data = new_data

            # Get predictions from API
            predictions = []
            for _, row in sample_data.iterrows():
                try:
                    # Prepare request data
                    request_data = row.drop(
                        "default_payment_next_month", errors="ignore"
                    ).to_dict()

                    # Convert numpy types to Python types
                    for key, value in request_data.items():
                        if isinstance(value, (np.integer, np.floating)):
                            request_data[key] = value.item()

                    # Send request
                    response = requests.post(
                        f"{self.api_url}/predict", json=request_data, timeout=10
                    )

                    if response.status_code == 200:
                        predictions.append(response.json()["default_probability"])

                except Exception as e:
                    print(f"Error getting prediction: {e}")
                    continue

            if predictions:
                # Compare with training predictions (if available)
                # For now, calculate statistics
                pred_stats = {
                    "mean": np.mean(predictions),
                    "std": np.std(predictions),
                    "min": np.min(predictions),
                    "max": np.max(predictions),
                    "sample_size": len(predictions),
                }

                return {
                    "prediction_stats": pred_stats,
                    "predictions": predictions[:10],  # Return first 10 for inspection
                }

        except Exception as e:
            print(f"Error in prediction monitoring: {e}")

        return None

    def generate_report(self, new_data_path, output_dir="reports/monitoring"):
        """Generate comprehensive drift report"""
        os.makedirs(output_dir, exist_ok=True)

        # Load new data
        new_data = pd.read_csv(new_data_path)

        # Generate drift report
        drift_report = self.monitor_drift(new_data)

        # Monitor predictions
        pred_report = self.monitor_predictions(new_data)

        # Combine reports
        full_report = {
            "drift_analysis": drift_report,
            "prediction_analysis": pred_report,
            "summary": {
                "total_alerts": len(drift_report["alerts"]),
                "high_severity_alerts": len(
                    [a for a in drift_report["alerts"] if a["severity"] == "HIGH"]
                ),
                "overall_drift_score": drift_report["overall_drift_score"],
                "monitoring_timestamp": datetime.now().isoformat(),
            },
        }

        # Save report
        report_path = os.path.join(
            output_dir, f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(full_report, f, indent=2)

        # Generate visualization
        self.plot_drift_summary(drift_report, output_dir)

        print(f"Drift report saved to: {report_path}")
        return full_report

    def plot_drift_summary(self, drift_report, output_dir):
        """Create drift visualization"""
        features = list(drift_report["feature_drift"].keys())[:10]  # Top 10 features
        psi_values = []

        for feature in features:
            if "psi" in drift_report["feature_drift"][feature]:
                psi_values.append(drift_report["feature_drift"][feature]["psi"])

        if psi_values:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(psi_values)), psi_values)

            # Color bars based on PSI value
            for i, bar in enumerate(bars):
                if psi_values[i] > 0.2:
                    bar.set_color("red")
                elif psi_values[i] > 0.1:
                    bar.set_color("orange")
                else:
                    bar.set_color("green")

            plt.axhline(
                y=0.1,
                color="orange",
                linestyle="--",
                alpha=0.7,
                label="Warning threshold (0.1)",
            )
            plt.axhline(
                y=0.2,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="Alert threshold (0.2)",
            )

            plt.xticks(range(len(features)), features, rotation=45, ha="right")
            plt.ylabel("PSI Score")
            plt.title("Feature Drift Analysis (PSI)")
            plt.legend()
            plt.tight_layout()

            plot_path = os.path.join(output_dir, "drift_summary.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()


def main():
    """Main monitoring function"""
    monitor = DriftMonitor()

    # Use test data for monitoring
    test_data_path = "data/processed/test.csv"

    if os.path.exists(test_data_path):
        print("Generating drift report...")
        report = monitor.generate_report(test_data_path)

        # Print summary
        print(f"\n{'='*60}")
        print("DRIFT MONITORING SUMMARY")
        print("=" * 60)
        print(f"Overall drift score: {report['summary']['overall_drift_score']:.4f}")
        print(f"Total alerts: {report['summary']['total_alerts']}")
        print(f"High severity alerts: {report['summary']['high_severity_alerts']}")

        if report["drift_analysis"]["alerts"]:
            print("\nAlerts:")
            for alert in report["drift_analysis"]["alerts"]:
                print(f"  - [{alert['severity']}] {alert['message']}")
    else:
        print(f"Test data not found at: {test_data_path}")


if __name__ == "__main__":
    main()
