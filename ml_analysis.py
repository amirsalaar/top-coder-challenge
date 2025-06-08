#!/usr/bin/env python3
"""
Machine Learning Analysis for Black Box Reimbursement System
==========================================================

This script analyzes the historical reimbursement data to reverse-engineer
the legacy system's logic using various ML techniques.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings("ignore")


class ReimbursementAnalyzer:
    def __init__(self, data_file="public_cases.json"):
        """Initialize the analyzer and load data."""
        self.data_file = data_file
        self.df = None
        self.models = {}
        self.results = {}

    def load_data(self):
        """Load and prepare the training data."""
        print("Loading training data...")
        with open(self.data_file, "r") as f:
            data = json.load(f)

        # Convert to DataFrame
        records = []
        for case in data:
            record = case["input"].copy()
            record["expected_output"] = case["expected_output"]
            records.append(record)

        self.df = pd.DataFrame(records)
        print(f"Loaded {len(self.df)} training cases")
        print("\nData shape:", self.df.shape)
        print("\nColumns:", self.df.columns.tolist())

    def exploratory_analysis(self):
        """Perform comprehensive EDA."""
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)

        # Basic statistics
        print("\nBasic Statistics:")
        print(self.df.describe())

        # Check for missing values
        print("\nMissing Values:")
        print(self.df.isnull().sum())

        # Correlation analysis
        print("\nCorrelation Matrix:")
        correlation_matrix = self.df.corr()
        print(correlation_matrix)

        # Create visualizations
        self.create_visualizations()

    def create_visualizations(self):
        """Create comprehensive visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Reimbursement System Analysis", fontsize=16, fontweight="bold")

        # 1. Distribution of target variable
        axes[0, 0].hist(self.df["expected_output"], bins=50, alpha=0.7, color="skyblue")
        axes[0, 0].set_title("Distribution of Reimbursement Amounts")
        axes[0, 0].set_xlabel("Reimbursement ($)")
        axes[0, 0].set_ylabel("Frequency")

        # 2. Reimbursement vs Trip Duration
        axes[0, 1].scatter(
            self.df["trip_duration_days"], self.df["expected_output"], alpha=0.6
        )
        axes[0, 1].set_title("Reimbursement vs Trip Duration")
        axes[0, 1].set_xlabel("Trip Duration (days)")
        axes[0, 1].set_ylabel("Reimbursement ($)")

        # 3. Reimbursement vs Miles
        axes[0, 2].scatter(
            self.df["miles_traveled"],
            self.df["expected_output"],
            alpha=0.6,
            color="orange",
        )
        axes[0, 2].set_title("Reimbursement vs Miles Traveled")
        axes[0, 2].set_xlabel("Miles Traveled")
        axes[0, 2].set_ylabel("Reimbursement ($)")

        # 4. Reimbursement vs Receipts
        axes[1, 0].scatter(
            self.df["total_receipts_amount"],
            self.df["expected_output"],
            alpha=0.6,
            color="green",
        )
        axes[1, 0].set_title("Reimbursement vs Receipt Amount")
        axes[1, 0].set_xlabel("Receipt Amount ($)")
        axes[1, 0].set_ylabel("Reimbursement ($)")

        # 5. Correlation heatmap
        correlation_matrix = self.df.corr()
        sns.heatmap(
            correlation_matrix, annot=True, cmap="coolwarm", center=0, ax=axes[1, 1]
        )
        axes[1, 1].set_title("Correlation Matrix")

        # 6. Per-mile reimbursement analysis
        self.df["per_mile_rate"] = (
            self.df["expected_output"] / self.df["miles_traveled"]
        )
        axes[1, 2].scatter(
            self.df["miles_traveled"], self.df["per_mile_rate"], alpha=0.6, color="red"
        )
        axes[1, 2].set_title("Per-Mile Rate vs Total Miles")
        axes[1, 2].set_xlabel("Miles Traveled")
        axes[1, 2].set_ylabel("Per-Mile Rate ($/mile)")

        plt.tight_layout()
        plt.savefig("reimbursement_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def feature_engineering(self):
        """Create derived features based on interview insights."""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

        # Basic derived features
        self.df["receipts_per_day"] = (
            self.df["total_receipts_amount"] / self.df["trip_duration_days"]
        )
        self.df["miles_per_day"] = (
            self.df["miles_traveled"] / self.df["trip_duration_days"]
        )

        # Trip length categories (based on Marcus's "5-6 day sweet spot")
        self.df["trip_category"] = pd.cut(
            self.df["trip_duration_days"],
            bins=[0, 2, 4, 6, 10, float("inf")],
            labels=["short", "medium", "sweet_spot", "long", "very_long"],
        )

        # Mileage categories (based on interview hints about non-linear rates)
        self.df["mileage_category"] = pd.cut(
            self.df["miles_traveled"],
            bins=[0, 200, 400, 600, 1000, float("inf")],
            labels=["local", "regional", "long_distance", "cross_country", "extreme"],
        )

        # Receipt spending levels
        self.df["receipt_category"] = pd.cut(
            self.df["total_receipts_amount"],
            bins=[0, 100, 500, 1000, 2000, float("inf")],
            labels=["minimal", "moderate", "high", "very_high", "extreme"],
        )

        # Efficiency metrics
        self.df["cost_efficiency"] = (
            self.df["total_receipts_amount"] / self.df["miles_traveled"]
        )

        print("Created derived features:")
        for col in [
            "receipts_per_day",
            "miles_per_day",
            "trip_category",
            "mileage_category",
            "receipt_category",
            "cost_efficiency",
        ]:
            print(f"  - {col}")

    def train_models(self):
        """Train various ML models."""
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)

        # Prepare features (only numeric for now)
        feature_cols = [
            "trip_duration_days",
            "miles_traveled",
            "total_receipts_amount",
            "receipts_per_day",
            "miles_per_day",
            "cost_efficiency",
        ]
        X = self.df[feature_cols]
        y = self.df["expected_output"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define models to try
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=1.0),
            "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }

        # Train and evaluate models
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y, cv=5, scoring="neg_mean_absolute_error"
            )
            cv_mae = -cv_scores.mean()

            self.models[name] = model
            self.results[name] = {"mae": mae, "mse": mse, "r2": r2, "cv_mae": cv_mae}

            print(f"  MAE: ${mae:.2f}")
            print(f"  R²: {r2:.3f}")
            print(f"  CV MAE: ${cv_mae:.2f}")

        # Try polynomial features
        print(f"\nTrying Polynomial Regression...")
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.transform(X_test)

        poly_model = LinearRegression()
        poly_model.fit(X_poly_train, y_train)
        y_pred_poly = poly_model.predict(X_poly_test)

        mae_poly = mean_absolute_error(y_test, y_pred_poly)
        r2_poly = r2_score(y_test, y_pred_poly)

        self.models["Polynomial Regression"] = (poly, poly_model)
        self.results["Polynomial Regression"] = {"mae": mae_poly, "r2": r2_poly}

        print(f"  MAE: ${mae_poly:.2f}")
        print(f"  R²: {r2_poly:.3f}")

    def analyze_best_model(self):
        """Analyze the best performing model in detail."""
        print("\n" + "=" * 60)
        print("BEST MODEL ANALYSIS")
        print("=" * 60)

        # Find best model
        best_model_name = min(self.results.keys(), key=lambda x: self.results[x]["mae"])
        best_model = self.models[best_model_name]

        print(f"Best Model: {best_model_name}")
        print(f"MAE: ${self.results[best_model_name]['mae']:.2f}")
        print(f"R²: {self.results[best_model_name]['r2']:.3f}")

        # Feature importance for tree-based models
        if hasattr(best_model, "feature_importances_"):
            feature_cols = [
                "trip_duration_days",
                "miles_traveled",
                "total_receipts_amount",
                "receipts_per_day",
                "miles_per_day",
                "cost_efficiency",
            ]

            importance_df = pd.DataFrame(
                {"feature": feature_cols, "importance": best_model.feature_importances_}
            ).sort_values("importance", ascending=False)

            print("\nFeature Importance:")
            for _, row in importance_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")

        # Decision tree rules (if applicable)
        if isinstance(best_model, DecisionTreeRegressor):
            print("\nDecision Tree Rules (first 20 lines):")
            tree_rules = export_text(
                best_model,
                feature_names=[
                    "trip_duration_days",
                    "miles_traveled",
                    "total_receipts_amount",
                    "receipts_per_day",
                    "miles_per_day",
                    "cost_efficiency",
                ],
            )
            print("\n".join(tree_rules.split("\n")[:20]))

    def hypothesis_testing(self):
        """Test specific hypotheses from employee interviews."""
        print("\n" + "=" * 60)
        print("HYPOTHESIS TESTING")
        print("=" * 60)

        # Hypothesis 1: Sweet spot around 5-6 days (Marcus)
        print("Hypothesis 1: Trip duration sweet spot (5-6 days)")
        for duration in range(1, 15):
            subset = self.df[self.df["trip_duration_days"] == duration]
            if len(subset) > 5:  # Only if we have enough samples
                avg_reimbursement = subset["expected_output"].mean()
                avg_per_day = avg_reimbursement / duration
                print(
                    f"  {duration} days: Avg ${avg_reimbursement:.2f} (${avg_per_day:.2f}/day) - {len(subset)} cases"
                )

        # Hypothesis 2: Non-linear mileage rates
        print("\nHypothesis 2: Non-linear mileage rates")
        mileage_bins = [
            (0, 200),
            (200, 400),
            (400, 600),
            (600, 1000),
            (1000, float("inf")),
        ]
        for min_miles, max_miles in mileage_bins:
            if max_miles == float("inf"):
                subset = self.df[self.df["miles_traveled"] >= min_miles]
                label = f"{min_miles}+ miles"
            else:
                subset = self.df[
                    (self.df["miles_traveled"] >= min_miles)
                    & (self.df["miles_traveled"] < max_miles)
                ]
                label = f"{min_miles}-{max_miles} miles"

            if len(subset) > 5:
                avg_per_mile = (
                    subset["expected_output"] / subset["miles_traveled"]
                ).mean()
                print(f"  {label}: ${avg_per_mile:.2f}/mile - {len(subset)} cases")

        # Hypothesis 3: Receipt spending caps/penalties
        print("\nHypothesis 3: Receipt amount patterns")
        receipt_bins = [(0, 500), (500, 1000), (1000, 2000), (2000, float("inf"))]
        for min_receipt, max_receipt in receipt_bins:
            if max_receipt == float("inf"):
                subset = self.df[self.df["total_receipts_amount"] >= min_receipt]
                label = f"${min_receipt}+ receipts"
            else:
                subset = self.df[
                    (self.df["total_receipts_amount"] >= min_receipt)
                    & (self.df["total_receipts_amount"] < max_receipt)
                ]
                label = f"${min_receipt}-{max_receipt} receipts"

            if len(subset) > 5:
                # Calculate "receipt efficiency" - how much reimbursement per receipt dollar
                avg_efficiency = (
                    subset["expected_output"] / subset["total_receipts_amount"]
                ).mean()
                print(
                    f"  {label}: {avg_efficiency:.2f}x multiplier - {len(subset)} cases"
                )

    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("REIMBURSEMENT SYSTEM REVERSE ENGINEERING")
        print("=" * 60)

        self.load_data()
        self.exploratory_analysis()
        self.feature_engineering()
        self.hypothesis_testing()
        self.train_models()
        self.analyze_best_model()

        # Summary
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)

        print("\nModel Performance Summary:")
        for model_name, metrics in self.results.items():
            print(f"  {model_name}: MAE ${metrics['mae']:.2f}, R² {metrics['r2']:.3f}")

        print("\nNext Steps:")
        print("  1. Implement the best performing model")
        print("  2. Fine-tune parameters for exact match accuracy")
        print("  3. Create the run.sh script")
        print("  4. Test against all 1,000 cases")


if __name__ == "__main__":
    analyzer = ReimbursementAnalyzer()
    analyzer.run_full_analysis()
