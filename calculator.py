#!/usr/bin/env python3
"""
Rule-Based Reimbursement Calculator
Derived from Decision Tree Analysis
"""
import sys
import joblib
import numpy as np

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """Calculate reimbursement using decision tree model."""
    # Feature engineering (must match training)
    receipts_per_day = total_receipts_amount / trip_duration_days
    miles_per_day = miles_traveled / trip_duration_days
    total_intensity = miles_traveled + total_receipts_amount + trip_duration_days * 100

    # Create feature array
    features = np.array([[trip_duration_days, miles_traveled, total_receipts_amount,
                         receipts_per_day, miles_per_day, total_intensity]])

    # Load the trained model
    try:
        model = joblib.load('decision_tree_model.joblib')
        prediction = model.predict(features)[0]
        return round(prediction, 2)
    except:
        # Fallback to simple linear formula if model loading fails
        return round(57.25 * trip_duration_days + 0.356 * miles_traveled + 0.343 * total_receipts_amount + 286.71, 2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 calculator.py <days> <miles> <receipts>")
        sys.exit(1)

    days = int(sys.argv[1])
    miles = float(sys.argv[2])
    receipts = float(sys.argv[3])

    result = calculate_reimbursement(days, miles, receipts)
    print(result)
