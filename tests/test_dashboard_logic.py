"""Small deterministic tests for Day 2 inference and decision logic."""
import numpy as np
import pandas as pd
import pytest
from src.dashboard_logic import feature_frame, risk_label, expected_offer_value, score_customers


SCHEMA = {
    "numeric": ["tenure", "MonthlyCharges", "TotalCharges"],
    "categorical": ["Contract", "SeniorCitizen"],
    "categorical_choices": {"Contract": ["Month-to-month", "One year"], "SeniorCitizen": ["0", "1"]},
}


class FakePipeline:
    def predict_proba(self, X):
        # Verify exact feature order, just like the trained pipeline requires.
        assert list(X) == SCHEMA["numeric"] + SCHEMA["categorical"]
        p = np.where(X["tenure"].to_numpy() > 12, 0.10, 0.80)
        return np.column_stack((1-p, p))


def sample():
    return pd.DataFrame([
        {"customerID": "A", "Churn": "No", "tenure": "6", "MonthlyCharges": "100", "TotalCharges": "600", "Contract": "Month-to-month", "SeniorCitizen": 0},
        {"customerID": "B", "Churn": "Yes", "tenure": "24", "MonthlyCharges": "50", "TotalCharges": "1200", "Contract": "One year", "SeniorCitizen": 1},
    ])


def test_feature_frame_drop_target_id_normalize():
    X = feature_frame(sample(), SCHEMA)
    assert list(X) == SCHEMA["numeric"] + SCHEMA["categorical"]
    assert X.iloc[0]["SeniorCitizen"] == "0"
    assert X.iloc[1]["MonthlyCharges"] == 50


def test_invalid_numeric_rejected():
    bad = sample()
    bad.loc[0, "tenure"] = "six months"
    with pytest.raises(ValueError, match="non-numeric"):
        feature_frame(bad, SCHEMA)


def test_missing_column_rejected():
    with pytest.raises(ValueError, match="Missing required"):
        feature_frame(sample().drop(columns=["Contract"]), SCHEMA)


def test_expected_value_math():
    # 0.8 * 0.25 * $100 * 0.5 * 12 - $25 = $95
    assert expected_offer_value(0.8, 100, 25, 0.25, 0.5, 12) == pytest.approx(95)


def test_score_customers_threshold_and_privacy():
    results = score_customers(FakePipeline(), sample(), SCHEMA)
    assert set(results["predicted_churn"]) == {0, 1}
    assert "Churn" not in results.columns
    assert "Contract" not in results.columns
    assert results.iloc[0]["customerID"] == "A"
    assert results.iloc[0]["churn_probability"] == pytest.approx(.8)


def test_risk_bands():
    assert risk_label(.1) == "Low"
    assert risk_label(.5) == "Medium"
    assert risk_label(.9) == "High"
