"""Day 2: testable prediction and business-scenario helpers (no Streamlit dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def feature_frame(source: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Create model input in the EXACT training column order, without identifiers/targets.

    A missing required column is an error; missing cell values remain missing so
    the trained sklearn pipeline's imputers can handle them.
    """
    required = schema["numeric"] + schema["categorical"]
    absent = [column for column in required if column not in source.columns]
    if absent:
        raise ValueError("Missing required feature columns: " + ", ".join(absent))

    data = source.loc[:, required].copy()
    for column in schema["numeric"]:
        original = data[column]
        numeric = pd.to_numeric(original, errors="coerce")
        bad = original.notna() & original.astype(str).str.strip().ne("") & numeric.isna()
        if bad.any():
            raise ValueError(f"'{column}' contains non-numeric values (first invalid row: {bad[bad].index[0]}).")
        data[column] = numeric

    for column in schema["categorical"]:
        if column == "SeniorCitizen":
            data[column] = data[column].apply(
                lambda v: str(int(v)) if pd.notna(v) and str(v).strip() in {"0", "1", "0.0", "1.0"} else v
            )
        data[column] = data[column].astype(object).where(pd.notna(data[column]), np.nan)
    return data


def risk_label(probability: float) -> str:
    """Illustrative risk bands; not calibrated to actual costs or outcomes."""
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be in [0, 1].")
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def expected_offer_value(
    probability: float | np.ndarray | pd.Series,
    monthly_revenue: float | np.ndarray | pd.Series,
    offer_cost: float = 25.0,
    assumed_success: float = 0.25,
    margin: float = 0.50,
    months: int = 12,
):
    """Illustrative net value under an UNVERIFIED success-rate assumption.

    E[value] = P(churn) * P(offer saves an otherwise lost customer | target)
               * revenue_per_month * gross_margin * months - per-contact offer cost.

    Does NOT estimate treatment effects or true ROI; causal impact needs an experiment.
    """
    if not 0 <= assumed_success <= 1 or not 0 <= margin <= 1:
        raise ValueError("Assumed success and margin must lie in [0, 1].")
    if offer_cost < 0 or months < 1:
        raise ValueError("Cost must be nonnegative and months positive.")
    p = np.asarray(probability, dtype=float)
    revenue = np.asarray(monthly_revenue, dtype=float)
    if np.any((p < 0) | (p > 1)) or np.any(revenue < 0):
        raise ValueError("Probability must be [0, 1] and monthly revenue nonnegative.")
    result = p * assumed_success * revenue * margin * months - offer_cost
    return float(result) if result.ndim == 0 else result


def score_customers(
    model,
    rows: pd.DataFrame,
    schema: dict,
    threshold: float = 0.50,
    offer_cost: float = 25.0,
    assumed_success: float = 0.25,
    margin: float = 0.50,
    months: int = 12,
) -> pd.DataFrame:
    """Score each row and return a limited export without user-supplied PII columns."""
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0, 1].")
    if rows.empty:
        raise ValueError("No rows to score.")
    X = feature_frame(rows, schema)
    probabilities = model.predict_proba(X)[:, 1]
    results = pd.DataFrame({
        "row_number": np.arange(1, len(rows) + 1),
        "churn_probability": probabilities,
        "predicted_churn": (probabilities >= threshold).astype(int),
        "risk_band": [risk_label(float(p)) for p in probabilities],
        "monthly_charges": X["MonthlyCharges"].to_numpy(),
    })
    if "customerID" in rows.columns:
        results.insert(1, "customerID", rows["customerID"].astype(str).to_numpy())
    results["illustrative_net_value"] = expected_offer_value(
        probabilities, X["MonthlyCharges"].fillna(0).to_numpy(),
        offer_cost=offer_cost, assumed_success=assumed_success,
        margin=margin, months=months,
    )
    results["illustrative_net_value"] = results["illustrative_net_value"].round(2)
    return results.sort_values("illustrative_net_value", ascending=False).reset_index(drop=True)


def feature_importances(model) -> pd.DataFrame:
    """Return descriptive (not causal) model feature signals."""
    preprocess = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    names = preprocess.get_feature_names_out()
    if hasattr(estimator, "coef_"):
        values = estimator.coef_[0]
        kind = "log-odds coefficient"
        ranking = np.abs(values)
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        kind = "impurity-based importance"
        ranking = values
    else:
        return pd.DataFrame(columns=["feature", "signal", "magnitude", "type"])
    result = pd.DataFrame({"feature": names, "signal": values, "magnitude": ranking})
    result["feature"] = result["feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False)
    result["type"] = kind
    return result.sort_values("magnitude", ascending=False).reset_index(drop=True)
