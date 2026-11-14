"""Shared paths, schema, and safe dataset cleaning."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
DB_PATH = ROOT / "data" / "processed" / "churn.db"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
OUTPUT = ROOT / "outputs"
CHARTS = OUTPUT / "charts"

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
DROP_COLUMNS = ["customerID", "Churn"]
EXPECTED_COLUMNS = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
}


def load_clean_data() -> pd.DataFrame:
    """Read the IBM CSV, convert charge and outcome columns, validate row IDs."""
    if not RAW_CSV.is_file():
        raise FileNotFoundError(
            f"Dataset missing at {RAW_CSV}. Run: python -m src.download_data"
        )
    df = pd.read_csv(RAW_CSV, dtype={"customerID": "string", "TotalCharges": "string"})
    missing = EXPECTED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Unexpected dataset schema. Missing columns: {sorted(missing)}")
    if df["customerID"].isna().any() or df["customerID"].duplicated().any():
        raise ValueError("Customer IDs cannot be blank or duplicated.")

    # The real file has empty strings in some TotalCharges cells.
    # Converting to NaN lets the ML pipeline impute from TRAINING data only.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].str.strip(), errors="coerce"
    ).astype(float)
    for column in ["tenure", "MonthlyCharges"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # SeniorCitizen is an indicator, not a continuous quantity.
    df["SeniorCitizen"] = df["SeniorCitizen"].astype("string")
    outcome = df["Churn"].astype("string").str.strip().map({"Yes": 1, "No": 0})
    if outcome.isna().any():
        raise ValueError("Churn must contain only 'Yes' or 'No'.")
    df["Churn"] = outcome.astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical predictor names consistently."""
    predictors = [c for c in df.columns if c not in DROP_COLUMNS]
    categorical = [c for c in predictors if c not in NUMERIC_COLUMNS]
    return NUMERIC_COLUMNS.copy(), categorical
