"""Check Day 2 prerequisites and inference without launching the web UI."""
from __future__ import annotations
import json
import joblib
import pandas as pd
from .common import MODEL_PATH, OUTPUT
from .dashboard_logic import score_customers


def main() -> None:
    schema_path = OUTPUT / "feature_schema.json"
    if not MODEL_PATH.is_file() or not schema_path.is_file():
        raise FileNotFoundError(
            "Missing trained model or feature schema. Run 'python -m src.train' "
            "(or copy these generated files from your completed Day 1 folder)."
        )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    # Use observed category choices so the input is compatible with the pipeline.
    example = {"tenure": 12, "MonthlyCharges": 85.0, "TotalCharges": 1020.0}
    for col in schema["categorical"]:
        example[col] = schema["categorical_choices"][col][0]
    model = joblib.load(MODEL_PATH)  # Only load an artifact you created/trust.
    scored = score_customers(model, pd.DataFrame([example]), schema)
    print("Day 2 backend is ready!")
    print(f"Example probability: {scored.iloc[0]['churn_probability']:.1%}")
    print(f"Example class at threshold 0.50: {scored.iloc[0]['predicted_churn']}")
    print("Next: python -m streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
