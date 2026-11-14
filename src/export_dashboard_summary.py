"""Make an aggregate-only CSV for the Overview tab (safe for demo deployment)."""
from pathlib import Path
from .common import OUTPUT, load_clean_data


def main() -> None:
    df = load_clean_data().copy()
    df["tenure_segment"] = df["tenure"].apply(
        lambda t: "0–12 months" if t <= 12 else ("13–36 months" if t <= 36 else "37+ months")
    )
    groupings = ["Contract", "InternetService", "tenure_segment", "PaymentMethod"]
    import pandas as pd
    parts = []
    for column in groupings:
        agg = df.groupby(column, dropna=False)["Churn"].agg(["size", "sum", "mean"]).reset_index()
        agg.columns = ["segment", "customers", "churners", "churn_rate"]
        agg.insert(0, "dimension", column)
        parts.append(agg)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT / "segment_summary.csv"
    pd.concat(parts, ignore_index=True).to_csv(output, index=False)
    print(f"Saved aggregated customer statistics to {output} (no individual customer data).")


if __name__ == "__main__":
    main()
