"""Exploratory data analysis. No target-dependent transformations for ML."""
import matplotlib
matplotlib.use("Agg")  # Save charts without a desktop GUI.
import matplotlib.pyplot as plt
import pandas as pd
from .common import CHARTS, OUTPUT, load_clean_data


def main() -> None:
    df = load_clean_data()
    CHARTS.mkdir(parents=True, exist_ok=True)

    summary = [
        f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns",
        f"Churn count: {df['Churn'].sum():,}",
        f"Churn rate: {df['Churn'].mean():.2%}",
        "\nMissing values by column:\n" + df.isna().sum().to_string(),
        "\nContract churn rates:\n" +
        (df.groupby('Contract')['Churn'].agg(['count', 'mean'])
         .sort_values('mean', ascending=False).to_string()),
    ]
    message = "\n".join(summary)
    print(message)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "eda_summary.txt").write_text(message, encoding="utf-8")

    # Chart 1 — does the positive class appear less frequently?
    counts = df['Churn'].map({0: 'Stayed', 1: 'Left'}).value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color=['#16857A', '#D37C51'])
    ax.set(title='Class distribution', ylabel='Number of customers')
    fig.tight_layout()
    fig.savefig(CHARTS / 'churn_distribution.png', dpi=150)
    plt.close(fig)

    # Chart 2 — compare churn proportions, not just raw counts.
    by_contract = (df.groupby('Contract')['Churn'].mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(by_contract.index, by_contract.values, color='#4371A5')
    ax.set(title='Churn rate by contract', xlabel='Churn rate (%)')
    fig.tight_layout()
    fig.savefig(CHARTS / 'churn_by_contract.png', dpi=150)
    plt.close(fig)

    # Chart 3 — tenure distribution by observed outcome.
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = list(range(0, 80, 6))
    ax.hist(df.loc[df.Churn == 0, 'tenure'].dropna(), bins=bins,
            alpha=.65, label='Stayed')
    ax.hist(df.loc[df.Churn == 1, 'tenure'].dropna(), bins=bins,
            alpha=.65, label='Left')
    ax.set(title='Tenure distributions by outcome', xlabel='Tenure (months)', ylabel='Count')
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / 'tenure_by_churn.png', dpi=150)
    plt.close(fig)

    # Chart 4 — billing amounts and their relationship to the outcome.
    stayed = df.loc[df.Churn == 0, 'MonthlyCharges'].dropna()
    left = df.loc[df.Churn == 1, 'MonthlyCharges'].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([stayed, left], tick_labels=['Stayed', 'Left'])
    ax.set(title='Monthly charges by outcome', ylabel='Monthly charges')
    fig.tight_layout()
    fig.savefig(CHARTS / 'monthly_charges_by_churn.png', dpi=150)
    plt.close(fig)
    print(f"\nSaved 4 charts to: {CHARTS}")


if __name__ == "__main__":
    main()
