"""Sanity check that the model artifact can score raw, unseen-form inputs."""
import joblib
from .common import MODEL_PATH, load_clean_data


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Model not found. Run: python -m src.train")
    model = joblib.load(MODEL_PATH)  # Only load artifacts you created or trust.
    data = load_clean_data().drop(columns=["customerID", "Churn"])
    examples = data.head(3)
    probabilities = model.predict_proba(examples)[:, 1]
    for i, p in enumerate(probabilities, start=1):
        print(f"Example customer {i}: predicted churn probability = {p:.1%}")
    print("These are demo predictions, not proof of out-of-sample performance.")


if __name__ == "__main__":
    main()
