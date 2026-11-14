"""Leakage-safe model comparison and untouched holdout evaluation."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score,
    roc_auc_score, RocCurveDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import (
    CHARTS, DROP_COLUMNS, MODEL_PATH, OUTPUT, feature_columns, load_clean_data
)

RANDOM_STATE = 42


def create_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Fit imputation, scaling, and encoding using training folds only."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    category_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", category_pipeline, categorical),
    ])


def main() -> None:
    df = load_clean_data()
    numeric, categorical = feature_columns(df)
    feature_names = numeric + categorical
    X = df[feature_names]
    y = df["Churn"]

    # Split BEFORE fitting imputers, scalers, encoders, or models.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    algorithms = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=160, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    # DummyClassifier makes clear why ROC-AUC ~0.5 is not meaningful progress.
    cv_scores = {}
    candidates = {"Majority baseline": DummyClassifier(strategy="prior"), **algorithms}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocess", create_preprocessor(numeric, categorical)),
            ("model", estimator),
        ])
        scores = cross_val_score(pipeline, X_train, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=1)
        cv_scores[name] = {
            "mean_roc_auc": float(scores.mean()),
            "std_roc_auc": float(scores.std()),
        }
        print(f"{name:20s}: CV ROC-AUC {scores.mean():.3f} ± {scores.std():.3f}")

    best_name = max(algorithms, key=lambda name: cv_scores[name]["mean_roc_auc"])
    best_pipeline = Pipeline([
        ("preprocess", create_preprocessor(numeric, categorical)),
        ("model", algorithms[best_name]),
    ])
    best_pipeline.fit(X_train, y_train)

    # Evaluate ONCE on holdout data. Never select or tune using these scores.
    predicted_probs = best_pipeline.predict_proba(X_test)[:, 1]
    predicted = (predicted_probs >= 0.50).astype(int)
    metrics = {
        "model": best_name,
        "selection_metric": "5-fold cross-validated training ROC-AUC",
        "cross_validation": cv_scores,
        "holdout_threshold": 0.50,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_positive_rate": float(y_test.mean()),
        "test_roc_auc": float(roc_auc_score(y_test, predicted_probs)),
        "test_average_precision": float(average_precision_score(y_test, predicted_probs)),
        "test_precision": float(precision_score(y_test, predicted, zero_division=0)),
        "test_recall": float(recall_score(y_test, predicted, zero_division=0)),
        "test_f1": float(f1_score(y_test, predicted, zero_division=0)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
    }
    print(f"\nSelected model: {best_name}")
    print(classification_report(y_test, predicted, target_names=["Stayed", "Left"], zero_division=0))
    print(f"Holdout ROC-AUC: {metrics['test_roc_auc']:.3f}")
    print(f"Holdout average precision: {metrics['test_average_precision']:.3f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix_labels_0_1']}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Store the required input schema for Day 2's Streamlit form.
    schema = {
        "numeric": numeric,
        "categorical": categorical,
        "categorical_choices": {
            column: sorted(str(v) for v in X_train[column].dropna().unique())
            for column in categorical
        },
        "target": "Churn",
        "positive_class": 1,
    }
    (OUTPUT / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    test_results = X_test.copy()
    test_results.insert(0, "actual_churn", y_test.to_numpy())
    test_results.insert(1, "predicted_churn", predicted)
    test_results.insert(2, "churn_probability", predicted_probs)
    test_results.to_csv(OUTPUT / "test_predictions.csv", index=False)

    cm = confusion_matrix(y_test, predicted, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Left"]).plot(cmap="Blues", values_format="d")
    plt.title(f"Holdout confusion matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(CHARTS / "confusion_matrix.png", dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, predicted_probs, name=best_name)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    plt.title("Holdout ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS / "roc_curve.png", dpi=150)
    plt.close()
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved metrics, schema and predictions to: {OUTPUT}")
    print("IMPORTANT: This is a sample-data demonstration, not a deployed business model.")


if __name__ == "__main__":
    main()
