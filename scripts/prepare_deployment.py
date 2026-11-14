"""Preflight checks + optional dependency pinning for Streamlit Community Cloud.

Run this locally, using the SAME Python environment used to train your model:
    python scripts/prepare_deployment.py
    python scripts/prepare_deployment.py --pin-deps

The script does not retrain, download data, push to GitHub, or deploy anything.
Only load a joblib file you trained or otherwise trust; joblib uses pickle.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED = {
    "trained sklearn pipeline": PROJECT_ROOT / "models" / "churn_pipeline.joblib",
    "input feature schema": PROJECT_ROOT / "outputs" / "feature_schema.json",
    "holdout metrics": PROJECT_ROOT / "outputs" / "metrics.json",
    "segment summary": PROJECT_ROOT / "outputs" / "segment_summary.csv",
    "Streamlit entrypoint": PROJECT_ROOT / "streamlit_app.py",
    "Python requirements": PROJECT_ROOT / "requirements.txt",
}
OPTIONAL = {
    "confusion matrix image": PROJECT_ROOT / "outputs" / "charts" / "confusion_matrix.png",
    "ROC image": PROJECT_ROOT / "outputs" / "charts" / "roc_curve.png",
}
PIN_PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "matplotlib", "streamlit"]


def preflight() -> bool:
    print("\nCUSTOMER CHURN — DEPLOYMENT PREFLIGHT\n")
    failures = []
    for title, path in REQUIRED.items():
        present = path.is_file() and path.stat().st_size > 0
        print(("[OK]" if present else "[MISSING]") + f" {title}: {path.relative_to(PROJECT_ROOT)}")
        if not present:
            failures.append(title)
    for title, path in OPTIONAL.items():
        print(("[OK]" if path.is_file() else "[OPTIONAL]") + f" {title}: {path.relative_to(PROJECT_ROOT)}")

    if failures:
        print("\nFIX: From your Day 2 folder, run:")
        print("  python -m src.download_data  # only if raw IBM CSV is missing")
        print("  python -m src.train")
        print("  python -m src.export_dashboard_summary")
        return False

    try:
        import joblib
        import pandas as pd
        import numpy as np

        schema = json.loads(REQUIRED["input feature schema"].read_text(encoding="utf-8"))
        metrics = json.loads(REQUIRED["holdout metrics"].read_text(encoding="utf-8"))
        columns = schema["numeric"] + schema["categorical"]
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("Feature schema is empty or contains duplicate feature names")
        row = {}
        for feature in schema["numeric"]:
            row[feature] = 12 if feature == "tenure" else (85.0 if feature == "MonthlyCharges" else 1020.0)
        for feature in schema["categorical"]:
            choices = schema["categorical_choices"][feature]
            if not choices:
                raise ValueError(f"No encoded options for {feature}")
            row[feature] = choices[0]
        # This file must come from YOUR OWN Day 1 training run.
        model = joblib.load(REQUIRED["trained sklearn pipeline"])
        probabilities = model.predict_proba(pd.DataFrame([row], columns=columns))
        if probabilities.shape != (1, 2) or not np.isfinite(probabilities).all():
            raise ValueError(f"Unexpected probability output: {probabilities}")
        risk = float(probabilities[0, 1])
        if not (0 <= risk <= 1):
            raise ValueError("Predicted risk outside [0, 1]")
        summary = pd.read_csv(REQUIRED["segment summary"])
        for c in ("dimension", "segment", "churn_rate"):
            if c not in summary.columns:
                raise ValueError(f"Segment summary missing {c}")
        auc = float(metrics["test_roc_auc"])
        if not math.isfinite(auc) or not 0 <= auc <= 1:
            raise ValueError("Invalid holdout ROC-AUC")
        print(f"\n[OK] Saved model scored one customer (example p={risk:.3f}).")
        print(f"[OK] Real holdout ROC-AUC from metrics.json: {auc:.3f}")
        print(f"[OK] Selected model: {metrics.get('model', 'not recorded')}")
        print("[OK] Segment summary and required schema are readable.")
        return True
    except Exception as exc:
        print(f"\n[ERROR] Could not verify deployment artifacts: {type(exc).__name__}: {exc}")
        print("Check that you have run training in this environment and trust the saved joblib file.")
        return False


def pin_dependencies() -> bool:
    print("\nPINNING DIRECT DEPENDENCIES TO THIS LOCAL ENVIRONMENT")
    pins = []
    for package in PIN_PACKAGES:
        try:
            pins.append(f"{package}=={version(package)}")
        except PackageNotFoundError:
            print(f"[ERROR] Not installed: {package}. Install requirements.txt first.")
            return False
    path = PROJECT_ROOT / "requirements.txt"
    path.write_text("# Generated from the environment used to train/check this model.\n" + "\n".join(pins) + "\n", encoding="utf-8")
    print(f"[OK] Saved {path.relative_to(PROJECT_ROOT)} with exact local package versions.")
    print("Deploy with the same Python MINOR version (recommended: 3.12).")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin-deps", action="store_true", help="Pin local versions into requirements.txt after passing preflight")
    args = parser.parse_args()
    ok = preflight()
    if ok and args.pin_deps:
        ok = pin_dependencies()
    if ok:
        print("\n[READY] Review git status, then commit the four demo artifacts and source files.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
