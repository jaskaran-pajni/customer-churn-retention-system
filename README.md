# Customer Churn Prediction & Retention Decision System

**Data science portfolio project | Python · SQL · Scikit-learn · Streamlit**

An end-to-end machine learning demonstration that predicts potential telecom customer churn, explains observed churn patterns, and explores **hypothetical** retention outreach scenarios in an interactive dashboard.

**Live Demo:** [Customer Churn Intelligence Dashboard](https://customer-churn-retention-system-5x.streamlit.app/)

**GitHub Repository:** [View Source Code](https://github.com/jaskaran-pajni/customer-churn-retention-system)

> **Data & ethics:** This project uses the public IBM hypothetical Telco Customer Churn dataset (7,043 records). Predictions are illustrative and are not approved for real customer decisions. The sample data cannot establish whether outreach or discounts cause customers to stay.

## Features

- **Overview:** churn by contract and tenure segment, plus holdout evaluation summary.
- **Single customer:** enter account attributes, estimate churn probability, vary the classification threshold.
- **Batch & retention:** upload CSV or use six fictional examples, rank customers by an assumption-based expected-value calculation, export scored rows.
- **Model diagnostics:** cross-validation results, holdout metrics, ROC/confusion matrix, and model feature signals.

### Project workflow


## System Architecture


flowchart TD
    A["IBM Telco Dataset"] --> B["Data Cleaning"]
    B --> C["EDA & SQL Analysis"]
    B --> D["80/20 Train-Test Split"]
    D --> E["Preprocessing + 5-Fold CV"]
    E --> F{"Model Comparison"}
    F --> G["Logistic Regression"]
    F --> H["Random Forest"]
    G --> I["Model Selection"]
    H --> I
    I --> J["Holdout Evaluation"]
    J --> K["Saved ML Pipeline"]
    K --> L["Streamlit Dashboard"]
    C --> L
    L --> M["Churn Prediction"]
    L --> N["Retention Analytics"]
    L --> O["Model Diagnostics"]

    classDef source fill:#dbeafe,stroke:#2563eb,color:#1e3a8a
    classDef model fill:#dcfce7,stroke:#16a34a,color:#14532d
    classDef app fill:#f3e8ff,stroke:#9333ea,color:#581c87

    class A,B source
    class E,F,G,H,I,J,K model
    class L,M,N,O app


The pipeline fits preprocessing within cross-validation, selects the model on training data, and evaluates it once on an untouched holdout set. The retention simulator uses hypothetical assumptions rather than measured intervention effects.
  

## Run locally

Use Python 3.12 when possible and run from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.download_data
python -m src.eda
python -m src.build_database
python -m src.train
python -m src.export_dashboard_summary
python -m src.check_dashboard
python -m streamlit run streamlit_app.py
```

If you've already trained Day 2, you can keep your existing `models/` and `outputs/` artifacts; the app does not retrain on each page load. Only load locally trained/trusted model files, because `joblib` uses pickle serialization.

## Measured model evaluation

Model selection: compare Logistic Regression and Random Forest using **stratified five-fold CV ROC-AUC on the training split only**; report precision, recall, F1, average precision, ROC-AUC, and confusion matrix on the untouched test split. The 0.50 holdout threshold is fixed for the reported confusion matrix.

**Your exact, reproducible evaluation numbers are generated in `outputs/metrics.json`.** Add the values to this README only after you have run your own model; this template does not invent a ROC-AUC or claimed business ROI.

| Measure | Where to find it |
| --- | --- |
| Selected model | `outputs/metrics.json` → `model` |
| Holdout ROC-AUC | `outputs/metrics.json` → `test_roc_auc` |
| Holdout precision | `outputs/metrics.json` → `test_precision` |
| Holdout recall | `outputs/metrics.json` → `test_recall` |
| Holdout F1 | `outputs/metrics.json` → `test_f1` |
| Confusion matrix | `outputs/charts/confusion_matrix.png` |

## Retention decision simulator (not measured uplift)

The dashboard evaluates a transparent scenario:

`P(churn) × assumed_success × monthly_revenue × assumed_gross_margin × months − offer_cost`

This is an **illustrative prioritization heuristic**, not a causal effect estimate, profit forecast, optimized offer policy, or demonstrated savings. A real deployment would require randomized trials, calibrated probabilities, temporal validation, consent/privacy safeguards, and ongoing monitoring.

## Technologies

- **Pandas, NumPy:** data processing, imputation-aware cleaning, numeric features.
- **SQLite / SQL:** churn and customer-segment queries.
- **Scikit-learn:** Logistic Regression, Random Forest, cross-validation, preprocessing pipeline, saved inference model.
- **Matplotlib:** EDA and model evaluation plots.
- **Streamlit:** interactive analytics, predictions, batch scoring, scenario UI.
- **GitHub Actions / pytest:** unit tests and optional deployment preflight.

## Deployment

Follow **[DEPLOY_DAY3.md](DEPLOY_DAY3.md)** for every step, including generating real model artifacts, pinning Python package versions, safe git staging, and deploying to Streamlit Community Cloud.

Preflight in your local model's training environment:

```powershell
python scripts/prepare_deployment.py
python scripts/prepare_deployment.py --pin-deps
python -m pytest -q tests
```

The four required demo artifacts are `models/churn_pipeline.joblib`, `outputs/feature_schema.json`, `outputs/metrics.json`, and `outputs/segment_summary.csv`. They are generated locally from the IBM public demonstration dataset; no invented model or scores are included in this template.

## Limitations

- IBM's dataset is cross-sectional and hypothetical, not a documented future-outcome production dataset.
- Random holdout performance may not transfer to future periods or other telecom customers.
- Logistic coefficients and forest feature importance describe model associations, not causal drivers.
- Raw probabilities may be miscalibrated; check calibration before communicating literal risk levels.
- Real retention effectiveness cannot be inferred without treatment assignment and follow-up outcomes.
- Do not upload private customer information to a public demonstration service.

## Data source

Public IBM example dataset: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv
