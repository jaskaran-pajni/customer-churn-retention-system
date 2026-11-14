"""Customer Churn & Retention Decision System — interactive Day 2 portfolio app.

Run from project root:
    python -m streamlit run streamlit_app.py

The app uses Day 1's trusted, locally trained sklearn pipeline. It never fits
or changes a model during interactive inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.common import MODEL_PATH, OUTPUT, CHARTS
from src.dashboard_logic import (
    feature_frame, feature_importances, risk_label, score_customers,
    expected_offer_value,
)

st.set_page_config(page_title="Churn Intelligence | Portfolio", page_icon="📊", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1300px;}
    h1,h2,h3 {letter-spacing: -0.035em;}
    div[data-testid="stMetric"] {border:1px solid rgba(125,140,165,.22);
        padding:14px 16px; border-radius:12px; background:rgba(110,125,165,.07);}
    .eyebrow {font-size:.79rem; font-weight:700; letter-spacing:.12em;
        text-transform:uppercase; color:#94A3B8; margin-bottom:6px;}
    .subhead {font-size:1.02rem; color:#94A3B8; margin-top:-8px; margin-bottom:16px;}
    </style>""", unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(path: str):
    # Joblib/pickle must NEVER load untrusted uploads. This file is created by Day 1 training.
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_segment_summary(path: str):
    return pd.read_csv(path)


def default_customer(schema: dict) -> dict:
    """Valid, internally consistent starting values for the form/template."""
    preferred = {
        "gender": "Female", "SeniorCitizen": "0", "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }
    row = {"tenure": 12, "MonthlyCharges": 85.0, "TotalCharges": 1020.0}
    for name in schema["categorical"]:
        options = schema["categorical_choices"].get(name, [])
        choice = preferred.get(name)
        row[name] = choice if choice in options else (options[0] if options else "")
    return row


def demo_rows(schema: dict) -> pd.DataFrame:
    base = default_customer(schema)
    samples = []
    for i, (tenure, charge, contract) in enumerate([
        (2, 95.0, "Month-to-month"), (8, 80.0, "Month-to-month"),
        (32, 65.0, "One year"), (68, 105.0, "Two year"),
        (15, 55.0, "One year"), (48, 110.0, "Month-to-month"),
    ], start=1):
        row = base.copy()
        row.update({"customerID": f"DEMO-{i:03d}", "tenure": tenure,
                    "MonthlyCharges": charge, "TotalCharges": round(tenure*charge, 2)})
        if contract in schema["categorical_choices"].get("Contract", []):
            row["Contract"] = contract
        samples.append(row)
    return pd.DataFrame(samples)


st.markdown('<div class="eyebrow">Data Science Portfolio / ML + Business Analytics</div>', unsafe_allow_html=True)
st.title("Customer Churn Intelligence")
st.markdown('<div class="subhead">Predict churn · Explore segments · Prioritize retention outreach</div>', unsafe_allow_html=True)
st.caption("Educational demo trained on IBM's hypothetical Telco Churn dataset — not production advice or validated retention uplift.")

schema_path = OUTPUT / "feature_schema.json"
metrics_path = OUTPUT / "metrics.json"
summary_path = OUTPUT / "segment_summary.csv"
missing = [p for p in (MODEL_PATH, schema_path) if not p.is_file()]
if missing:
    st.error("The Day 1 model or feature schema is missing. Train locally before opening this dashboard.")
    st.code("python -m src.download_data\npython -m src.train\npython -m src.export_dashboard_summary\npython -m streamlit run streamlit_app.py", language="powershell")
    st.info("Keep the saved model and feature_schema.json in their original folders. No model will be downloaded from the internet.")
    st.stop()

model = load_model(str(MODEL_PATH))
schema = load_json(str(schema_path))
metrics = load_json(str(metrics_path)) if metrics_path.is_file() else {}

with st.sidebar:
    st.subheader("Decision assumptions")
    threshold = st.slider("Classification threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05,
                          help="Changes predicted labels, not model probabilities. Not tuned on the holdout set.")
    offer_cost = st.number_input("Offer/contact cost ($)", min_value=0.0, max_value=1000.0, value=25.0, step=5.0)
    assumed_success_pct = st.slider("Assumed intervention success (%)", 0, 100, 25, 5,
                                    help="Hypothetical uplift parameter. This dataset cannot estimate the true causal impact of an offer.")
    margin_pct = st.slider("Assumed gross margin (%)", 0, 100, 50, 5)
    months = st.slider("Value horizon (months)", 1, 24, 12, 1)
    assumed_success = assumed_success_pct / 100
    margin = margin_pct / 100
    st.divider()
    st.caption("Model: " + str(metrics.get("model", type(model.named_steps['model']).__name__)))
    st.caption("Training source: IBM hypothetical telco sample. Only load model files you created or trust.")


overview_tab, single_tab, batch_tab, diagnostics_tab = st.tabs([
    "📈 Overview", "👤 Single customer", "📋 Batch & retention", "🔬 Model diagnostics",
])

with overview_tab:
    st.subheader("Snapshot of customer churn")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Holdout ROC-AUC", f"{metrics['test_roc_auc']:.3f}" if "test_roc_auc" in metrics else "—")
    c2.metric("Holdout recall", f"{metrics['test_recall']:.1%}" if "test_recall" in metrics else "—")
    c3.metric("Holdout precision", f"{metrics['test_precision']:.1%}" if "test_precision" in metrics else "—")
    c4.metric("Test customers", f"{metrics['test_rows']:,}" if "test_rows" in metrics else "—")
    st.caption("These figures are from a single untouched holdout evaluation at a 50% threshold; the sidebar threshold does not recompute them.")
    if summary_path.is_file():
        segments = read_segment_summary(str(summary_path))
        lcol, rcol = st.columns(2)
        for box, dimension, title in [(lcol, "Contract", "Churn rate by contract"),
                                       (rcol, "tenure_segment", "Churn rate by tenure")]:
            with box:
                st.markdown(f"**{title}**")
                chart_data = segments.loc[segments.dimension == dimension, ["segment", "churn_rate"]].copy()
                chart_data["churn_rate"] *= 100
                st.bar_chart(chart_data.set_index("segment"), horizontal=True,
                             x_label="Churn rate (%)", y_label="")
        st.caption("Observed group-level proportions in the IBM sample; these are associations, not causal explanations.")
        with st.expander("View segment table"):
            display = segments.copy()
            display["churn_rate_pct"] = (display["churn_rate"] * 100).round(2)
            st.dataframe(display.drop(columns="churn_rate"), use_container_width=True, hide_index=True)
    else:
        st.info("Optional segment charts aren't generated yet. Run `python -m src.export_dashboard_summary` after downloading Day 1 data.")
    st.info("Prediction is not proof that a retention discount will work. Such effects require experimentation.")

with single_tab:
    st.subheader("Predict one customer's churn risk")
    st.write("Enter customer attributes as they would appear **at prediction time**. The same Day 1 preprocessing pipeline will handle encoding and scaling.")
    defaults = default_customer(schema)
    with st.form("prediction_form"):
        numeric_cols = st.columns(3)
        row = {}
        for index, (name, label, fmt) in enumerate([
            ("tenure", "Tenure (months)", "%d"),
            ("MonthlyCharges", "Monthly charges ($)", "%.2f"),
            ("TotalCharges", "Total charges to date ($)", "%.2f"),
        ]):
            if name not in schema["numeric"]:
                continue
            with numeric_cols[index]:
                row[name] = st.number_input(label, min_value=0 if name == "tenure" else 0.0,
                                            max_value=1000 if name == "tenure" else 100000.0,
                                            value=defaults[name], format=fmt, key=f"single_{name}")
        st.caption("Use TotalCharges as the total historical amount billed, not the current monthly charge.")
        with st.expander("Customer, service, and billing features", expanded=True):
            cols = st.columns(3)
            for idx, name in enumerate(schema["categorical"]):
                options = schema["categorical_choices"].get(name, [])
                if not options:
                    st.error(f"No known training choices for '{name}'.")
                    st.stop()
                chosen = defaults.get(name, options[0])
                with cols[idx % 3]:
                    row[name] = st.selectbox(name.replace("_", " "), options,
                                             index=options.index(chosen), key=f"single_{name}")
        submitted = st.form_submit_button("Calculate churn probability", type="primary", use_container_width=True)
    if submitted:
        try:
            X = feature_frame(pd.DataFrame([row]), schema)
            prob = float(model.predict_proba(X)[0, 1])
            st.session_state.single_result = {"probability": prob, "monthly": float(row["MonthlyCharges"])}
        except (ValueError, TypeError) as exc:
            st.error(f"Could not score this customer: {exc}")
    if "single_result" in st.session_state:
        result = st.session_state.single_result
        probability = result["probability"]
        st.divider()
        left, right = st.columns([1, 2])
        with left:
            st.metric("Predicted probability of churn", f"{probability:.1%}")
            st.progress(min(100, max(0, int(round(probability*100)))))
            st.write(f"**Risk band:** {risk_label(probability)} (illustrative)")
            st.write(f"**Classification:** {'1 — predicted churn' if probability >= threshold else '0 — predicted stay'} at {threshold:.0%} threshold")
        with right:
            scenario_value = expected_offer_value(
                probability, result["monthly"], offer_cost, assumed_success, margin, months
            )
            st.metric("Illustrative net value of offering retention", f"${scenario_value:,.2f}")
            st.caption("Calculation: P(churn) × assumed offer success × monthly charges × assumed gross margin × months − offer cost.")
            st.caption("Not measured ROI; neither price sensitivity nor intervention efficacy is identified by this dataset.")
        st.info("A probability is a model estimate, not a guarantee. These probabilities have not been separately calibrated.")

with batch_tab:
    st.subheader("Batch prediction & retention prioritization")
    st.write("Upload a CSV with the original Day 1 feature columns. `customerID` is optional; `Churn` is not required. For a quick test, use the demo records below.")
    template = pd.DataFrame([{"customerID": "EXAMPLE-001", **default_customer(schema)}])
    st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"),
                       file_name="customer_upload_template.csv", mime="text/csv")
    upload = st.file_uploader("Choose customer CSV", type=["csv"], help="Up to 5 MB and 2,000 records, for demo purposes")
    use_examples = st.checkbox("Use six fictional example customers instead")
    candidate_rows = None
    if use_examples:
        candidate_rows = demo_rows(schema)
        st.caption("These six inputs are fabricated for UI demonstration, not actual observed customers.")
    elif upload is not None:
        if upload.size > 5 * 1024 * 1024:
            st.error("CSV is larger than 5 MB. Please use a smaller sample.")
        else:
            try:
                candidate_rows = pd.read_csv(upload)
            except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
                st.error(f"Could not read the CSV file: {exc}")
    if candidate_rows is not None:
        if len(candidate_rows) > 2000:
            st.error("Please limit demo batches to 2,000 rows.")
        elif candidate_rows.empty:
            st.error("Upload at least one row.")
        else:
            with st.expander("Preview input rows"):
                st.dataframe(candidate_rows.head(10), use_container_width=True, hide_index=True)
            try:
                scores = score_customers(
                    model, candidate_rows, schema, threshold=threshold,
                    offer_cost=offer_cost, assumed_success=assumed_success,
                    margin=margin, months=months,
                )
                count_flagged = int((scores.predicted_churn == 1).sum())
                m1, m2, m3 = st.columns(3)
                m1.metric("Customers scored", f"{len(scores):,}")
                m2.metric("Flagged for churn", f"{count_flagged:,}")
                m3.metric("Mean predicted churn risk", f"{scores.churn_probability.mean():.1%}")
                st.subheader("Priority list")
                budget = st.number_input("Maximum customers to contact", min_value=1,
                                         max_value=max(1, len(scores)), value=min(20, len(scores)), step=1)
                eligible = scores[(scores.predicted_churn == 1) & (scores.illustrative_net_value > 0)]
                prioritized = eligible.head(budget)
                st.caption("Sorted by illustrative expected net value, filtered by chosen threshold and positive estimated value. Uses hypothetical, unvalidated assumptions.")
                st.dataframe(prioritized, use_container_width=True, hide_index=True, column_config={
                    "churn_probability": st.column_config.ProgressColumn("Churn risk", min_value=0, max_value=1, format="percent"),
                    "illustrative_net_value": st.column_config.NumberColumn("Scenario net value ($)", format="$%.2f"),
                })
                st.write(f"**{len(prioritized)}** contacts selected · total illustrative net value: **${prioritized.illustrative_net_value.sum():,.2f}**")
                st.download_button("Download all scored customers", scores.to_csv(index=False).encode("utf-8"),
                                   file_name="scored_customers.csv", mime="text/csv")
                st.warning("Rankings are scenario simulations, not real-world profit estimates. To prove that offers reduce churn, run an A/B test.")
            except (ValueError, TypeError) as exc:
                st.error(f"Could not score these rows: {exc}")

with diagnostics_tab:
    st.subheader("How trustworthy is the model?")
    if metrics:
        st.write(f"**Selected algorithm:** {metrics.get('model', 'Unknown')}")
        st.write("**Selection method:** training-only stratified five-fold cross-validation, ranked by ROC-AUC.")
        cv = metrics.get("cross_validation", {})
        if cv:
            cv_display = pd.DataFrame([
                {"model": k, "CV ROC-AUC (mean)": v["mean_roc_auc"],
                 "CV ROC-AUC (std)": v["std_roc_auc"]} for k, v in cv.items()
            ])
            st.dataframe(cv_display, hide_index=True, use_container_width=True)
        cols = st.columns(4)
        for col, label, key in zip(cols,
                                   ["ROC-AUC", "Avg precision", "F1 (0.50)", "Recall (0.50)"],
                                   ["test_roc_auc", "test_average_precision", "test_f1", "test_recall"]):
            col.metric(label, f"{metrics[key]:.3f}" if key in metrics else "—")
        st.caption("The holdout results always use a 0.50 threshold. Changing the sidebar only changes predictions made in this demo.")
    else:
        st.info("No outputs/metrics.json available; re-run Day 1 training to see evaluation scores.")
    left, right = st.columns(2)
    for col, file, title in [
        (left, CHARTS / "confusion_matrix.png", "Holdout confusion matrix"),
        (right, CHARTS / "roc_curve.png", "Holdout ROC curve"),
    ]:
        with col:
            st.markdown(f"**{title}**")
            if file.is_file():
                st.image(str(file), use_container_width=True)
            else:
                st.caption("Not bundled for cloud deployment; run Day 1 locally to generate this image.")
    try:
        importances = feature_importances(model)
        if not importances.empty:
            st.subheader("Features with the strongest model signal")
            top = importances.head(12).iloc[::-1]
            st.bar_chart(top.set_index("feature")["magnitude"], horizontal=True,
                         x_label="Absolute coefficient or relative importance", y_label="")
            st.caption("For Logistic Regression, values are absolute log-odds coefficients after training-time transformations. For Random Forest, values are impurity-based importances. Neither represents a causal effect.")
    except (KeyError, AttributeError, ValueError) as exc:
        st.caption(f"Feature summary unavailable: {exc}")
    with st.expander("Limitations and interview talking points"):
        st.markdown("""
        - The dataset is a hypothetical, cross-sectional snapshot without a defined future churn window.
        - There is no leakage from fitting scalers/encoders before data splitting; they are fitted inside cross-validation folds.
        - Random splitting evaluates held-out rows, not real temporal drift. Production validation should use time-based splits.
        - A risk model does not measure the causal effect of offering discounts. Retention strategy requires experimentation.
        - Probabilities should be checked for calibration before relying on exact percentage interpretations.
        - Model-feature associations are not causal explanations; consider fairness, subgroup performance, privacy, and drift.
        """)
