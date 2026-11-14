# DAY 2 — Interactive Churn Dashboard (Windows / VS Code)

**What you will build:** four working pages in Streamlit: `Overview`, `Single customer`, `Batch & retention`, `Model diagnostics`. You already built the model on Day 1; Day 2 **reuses** its trained `joblib` pipeline instead of fitting on app reruns.

> If you finished Day 1 locally, you can copy `streamlit_app.py`, `src/dashboard_logic.py`, `src/export_dashboard_summary.py`, `.streamlit/config.toml`, and the updated `requirements.txt` into **your existing Day 1 folder**. Alternatively, this ZIP already contains all Day 1 source files as well, so you can use it as a complete replacement, while copying over your locally generated model, data, and outputs.

## 1. Open the project

1. Extract ZIP, open `customer-churn-day2` in VS Code.
2. Open **Terminal → New Terminal** (PowerShell).
3. If you already have a `.venv` in the old project, either continue in that old folder with these new files or create a fresh one here:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.12 is recommended for matching Streamlit Community Cloud; if you have it installed, you can use `py -3.12 -m venv .venv` instead of `py -m venv .venv`.

If activation is blocked, temporarily run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then try again. In VS Code use **Python: Select Interpreter** → `.venv`.

## 2. Make sure Day 1 files are present

The app requires:

- `models/churn_pipeline.joblib` — saved complete preprocessing+classifier.
- `outputs/feature_schema.json` — names/categories from training data.
- `outputs/metrics.json` — recommended to display scores.

If you are on a *new* extracted project, or haven't trained Day 1 yet:

```powershell
python -m src.download_data
python -m src.eda
python -m src.build_database
python -m src.train
```

If you already completed Day 1, and want to **avoid retraining**, copy over your existing `models/churn_pipeline.joblib` and the whole `outputs` folder from your existing Day 1 directory. You may also copy over your raw IBM CSV, but it isn't necessary to run the app.

## 3. Generate aggregate dashboard insights

```powershell
python -m src.export_dashboard_summary
```

This creates `outputs/segment_summary.csv` containing group-level churn rates (no individual records), powering the Overview charts. It requires `data/raw/Telco-Customer-Churn.csv`, which the Day 1 downloader saves.

Before launching the web UI, verify that model inference works:

```powershell
python -m src.check_dashboard
```

## 4. Start the dashboard

```powershell
python -m streamlit run streamlit_app.py
```

Open the **Local URL** displayed by Streamlit (commonly `http://localhost:8501`). Press **Ctrl+C** in the terminal to stop the server. Streamlit will rerun the script on user interactions, but `@st.cache_resource` keeps the same trusted, trained model loaded for subsequent interactions.

## 5. How to demonstrate the app

1. **Overview:** inspect trained model holdout scores and churn differences by contract and tenure. Ask: "Which segments have higher observed churn?"
2. **Single customer:** change `tenure`, contract type, or monthly charges and click **Calculate churn probability**. Show predicted probability, label, and illustrative business value.
3. **Batch & retention:** check **Use six fictional example customers instead** or upload the CSV template. Use sidebar to change `Classification threshold`, `Offer/contact cost`, `Assumed intervention success`, `Gross margin`, and `Value horizon`.
4. **Model diagnostics:** show five-fold CV comparison, holdout ROC-AUC, confusion matrix, and model features.

## 6. Probability vs classification vs decision

```python
probability = model.predict_proba(customer_features)[:, 1]
predicted_label = (probability >= threshold).astype(int)
```

- **Probability:** model estimate of churn (e.g., 0.78).
- **Predicted class:** a binary decision using a threshold (e.g., 1 at threshold 0.50).
- **Retention priority:** a separate *hypothetical* scenario based on probability, assumed effectiveness, gross margin, horizon, and offer cost.

Illustrative net value per customer:

```
P(churn) × assumed_offer_success × monthly_revenue × gross_margin × months − offer_cost
```

**Crucial limitation:** The dataset can estimate churn associations, but provides no intervention outcome. Therefore the app **does not know** whether discounts cause customers to stay. Any success-rate assumption is illustrative, not a measured lift. Demonstrating this distinction in an interview makes the project stronger.

## 7. CSV upload: required columns

All Day 1 model features must appear; `customerID` is optional, `Churn` is optional and is ignored during inference. The app refuses missing *columns* and bad numeric strings, while the trained imputer handles missing *cell values*. Upload a file under 5 MB with no more than 2,000 records. Use the **Download CSV template** button to see exact column spellings.

## 8. Tests

```powershell
python -m pip install pytest
python -m pytest -q tests
```

The tests cover exact feature order, target/ID leakage, bad numeric values, decision values, and scoring. Use the saved pipeline and a **new** dataset if you later want additional real-world integration tests.

## 9. Deploy on Streamlit Community Cloud (optional, Day 3)

1. Run Day 1 training and `python -m src.export_dashboard_summary` locally **using the IBM public demo data only**.
2. Create your own GitHub repository and commit source code **plus** the small generated demo assets:
   - `models/churn_pipeline.joblib`
   - `outputs/feature_schema.json`
   - `outputs/metrics.json`
   - `outputs/segment_summary.csv`
3. Push the repository to GitHub. Set the Streamlit entrypoint to **`streamlit_app.py`**, at the repository root. `requirements.txt` is also at the root.
4. Visit https://share.streamlit.io/ and choose **Create app** → **I have an app** → choose your repo/branch/entrypoint → **Deploy**.
5. **Do not** commit uploaded private customers, private models, `.venv`, `.streamlit/secrets.toml`, or any confidential data. Joblib uses pickle, so only load model artifacts **you trained or explicitly trust**. For deployment, use Python/package versions compatible with your training environment. If the app reports missing model/schema, verify those four files are committed and paths match.

Note: the `.gitignore` lets the four public-demo artifacts be committed while continuing to ignore raw customer files and local intermediate outputs. If you adapt this for private data, change the storage/deployment architecture and permissions first.

## 10. Interview explanation (practice aloud)

> "I built a Streamlit application around the same leakage-safe scikit-learn pipeline I evaluated on Day 1. It takes raw customer features, applies exactly the learned preprocessing, and returns predicted churn probabilities and threshold-based labels. For batch outreach planning, I separated the churn model from an assumption-driven retention-value simulation. I included a model diagnostics tab and documented that the estimated retention uplift is unvalidated because the dataset lacks treatment outcomes."

**Resume line (only when the project is complete):** "Developed a Streamlit decision-support dashboard for individual and batch customer churn scoring, integrating a validated scikit-learn pipeline, model performance diagnostics, and scenario-based retention prioritization."
