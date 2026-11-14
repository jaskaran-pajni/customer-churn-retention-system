# Day 3 — GitHub + Streamlit deployment (Windows / VS Code)

Use **your existing Day 2 model**. The ZIP does **not** include a made-up model or invented scores. Only share a model trained on the **public IBM hypothetical telco dataset**.

## 1. Start in the right project

Extract this project or copy the Day 3 files into your existing `customer-churn-day2` folder. If you already trained Day 2 locally, the simplest option is to **keep that folder**, not overwrite your actual artifacts. Copy `DEPLOY_DAY3.md`, `scripts/`, `.github/`, `.gitignore`, and the updated `README.md` and `.streamlit/config.toml` into that folder. Do not overwrite trained model/metrics. 

Open VS Code in the folder containing `streamlit_app.py`.

Activate your existing environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

We recommend **Python 3.12** for local and Community Cloud to reduce compatibility issues. If you trained using a different Python minor version, you may need to recreate a 3.12 environment, retrain, and pin from that environment.

## 2. Confirm the four generated public-demo artifacts exist

```text
models/churn_pipeline.joblib
outputs/feature_schema.json
outputs/metrics.json
outputs/segment_summary.csv
```

If any are missing:

```powershell
python -m src.download_data
python -m src.train
python -m src.export_dashboard_summary
```

`src.train` evaluates the holdout set and writes **actual** model metrics. `src.export_dashboard_summary` makes aggregate counts/rates. Do not substitute synthetic or made-up scores.

Optional charts (for the Diagnostics tab):

```text
outputs/charts/confusion_matrix.png
outputs/charts/roc_curve.png
```

## 3. Test your local dashboard and deployment readiness

```powershell
python -m src.check_dashboard
python scripts/prepare_deployment.py
python -m pip install pytest
python -m pytest -q tests
python -m streamlit run streamlit_app.py
```

In the open app, check all four tabs and try the six **fictional** customers in the Batch tab. Press Ctrl+C to end the Streamlit process.

## 4. Pin dependencies to the versions that work with YOUR model

The saved scikit-learn `joblib` pipeline may not be portable across mismatched versions. With the **same active environment you used for training**, run:

```powershell
python scripts/prepare_deployment.py --pin-deps
```

This replaces `requirements.txt` with exact installed versions of the direct project dependencies. If you switch to Python 3.12 after training, retrain under 3.12 first, then re-run this command. Select Python **3.12** in the Streamlit deployment settings. If any package is missing, reinstall requirements before pinning.

## 5. Inspect files before publishing

**Public repo may include:** Python source, SQL files, your explanatory notebook, `requirements.txt`, the *public IBM demo* `models/churn_pipeline.joblib`, schema, metrics, aggregate segment summary, optional charts.

**Keep out:** `.venv`, `.env`, `.streamlit/secrets.toml`, raw CSV, local SQLite database, `outputs/test_predictions.csv`, real client data, private customer IDs/uploads, credentials. Even if a private customer dataset looks anonymized, don't commit it without proper clearance. Joblib is pickle-based: **only load artifacts you created or trust**.

From your project root:

```powershell
git init
git add .
git status
```

Verify the four required demo artifacts are **staged** and sensitive files are **not staged**. If you have a public repo with a different purpose, don't push its files here. No CLI push is automatic.

## 6. Publish to GitHub

In GitHub, create a **new empty public repository** named `customer-churn-retention` (do not add a README online because this folder already has one).

Then run from your local project:

```powershell
git commit -m "Build customer churn portfolio app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-retention.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username. If `origin` already exists, inspect `git remote -v` instead of adding a duplicate. If you don't have `git`, install Git for Windows or upload the allowed files via GitHub's browser interface. Check your GitHub repository in the browser to confirm the required model/JSON/CSV paths are present.

## 7. Deploy your Streamlit app

1. Visit https://share.streamlit.io and sign in with GitHub.
2. Choose **Create app** and the option to deploy an **existing app**.
3. Select your GitHub repository, branch `main`, and entrypoint `streamlit_app.py`.
4. Under advanced settings, choose **Python 3.12** (matching your local training environment).
5. Choose an optional memorable app URL; click **Deploy**.
6. Test all four tabs, individual scoring, the six demo customers, and the diagnostics figures.

The app's public URL will be an address ending in `.streamlit.app`. Add the *actual* URL to the README and your résumé after the app works. Do not claim deployment until you can open it in a browser.

## 8. Common errors and quick fixes

| Error | Likely cause | Fix |
| --- | --- | --- |
| Missing model or schema | Generated files not staged/pushed | Run preflight, `git status`, confirm GitHub contains four artifacts |
| `ModuleNotFoundError` | Requirements not installed or wrong root path | Keep `requirements.txt` and `streamlit_app.py` at repository root |
| scikit-learn inconsistent version warning | Different training and cloud dependencies | Rebuild under Python 3.12 and `--pin-deps`; recommit model and requirements |
| App works locally but not online | Cloud doesn't have local files | Commit allowed demo artifacts; don't rely on files on `C:` |
| Charts missing | Optional PNGs not generated or not staged | Run `python -m src.train` and commit only the two allowed chart images |
| GitHub rejects large files | Large binary artifact | Review model size and consider model compression/LFS if truly needed |

## 9. Final résumé bullets (replace placeholders with real results)

**Customer Churn Prediction & Retention Decision System** — Python, SQL, Scikit-learn, Streamlit

- Developed an end-to-end churn prediction pipeline on IBM's 7,043-record demonstration dataset, with leakage-safe preprocessing and stratified five-fold model comparison.
- Evaluated Logistic Regression and Random Forest on a held-out test set, achieving an ROC-AUC of **[YOUR ACTUAL SCORE]** (and precision **[YOUR ACTUAL SCORE]** at 50% threshold).
- Deployed a Streamlit dashboard featuring individual and batch prediction, diagnostic visuals, and scenario-based retention prioritization.

**Use the deployment bullet only after your live app is online.** Retention offer performance is hypothetical, not measured uplift. 

## 10. Interview-ready summary

> "I selected a churn classifier using cross-validated ROC-AUC within the training set and evaluated it once on an untouched test set. I preserved the same preprocessing pipeline at inference, deployed the trained model behind a Streamlit UI, and designed a retention planning simulation using transparent assumptions. The dataset doesn't contain outcomes of retention offers, so I explicitly avoid claiming causal impact."

### Sources

- Streamlit Community Cloud deploy guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Streamlit dependency instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- Streamlit file organization: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
