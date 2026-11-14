# Start here — Windows / VS Code

1. Extract the ZIP and open the `customer-churn-day1` folder in VS Code.
2. Open Terminal → New Terminal (PowerShell).
3. Set up your Python environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then retry activation.

4. Execute in this order, **from the folder containing `requirements.txt`**:

```powershell
python -m src.download_data
python -m src.eda
python -m src.build_database
python -m src.train
python -m src.predict_sample
```

5. Examine `outputs/eda_summary.txt`, `outputs/sql_results.txt`, `outputs/metrics.json` and the figures in `outputs/charts/`.
6. Read `README.md` for concepts and practice answering its interview questions.

If the download is blocked by your network, manually download the raw original IBM CSV from https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv and save it as `data/raw/Telco-Customer-Churn.csv`.

**Do not claim a model performance number until you have run it against the actual IBM dataset.**
