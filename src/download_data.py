"""Fetch the original IBM example CSV once; no Kaggle account required."""
from urllib.request import urlopen
from urllib.error import URLError
from .common import RAW_CSV, EXPECTED_COLUMNS
import pandas as pd

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def main() -> None:
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists():
        print(f"Dataset already exists: {RAW_CSV}")
        return
    print(f"Downloading original IBM dataset from:\n{URL}")
    tmp_path = RAW_CSV.with_suffix(".part")
    try:
        with urlopen(URL, timeout=30) as response, tmp_path.open("wb") as dest:
            while chunk := response.read(64 * 1024):
                dest.write(chunk)
        sample = pd.read_csv(tmp_path, nrows=5)
        missing = EXPECTED_COLUMNS.difference(sample.columns)
        if missing:
            raise ValueError(f"Download does not match IBM schema: {sorted(missing)}")
        tmp_path.replace(RAW_CSV)
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(
            "Download failed. Manually save the CSV from the original IBM GitHub "
            f"URL into data/raw/Telco-Customer-Churn.csv.\nReason: {exc}"
        )
    print(f"Saved: {RAW_CSV}")


if __name__ == "__main__":
    main()
