"""Checks outputs/submission.csv against the rules and builds submission_package/ :
    python make_submission.py
Creates submission_package/submission.csv and submission_package/source.zip (code only)."""
import shutil
import sys
import zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "submission_package"
INCLUDE = ["pipeline", "dashboard", ".streamlit", "outputs", "run_pipeline.py", "make_submission.py",
           "requirements.txt", "README.md", ".gitignore"]   # outputs/ lets reviewers open the dashboard without the raw data
SKIP = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def check_submission():
    sub = pd.read_csv(ROOT / "outputs" / "submission.csv")
    test = pd.read_csv(ROOT / "data" / "raw" / "test.csv")
    problems = []
    if list(sub.columns) != ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]:
        problems.append(f"columns are {list(sub.columns)}")
    if len(sub) != len(test):
        problems.append(f"{len(sub)} rows, test.csv has {len(test)}")
    if sub["Units_Sold_Forecast"].isna().any():
        problems.append("blank forecasts")
    if (sub["Units_Sold_Forecast"] < 0).any() or (sub["Units_Sold_Forecast"] % 1 != 0).any():
        problems.append("forecasts must be non-negative integers")
    if sub.duplicated(["Date", "SKU_ID", "Region"]).any():
        problems.append("duplicate keys")
    k = lambda d: set(zip(d["Date"].astype(str), d["SKU_ID"].astype(str).str.strip(), d["Region"].astype(str).str.strip()))
    if len(k(test) ^ k(sub)) > 0:
        problems.append("keys do not match test.csv (check SKU/region spelling in test.csv vs the submission)")
    return sub, problems


def build_zip():
    PKG.mkdir(exist_ok=True)
    with zipfile.ZipFile(PKG / "source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            p = ROOT / item
            if not p.exists():
                print(f"  (skipped, not found: {item})")
                continue
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
            for f in files:
                if SKIP & set(f.parts) or f.suffix == ".pyc":
                    continue
                z.write(f, f.relative_to(ROOT))
        z.write(ROOT / "outputs" / "submission.csv", "submission.csv")      # prediction file also at the top level of the zip
    return PKG / "source.zip"


if __name__ == "__main__":
    sub, problems = check_submission()
    if problems:
        print("SUBMISSION PROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    shutil.copy(ROOT / "outputs" / "submission.csv", PKG.mkdir(exist_ok=True) or PKG / "submission.csv")
    z = build_zip()
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
    print(f"submission.csv OK: {len(sub):,} rows, total {int(sub['Units_Sold_Forecast'].sum()):,} units")
    mb = z.stat().st_size / 1048576
    print(f"source.zip: {len(names)} files, {mb:.1f} MB" + ("   <-- OVER the 50 MB form limit!" if mb > 50 else "   (limit 50 MB)"))
    print("Upload both files from:", PKG)