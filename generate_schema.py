# generate_schema.py
# Run once: python generate_schema.py
# Produces schema_sample.txt — the agent's sole data reference.

import polars as pl
from pathlib import Path

DATA_DIR = Path("./data/raw")
OUTPUT_FILE = Path("./schema_sample.txt")

TABLE_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
]

with OUTPUT_FILE.open("w") as f:
    for fname in TABLE_FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            f.write(f"\n{'='*60}\n{fname}: FILE NOT FOUND\n")
            continue

        df = pl.read_csv(fpath, n_rows=5, infer_schema_length=10000)
        f.write(f"\n{'='*60}\n")
        f.write(f"TABLE: {fname}\n")
        f.write(f"Shape (sample): {df.shape}\n\n")
        f.write("--- dtypes ---\n")
        for col, dtype in zip(df.columns, df.dtypes):
            f.write(f"  {col}: {dtype}\n")
        f.write("\n--- head(5) ---\n")
        f.write(df.to_pandas().to_string(index=False))
        f.write("\n")

print(f"Schema written to {OUTPUT_FILE}")
