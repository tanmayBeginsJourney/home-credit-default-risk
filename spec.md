# HOME CREDIT DEFAULT RISK — PIPELINE SPECIFICATION
**Version:** 2.1 | **Companion:** `.cursorrules`  
**Target:** >= 0.80 OOF ROC-AUC | **Hardware:** 16 GB RAM / 8 GB VRAM

> **Agent:** Before reading anything else, confirm `.cursorrules` is
> loaded and active. The execution contract, OOM rules, and
> Collaborative Loop defined there take precedence over everything here.

---

## TABLE OF CONTENTS

1. [Project Structure](#1-project-structure)
2. [Bootstrap: Schema Generation (First-Run Task)](#2-bootstrap-schema-generation)
3. [Data Schema & Relational Join Map](#3-data-schema--relational-join-map)
4. [Notebook & Terminal Execution Commands](#4-notebook--terminal-execution-commands)
5. [Configuration System (config.py)](#5-configuration-system-configpy)
6. [Pipeline Component Blueprints](#6-pipeline-component-blueprints)
7. [Research-Backed Novel Feature Set](#7-research-backed-novel-feature-set)
8. [Agent Development Loop Reference](#8-agent-development-loop-reference)

---

## 1. Project Structure

The agent must enforce this exact directory layout. Do not create
files outside this structure without explicit user instruction.

```
project_root/
├── .cursorrules              ← Immutable system contract (read-only)
├── spec.md                   ← This file (read-only)
├── schema_sample.txt         ← GENERATED on first run (see §2)
├── experiment.log            ← Runtime log (never delete)
│
├── data/
│   └── raw/                  ← 7 Kaggle CSVs + converted .parquet files
│
├── cache/                    ← Parquet cache for expensive stages
│
├── pipeline/
│   ├── __init__.py
│   ├── config.py             ← ALL parameters live here
│   ├── utils.py              ← Caching, seeding, memory profiling, logging
│   ├── data.py               ← Load, clean, downcast (Polars)
│   ├── aggregations.py       ← Multi-table time-windowed aggs (Polars LazyFrame)
│   ├── features.py           ← Domain feature engineering
│   ├── caafe.py              ← LLM-guided feature generation loop
│   └── model.py              ← Trinity ensemble + OOF eval + adversarial val
│
├── entrypoints/
│   ├── __init__.py
│   ├── run_fast.py           ← DEBUG_MODE=True, 15% sample, 3 folds
│   └── run_full.py           ← Full pipeline for final Kaggle submission
│
└── notebooks/
    └── experiment.ipynb      ← User execution playground
```

---

## 2. Bootstrap: Schema Generation (First-Run Task)

**This is the agent's FIRST task on a fresh project.**  
The `schema_sample.txt` file does not exist yet. The agent must
create and run this script before any feature engineering is proposed.

### Agent Action
Create the file `generate_schema.py` in `project_root/` with the
following content, then issue a HANDOFF BLOCK for the user to run it.

```python
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
```

### HANDOFF BLOCK (Agent must output this after creating the file)
```
┌─────────────────────────────────────────────────────────┐
│  ⚙️  AGENT HANDOFF — USER ACTION REQUIRED               │
├─────────────────────────────────────────────────────────┤
│  Run this in your terminal (mlpr env active):           │
│                                                         │
│    python generate_schema.py                            │
│                                                         │
│  Estimated runtime: < 60 seconds                        │
│  Watch for: "Schema written to schema_sample.txt"       │
│  When done: Confirm schema_sample.txt exists, then      │
│             paste first 20 lines back to agent.         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Data Schema & Relational Join Map

> **Anti-hallucination rule:** The agent MUST verify every column
> name used in features.py or aggregations.py against schema_sample.txt
> before writing code. Never assume a column exists.

### Relational Hierarchy

```
Level 0 — Base (Key: SK_ID_CURR)
  ├── application_train.csv
  └── application_test.csv

Level 1 — Direct join to Level 0 on SK_ID_CURR
  ├── bureau.csv
  └── previous_application.csv

Level 2 — Must be aggregated FIRST, then joined to Level 1
  ├── bureau_balance.csv          → agg → join bureau ON SK_ID_BUREAU
  ├── POS_CASH_balance.csv        → agg → join previous_application ON SK_ID_PREV
  ├── installments_payments.csv   → agg → join previous_application ON SK_ID_PREV
  └── credit_card_balance.csv     → agg → join previous_application ON SK_ID_PREV
```

### Join Execution Order (STRICT)
1. `agg_bureau_balance()` → produces bureau-level features
2. `agg_bureau()` + bureau-level features → merge to application on `SK_ID_CURR`
3. `agg_pos_cash()`, `agg_installments()`, `agg_credit_card()` → prev_app-level features
4. `agg_previous_application()` + above → merge to application on `SK_ID_CURR`
5. `fe_application()` → application-level cross-feature engineering
6. Final matrix → convert to Pandas → model training

---

## 4. Notebook & Terminal Execution Commands

### 4.1 Jupyter Notebook (notebooks/experiment.ipynb)

Use these exact cells. Cell magic `!` runs shell commands.
Using module syntax (`-m`) ensures Python path resolution is correct.

```python
# Cell 1 — Bootstrap (run once on project setup)
!python generate_schema.py
```

```python
# Cell 2 — Fast iteration run (DEBUG_MODE=True, ~15 min)
!python -m entrypoints.run_fast
```

```python
# Cell 3 — Full pipeline run (DEBUG_MODE=False, ~2-4 hrs)
# Only run on Kaggle or when submitting. Do NOT run locally mid-dev.
!python -m entrypoints.run_full
```

```python
# Cell 4 — Tail experiment log (run after any execution)
!tail -50 experiment.log
```

```python
# Cell 5 — Check cache directory
import os
cache_files = sorted(os.listdir("./cache"))
for f in cache_files:
    size_mb = os.path.getsize(f"./cache/{f}") / (1024**2)
    print(f"{f:60s}  {size_mb:6.1f} MB")
```

```python
# Cell 6 — Alternative: import-based run (if shell is unavailable)
import importlib
import entrypoints.run_fast as rf
importlib.reload(rf)
rf.main()
```

### 4.2 Terminal Commands (mlpr env active)

```bash
# Fast dev run
python -m entrypoints.run_fast

# Full run (Kaggle or overnight)
python -m entrypoints.run_full

# Monitor live log output
tail -f experiment.log

# Check memory during run (separate terminal)
watch -n 2 "free -h && ps aux --sort=-%mem | head -5"

# Clear stale cache (use with caution)
rm -rf ./cache/*.parquet
```

### 4.3 Estimated Runtimes (Local, 16 GB RAM)

| Stage                          | DEBUG_MODE=True | DEBUG_MODE=False |
|--------------------------------|-----------------|------------------|
| Schema generation              | < 1 min         | < 1 min          |
| CSV → Parquet conversion       | 2–4 min         | 2–4 min          |
| Bureau aggregation             | 1–2 min         | 5–8 min          |
| Installments aggregation       | 1–2 min         | 8–12 min         |
| Full aggregation pipeline      | 5–8 min         | 25–35 min        |
| LightGBM+CatBoost (3-fold)     | 8–12 min        | 45–70 min        |
| LightGBM+CatBoost (5-fold)     | N/A             | 75–110 min       |
| Optuna HPO (100 trials)        | N/A             | 90–180 min       |

---

## 5. Configuration System (config.py)

**Every parameter lives here. Zero hardcoding elsewhere.**

```python
# pipeline/config.py
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
DATA_DIR  = Path("./data/raw")
CACHE_DIR = Path("./cache")
LOG_FILE  = Path("./experiment.log")

# ── Development Modes ────────────────────────────────────────
DEBUG_MODE   = True   # Agent toggles this for fast iteration
RUN_OPTUNA   = False  # NEVER True while DEBUG_MODE is True

# ── Pipeline Toggles ─────────────────────────────────────────
USE_CAAFE    = False
USE_OPENFE   = False
DROP_ADVERS  = True   # Auto-drop high-drift features post adv. val.
ADV_AUC_THRESHOLD = 0.65  # Threshold to flag train/test drift

# ── Sampling & Folds ─────────────────────────────────────────
N_FOLDS = 5
SEED    = 42

# ── Ensemble Weights (must sum to 1.0) ───────────────────────
LGBM_WEIGHT   = 0.40
CATBOOST_WEIGHT = 0.35
XGB_WEIGHT    = 0.25

# ── Model Defaults (used when RUN_OPTUNA=False) ──────────────
LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}
CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_seed": SEED,
    "verbose": 0,
    "task_type": "CPU",
}
XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

# ── CAAFE ─────────────────────────────────────────────────────
CAAFE_ITERS = 3

# ── Early Stopping ────────────────────────────────────────────
EARLY_STOPPING_ROUNDS = 30

# ── Apply DEBUG overrides ─────────────────────────────────────
if DEBUG_MODE:
    assert not RUN_OPTUNA, "Cannot run Optuna in DEBUG_MODE."
    DEBUG_SAMPLE_FRAC = 0.15
    N_FOLDS = 3
    CAAFE_ITERS = 1
    USE_OPENFE  = False
    LGBM_PARAMS["n_estimators"]     = 300
    CATBOOST_PARAMS["iterations"]   = 300
    XGB_PARAMS["n_estimators"]      = 300
```

---

## 6. Pipeline Component Blueprints

### 6.1 pipeline/utils.py

```python
import os, gc, random, logging
import psutil
from functools import wraps
from pathlib import Path
import numpy as np
import polars as pl
from pipeline import config

logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("home_credit")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    pl.set_random_seed(seed)


def profile_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        proc = psutil.Process(os.getpid())
        mb_before = proc.memory_info().rss / (1024 ** 2)
        result = func(*args, **kwargs)
        mb_after = proc.memory_info().rss / (1024 ** 2)
        logger.info(
            f"[MEM] {func.__name__}: {mb_before:.1f} MB → "
            f"{mb_after:.1f} MB (Δ {mb_after - mb_before:+.1f} MB)"
        )
        return result
    return wrapper


def cache_key(*parts) -> str:
    """Build a deterministic cache filename from config-relevant parts."""
    tag = "_".join(str(p) for p in parts)
    return f"{tag}_debug{int(config.DEBUG_MODE)}_seed{config.SEED}.parquet"


def ensure_parquet_format(data_dir: Path) -> None:
    for csv_path in data_dir.glob("*.csv"):
        pq_path = csv_path.with_suffix(".parquet")
        if not pq_path.exists():
            logger.info(f"Converting {csv_path.name} → parquet...")
            df = pl.read_csv(csv_path, infer_schema_length=100_000)
            df.write_parquet(pq_path)
            size_mb = pq_path.stat().st_size / (1024 ** 2)
            logger.info(f"  → {pq_path.name} ({size_mb:.1f} MB)")
```

### 6.2 pipeline/data.py

```python
import polars as pl
from pipeline import config
from pipeline.utils import logger, profile_memory


@profile_memory
def load_and_clean_application(is_train: bool = True) -> pl.DataFrame:
    fname = "application_train.parquet" if is_train else "application_test.parquet"
    df = pl.read_parquet(config.DATA_DIR / fname)

    if config.DEBUG_MODE and is_train:
        df = df.sample(fraction=config.DEBUG_SAMPLE_FRAC, seed=config.SEED)
        logger.info(f"DEBUG sample: {len(df):,} rows")

    # ── Cast all string cols to Categorical immediately ───────
    cat_cols = [c for c, d in zip(df.columns, df.dtypes) if d == pl.Utf8]
    df = df.with_columns([pl.col(c).cast(pl.Categorical) for c in cat_cols])

    # ── Domain cleaning ───────────────────────────────────────
    df = df.with_columns([
        (pl.col("DAYS_EMPLOYED") == 365243).cast(pl.Int32).alias("DAYS_EMPLOYED_ANOM"),
        pl.when(pl.col("DAYS_EMPLOYED") == 365243)
            .then(None)
            .otherwise(pl.col("DAYS_EMPLOYED"))
            .alias("DAYS_EMPLOYED"),
        pl.when(pl.col("CODE_GENDER") == "XNA")
            .then(None)
            .otherwise(pl.col("CODE_GENDER"))
            .alias("CODE_GENDER"),
    ])

    days_cols = [c for c in df.columns if c.startswith("DAYS_")]
    df = df.with_columns([pl.col(c).abs() for c in days_cols])

    p99 = df.select(pl.col("OWN_CAR_AGE").quantile(0.99)).item()
    df = df.with_columns([
        pl.col("OWN_CAR_AGE").clip(upper_bound=p99),
        pl.col("AMT_INCOME_TOTAL").log1p().alias("AMT_INCOME_TOTAL_LOG"),
    ])

    # ── Downcast immediately ──────────────────────────────────
    df = df.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.Int64).cast(pl.Int32),
    ])

    logger.info(f"Loaded {'train' if is_train else 'test'}: {df.shape}")
    return df
```

### 6.3 pipeline/aggregations.py

```python
import gc
import polars as pl
from pathlib import Path
from pipeline import config
from pipeline.utils import logger, profile_memory, cache_key

CACHE = config.CACHE_DIR


@profile_memory
def agg_bureau(bureau_path: Path, bureau_bal_path: Path) -> pl.DataFrame:
    cache_path = CACHE / cache_key("bureau")
    if cache_path.exists():
        logger.info(f"[CACHE HIT] {cache_path.name}")
        return pl.read_parquet(cache_path)

    bb = pl.scan_parquet(bureau_bal_path)
    bb_agg = bb.group_by("SK_ID_BUREAU").agg([
        pl.len().alias("BB_MONTHS_COUNT"),
        (pl.col("STATUS") == "C").mean().alias("BB_STATUS_C_FRAC"),
        (pl.col("STATUS").is_in(["1","2","3","4","5"])).sum().alias("BB_STATUS_LATE_COUNT"),
    ])

    bur = pl.scan_parquet(bureau_path).join(bb_agg, on="SK_ID_BUREAU", how="left")

    active = bur.filter(pl.col("CREDIT_ACTIVE") == "Active")
    active_agg = active.group_by("SK_ID_CURR").agg([
        pl.col("AMT_CREDIT_SUM").sum().alias("ACTIVE_CREDIT_SUM"),
        pl.len().alias("ACTIVE_CREDIT_COUNT"),
    ])

    result = bur.group_by("SK_ID_CURR").agg([
        pl.col("DAYS_CREDIT").min().alias("BUREAU_DAYS_CREDIT_MIN"),
        pl.col("DAYS_CREDIT").mean().alias("BUREAU_DAYS_CREDIT_MEAN"),
        pl.col("AMT_CREDIT_SUM").sum().alias("BUREAU_AMT_CREDIT_SUM"),
        pl.col("AMT_CREDIT_SUM").mean().alias("BUREAU_AMT_CREDIT_MEAN"),
        pl.col("AMT_CREDIT_SUM_DEBT").sum().alias("BUREAU_DEBT_SUM"),
        pl.col("CREDIT_DAY_OVERDUE").max().alias("BUREAU_MAX_OVERDUE"),
        pl.col("CNT_CREDIT_PROLONG").sum().alias("BUREAU_PROLONG_SUM"),
        (pl.col("CREDIT_ACTIVE") == "Active").sum().alias("BUREAU_ACTIVE_COUNT"),
        pl.col("BB_MONTHS_COUNT").mean().alias("BUREAU_BB_MONTHS_MEAN"),
        pl.col("BB_STATUS_LATE_COUNT").sum().alias("BUREAU_LATE_STATUS_SUM"),
    ]).join(active_agg, on="SK_ID_CURR", how="left")

    out = result.collect()
    out = out.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.Int64).cast(pl.Int32),
    ])
    out.write_parquet(cache_path)
    logger.info(f"[CACHE WRITE] {cache_path.name}: {out.shape}")
    return out


@profile_memory
def agg_installments(inst_path: Path) -> pl.DataFrame:
    cache_path = CACHE / cache_key("installments")
    if cache_path.exists():
        logger.info(f"[CACHE HIT] {cache_path.name}")
        return pl.read_parquet(cache_path)

    inst = pl.scan_parquet(inst_path).with_columns([
        (pl.col("AMT_INSTALMENT") - pl.col("AMT_PAYMENT")).alias("PAYMENT_DIFF"),
        (pl.col("AMT_PAYMENT") / (pl.col("AMT_INSTALMENT") + 1e-8)).alias("PAYMENT_RATIO"),
        (pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")).alias("DAYS_LATE"),
        (pl.col("DAYS_ENTRY_PAYMENT") > pl.col("DAYS_INSTALMENT")).alias("IS_LATE"),
    ])

    out = inst.group_by("SK_ID_CURR").agg([
        pl.col("PAYMENT_DIFF").mean().alias("INST_PAYMENT_DIFF_MEAN"),
        pl.col("PAYMENT_DIFF").sum().alias("INST_PAYMENT_DIFF_SUM"),
        pl.col("PAYMENT_RATIO").mean().alias("INST_PAYMENT_RATIO_MEAN"),
        pl.col("DAYS_LATE").max().alias("INST_DAYS_LATE_MAX"),
        pl.col("DAYS_LATE").mean().alias("INST_DAYS_LATE_MEAN"),
        pl.col("IS_LATE").mean().alias("INST_LATE_FRAC"),
        pl.col("AMT_INSTALMENT").max().alias("INST_MAX_INSTALMENT"),
        pl.col("AMT_PAYMENT").sum().alias("INST_TOTAL_PAID"),
    ]).collect()

    out = out.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.Int64).cast(pl.Int32),
    ])
    out.write_parquet(cache_path)
    logger.info(f"[CACHE WRITE] {cache_path.name}: {out.shape}")
    return out


@profile_memory
def agg_credit_card(cc_path: Path) -> pl.DataFrame:
    cache_path = CACHE / cache_key("credit_card")
    if cache_path.exists():
        logger.info(f"[CACHE HIT] {cache_path.name}")
        return pl.read_parquet(cache_path)

    cc = pl.scan_parquet(cc_path).filter(
        pl.col("AMT_CREDIT_LIMIT_ACTUAL") > 0
    ).with_columns([
        (pl.col("AMT_BALANCE") / (pl.col("AMT_CREDIT_LIMIT_ACTUAL") + 1e-8)).alias("UTIL_RATIO"),
    ])

    recent = cc.filter(pl.col("MONTHS_BALANCE") >= -12)
    recent_agg = recent.group_by("SK_ID_CURR").agg([
        pl.col("UTIL_RATIO").mean().alias("CC_UTIL_MEAN_12M"),
        pl.col("UTIL_RATIO").quantile(0.95).alias("CC_UTIL_P95_12M"),
        pl.col("AMT_DRAWINGS_CURRENT").sum().alias("CC_DRAWINGS_12M"),
    ])

    out = cc.group_by("SK_ID_CURR").agg([
        pl.col("UTIL_RATIO").mean().alias("CC_UTIL_MEAN_ALL"),
        pl.col("UTIL_RATIO").max().alias("CC_UTIL_MAX_ALL"),
        pl.col("AMT_PAYMENT_CURRENT").mean().alias("CC_PAYMENT_MEAN"),
        pl.col("SK_DPD").max().alias("CC_DPD_MAX"),
        pl.col("SK_DPD").mean().alias("CC_DPD_MEAN"),
        pl.col("CNT_DRAWINGS_CURRENT").sum().alias("CC_DRAWINGS_COUNT"),
    ]).join(recent_agg, on="SK_ID_CURR", how="left").collect()

    out = out.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.Int64).cast(pl.Int32),
    ])
    out.write_parquet(cache_path)
    logger.info(f"[CACHE WRITE] {cache_path.name}: {out.shape}")
    return out


@profile_memory
def agg_pos_cash(pos_path: Path) -> pl.DataFrame:
    cache_path = CACHE / cache_key("pos_cash")
    if cache_path.exists():
        logger.info(f"[CACHE HIT] {cache_path.name}")
        return pl.read_parquet(cache_path)

    pos = pl.scan_parquet(pos_path)

    out = pos.group_by("SK_ID_CURR").agg([
        pl.col("SK_DPD").max().alias("POS_DPD_MAX"),
        pl.col("SK_DPD").mean().alias("POS_DPD_MEAN"),
        pl.col("SK_DPD_DEF").max().alias("POS_DPD_DEF_MAX"),
        pl.col("CNT_INSTALMENT_FUTURE").mean().alias("POS_CNT_INSTALMENT_FUTURE_MEAN"),
        pl.col("MONTHS_BALANCE").count().alias("POS_MONTHS_COUNT"),
        (pl.col("NAME_CONTRACT_STATUS") == "Completed").mean().alias("POS_COMPLETED_FRAC"),
    ]).collect()

    out = out.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.Int64).cast(pl.Int32),
    ])
    out.write_parquet(cache_path)
    logger.info(f"[CACHE WRITE] {cache_path.name}: {out.shape}")
    return out


@profile_memory
def agg_previous_application(prev_path: Path) -> pl.DataFrame:
    cache_path = CACHE / cache_key("prev_app")
    if cache_path.exists():
        logger.info(f"[CACHE HIT] {cache_path.name}")
        return pl.read_parquet(cache_path)

    prev = pl.scan_parquet(prev_path)

    recent = prev.filter(pl.col("DAYS_DECISION") >= -730)
    recent_status = recent.group_by(["SK_ID_CURR", "NAME_CONTRACT_STATUS"]).agg(
        pl.len().alias("CNT")
    ).collect().pivot(
        values="CNT", index="SK_ID_CURR", on="NAME_CONTRACT_STATUS"
    ).fill_null(0)

    # Rename columns safely (only those that exist)
    rename_map = {}
    for s in ["Approved", "Refused", "Canceled", "Unused offer"]:
        col = s.replace(" ", "_")
        if s in recent_status.columns:
            rename_map[s] = f"PREV_{col}_CNT_2Y"
    if rename_map:
        recent_status = recent_status.rename(rename_map)

    out = prev.group_by("SK_ID_CURR").agg([
        pl.col("AMT_CREDIT").mean().alias("PREV_AMT_CREDIT_MEAN"),
        pl.col("AMT_ANNUITY").mean().alias("PREV_AMT_ANNUITY_MEAN"),
        pl.col("AMT_DOWN_PAYMENT").mean().alias("PREV_DOWN_PAYMENT_MEAN"),
        pl.col("DAYS_DECISION").max().alias("PREV_DAYS_DECISION_MAX"),
        pl.col("CNT_PAYMENT").mean().alias("PREV_CNT_PAYMENT_MEAN"),
        pl.len().alias("PREV_TOTAL_APPS"),
        (pl.col("NAME_CONTRACT_STATUS") == "Refused")
            .mean().alias("PREV_REFUSED_FRAC"),
        pl.col("RATE_DOWN_PAYMENT").mean().alias("PREV_RATE_DOWN_MEAN"),
    ]).collect()

    out = out.join(
        pl.from_pandas(recent_status.to_pandas()),
        on="SK_ID_CURR", how="left"
    )
    out = out.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.Int64).cast(pl.Int32),
    ])
    out.write_parquet(cache_path)
    logger.info(f"[CACHE WRITE] {cache_path.name}: {out.shape}")
    return out
```

### 6.4 pipeline/features.py

```python
import polars as pl
from pipeline.utils import logger, profile_memory


@profile_memory
def fe_application(df: pl.DataFrame) -> pl.DataFrame:
    """Application-level feature engineering. All Polars expressions."""

    df = df.with_columns([
        # ── EXT SOURCE interactions ───────────────────────────
        (pl.col("EXT_SOURCE_1") * pl.col("EXT_SOURCE_2") * pl.col("EXT_SOURCE_3"))
            .alias("EXT_SRC_PROD"),
        pl.mean_horizontal(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
            .alias("EXT_SRC_MEAN"),
        pl.min_horizontal(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
            .alias("EXT_SRC_MIN"),

        # ── Macro Capacity (DTI) ──────────────────────────────
        (pl.col("AMT_CREDIT") / (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("CREDIT_INCOME_RATIO"),
        (pl.col("AMT_ANNUITY") / (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("ANNUITY_INCOME_RATIO"),
        (pl.col("AMT_GOODS_PRICE") / (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("GOODS_INCOME_RATIO"),

        # ── Loan-to-Value Proxy ───────────────────────────────
        (pl.col("AMT_CREDIT") / (pl.col("AMT_GOODS_PRICE") + 1))
            .alias("GOODS_CREDIT_RATIO"),

        # ── Downpayment Strain ────────────────────────────────
        ((pl.col("AMT_GOODS_PRICE") - pl.col("AMT_CREDIT")) /
         (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("DOWNPAYMENT_INCOME_STRAIN"),

        # ── Credit Velocity ───────────────────────────────────
        (pl.col("AMT_REQ_CREDIT_BUREAU_MON") /
         ((pl.col("AMT_REQ_CREDIT_BUREAU_YEAR") / 12) + 1e-8))
            .alias("CREDIT_SEEK_VELOCITY"),

        # ── Repayment burden ─────────────────────────────────
        (pl.col("AMT_ANNUITY") * 12 / (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("ANNUAL_ANNUITY_INCOME_RATIO"),

        # ── Employment quality ────────────────────────────────
        (pl.col("DAYS_EMPLOYED") / (pl.col("DAYS_BIRTH") + 1))
            .alias("EMPLOYMENT_DURATION_FRAC"),

        # ── Document completeness proxy ───────────────────────
        pl.sum_horizontal([
            pl.col(c).is_not_null().cast(pl.Int32)
            for c in [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)
                      if f"FLAG_DOCUMENT_{i}" in df.columns]
        ]).alias("DOCUMENT_COUNT"),
    ])

    logger.info(f"fe_application: {df.shape[1]} cols after FE")
    return df


@profile_memory
def fe_bureau_derived(df: pl.DataFrame) -> pl.DataFrame:
    """Derived features from bureau aggregates already merged into df."""

    if "ACTIVE_CREDIT_SUM" not in df.columns:
        logger.warning("fe_bureau_derived: bureau aggs not merged, skipping.")
        return df

    df = df.with_columns([
        # ── Time-decayed DTI ──────────────────────────────────
        ((pl.col("ACTIVE_CREDIT_SUM") + pl.col("AMT_CREDIT")) /
         (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("TOTAL_DTI"),

        # ── Fractional debt acceleration rate ────────────────
        (pl.col("BUREAU_AMT_CREDIT_SUM") /
         (pl.col("BUREAU_AMT_CREDIT_MEAN") * 2 + 1e-8))
            .alias("DEBT_ACCEL_RATE"),

        # ── Late payment to active credit ratio ──────────────
        (pl.col("INST_LATE_FRAC") /
         (pl.col("BUREAU_ACTIVE_COUNT").cast(pl.Float32) + 1e-8))
            .alias("LATE_TO_ACTIVE_RATIO"),

        # ── Annuity to historical peak installment ────────────
        (pl.col("AMT_ANNUITY") /
         (pl.col("INST_MAX_INSTALMENT") + 1e-8))
            .alias("ANNUITY_TO_PEAK_INST_RATIO"),
    ])
    return df
```

### 6.5 pipeline/model.py

```python
import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pipeline import config
from pipeline.utils import logger


def _adversarial_validation(X_train: pd.DataFrame, X_test: pd.DataFrame) -> float:
    X_adv = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_adv = np.array([0] * len(X_train) + [1] * len(X_test))
    cat_cols = X_train.select_dtypes(include=["category"]).columns.tolist()
    for c in cat_cols:
        X_adv[c] = X_adv[c].astype(str)

    adv_model = lgb.LGBMClassifier(n_estimators=200, random_state=config.SEED, verbose=-1)
    adv_model.fit(X_adv, y_adv)
    auc = roc_auc_score(y_adv, adv_model.predict_proba(X_adv)[:, 1])
    logger.info(f"Adversarial Validation AUC: {auc:.4f}")
    if auc > config.ADV_AUC_THRESHOLD:
        logger.warning(f"HIGH DRIFT DETECTED — Adversarial AUC {auc:.4f} > {config.ADV_AUC_THRESHOLD}")
    return auc


def evaluate_model(
    X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame
) -> tuple[float, list[str]]:

    _adversarial_validation(X, X_test)

    skf   = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    cat_cols = X.select_dtypes(include=["category"]).columns.tolist()
    oof_preds = np.zeros(len(X))
    fold_aucs: list[float] = []
    zero_imp_counts: dict[str, int] = {}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        # ── LightGBM ──────────────────────────────────────────
        lgb_m = lgb.LGBMClassifier(**config.LGBM_PARAMS)
        lgb_m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False),
                       lgb.log_evaluation(-1)],
        )

        # ── CatBoost ──────────────────────────────────────────
        cb_m = cb.CatBoostClassifier(**config.CATBOOST_PARAMS, cat_features=cat_cols)
        cb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                 early_stopping_rounds=config.EARLY_STOPPING_ROUNDS)

        # ── XGBoost ───────────────────────────────────────────
        X_tr_xgb  = X_tr.copy();  X_tr_xgb[cat_cols]  = X_tr_xgb[cat_cols].apply(lambda c: c.cat.codes)
        X_val_xgb = X_val.copy(); X_val_xgb[cat_cols] = X_val_xgb[cat_cols].apply(lambda c: c.cat.codes)
        xgb_m = xgb.XGBClassifier(**config.XGB_PARAMS)
        xgb_m.fit(X_tr_xgb, y_tr,
                  eval_set=[(X_val_xgb, y_val)],
                  early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
                  verbose=False)

        # ── Ensemble fold predictions ─────────────────────────
        p_lgb = lgb_m.predict_proba(X_val)[:, 1]
        p_cb  = cb_m.predict_proba(X_val)[:, 1]
        p_xgb = xgb_m.predict_proba(X_val_xgb)[:, 1]
        oof_preds[val_idx] = (
            config.LGBM_WEIGHT    * p_lgb  +
            config.CATBOOST_WEIGHT * p_cb  +
            config.XGB_WEIGHT     * p_xgb
        )
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_aucs.append(fold_auc)
        logger.info(f"Fold {fold}/{config.N_FOLDS} AUC: {fold_auc:.5f}")

        # ── Track zero-importance (LightGBM only) ────────────
        for feat, imp in zip(X.columns, lgb_m.feature_importances_):
            if imp == 0:
                zero_imp_counts[feat] = zero_imp_counts.get(feat, 0) + 1

        del X_tr, X_val, X_tr_xgb, X_val_xgb, lgb_m, cb_m, xgb_m
        gc.collect()

    final_auc = float(roc_auc_score(y, oof_preds))
    logger.info(f"OOF Ensemble AUC (all folds): {final_auc:.5f}")
    logger.info(f"Per-fold AUCs: {[round(a,5) for a in fold_aucs]}")

    # Zero-importance = 0 importance in ALL folds
    zero_imp_feats = [f for f, cnt in zero_imp_counts.items() if cnt == config.N_FOLDS]
    logger.info(f"Zero-importance features ({len(zero_imp_feats)}): {zero_imp_feats[:20]}")

    return final_auc, zero_imp_feats
```

### 6.6 entrypoints/run_fast.py

```python
import gc
from pathlib import Path
from pipeline import config
from pipeline.utils import seed_everything, ensure_parquet_format, logger
from pipeline.data import load_and_clean_application
from pipeline.features import fe_application, fe_bureau_derived
from pipeline.aggregations import (
    agg_bureau, agg_installments, agg_credit_card,
    agg_pos_cash, agg_previous_application,
)
from pipeline.model import evaluate_model
import polars as pl


def main() -> None:
    seed_everything(config.SEED)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_parquet_format(config.DATA_DIR)

    logger.info(
        f"=== Pipeline Start | DEBUG={config.DEBUG_MODE} | "
        f"FOLDS={config.N_FOLDS} | SEED={config.SEED} ==="
    )

    # ── Load base tables ──────────────────────────────────────
    train = load_and_clean_application(is_train=True)
    test  = load_and_clean_application(is_train=False)

    # ── Aggregations ──────────────────────────────────────────
    D = config.DATA_DIR
    bur_feats  = agg_bureau(D/"bureau.parquet", D/"bureau_balance.parquet")
    inst_feats = agg_installments(D/"installments_payments.parquet")
    cc_feats   = agg_credit_card(D/"credit_card_balance.parquet")
    pos_feats  = agg_pos_cash(D/"POS_CASH_balance.parquet")
    prev_feats = agg_previous_application(D/"previous_application.parquet")

    # ── Merge all agg tables ──────────────────────────────────
    for agg_df, tag in [
        (bur_feats, "bureau"), (inst_feats, "installments"),
        (cc_feats, "credit_card"), (pos_feats, "pos_cash"),
        (prev_feats, "prev_app"),
    ]:
        train = train.join(agg_df, on="SK_ID_CURR", how="left")
        test  = test.join(agg_df,  on="SK_ID_CURR", how="left")
        del agg_df
        gc.collect()
        logger.info(f"Merged {tag}: train={train.shape}, test={test.shape}")

    # ── Feature engineering ───────────────────────────────────
    train = fe_application(train)
    test  = fe_application(test)
    train = fe_bureau_derived(train)
    test  = fe_bureau_derived(test)

    # ── Convert to Pandas for modeling ────────────────────────
    logger.info("Converting Polars → Pandas...")
    train_pd = train.to_pandas()
    test_pd  = test.to_pandas()
    del train, test
    gc.collect()

    y_train = train_pd.pop("TARGET")
    X_train = train_pd.drop(columns=["SK_ID_CURR"], errors="ignore")
    X_test  = test_pd.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")
    del train_pd, test_pd
    gc.collect()

    # ── Evaluate ──────────────────────────────────────────────
    logger.info("Starting model evaluation...")
    auc, zero_feats = evaluate_model(X_train, y_train, X_test)

    if zero_feats:
        logger.info(f"ACTION REQUIRED: Drop {len(zero_feats)} zero-importance features.")

    logger.info(f"=== Pipeline Complete | Final OOF AUC: {auc:.5f} ===")
    return auc


if __name__ == "__main__":
    main()
```

---

## 7. Research-Backed Novel Feature Set

> **Implementation rule:** Verify every column name in
> `schema_sample.txt` before coding. All features use Polars syntax.
> Implement in `features.py` (application-level) or `aggregations.py`
> (cross-table). Add ONE feature family per iteration and validate.

### Priority Tier 1 — Macro Capacity

| Feature | Source | Specification |
|---------|--------|---------------|
| **Total DTI** | bureau + application | Filter bureau for `CREDIT_ACTIVE == 'Active'`. Sum `AMT_CREDIT_SUM` per `SK_ID_CURR`. Add current `AMT_CREDIT`. Divide by `AMT_INCOME_TOTAL`. |
| **Time-Decayed DTI** | bureau | As above, but multiply `AMT_CREDIT_SUM` by `exp(-0.01 × abs(DAYS_CREDIT))` before summing. |
| **Goods-to-Credit Gap** | application | `AMT_CREDIT / AMT_GOODS_PRICE` |
| **Downpayment Strain** | application | `(AMT_GOODS_PRICE - AMT_CREDIT) / AMT_INCOME_TOTAL` |

### Priority Tier 2 — Liquidity Stress

| Feature | Source | Specification |
|---------|--------|---------------|
| **Revolving Utilization (P95)** | credit_card_balance | Filter `AMT_CREDIT_LIMIT_ACTUAL > 0`. Compute `AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL`. Group by `SK_ID_CURR` → 95th percentile and time-weighted mean over last 12 months. |
| **Fractional Payment Flag** | installments_payments | Flag rows where `0 < AMT_PAYMENT < AMT_INSTALMENT`. Sum per `SK_ID_CURR`. |
| **Temporal Installment Deficit** | installments_payments | Filter `DAYS_INSTALMENT > -1000`. Compute `AMT_PAYMENT - AMT_INSTALMENT`. Mean of means per `SK_ID_CURR`. |

### Priority Tier 3 — Temporal Risk

| Feature | Source | Specification |
|---------|--------|---------------|
| **EWMA Recency Decay** | installments_payments | `DAYS_LATE × exp(-0.01 × abs(DAYS_INSTALMENT))`. Penalises recent delays exponentially. |
| **DPD Acceleration (Markov)** | POS_CASH_balance | Max `SK_DPD` last 6 months minus max `SK_DPD` prior to 6 months. |
| **Sequential POS-Cash EMA** | POS_CASH_balance | Sort by `SK_ID_CURR`, `MONTHS_BALANCE`. EMA of `SK_DPD` per `SK_ID_CURR` with 3-month half-life. |

### Priority Tier 4 — Behavioral Signals

| Feature | Source | Specification |
|---------|--------|---------------|
| **Refusal Ratio** | previous_application | Count `NAME_CONTRACT_STATUS == 'Refused'` / total applications per `SK_ID_CURR`. |
| **Approved vs. Refused Velocity (2Y)** | previous_application | Filter `DAYS_DECISION >= -730`. Pivot on `NAME_CONTRACT_STATUS`. Compute `Refused / Total`. |
| **Loan Restructuring Flag** | installments_payments | Count distinct `NUM_INSTALMENT_VERSION` per `SK_ID_PREV`. Flag if > 2. |
| **Credit-Seeking Velocity** | application | `AMT_REQ_CREDIT_BUREAU_MON / (AMT_REQ_CREDIT_BUREAU_YEAR / 12)` |

### Priority Tier 5 — Behavioral Competence

| Feature | Source | Specification |
|---------|--------|---------------|
| **Annuity to Peak Installment** | installments + application | `max(AMT_INSTALMENT)` per `SK_ID_CURR`. Then `AMT_ANNUITY / max_instalment`. |
| **Late Payment to Active Credit** | installments + bureau | `late_fraction / active_credit_count`. Penalises inability to manage multiple active credits. |
| **Fractional Debt Acceleration** | bureau | `recent_debt (DAYS_CREDIT >= -365)` / `(historical_debt (DAYS_CREDIT in [-1095, -365]) / 2)`. |

### Priority Tier 6 — Advanced / Last-Mile

| Feature | Source | Specification |
|---------|--------|---------------|
| **KNN Target Imputation** | EXT_SOURCE + application | Use `EXT_SOURCE_1/2/3` + `AMT_CREDIT/AMT_ANNUITY` to build KD-Tree. For each row, mean TARGET of 500 nearest training neighbors. **Only add to training set post-split.** |

---

## 8. Agent Development Loop Reference

This mirrors §2 of `.cursorrules` for quick in-context reference.

```
PROFILE  → Read experiment.log + config.py + schema_sample.txt
PROPOSE  → State ONE change, expected AUC impact, target file(s)
MODIFY   → Edit the file; keep changes isolated
HANDOFF  → Output the HANDOFF BLOCK; wait for user to run & paste logs
EVALUATE → Parse logs; accept if ΔAUC > +0.0001; revert if negative/OOM
PRUNE    → Drop zero-importance features immediately
ITERATE  → Return to PROFILE
```

### Decision Tree for OOM
```
OOM occurred?
  └─ YES → 1. Revert last change
            2. Add chunking OR reduce DEBUG_SAMPLE_FRAC
            3. Add explicit del + gc.collect() after each merge
            4. Re-propose the safe version
  └─ NO  → Continue iteration
```

### AUC Milestone Targets

| Stage | Expected OOF AUC |
|-------|-----------------|
| Baseline (application only, LightGBM) | 0.745 – 0.760 |
| + Bureau aggregations | 0.768 – 0.778 |
| + All table aggregations | 0.778 – 0.790 |
| + Novel feature set (Tier 1–3) | 0.790 – 0.800 |
| + Full Trinity ensemble + Tier 4–5 | 0.800 – 0.810 |
| + Optuna HPO + KNN imputation | 0.810 – 0.820 |

---

*End of spec.md — companion file: `.cursorrules`*
