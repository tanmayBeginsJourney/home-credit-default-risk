# pipeline/config.py
# LOCAL PERF NOTE: cache rebuild < 5 sec, full run ~ 90-120 sec (DEBUG).
# Never skip a feature to avoid cache invalidation — it is free.
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
DATA_DIR  = Path("./data/raw")
CACHE_DIR = Path("./cache")
# Single project log (append). Do not add alternate log paths — all runs go here.
LOG_FILE = Path("./experiment.log")

# ── Development Modes ────────────────────────────────────────
# `apply_mode_debug` / `apply_mode_full` (called from run_fast / run_full) set
# DEBUG_MODE, N_FOLDS, and tree budget. Do not rely on this module's initial
# DEBUG_MODE for behavior — use the right entrypoint.
#   run_fast → 15% sample, 3 folds, no Optuna
#   run_full → 100% data, 5-fold OOF (set RUN_OPTUNA True here for HPO on full data)
DEBUG_MODE = False
RUN_OPTUNA = False
DEBUG_SAMPLE_FRAC = 0.15

# ── Pipeline Toggles ─────────────────────────────────────────
USE_CAAFE    = False
USE_OPENFE   = False
DROP_ADVERS  = True   # Auto-drop high-drift features post adv. val.
ADV_AUC_THRESHOLD = 0.65  # Threshold to flag train/test drift

# ── Sampling & Folds ─────────────────────────────────────────
N_FOLDS = 5
SEED    = 42

# ── Ensemble Weights (must sum to 1.0) ───────────────────────
LGBM_WEIGHT     = 0.40
CATBOOST_WEIGHT = 0.35
XGB_WEIGHT      = 0.25

# ── Model Defaults (used when RUN_OPTUNA=False) ──────────────
# LGBM row tuned: Optuna study inner AUC 0.77676 (experiment.log 2026-04-24, subsample CV).
LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.031823148246266635,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 88,
    "subsample": 0.6531367953827713,
    "colsample_bytree": 0.44449518484544653,
    "reg_alpha": 8.394296984616961,
    "reg_lambda": 0.005560393707943842,
    "random_state": SEED,
    "device": "gpu",
    "verbose": -1,
}
CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3,
    "random_seed": SEED,
    "verbose": 0,
    "task_type": "GPU",
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
    "device": "cuda",
    "verbosity": 0,
}

# ── Zero-importance features to drop (updated each run) ───────
# Confirmed zero-importance across all 3 folds (DEBUG run 1).
# Re-run to surface remaining 5 (log was truncated at 20).
COLS_TO_DROP: list[str] = [
    # Run 1 — zero-importance across all 3 folds
    "FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_CONT_MOBILE",
    "FONDKAPREMONT_MODE", "NAME_INCOME_TYPE",
    "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5",
    "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10",
    "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
    "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
    "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20",
    # Run 2 — zero-importance across all 3 folds
    "NAME_HOUSING_TYPE", "EMERGENCYSTATE_MODE", "FLAG_DOCUMENT_21",
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
    "DOCUMENT_COUNT",
    # Run 3 — zero-importance across all 3 folds
    "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "DAYS_EMPLOYED_ANOM",
    # Run 4 — zero-importance across all 3 folds
    "REG_REGION_NOT_LIVE_REGION",
    # Run 12 — zero-importance across all 3 folds
    "FLAG_EMAIL",
    # Zero importance after meta-features
    "BUREAU_BB_STATUS_3_COUNT_SUM", "BUREAU_BB_STATUS_4_COUNT_SUM", "BUREAU_BB_STATUS_5_COUNT_SUM",
]

# ── KNN Target Imputation (Tier 6) ───────────────────────
USE_KNN         = False
KNN_N_NEIGHBORS = 500
KNN_COLS        = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# ── CAAFE ─────────────────────────────────────────────────────
CAAFE_ITERS = 3

# ── Optuna ────────────────────────────────────────────────────
OPTUNA_N_TRIALS       = 60   # Trials per model (LGBM → CatBoost → XGBoost)
OPTUNA_SUBSAMPLE_FRAC = 1.0  # Fraction of train rows per trial (100% for highest fidelity)
OPTUNA_N_FOLDS        = 5    # CV folds inside each Optuna trial
# After LGBM Optuna completes, set True to reuse LGBM_PARAMS and only tune CatBoost + XGBoost.
OPTUNA_SKIP_LGBM_STUDY = False

# ── Early Stopping ────────────────────────────────────────────
EARLY_STOPPING_ROUNDS = 30

# ── Target Encoding ───────────────────────────────────────────
# Applied to LGBM + XGBoost only; CatBoost uses native cat support.
TARGET_ENCODE_ALPHA = 10.0   # smoothing strength (higher = stronger regularisation)
TE_CAT_COLS: list[str] = [
    "CODE_GENDER",
    "NAME_CONTRACT_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_TYPE_SUITE",
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
]

def apply_mode_debug() -> None:
    """
    15% train sample, 3 folds, smaller trees, Optuna off.
    Used only by `python -m entrypoints.run_fast` (or `main(full=False)`).
    """
    global DEBUG_MODE, N_FOLDS, CAAFE_ITERS, USE_OPENFE, RUN_OPTUNA
    DEBUG_MODE = True
    N_FOLDS = 3
    CAAFE_ITERS = 1
    USE_OPENFE = False
    RUN_OPTUNA = False
    LGBM_PARAMS["n_estimators"] = 300
    CATBOOST_PARAMS["iterations"] = 300
    XGB_PARAMS["n_estimators"] = 300


def apply_mode_full() -> None:
    """
    100% rows, 5 folds, full tree budgets. Does not change RUN_OPTUNA (edit in this file to run HPO).
    Used only by `python -m entrypoints.run_full` (or `main(full=True)`).
    """
    global DEBUG_MODE, N_FOLDS, CAAFE_ITERS
    DEBUG_MODE = False
    N_FOLDS = 5
    CAAFE_ITERS = 3
    LGBM_PARAMS["n_estimators"] = 1000
    CATBOOST_PARAMS["iterations"] = 1000
    XGB_PARAMS["n_estimators"] = 1000
