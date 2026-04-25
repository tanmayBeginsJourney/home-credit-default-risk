import gc
import uuid
from datetime import datetime, timezone

import numpy as np
from pathlib import Path
from pipeline import config
from pipeline.utils import seed_everything, ensure_parquet_format, logger, flush_logger
from pipeline.data import load_and_clean_application
from pipeline.features import fe_application, fe_bureau_derived
from pipeline.aggregations import (
    agg_bureau, agg_installments, agg_credit_card,
    agg_pos_cash, agg_previous_application,
)
from pipeline.model import evaluate_model, run_optuna


def main(full: bool = False) -> float:
    """
    :param full: If True (only via `entrypoints.run_full`), 100% data, 5 folds,
        production tree budgets. If False, 15% sample, 3 folds, fast iteration.
    """
    if full:
        config.apply_mode_full()
    else:
        config.apply_mode_debug()

    seed_everything(config.SEED)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_parquet_format(config.DATA_DIR)

    run_id = uuid.uuid4().hex
    started_utc = datetime.now(timezone.utc).isoformat()
    logger.info(
        f"Run session | id={run_id} | started_utc={started_utc} | "
        f"log={config.LOG_FILE} | "
        f"DEBUG={config.DEBUG_MODE} | FOLDS={config.N_FOLDS} | SEED={config.SEED} | "
        f"RUN_OPTUNA={config.RUN_OPTUNA}"
    )
    logger.info(
        f"=== Pipeline Start | DEBUG={config.DEBUG_MODE} | "
        f"FOLDS={config.N_FOLDS} | SEED={config.SEED} ==="
    )

    # Load base tables
    train = load_and_clean_application(is_train=True)
    test  = load_and_clean_application(is_train=False)

    # Aggregations
    D = config.DATA_DIR
    bur_feats  = agg_bureau(D / "bureau.parquet", D / "bureau_balance.parquet")
    inst_feats = agg_installments(D / "installments_payments.parquet")
    cc_feats   = agg_credit_card(D / "credit_card_balance.parquet")
    pos_feats  = agg_pos_cash(D / "POS_CASH_balance.parquet")
    prev_feats = agg_previous_application(D / "previous_application.parquet")

    # Merge all agg tables
    for agg_df, tag in [
        (bur_feats,  "bureau"),
        (inst_feats, "installments"),
        (cc_feats,   "credit_card"),
        (pos_feats,  "pos_cash"),
        (prev_feats, "prev_app"),
    ]:
        train = train.join(agg_df, on="SK_ID_CURR", how="left")
        test  = test.join(agg_df,  on="SK_ID_CURR", how="left")
        del agg_df
        gc.collect()
        logger.info(f"Merged {tag}: train={train.shape}, test={test.shape}")

    # Meta-features
    import polars as pl
    if (config.CACHE_DIR / "meta_bureau.parquet").exists():
        meta_bureau = pl.read_parquet(config.CACHE_DIR / "meta_bureau.parquet")
        train = train.join(meta_bureau, on="SK_ID_CURR", how="left")
        test  = test.join(meta_bureau,  on="SK_ID_CURR", how="left")
        logger.info(f"Merged meta_bureau: train={train.shape}, test={test.shape}")
        del meta_bureau

    if (config.CACHE_DIR / "meta_prev.parquet").exists():
        meta_prev = pl.read_parquet(config.CACHE_DIR / "meta_prev.parquet")
        train = train.join(meta_prev, on="SK_ID_CURR", how="left")
        test  = test.join(meta_prev,  on="SK_ID_CURR", how="left")
        logger.info(f"Merged meta_prev: train={train.shape}, test={test.shape}")
        del meta_prev

    if (config.CACHE_DIR / "meta_installments.parquet").exists():
        meta_inst = pl.read_parquet(config.CACHE_DIR / "meta_installments.parquet")
        train = train.join(meta_inst, on="SK_ID_CURR", how="left")
        test  = test.join(meta_inst,  on="SK_ID_CURR", how="left")
        logger.info(f"Merged meta_installments: train={train.shape}, test={test.shape}")
        del meta_inst
        
    if (config.CACHE_DIR / "meta_pos_cash.parquet").exists():
        meta_pos = pl.read_parquet(config.CACHE_DIR / "meta_pos_cash.parquet")
        train = train.join(meta_pos, on="SK_ID_CURR", how="left")
        test  = test.join(meta_pos,  on="SK_ID_CURR", how="left")
        logger.info(f"Merged meta_pos_cash: train={train.shape}, test={test.shape}")
        del meta_pos

    if (config.CACHE_DIR / "meta_ext_train.parquet").exists():
        meta_ext_train = pl.read_parquet(config.CACHE_DIR / "meta_ext_train.parquet")
        meta_ext_test = pl.read_parquet(config.CACHE_DIR / "meta_ext_test.parquet")
        train = train.join(meta_ext_train, on="SK_ID_CURR", how="left")
        test  = test.join(meta_ext_test,  on="SK_ID_CURR", how="left")
        
        # Interaction residuals
        for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            if c in train.columns and f"PRED_{c}" in train.columns:
                train = train.with_columns(
                    (pl.col(c) - pl.col(f"PRED_{c}")).alias(f"{c}_RESIDUAL")
                )
                test = test.with_columns(
                    (pl.col(c) - pl.col(f"PRED_{c}")).alias(f"{c}_RESIDUAL")
                )
        logger.info(f"Merged meta_ext_sources: train={train.shape}, test={test.shape}")
        del meta_ext_train, meta_ext_test
        
    gc.collect()

    # Feature engineering
    train = fe_application(train)
    test  = fe_application(test)
    train = fe_bureau_derived(train)
    test  = fe_bureau_derived(test)

    # Drop any COLS_TO_DROP columns that were created by FE (not present in raw data)
    fe_drop = [c for c in config.COLS_TO_DROP if c in train.columns]
    if fe_drop:
        train = train.drop(fe_drop)
        test  = test.drop([c for c in fe_drop if c in test.columns])
        logger.info(f"Post-FE drop {len(fe_drop)} cols: {fe_drop}")

    # Convert to Pandas for modeling
    logger.info("Converting Polars → Pandas...")
    train_pd = train.to_pandas()
    test_pd  = test.to_pandas()
    del train, test
    gc.collect()

    # Polars Categorical → pandas StringDtype on conversion; re-cast to category
    str_cols = [c for c in train_pd.columns if str(train_pd[c].dtype) in ("string", "object")]
    for c in str_cols:
        train_pd[c] = train_pd[c].astype("category")
        if c in test_pd.columns:
            test_pd[c] = test_pd[c].astype("category")
    logger.info(f"Re-cast {len(str_cols)} string cols to pandas category")

    y_train  = train_pd.pop("TARGET")
    X_train  = train_pd.drop(columns=["SK_ID_CURR"], errors="ignore")
    X_test   = test_pd.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")
    del train_pd, test_pd
    gc.collect()

    # KNN Target Imputation (Tier 6 — fold-external, leave-one-out on train)
    if config.USE_KNN:
        from sklearn.neighbors import KDTree
        avail = [c for c in config.KNN_COLS if c in X_train.columns]
        if avail:
            knn_train_df = X_train[avail].copy()
            knn_test_df = X_test[avail].copy()
            if getattr(config, "KNN_USE_CREDIT_ANNUITY_RATIO", False):
                ratio_col = "KNN_CREDIT_ANNUITY_RATIO"
                knn_train_df[ratio_col] = (
                    X_train["AMT_CREDIT"] / (X_train["AMT_ANNUITY"] + 1e-8)
                )
                knn_test_df[ratio_col] = (
                    X_test["AMT_CREDIT"] / (X_test["AMT_ANNUITY"] + 1e-8)
                )
            med = knn_train_df.median()
            tr_knn = knn_train_df.fillna(med).values.astype("float32")
            te_knn = knn_test_df.fillna(med).values.astype("float32")
            tree = KDTree(tr_knn)
            ks = list(getattr(config, "KNN_NEIGHBORS_LIST", [])) or [config.KNN_N_NEIGHBORS]
            ks = sorted({int(k) for k in ks if int(k) > 1})
            X_train = X_train.copy()
            X_test = X_test.copy()
            for k_raw in ks:
                k = min(k_raw, len(X_train) - 1)
                if k < 2:
                    continue
                # Leave-one-out: query k+1, skip index 0 (self)
                _, tr_idx = tree.query(tr_knn, k=k + 1)
                feat_name = f"KNN_TARGET_MEAN_K{k}"
                X_train[feat_name] = [y_train.iloc[i[1:]].mean() for i in tr_idx]
                # Test: all training neighbors (no leakage)
                _, te_idx = tree.query(te_knn, k=k)
                X_test[feat_name] = y_train.values[te_idx].mean(axis=1)
            logger.info(
                "KNN target imputation: "
                f"k_list={ks}, cols={list(knn_train_df.columns)}"
            )

    # Optuna HPO (only when RUN_OPTUNA=True — gate strictly)
    tuned_params = None
    if config.RUN_OPTUNA:
        assert not config.DEBUG_MODE, "Optuna must not run in DEBUG_MODE."
        cat_cols_opt = X_train.select_dtypes(include=["category"]).columns.tolist()
        logger.info(
            f"Starting Optuna HPO | trials={config.OPTUNA_N_TRIALS} "
            f"subsample={config.OPTUNA_SUBSAMPLE_FRAC} folds={config.OPTUNA_N_FOLDS}"
        )
        flush_logger()
        tuned_params = run_optuna(X_train, y_train, cat_cols_opt)
        logger.info("Optuna complete. Running final evaluation with tuned params...")

    # Evaluate
    logger.info("Starting model evaluation...")
    auc, zero_feats = evaluate_model(X_train, y_train, X_test, tuned_params=tuned_params)

    if zero_feats:
        logger.info(f"ACTION REQUIRED: Drop {len(zero_feats)} zero-importance features.")

    logger.info(
        f"=== Pipeline Complete | run_id={run_id} | Final OOF AUC: {auc:.5f} ==="
    )
    return auc


if __name__ == "__main__":
    main()
