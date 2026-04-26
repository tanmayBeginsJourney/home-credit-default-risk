import gc
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from pipeline.utils import logger, ensure_parquet_format
from pipeline import config

def _cast_non_numeric(df: pd.DataFrame) -> pd.DataFrame:
    cast_df = df.copy()
    non_num_cols = cast_df.select_dtypes(include=["object", "string", "category"]).columns
    for c in non_num_cols:
        cast_df[c] = cast_df[c].astype("category").cat.codes.astype(np.int32)
    return cast_df


def generate_meta_features(table_name: str, id_col: str, drop_cols: list) -> pd.DataFrame:
    logger.info(f"Generating meta-features for {table_name}...")
    
    # Load targets
    app_train = pd.read_parquet(config.DATA_DIR / "application_train.parquet", columns=["SK_ID_CURR", "TARGET"])
    app_test = pd.read_parquet(config.DATA_DIR / "application_test.parquet", columns=["SK_ID_CURR"])
    
    # Load secondary table
    df = pd.read_parquet(config.DATA_DIR / f"{table_name}.parquet")
    
    if table_name == "bureau_balance":
        bureau_keys = pd.read_parquet(config.DATA_DIR / "bureau.parquet", columns=["SK_ID_BUREAU", "SK_ID_CURR"])
        df = df.merge(bureau_keys, on="SK_ID_BUREAU", how="left")
        del bureau_keys
        gc.collect()
        df = df[df["SK_ID_CURR"].notna()].copy()
        
    # Split train/test
    train_df = df[df["SK_ID_CURR"].isin(app_train["SK_ID_CURR"])].copy()
    test_df = df[df["SK_ID_CURR"].isin(app_test["SK_ID_CURR"])].copy()
    train_df["SK_ID_CURR"] = train_df["SK_ID_CURR"].astype(np.int32)
    test_df["SK_ID_CURR"] = test_df["SK_ID_CURR"].astype(np.int32)
    
    # Merge TARGET to train_df
    train_df = train_df.merge(app_train, on="SK_ID_CURR", how="left")
    
    y = train_df["TARGET"]
    groups = train_df["SK_ID_CURR"]
    X = train_df.drop(columns=["TARGET", "SK_ID_CURR", id_col] + drop_cols, errors="ignore")
    X_test = test_df.drop(columns=["SK_ID_CURR", id_col] + drop_cols, errors="ignore")
    X = _cast_non_numeric(X).replace([np.inf, -np.inf], np.nan).fillna(-999)
    X_test = _cast_non_numeric(X_test).replace([np.inf, -np.inf], np.nan).fillna(-999)
    test_groups = test_df["SK_ID_CURR"]
    
    # Initialize OOF array and test predictions
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    # Use StratifiedGroupKFold to keep all records for a user in the same fold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    
    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=config.SEED,
            verbose=-1,
            device="gpu" if config.LGBM_PARAMS.get("device") == "gpu" else "cpu"
        )
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        
        oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / 5
        
    meta_col = f"META_{table_name}_PRED".upper()
    train_df[meta_col] = oof_preds
    test_df[meta_col] = test_preds
    
    # Aggregate predictions by user
    train_agg = train_df.groupby("SK_ID_CURR")[meta_col].agg(["mean", "max", "min", "std"]).reset_index()
    test_agg = test_df.groupby("SK_ID_CURR")[meta_col].agg(["mean", "max", "min", "std"]).reset_index()
    
    # Rename columns
    prefix = f"META_{table_name}".upper()
    train_agg.columns = ["SK_ID_CURR", f"{prefix}_MEAN", f"{prefix}_MAX", f"{prefix}_MIN", f"{prefix}_STD"]
    test_agg.columns = ["SK_ID_CURR", f"{prefix}_MEAN", f"{prefix}_MAX", f"{prefix}_MIN", f"{prefix}_STD"]
    
    res = pd.concat([train_agg, test_agg], axis=0).reset_index(drop=True)
    res["SK_ID_CURR"] = res["SK_ID_CURR"].astype(np.int32)
    
    logger.info(f"Generated meta-features for {table_name}: shape {res.shape}")
    return res


def generate_meta_ext_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Generating EXT_SOURCE meta-features...")
    train = pd.read_parquet(config.DATA_DIR / "application_train.parquet")
    test = pd.read_parquet(config.DATA_DIR / "application_test.parquet")

    feature_candidates = [
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "REGION_POPULATION_RELATIVE",
        "DAYS_ID_PUBLISH",
        "DAYS_REGISTRATION",
        "CNT_FAM_MEMBERS",
        "CNT_CHILDREN",
    ]
    use_feats = [c for c in feature_candidates if c in train.columns and c in test.columns]

    x_train = train[use_feats].copy()
    x_test = test[use_feats].copy()
    x_train = _cast_non_numeric(x_train).replace([np.inf, -np.inf], np.nan).fillna(-999)
    x_test = _cast_non_numeric(x_test).replace([np.inf, -np.inf], np.nan).fillna(-999)

    train_out = pd.DataFrame({"SK_ID_CURR": train["SK_ID_CURR"].values})
    test_out = pd.DataFrame({"SK_ID_CURR": test["SK_ID_CURR"].values})

    for ext_col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        if ext_col not in train.columns:
            continue
        y = train[ext_col]
        known_mask = y.notna()
        if known_mask.sum() < 1000:
            continue

        x_known = x_train.loc[known_mask]
        y_known = y.loc[known_mask]
        oof = np.full(len(train), np.nan, dtype=np.float32)
        pred_test = np.zeros(len(test), dtype=np.float32)

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED)
        y_bins = pd.qcut(y_known, q=10, labels=False, duplicates="drop")
        groups = train.loc[known_mask, "SK_ID_CURR"]
        for tr_idx, va_idx in sgkf.split(x_known, y_bins, groups=groups):
            x_tr = x_known.iloc[tr_idx]
            y_tr = y_known.iloc[tr_idx]
            x_va = x_known.iloc[va_idx]

            reg = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=63,
                random_state=config.SEED,
                verbose=-1,
                device="gpu" if config.LGBM_PARAMS.get("device") == "gpu" else "cpu",
            )
            reg.fit(x_tr, y_tr)
            oof_idx = y_known.index[va_idx]
            oof[oof_idx] = reg.predict(x_va).astype(np.float32)
            pred_test += reg.predict(x_test).astype(np.float32) / 5.0

        full_reg = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            random_state=config.SEED,
            verbose=-1,
            device="gpu" if config.LGBM_PARAMS.get("device") == "gpu" else "cpu",
        )
        full_reg.fit(x_known, y_known)
        missing_mask = y.isna()
        if missing_mask.any():
            oof[missing_mask] = full_reg.predict(x_train.loc[missing_mask]).astype(np.float32)

        train_out[f"PRED_{ext_col}"] = oof
        test_out[f"PRED_{ext_col}"] = pred_test

    logger.info(
        "Generated EXT_SOURCE meta-features: "
        f"train_shape={train_out.shape}, test_shape={test_out.shape}"
    )
    train_out["SK_ID_CURR"] = train_out["SK_ID_CURR"].astype(np.int32)
    test_out["SK_ID_CURR"] = test_out["SK_ID_CURR"].astype(np.int32)
    return train_out, test_out

if __name__ == "__main__":
    ensure_parquet_format(config.DATA_DIR)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Bureau
    if not (config.CACHE_DIR / "meta_bureau.parquet").exists():
        bureau_meta = generate_meta_features("bureau", "SK_ID_BUREAU", [])
        bureau_meta.to_parquet(config.CACHE_DIR / "meta_bureau.parquet")
    
    # Previous Application
    if not (config.CACHE_DIR / "meta_prev.parquet").exists():
        prev_meta = generate_meta_features("previous_application", "SK_ID_PREV", [])
        prev_meta.to_parquet(config.CACHE_DIR / "meta_prev.parquet")
        
    # Installments Payments
    if not (config.CACHE_DIR / "meta_installments.parquet").exists():
        inst_meta = generate_meta_features("installments_payments", "SK_ID_PREV", [])
        inst_meta.to_parquet(config.CACHE_DIR / "meta_installments.parquet")
        
    # POS CASH Balance
    if not (config.CACHE_DIR / "meta_pos_cash.parquet").exists():
        pos_meta = generate_meta_features("POS_CASH_balance", "SK_ID_PREV", [])
        pos_meta.to_parquet(config.CACHE_DIR / "meta_pos_cash.parquet")

    # Credit card meta model
    if not (config.CACHE_DIR / "meta_credit_card.parquet").exists():
        cc_meta = generate_meta_features("credit_card_balance", "SK_ID_PREV", [])
        cc_meta.to_parquet(config.CACHE_DIR / "meta_credit_card.parquet")

    # Bureau balance meta model
    if not (config.CACHE_DIR / "meta_bureau_balance.parquet").exists():
        bb_meta = generate_meta_features("bureau_balance", "SK_ID_BUREAU", [])
        bb_meta.to_parquet(config.CACHE_DIR / "meta_bureau_balance.parquet")

    # EXT source reconstruction metas
    if not (config.CACHE_DIR / "meta_ext_train.parquet").exists():
        meta_ext_train, meta_ext_test = generate_meta_ext_sources()
        meta_ext_train.to_parquet(config.CACHE_DIR / "meta_ext_train.parquet")
        meta_ext_test.to_parquet(config.CACHE_DIR / "meta_ext_test.parquet")
    
    logger.info("Meta-features successfully generated and cached.")
