import gc
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from pipeline.utils import logger, ensure_parquet_format
from pipeline import config

def generate_meta_features(table_name: str, id_col: str, drop_cols: list) -> pd.DataFrame:
    logger.info(f"Generating meta-features for {table_name}...")
    
    # Load targets
    app_train = pd.read_parquet(config.DATA_DIR / "application_train.parquet", columns=["SK_ID_CURR", "TARGET"])
    app_test = pd.read_parquet(config.DATA_DIR / "application_test.parquet", columns=["SK_ID_CURR"])
    
    # Load secondary table
    df = pd.read_parquet(config.DATA_DIR / f"{table_name}.parquet")
    
    # Simple preprocessing: label encode categorical columns, fill NaNs
    cat_cols = df.select_dtypes(include=['object', 'string']).columns
    for c in cat_cols:
        df[c] = df[c].astype('category')
        
    # Split train/test
    train_df = df[df["SK_ID_CURR"].isin(app_train["SK_ID_CURR"])].copy()
    test_df = df[df["SK_ID_CURR"].isin(app_test["SK_ID_CURR"])].copy()
    
    # Merge TARGET to train_df
    train_df = train_df.merge(app_train, on="SK_ID_CURR", how="left")
    
    y = train_df["TARGET"]
    groups = train_df["SK_ID_CURR"]
    X = train_df.drop(columns=["TARGET", "SK_ID_CURR", id_col] + drop_cols, errors="ignore")
    
    X_test = test_df.drop(columns=["SK_ID_CURR", id_col] + drop_cols, errors="ignore")
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
        
    train_df[f"META_{table_name}_PRED"] = oof_preds
    test_df[f"META_{table_name}_PRED"] = test_preds
    
    # Aggregate predictions by user
    train_agg = train_df.groupby("SK_ID_CURR")[f"META_{table_name}_PRED"].agg(['mean', 'max', 'min']).reset_index()
    test_agg = test_df.groupby("SK_ID_CURR")[f"META_{table_name}_PRED"].agg(['mean', 'max', 'min']).reset_index()
    
    # Rename columns
    train_agg.columns = ["SK_ID_CURR", f"META_{table_name}_MEAN", f"META_{table_name}_MAX", f"META_{table_name}_MIN"]
    test_agg.columns = ["SK_ID_CURR", f"META_{table_name}_MEAN", f"META_{table_name}_MAX", f"META_{table_name}_MIN"]
    
    res = pd.concat([train_agg, test_agg], axis=0).reset_index(drop=True)
    
    logger.info(f"Generated meta-features for {table_name}: shape {res.shape}")
    return res

if __name__ == "__main__":
    ensure_parquet_format(config.DATA_DIR)
    
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
    
    logger.info("Meta-features successfully generated and cached.")
