import gc
import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from pipeline import config
from pipeline.utils import logger, flush_logger


# ── Adversarial Validation ────────────────────────────────────────────────────

def _adversarial_validation(X_train: pd.DataFrame, X_test: pd.DataFrame) -> float:
    X_adv = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_adv = np.array([0] * len(X_train) + [1] * len(X_test))
    # pd.concat can drop category dtype when train/test have different category sets;
    # use pd.factorize on all non-numeric cols — dtype-agnostic and always safe
    for c in X_adv.select_dtypes(exclude=["number", "bool"]).columns:
        X_adv[c] = pd.factorize(X_adv[c])[0]

    adv_skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    adv_oof = np.zeros(len(X_adv), dtype=np.float32)

    adv_importances = np.zeros(len(X_adv.columns))

    for tr_idx, val_idx in adv_skf.split(X_adv, y_adv):
        X_tr_adv, X_val_adv = X_adv.iloc[tr_idx], X_adv.iloc[val_idx]
        y_tr_adv = y_adv[tr_idx]

        adv_model = lgb.LGBMClassifier(
            n_estimators=200,
            random_state=config.SEED,
            verbose=-1,
        )
        adv_model.fit(X_tr_adv, y_tr_adv)
        adv_oof[val_idx] = adv_model.predict_proba(X_val_adv)[:, 1]
        adv_importances += adv_model.feature_importances_ / config.N_FOLDS

    auc = roc_auc_score(y_adv, adv_oof)
    logger.info(f"Adversarial Validation AUC: {auc:.4f}")
    
    # Log top 15 high-drift features
    imp_df = pd.DataFrame({'feature': X_adv.columns, 'importance': adv_importances})
    imp_df = imp_df.sort_values(by='importance', ascending=False).head(15)
    logger.info(f"Top 15 adversarial features (drift sources):\n{imp_df.to_string(index=False)}")

    if auc > config.ADV_AUC_THRESHOLD:
        logger.warning(
            f"HIGH DRIFT DETECTED — Adversarial AUC {auc:.4f} > {config.ADV_AUC_THRESHOLD}"
        )
    return auc


# ── Target-Encoding Helpers ───────────────────────────────────────────────────

def _compute_te_map(
    col_series: pd.Series,
    y_series: pd.Series,
    alpha: float,
    global_mean: float,
) -> pd.Series:
    """
    Return a Series mapping category → smoothed mean.
    Formula: (sum_i + alpha * global_mean) / (n_i + alpha)
    """
    df = pd.DataFrame({"cat": col_series.astype(object).values, "target": y_series.values})
    agg = df.groupby("cat")["target"].agg(["sum", "count"])
    return (agg["sum"] + alpha * global_mean) / (agg["count"] + alpha)


def _apply_te_maps(
    X: pd.DataFrame,
    te_cols: list[str],
    te_maps: dict[str, pd.Series],
    global_mean: float,
) -> pd.DataFrame:
    """
    Drop te_cols from X and append smoothed _TE float columns.
    Unseen categories are mapped to global_mean.
    """
    X_out = X.drop(columns=te_cols)
    for c in te_cols:
        X_out = X_out.copy()   # avoid pandas SettingWithCopyWarning on first iter
        X_out[c + "_TE"] = (
            X[c].astype(object).map(te_maps[c]).fillna(global_mean).astype(np.float32)
        )
    return X_out


# ── Optuna HPO ────────────────────────────────────────────────────────────────

def run_optuna(
    X: pd.DataFrame, y: pd.Series, cat_cols: list[str]
) -> dict:
    """
    Tune LGBM, CatBoost, and XGBoost sequentially via Optuna.
    LGBM and XGBoost use fold-safe TE inside each trial; CatBoost uses native cats.
    Returns a dict of best params for each model.
    """
    import optuna
    from optuna.pruners import MedianPruner
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # sklearn rejects train_size=1.0 as float (must be in (0, 1)); use full data without splitting.
    frac = float(config.OPTUNA_SUBSAMPLE_FRAC)
    if frac >= 1.0:
        X_opt, y_opt = X, y
    else:
        X_opt, _, y_opt, _ = train_test_split(
            X, y,
            train_size=frac,
            stratify=y,
            random_state=config.SEED,
        )
    skf = StratifiedKFold(n_splits=config.OPTUNA_N_FOLDS, shuffle=True, random_state=config.SEED)

    alpha                = config.TARGET_ENCODE_ALPHA
    te_cols_opt          = [c for c in config.TE_CAT_COLS if c in X_opt.columns]
    remaining_cat_opt    = [c for c in cat_cols if c not in te_cols_opt]

    # ── LightGBM ─────────────────────────────────────────────────────────────
    def lgbm_objective(trial):
        params = {
            **config.LGBM_PARAMS,
            "num_leaves":        trial.suggest_int("num_leaves", 31, 511),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
            "subsample":         trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        }
        aucs = []
        gm = float(y_opt.mean())
        for tr_i, va_i in skf.split(X_opt, y_opt):
            X_tr_o, X_va_o = X_opt.iloc[tr_i], X_opt.iloc[va_i]
            y_tr_o, y_va_o = y_opt.iloc[tr_i], y_opt.iloc[va_i]
            te_maps = {c: _compute_te_map(X_tr_o[c], y_tr_o, alpha, gm) for c in te_cols_opt}
            X_tr_enc = _apply_te_maps(X_tr_o, te_cols_opt, te_maps, gm)
            X_va_enc = _apply_te_maps(X_va_o, te_cols_opt, te_maps, gm)
            m = lgb.LGBMClassifier(**params)
            m.fit(
                X_tr_enc, y_tr_o,
                eval_set=[(X_va_enc, y_va_o)],
                callbacks=[
                    lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            aucs.append(roc_auc_score(y_va_o, m.predict_proba(X_va_enc)[:, 1]))
        return float(np.mean(aucs))

    T = config.OPTUNA_N_TRIALS

    def _optuna_trial_log(tag: str):
        def _cb(study, trial) -> None:
            n = trial.number + 1
            if n == 1 or n % 5 == 0 or n == T:
                logger.info(
                    f"Optuna {tag}: trial {n}/{T} | intermed_best_auc={study.best_value:.5f}"
                )
                flush_logger()

        return _cb

    if getattr(config, "OPTUNA_SKIP_LGBM_STUDY", False):
        best_lgbm = {**config.LGBM_PARAMS}
        logger.info(
            "Optuna: skipping LGBM study (OPTUNA_SKIP_LGBM_STUDY=True); "
            f"using LGBM_PARAMS from config: "
            f"num_leaves={best_lgbm['num_leaves']}, min_child_samples={best_lgbm['min_child_samples']}, "
            f"lr={best_lgbm['learning_rate']:.6f}, subsample={best_lgbm['subsample']:.4f}, "
            f"colsample={best_lgbm['colsample_bytree']:.4f}, reg_alpha={best_lgbm['reg_alpha']:.4f}, "
            f"reg_lambda={best_lgbm['reg_lambda']:.6f}"
        )
        flush_logger()
    else:
        lgbm_study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=5))
        logger.info(
            f"Optuna: starting LGBM study | n_trials={T} "
            f"cv_folds={config.OPTUNA_N_FOLDS} (TE + fits per trial can take 10–25 min)..."
        )
        flush_logger()
        lgbm_study.optimize(
            lgbm_objective,
            n_trials=T,
            show_progress_bar=False,
            callbacks=[_optuna_trial_log("LGBM")],
        )
        best_lgbm = {**config.LGBM_PARAMS, **lgbm_study.best_params}
        logger.info(
            f"Optuna LGBM best AUC: {lgbm_study.best_value:.5f} | params: {lgbm_study.best_params}"
        )

    # ── CatBoost ─────────────────────────────────────────────────────────────
    def cb_objective(trial):
        params = {
            **config.CATBOOST_PARAMS,
            "depth":               trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count":        trial.suggest_int("border_count", 32, 255),
            "learning_rate":       trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        }
        X_cb = X_opt.copy()
        for c in cat_cols:
            X_cb[c] = X_cb[c].astype(object).fillna("MISSING")

        aucs = []
        for tr_i, va_i in skf.split(X_opt, y_opt):
            m = cb.CatBoostClassifier(**params, cat_features=cat_cols)
            m.fit(
                X_cb.iloc[tr_i], y_opt.iloc[tr_i],
                eval_set=(X_cb.iloc[va_i], y_opt.iloc[va_i]),
                early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
                verbose=False,
            )
            aucs.append(roc_auc_score(y_opt.iloc[va_i], m.predict_proba(X_cb.iloc[va_i])[:, 1]))
        return float(np.mean(aucs))

    cb_study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=5))
    logger.info("Optuna: starting CatBoost study (often the longest phase on GPU)...")
    flush_logger()
    cb_study.optimize(
        cb_objective,
        n_trials=T,
        show_progress_bar=False,
        callbacks=[_optuna_trial_log("CatBoost")],
    )
    best_cb = {**config.CATBOOST_PARAMS, **cb_study.best_params}
    logger.info(f"Optuna CatBoost best AUC: {cb_study.best_value:.5f} | params: {cb_study.best_params}")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    def xgb_objective(trial):
        params = {
            **config.XGB_PARAMS,
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 50),
            "subsample":         trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        }
        aucs = []
        gm = float(y_opt.mean())
        for tr_i, va_i in skf.split(X_opt, y_opt):
            X_tr_o, X_va_o = X_opt.iloc[tr_i], X_opt.iloc[va_i]
            y_tr_o, y_va_o = y_opt.iloc[tr_i], y_opt.iloc[va_i]
            te_maps = {c: _compute_te_map(X_tr_o[c], y_tr_o, alpha, gm) for c in te_cols_opt}
            X_tr_enc = _apply_te_maps(X_tr_o, te_cols_opt, te_maps, gm)
            X_va_enc = _apply_te_maps(X_va_o, te_cols_opt, te_maps, gm)
            # Factorize remaining cats for XGBoost
            for c in remaining_cat_opt:
                codes, uniques = pd.factorize(X_tr_enc[c])
                X_tr_enc[c] = codes
                X_va_enc[c] = pd.Categorical(X_va_enc[c], categories=uniques).codes
            m = xgb.XGBClassifier(**params, early_stopping_rounds=config.EARLY_STOPPING_ROUNDS)
            m.fit(
                X_tr_enc, y_tr_o,
                eval_set=[(X_va_enc, y_va_o)],
                verbose=False,
            )
            aucs.append(roc_auc_score(y_va_o, m.predict_proba(X_va_enc)[:, 1]))
        return float(np.mean(aucs))

    xgb_study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=5))
    logger.info("Optuna: starting XGBoost study...")
    flush_logger()
    xgb_study.optimize(
        xgb_objective,
        n_trials=T,
        show_progress_bar=False,
        callbacks=[_optuna_trial_log("XGBoost")],
    )
    best_xgb = {**config.XGB_PARAMS, **xgb_study.best_params}
    logger.info(f"Optuna XGBoost best AUC: {xgb_study.best_value:.5f} | params: {xgb_study.best_params}")

    return {"lgbm": best_lgbm, "catboost": best_cb, "xgboost": best_xgb}


# ── Main Evaluation ───────────────────────────────────────────────────────────

def evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    tuned_params: dict | None = None,
    *,
    catboost_in_ensemble: bool = True,
) -> tuple[float, list[str]]:
    """
    Surgical TE: TE columns for LGBM + XGB only; CatBoost uses raw categoricals when
    catboost_in_ensemble=True (production). Set catboost_in_ensemble=False only for ablations.
    """

    _adversarial_validation(X, X_test)

    if not catboost_in_ensemble:
        logger.info(
            "OOF ensemble: LGBM + XGBoost only (surgical TE on tree models; "
            "CatBoost skipped). Weights renormalized from LGBM_WEIGHT/XGB_WEIGHT."
        )

    skf      = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    cat_cols = X.select_dtypes(include=["category"]).columns.tolist()

    # TE setup — LGBM + XGBoost receive _TE floats; CatBoost keeps original cats
    alpha       = config.TARGET_ENCODE_ALPHA
    te_cols     = [c for c in config.TE_CAT_COLS if c in X.columns]
    global_mean = float(y.mean())

    # Build test-set TE using all training labels (no leakage — test has no TARGET)
    te_maps_full = {
        c: _compute_te_map(X[c], y, alpha, global_mean) for c in te_cols
    }
    X_test_lgb_xgb = _apply_te_maps(X_test, te_cols, te_maps_full, global_mean)

    # Categorical columns NOT handled by TE — XGBoost needs these factorized
    remaining_cat_cols = [c for c in cat_cols if c not in te_cols]

    # Store separate OOF predictions for stacking
    oof_lgb  = np.zeros(len(X))
    oof_dart = np.zeros(len(X))
    oof_xgb  = np.zeros(len(X))
    oof_cb   = np.zeros(len(X))

    fold_aucs:  list[float] = []
    zero_imp_counts: dict[str, int] = {}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        lgbm_p = tuned_params["lgbm"]     if tuned_params else config.LGBM_PARAMS
        dart_p = config.LGBM_DART_PARAMS
        cb_p   = tuned_params["catboost"] if tuned_params else config.CATBOOST_PARAMS
        xgb_p  = tuned_params["xgboost"]  if tuned_params else config.XGB_PARAMS

        # ── Fold-safe TE for LGBM + XGBoost ─────────────────────────────────
        te_maps_fold = {
            c: _compute_te_map(X_tr[c], y_tr, alpha, global_mean) for c in te_cols
        }
        X_tr_lgb  = _apply_te_maps(X_tr, te_cols, te_maps_fold, global_mean)
        X_val_lgb = _apply_te_maps(X_val, te_cols, te_maps_fold, global_mean)

        X_tr_xgb  = X_tr_lgb.copy()
        X_val_xgb = X_val_lgb.copy()
        for c in remaining_cat_cols:
            codes, uniques = pd.factorize(X_tr_xgb[c])
            X_tr_xgb[c] = codes
            X_val_xgb[c] = pd.Categorical(X_val_xgb[c], categories=uniques).codes

        X_tr_cb = X_val_cb = None
        if catboost_in_ensemble:
            X_tr_cb = X_tr.copy()
            X_val_cb = X_val.copy()
            for c in cat_cols:
                X_tr_cb[c] = X_tr_cb[c].astype(object).fillna("MISSING")
                X_val_cb[c] = X_val_cb[c].astype(object).fillna("MISSING")

        # ── LightGBM ─────────────────────────────────────────────────────────
        lgb_m = lgb.LGBMClassifier(**lgbm_p)
        lgb_m.fit(
            X_tr_lgb, y_tr,
            eval_set=[(X_val_lgb, y_val)],
            callbacks=[
                lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        # ── LightGBM DART ────────────────────────────────────────────────────
        dart_m = lgb.LGBMClassifier(**dart_p)
        dart_m.fit(
            X_tr_lgb, y_tr,
            eval_set=[(X_val_lgb, y_val)],
            callbacks=[
                # DART doesn't support early stopping well, but we provide it just in case
                lgb.early_stopping(config.EARLY_STOPPING_ROUNDS * 2, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )

        cb_m = None
        if catboost_in_ensemble:
            cb_m = cb.CatBoostClassifier(**cb_p, cat_features=cat_cols)
            cb_m.fit(
                X_tr_cb, y_tr,
                eval_set=[(X_val_cb, y_val)],
                early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
                verbose=False
            )

        # ── XGBoost ──────────────────────────────────────────────────────────
        xgb_m = xgb.XGBClassifier(**xgb_p, early_stopping_rounds=config.EARLY_STOPPING_ROUNDS)
        xgb_m.fit(
            X_tr_xgb, y_tr,
            eval_set=[(X_val_xgb, y_val)],
            verbose=False,
        )

        # ── Store base predictions ────────────────────────────────
        p_lgb = lgb_m.predict_proba(X_val_lgb)[:, 1]
        p_dart = dart_m.predict_proba(X_val_lgb)[:, 1]
        p_xgb = xgb_m.predict_proba(X_val_xgb)[:, 1]
        oof_lgb[val_idx] = p_lgb
        oof_dart[val_idx] = p_dart
        oof_xgb[val_idx] = p_xgb
        
        if catboost_in_ensemble:
            p_cb = cb_m.predict_proba(X_val_cb)[:, 1]
            oof_cb[val_idx] = p_cb
            # Temporary static weight for log
            fold_oof = (
                config.LGBM_WEIGHT * p_lgb 
                + config.LGBM_DART_WEIGHT * p_dart
                + config.CATBOOST_WEIGHT * p_cb 
                + config.XGB_WEIGHT * p_xgb
            )
        else:
            wsum = config.LGBM_WEIGHT + config.LGBM_DART_WEIGHT + config.XGB_WEIGHT
            wl = config.LGBM_WEIGHT / wsum
            wd = config.LGBM_DART_WEIGHT / wsum
            wx = config.XGB_WEIGHT / wsum
            fold_oof = wl * p_lgb + wd * p_dart + wx * p_xgb

        fold_auc = roc_auc_score(y_val, fold_oof)
        fold_aucs.append(fold_auc)
        logger.info(f"Fold {fold}/{config.N_FOLDS} Static Weights AUC: {fold_auc:.5f}")
        flush_logger()

        for feat, imp in zip(lgb_m.feature_name_, lgb_m.feature_importances_):
            if imp == 0:
                zero_imp_counts[feat] = zero_imp_counts.get(feat, 0) + 1

        del (X_tr, X_val, X_tr_lgb, X_val_lgb, X_tr_xgb, X_val_xgb, lgb_m, dart_m, xgb_m)
        if catboost_in_ensemble:
            del X_tr_cb, X_val_cb, cb_m
        gc.collect()

    # ── Level 2 Stacking ──────────────────────────────────────────
    logger.info("Training Level 2 Stacker (RidgeRegression)...")
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict

    if catboost_in_ensemble:
        S_train = np.column_stack((oof_lgb, oof_dart, oof_cb, oof_xgb))
    else:
        S_train = np.column_stack((oof_lgb, oof_dart, oof_xgb))

    meta_model = Ridge(alpha=1.0)
    # Use cross_val_predict on the OOF predictions to get the final stacked OOF
    stacked_oof = cross_val_predict(meta_model, S_train, y, cv=skf, method='predict')
    
    final_auc = float(roc_auc_score(y, stacked_oof))
    if catboost_in_ensemble:
        static_oof = config.LGBM_WEIGHT * oof_lgb + config.LGBM_DART_WEIGHT * oof_dart + config.CATBOOST_WEIGHT * oof_cb + config.XGB_WEIGHT * oof_xgb
    else:
        wsum = config.LGBM_WEIGHT + config.LGBM_DART_WEIGHT + config.XGB_WEIGHT
        wl = config.LGBM_WEIGHT / wsum
        wd = config.LGBM_DART_WEIGHT / wsum
        wx = config.XGB_WEIGHT / wsum
        static_oof = wl * oof_lgb + wd * oof_dart + wx * oof_xgb

    static_auc = float(roc_auc_score(y, static_oof))

    logger.info(f"OOF Static Weights AUC: {static_auc:.5f}")
    logger.info(f"OOF Stacked Ridge AUC: {final_auc:.5f}")
    
    # We will return the best of either static weights or stacked model
    if static_auc > final_auc:
        logger.info("Static weights outperformed Ridge Stacker. Using Static Weights.")
        final_auc = static_auc

    zero_imp_feats = [f for f, cnt in zero_imp_counts.items() if cnt == config.N_FOLDS]
    logger.info(f"Zero-importance features ({len(zero_imp_feats)}): {zero_imp_feats}")
    flush_logger()

    return final_auc, zero_imp_feats
