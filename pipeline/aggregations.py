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

    bb = pl.scan_parquet(bureau_bal_path).with_columns(
        pl.when(pl.col("STATUS") == "C").then(-1)
        .when(pl.col("STATUS") == "X").then(0)
        .otherwise(pl.col("STATUS").cast(pl.Int32, strict=False).fill_null(0))
        .alias("STATUS_SEVERITY")
    )
    bb_agg = bb.group_by("SK_ID_BUREAU").agg([
        pl.len().alias("BB_MONTHS_COUNT"),
        pl.col("MONTHS_BALANCE").min().alias("BB_MONTHS_BALANCE_MIN"),
        pl.col("MONTHS_BALANCE").max().alias("BB_MONTHS_BALANCE_MAX"),
        (pl.col("STATUS") == "C").mean().alias("BB_STATUS_C_FRAC"),
        (pl.col("STATUS").is_in(["1", "2", "3", "4", "5"])).sum().alias("BB_STATUS_LATE_COUNT"),
        (pl.col("STATUS").is_in(["1", "2", "3", "4", "5"])).mean().alias("BB_STATUS_LATE_FRAC"),
        (pl.col("STATUS").is_in(["3", "4", "5"])).sum().alias("BB_STATUS_SEVERE_COUNT"),
        (pl.col("STATUS").is_in(["3", "4", "5"])).mean().alias("BB_STATUS_SEVERE_FRAC"),
        (pl.col("STATUS") == "0").sum().alias("BB_STATUS_0_COUNT"),
        (pl.col("STATUS") == "1").sum().alias("BB_STATUS_1_COUNT"),
        (pl.col("STATUS") == "2").sum().alias("BB_STATUS_2_COUNT"),
        (pl.col("STATUS") == "3").sum().alias("BB_STATUS_3_COUNT"),
        (pl.col("STATUS") == "4").sum().alias("BB_STATUS_4_COUNT"),
        (pl.col("STATUS") == "5").sum().alias("BB_STATUS_5_COUNT"),
        (pl.col("STATUS") == "0").mean().alias("BB_STATUS_0_FRAC"),
        (pl.col("STATUS") == "1").mean().alias("BB_STATUS_1_FRAC"),
        (pl.col("STATUS") == "2").mean().alias("BB_STATUS_2_FRAC"),
        (pl.col("STATUS") == "3").mean().alias("BB_STATUS_3_FRAC"),
        (pl.col("STATUS") == "4").mean().alias("BB_STATUS_4_FRAC"),
        (pl.col("STATUS") == "5").mean().alias("BB_STATUS_5_FRAC"),
        pl.col("STATUS_SEVERITY").sort_by("MONTHS_BALANCE").first().alias("BB_FIRST_STATUS_SEVERITY"),
        pl.col("STATUS_SEVERITY").sort_by("MONTHS_BALANCE").last().alias("BB_LAST_STATUS_SEVERITY"),
        pl.when(pl.col("STATUS") == "C")
            .then(pl.col("MONTHS_BALANCE").abs())
            .otherwise(None)
            .min()
            .alias("BB_WHEN_CLOSED_MONTH_ABS"),
    ]).with_columns(
        (
            pl.col("BB_MONTHS_BALANCE_MAX").abs()
            - pl.col("BB_WHEN_CLOSED_MONTH_ABS")
        ).alias("BB_MONTH_CLOSED_TO_END")
    )

    bur = (
        pl.scan_parquet(bureau_path)
        .join(bb_agg, on="SK_ID_BUREAU", how="left")
        .with_columns([
            (pl.col("AMT_CREDIT_SUM") - pl.col("AMT_CREDIT_SUM_DEBT"))
                .alias("BUREAU_CREDIT_MINUS_DEBT"),
            (pl.col("AMT_CREDIT_SUM") - pl.col("AMT_CREDIT_SUM_LIMIT"))
                .alias("BUREAU_CREDIT_MINUS_LIMIT"),
            (pl.col("AMT_CREDIT_SUM") - pl.col("AMT_CREDIT_SUM_OVERDUE"))
                .alias("BUREAU_CREDIT_MINUS_OVERDUE"),
            (pl.col("AMT_CREDIT_SUM_DEBT") / (pl.col("AMT_CREDIT_SUM") + 1e-8))
                .alias("BUREAU_DEBT_CREDIT_RATIO"),
            (pl.col("AMT_CREDIT_SUM_OVERDUE") / (pl.col("AMT_CREDIT_SUM") + 1e-8))
                .alias("BUREAU_OVERDUE_CREDIT_RATIO"),
            (pl.col("DAYS_CREDIT") - pl.col("CREDIT_DAY_OVERDUE"))
                .alias("BUREAU_DAYS_CREDIT_OVERDUE_GAP"),
            (pl.col("DAYS_CREDIT") - pl.col("DAYS_CREDIT_ENDDATE"))
                .alias("BUREAU_DAYS_CREDIT_ENDDATE_GAP"),
            (pl.col("DAYS_CREDIT") - pl.col("DAYS_ENDDATE_FACT"))
                .alias("BUREAU_DAYS_CREDIT_FACT_GAP"),
            (pl.col("DAYS_CREDIT_ENDDATE") - pl.col("DAYS_ENDDATE_FACT"))
                .alias("BUREAU_ENDDATE_FACT_GAP"),
            (pl.col("DAYS_CREDIT_UPDATE") - pl.col("DAYS_CREDIT_ENDDATE"))
                .alias("BUREAU_UPDATE_ENDDATE_GAP"),
        ])
    )

    active = bur.filter(pl.col("CREDIT_ACTIVE") == "Active")
    active_agg = active.group_by("SK_ID_CURR").agg([
        pl.col("AMT_CREDIT_SUM").sum().alias("ACTIVE_CREDIT_SUM"),
        pl.len().alias("ACTIVE_CREDIT_COUNT"),
        (pl.col("AMT_CREDIT_SUM") * (pl.col("DAYS_CREDIT").abs() * -0.01).exp())
            .sum().alias("ACTIVE_CREDIT_SUM_DECAYED"),
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
        pl.col("BB_STATUS_LATE_FRAC").mean().alias("BUREAU_BB_LATE_FRAC_MEAN"),
        pl.col("BB_STATUS_SEVERE_COUNT").sum().alias("BUREAU_BB_SEVERE_STATUS_SUM"),
        pl.col("BB_STATUS_SEVERE_FRAC").mean().alias("BUREAU_BB_SEVERE_FRAC_MEAN"),
        pl.col("BB_FIRST_STATUS_SEVERITY").mean().alias("BUREAU_BB_FIRST_STATUS_SEV_MEAN"),
        pl.col("BB_LAST_STATUS_SEVERITY").mean().alias("BUREAU_BB_LAST_STATUS_SEV_MEAN"),
        pl.col("BB_LAST_STATUS_SEVERITY").max().alias("BUREAU_BB_LAST_STATUS_SEV_MAX"),
        pl.col("BB_MONTH_CLOSED_TO_END").mean().alias("BUREAU_BB_MONTH_CLOSED_TO_END_MEAN"),
        pl.col("BB_STATUS_0_COUNT").sum().alias("BUREAU_BB_STATUS_0_COUNT_SUM"),
        pl.col("BB_STATUS_1_COUNT").sum().alias("BUREAU_BB_STATUS_1_COUNT_SUM"),
        pl.col("BB_STATUS_2_COUNT").sum().alias("BUREAU_BB_STATUS_2_COUNT_SUM"),
        pl.col("BB_STATUS_3_COUNT").sum().alias("BUREAU_BB_STATUS_3_COUNT_SUM"),
        pl.col("BB_STATUS_4_COUNT").sum().alias("BUREAU_BB_STATUS_4_COUNT_SUM"),
        pl.col("BB_STATUS_5_COUNT").sum().alias("BUREAU_BB_STATUS_5_COUNT_SUM"),
        pl.col("BB_STATUS_0_FRAC").mean().alias("BUREAU_BB_STATUS_0_FRAC_MEAN"),
        pl.col("BB_STATUS_1_FRAC").mean().alias("BUREAU_BB_STATUS_1_FRAC_MEAN"),
        pl.col("BB_STATUS_2_FRAC").mean().alias("BUREAU_BB_STATUS_2_FRAC_MEAN"),
        pl.col("BB_STATUS_3_FRAC").mean().alias("BUREAU_BB_STATUS_3_FRAC_MEAN"),
        pl.col("BB_STATUS_4_FRAC").mean().alias("BUREAU_BB_STATUS_4_FRAC_MEAN"),
        pl.col("BB_STATUS_5_FRAC").mean().alias("BUREAU_BB_STATUS_5_FRAC_MEAN"),
        pl.col("BUREAU_CREDIT_MINUS_DEBT").mean().alias("BUREAU_CREDIT_MINUS_DEBT_MEAN"),
        pl.col("BUREAU_CREDIT_MINUS_DEBT").sum().alias("BUREAU_CREDIT_MINUS_DEBT_SUM"),
        pl.col("BUREAU_CREDIT_MINUS_LIMIT").mean().alias("BUREAU_CREDIT_MINUS_LIMIT_MEAN"),
        pl.col("BUREAU_CREDIT_MINUS_LIMIT").sum().alias("BUREAU_CREDIT_MINUS_LIMIT_SUM"),
        pl.col("BUREAU_CREDIT_MINUS_OVERDUE").mean().alias("BUREAU_CREDIT_MINUS_OVERDUE_MEAN"),
        pl.col("BUREAU_DEBT_CREDIT_RATIO").mean().alias("BUREAU_DEBT_CREDIT_RATIO_MEAN"),
        pl.col("BUREAU_DEBT_CREDIT_RATIO").max().alias("BUREAU_DEBT_CREDIT_RATIO_MAX"),
        pl.col("BUREAU_OVERDUE_CREDIT_RATIO").mean().alias("BUREAU_OVERDUE_CREDIT_RATIO_MEAN"),
        pl.col("BUREAU_DAYS_CREDIT_OVERDUE_GAP").mean().alias("BUREAU_DAYS_CREDIT_OVERDUE_GAP_MEAN"),
        pl.col("BUREAU_DAYS_CREDIT_ENDDATE_GAP").mean().alias("BUREAU_DAYS_CREDIT_ENDDATE_GAP_MEAN"),
        pl.col("BUREAU_DAYS_CREDIT_FACT_GAP").mean().alias("BUREAU_DAYS_CREDIT_FACT_GAP_MEAN"),
        pl.col("BUREAU_ENDDATE_FACT_GAP").mean().alias("BUREAU_ENDDATE_FACT_GAP_MEAN"),
        pl.col("BUREAU_UPDATE_ENDDATE_GAP").mean().alias("BUREAU_UPDATE_ENDDATE_GAP_MEAN"),
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
        # Fractional Payment Flag: paid something but less than required
        ((pl.col("AMT_PAYMENT") > 0) & (pl.col("AMT_PAYMENT") < pl.col("AMT_INSTALMENT")))
            .alias("IS_FRACTIONAL"),
        # EWMA Recency Decay: decays exponentially for older instalments
        # Recent late payments get high weight; old ones approach zero
        ((pl.col("DAYS_ENTRY_PAYMENT") - pl.col("DAYS_INSTALMENT")) *
         (pl.col("DAYS_INSTALMENT").abs() * -0.01).exp())
            .alias("EWMA_DAYS_LATE"),
    ])
                                                    
    # Temporal Installment Payment Deficit: mean recent shortfall (last ~3 years)
    recent_deficit = (
        inst.filter(pl.col("DAYS_INSTALMENT") > -1000)
        .group_by("SK_ID_CURR")
        .agg(
            (pl.col("AMT_PAYMENT") - pl.col("AMT_INSTALMENT")).mean()
                .alias("INST_DEFICIT_MEAN_3Y")
        )
    )

    # Kept A2 baseline feature: recent installment windows.
    inst_1yr = (
        inst.filter(pl.col("DAYS_INSTALMENT") >= -365)
        .group_by("SK_ID_CURR")
        .agg([
            pl.col("PAYMENT_RATIO").mean().alias("INST_1YR_PAYMENT_RATIO_MEAN"),
            pl.col("PAYMENT_RATIO").min().alias("INST_1YR_PAYMENT_RATIO_MIN"),
            pl.col("DAYS_LATE").mean().alias("INST_1YR_DAYS_LATE_MEAN"),
            pl.col("DAYS_LATE").max().alias("INST_1YR_DAYS_LATE_MAX"),
            pl.col("IS_LATE").mean().alias("INST_1YR_LATE_FRAC"),
            pl.len().alias("INST_1YR_COUNT"),
        ])
    )

    inst_2yr = (
        inst.filter(pl.col("DAYS_INSTALMENT") >= -730)
        .group_by("SK_ID_CURR")
        .agg([
            pl.col("PAYMENT_RATIO").mean().alias("INST_2YR_PAYMENT_RATIO_MEAN"),
            pl.col("PAYMENT_RATIO").min().alias("INST_2YR_PAYMENT_RATIO_MIN"),
            pl.col("DAYS_LATE").mean().alias("INST_2YR_DAYS_LATE_MEAN"),
            pl.col("DAYS_LATE").max().alias("INST_2YR_DAYS_LATE_MAX"),
            pl.col("IS_LATE").mean().alias("INST_2YR_LATE_FRAC"),
            pl.len().alias("INST_2YR_COUNT"),
        ])
    )

    # Loan Restructuring Flag: unique instalment versions per SK_ID_PREV > 2 = renegotiated
    loan_restructure = (
        pl.scan_parquet(inst_path)
        .group_by("SK_ID_PREV")
        .agg([
            pl.col("NUM_INSTALMENT_VERSION").n_unique().alias("N_INST_VERSIONS"),
            pl.first("SK_ID_CURR").alias("SK_ID_CURR"),
        ])
        .with_columns(
            (pl.col("N_INST_VERSIONS") > 2).cast(pl.Int32).alias("IS_RESTRUCTURED")
        )
        .group_by("SK_ID_CURR")
        .agg([
            pl.col("IS_RESTRUCTURED").sum().alias("INST_RESTRUCTURED_COUNT"),
            pl.col("IS_RESTRUCTURED").mean().alias("INST_RESTRUCTURED_FRAC"),
        ])
    )

    out = (
        inst.group_by("SK_ID_CURR").agg([
            pl.col("PAYMENT_DIFF").mean().alias("INST_PAYMENT_DIFF_MEAN"),
            pl.col("PAYMENT_DIFF").sum().alias("INST_PAYMENT_DIFF_SUM"),
            pl.col("PAYMENT_RATIO").mean().alias("INST_PAYMENT_RATIO_MEAN"),
            pl.col("DAYS_LATE").max().alias("INST_DAYS_LATE_MAX"),
            pl.col("DAYS_LATE").mean().alias("INST_DAYS_LATE_MEAN"),
            pl.col("IS_LATE").mean().alias("INST_LATE_FRAC"),
            pl.col("AMT_INSTALMENT").max().alias("INST_MAX_INSTALMENT"),
            pl.col("AMT_PAYMENT").sum().alias("INST_TOTAL_PAID"),
        pl.col("IS_FRACTIONAL").sum().alias("INST_FRACTIONAL_PAY_COUNT"),
        pl.col("IS_FRACTIONAL").mean().alias("INST_FRACTIONAL_PAY_FRAC"),
        pl.col("EWMA_DAYS_LATE").sum().alias("INST_EWMA_LATE_SUM"),
        pl.col("EWMA_DAYS_LATE").mean().alias("INST_EWMA_LATE_MEAN"),
    ])
        .join(recent_deficit, on="SK_ID_CURR", how="left")
        .join(inst_1yr, on="SK_ID_CURR", how="left")
        .join(inst_2yr, on="SK_ID_CURR", how="left")
        .join(loan_restructure, on="SK_ID_CURR", how="left")
        .collect()
    )

    out = out.with_columns(
        (
            pl.col("INST_1YR_LATE_FRAC").fill_null(0)
            - pl.col("INST_LATE_FRAC").fill_null(0)
        ).alias("INST_1YR_LATE_FRAC_DELTA")
    )

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

    # AMT_CREDIT_LIMIT_ACTUAL is Int64 — cast to Float32 before division
    cc = pl.scan_parquet(cc_path).filter(
        pl.col("AMT_CREDIT_LIMIT_ACTUAL") > 0
    ).with_columns([
        (pl.col("AMT_BALANCE") /
         (pl.col("AMT_CREDIT_LIMIT_ACTUAL").cast(pl.Float32) + 1e-8))
            .alias("UTIL_RATIO"),
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

    # DPD Acceleration: recent 6-month max minus prior max (Markov trajectory)
    recent_dpd = pos.filter(pl.col("MONTHS_BALANCE") >= -6).group_by("SK_ID_CURR").agg(
        pl.col("SK_DPD").max().alias("POS_DPD_MAX_6M")
    )
    prior_dpd = pos.filter(pl.col("MONTHS_BALANCE") < -6).group_by("SK_ID_CURR").agg(
        pl.col("SK_DPD").max().alias("POS_DPD_MAX_PRIOR")
    )

    out = (
        pos.group_by("SK_ID_CURR").agg([
            pl.col("SK_DPD").max().alias("POS_DPD_MAX"),
            pl.col("SK_DPD").mean().alias("POS_DPD_MEAN"),
            pl.col("SK_DPD_DEF").max().alias("POS_DPD_DEF_MAX"),
            pl.col("CNT_INSTALMENT_FUTURE").mean().alias("POS_CNT_INSTALMENT_FUTURE_MEAN"),
            pl.col("MONTHS_BALANCE").count().alias("POS_MONTHS_COUNT"),
            (pl.col("NAME_CONTRACT_STATUS") == "Completed").mean().alias("POS_COMPLETED_FRAC"),
            # Sequential POS-Cash Arrears Severity EMA (halving life = 3 months)
            pl.col("SK_DPD").sort_by(pl.col("MONTHS_BALANCE"))
                .ewm_mean(half_life=3.0).last().alias("POS_DPD_EMA_3M"),
        ])
        .join(recent_dpd, on="SK_ID_CURR", how="left")
        .join(prior_dpd, on="SK_ID_CURR", how="left")
        .collect()
    )

    out = out.with_columns(
        (pl.col("POS_DPD_MAX_6M").fill_null(0) - pl.col("POS_DPD_MAX_PRIOR").fill_null(0))
            .alias("POS_DPD_ACCELERATION"),
    ).with_columns([
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
    ])

    # C3: Refused velocity (2Y)
    recent_prev = prev.filter(pl.col("DAYS_DECISION") >= -730)
    recent_agg = recent_prev.group_by("SK_ID_CURR").agg([
        pl.len().alias("PREV_2YR_TOTAL_APPS"),
        (pl.col("NAME_CONTRACT_STATUS") == "Refused").sum().alias("PREV_2YR_REFUSED_COUNT"),
    ])
    
    out = out.join(recent_agg, on="SK_ID_CURR", how="left").collect()
    
    out = out.with_columns([
        (pl.col("PREV_2YR_REFUSED_COUNT").fill_null(0) / 
         (pl.col("PREV_2YR_TOTAL_APPS").fill_null(0) + 1e-8))
            .alias("PREV_2YR_REFUSED_VELOCITY")
    ])
    out = out.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.Int64).cast(pl.Int32),
    ])
    out.write_parquet(cache_path)
    logger.info(f"[CACHE WRITE] {cache_path.name}: {out.shape}")
    return out
