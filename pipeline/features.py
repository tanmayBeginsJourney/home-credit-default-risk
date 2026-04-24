import polars as pl
from pipeline.utils import logger, profile_memory


@profile_memory
def fe_application(df: pl.DataFrame) -> pl.DataFrame:
    """Application-level feature engineering. All Polars expressions."""

    doc_cols = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22) if f"FLAG_DOCUMENT_{i}" in df.columns]

    # NAN_COUNT: row-level null count on raw columns — computed first before new cols are added.
    # Missingness pattern is a proven top-10 feature in every Home Credit winning solution.
    base_cols = [c for c in df.columns if c not in ("TARGET", "SK_ID_CURR")]
    df = df.with_columns(
        pl.sum_horizontal([pl.col(c).is_null() for c in base_cols])
            .cast(pl.Int32).alias("NAN_COUNT")
    )

    df = df.with_columns([
        # EXT SOURCE interactions
        (pl.col("EXT_SOURCE_1") * pl.col("EXT_SOURCE_2") * pl.col("EXT_SOURCE_3"))
            .alias("EXT_SRC_PROD"),
        pl.mean_horizontal(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
            .alias("EXT_SRC_MEAN"),
        pl.min_horizontal(["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
            .alias("EXT_SRC_MIN"),

        # Loan term in months — #1 feature in virtually every top solution
        (pl.col("AMT_CREDIT") / (pl.col("AMT_ANNUITY") + 1e-8))
            .alias("LOAN_TERM_MONTHS"),

        # Per-capita household income (more honest than raw income for multi-person families)
        (pl.col("AMT_INCOME_TOTAL") / (pl.col("CNT_FAM_MEMBERS") + 1))
            .alias("INCOME_PER_FAMILY_MEMBER"),

        # Recency of phone change relative to age — frequent contact info changes signal instability
        (pl.col("DAYS_LAST_PHONE_CHANGE") / (pl.col("DAYS_BIRTH") + 1))
            .alias("PHONE_CHANGE_STABILITY"),

        # Macro Capacity (DTI)
        (pl.col("AMT_CREDIT") / (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("CREDIT_INCOME_RATIO"),
        (pl.col("AMT_ANNUITY") / (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("ANNUITY_INCOME_RATIO"),
        (pl.col("AMT_GOODS_PRICE") / (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("GOODS_INCOME_RATIO"),

        # Loan-to-Value Proxy
        (pl.col("AMT_CREDIT") / (pl.col("AMT_GOODS_PRICE") + 1))
            .alias("GOODS_CREDIT_RATIO"),

        # Downpayment Strain
        ((pl.col("AMT_GOODS_PRICE") - pl.col("AMT_CREDIT")) /
         (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("DOWNPAYMENT_INCOME_STRAIN"),

        # Credit Velocity
        (pl.col("AMT_REQ_CREDIT_BUREAU_MON") /
         ((pl.col("AMT_REQ_CREDIT_BUREAU_YEAR") / 12) + 1e-8))
            .alias("CREDIT_SEEK_VELOCITY"),

        # Repayment burden
        (pl.col("AMT_ANNUITY") * 12 / (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("ANNUAL_ANNUITY_INCOME_RATIO"),

        # Employment quality
        (pl.col("DAYS_EMPLOYED") / (pl.col("DAYS_BIRTH") + 1))
            .alias("EMPLOYMENT_DURATION_FRAC"),

        # Document completeness proxy
        pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in doc_cols])
            .alias("DOCUMENT_COUNT"),
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
        # Simple DTI (undecayed)
        ((pl.col("ACTIVE_CREDIT_SUM") + pl.col("AMT_CREDIT")) /
         (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("TOTAL_DTI"),

        # Time-Decayed DTI: exponential decay weight exp(-0.01 * |DAYS_CREDIT|)
        # penalises older active credits; more recent debt counts more
        ((pl.col("ACTIVE_CREDIT_SUM_DECAYED") + pl.col("AMT_CREDIT")) /
         (pl.col("AMT_INCOME_TOTAL") + 1))
            .alias("TIME_DECAYED_DTI"),

        # Fractional debt acceleration rate
        (pl.col("BUREAU_AMT_CREDIT_SUM") /
         (pl.col("BUREAU_AMT_CREDIT_MEAN") * 2 + 1e-8))
            .alias("DEBT_ACCEL_RATE"),

        # Late payment to active credit ratio
        (pl.col("INST_LATE_FRAC") /
         (pl.col("BUREAU_ACTIVE_COUNT").cast(pl.Float32) + 1e-8))
            .alias("LATE_TO_ACTIVE_RATIO"),

        # Annuity to historical peak installment
        (pl.col("AMT_ANNUITY") /
         (pl.col("INST_MAX_INSTALMENT") + 1e-8))
            .alias("ANNUITY_TO_PEAK_INST_RATIO"),
    ])

    logger.info(f"fe_bureau_derived: {df.shape[1]} cols after derived FE")
    return df
