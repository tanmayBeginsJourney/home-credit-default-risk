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

    # Cast all string cols to Categorical immediately
    cat_cols = [c for c, d in zip(df.columns, df.dtypes) if d == pl.String]
    df = df.with_columns([pl.col(c).cast(pl.Categorical) for c in cat_cols])

    # Domain cleaning
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

    # Downcast immediately
    df = df.with_columns([
        pl.col(pl.Float64).cast(pl.Float32),
        pl.col(pl.Int64).cast(pl.Int32),
    ])

    # Drop confirmed zero-importance features
    drop = [c for c in config.COLS_TO_DROP if c in df.columns]
    if drop:
        df = df.drop(drop)
        logger.info(f"Dropped {len(drop)} zero-importance cols")

    logger.info(f"Loaded {'train' if is_train else 'test'}: {df.shape}")
    return df
