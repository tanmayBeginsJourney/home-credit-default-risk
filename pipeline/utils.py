import os
import gc
import random
import logging
import psutil
from functools import wraps
from pathlib import Path
import numpy as np
import polars as pl
from pipeline import config

logging.basicConfig(
    filename=config.LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    force=True,   # override any handlers set by 3rd-party libs before us
)
logger = logging.getLogger("home_credit")


def flush_logger() -> None:
    """Force log lines to disk (FileHandler can otherwise buffer for minutes)."""
    for log in (logger, logging.getLogger()):
        for h in log.handlers:
            h.flush()


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
            df = pl.read_csv(csv_path, infer_schema_length=100_000, encoding="utf8-lossy")
            df.write_parquet(pq_path)
            size_mb = pq_path.stat().st_size / (1024 ** 2)
            logger.info(f"  → {pq_path.name} ({size_mb:.1f} MB)")
