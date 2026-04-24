# entrypoints/run_full.py
# 100% train/test rows, 5-fold OOF, surgical TE + LGBM + CatBoost + XGB.
# Default: no Optuna (set RUN_OPTUNA = True in pipeline/config.py for HPO on full data).
# DEBUG caches (debug1_*) are separate from full caches (debug0_*) — no conflicts.
from entrypoints.run_fast import main

if __name__ == "__main__":
    main(full=True)
