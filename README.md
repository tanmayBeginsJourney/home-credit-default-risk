# Home Credit ML Pipeline

Production-style, config-driven ML pipeline for Kaggle Home Credit Default Risk.

## Goal
- Iterate quickly and safely toward **>= 0.80 ROC-AUC** using strict OOF validation.

## Quick Start
1. Use the existing `mlpr` environment (no new venv/pip setup in this repo).
2. Put Kaggle data in `data/raw`.
   - Preferred: `.parquet`
   - `.csv` also works: first run auto-converts to parquet.
3. **Debug / fast (15% train sample, 3 folds):** `python -m entrypoints.run_fast` (or `from entrypoints.run_fast import main; main()`).
4. **Full 5-fold OOF on 100% data** (same as production evaluation path, default `RUN_OPTUNA=False` in `config.py`): `python -m entrypoints.run_full` (or `from entrypoints.run_full import main; main()`).
5. **Hyperparameter search:** set `RUN_OPTUNA = True` in `pipeline/config.py`, then use **`run_full` only** (never Optuna in debug).
6. Check output in **`experiment.log`**. For context, read **`AGENT_HANDOFF.md`**.

## Entry Points
- **`entrypoints.run_fast`:** debug — 15% sample, 3 folds, small trees, Optuna off.
- **`entrypoints.run_full`:** 100% data, 5-fold OOF, trinity + surgical TE; Optuna only if `RUN_OPTUNA=True` in `config.py`.

## Project Layout
- `pipeline/data.py`: loading, cleaning, type casting, downcasting.
- `pipeline/aggregations.py`: table-level and temporal aggregations (Polars).
- `pipeline/features.py`: applicant-level + derived features.
- `pipeline/model.py`: adversarial validation, CV training, ensembling, Optuna hooks.
- `pipeline/utils.py`: logging, seeding, parquet conversion, runtime helpers.
- `pipeline/config.py`: single source of truth for toggles, params, paths, seeds.
- `entrypoints/run_fast.py`: debug-mode orchestrator.
- `entrypoints/run_full.py`: full-mode orchestrator.
- `experiment.log`: **only** log file; full run history and OOF (git-ignored with `*.log`).

## Config Modes (`pipeline/config.py`)
- **Mode is chosen by the entrypoint** (`apply_mode_debug` / `apply_mode_full`); do not rely on `DEBUG_MODE` in the file alone.
- `RUN_OPTUNA=True`: run Optuna on **full data** when using `run_full` (forbidden in `run_fast`).
- `USE_KNN`, `DROP_ADVERS`, `COLS_TO_DROP`, ensemble weights, and model params are all config-controlled.

## Team Iteration Loop
1. Make one focused change (single feature family, one aggregation block, one HPO tweak).
2. Run `run_fast`.
3. Compare OOF ROC-AUC and fold behavior from `experiment.log`.
4. Keep change only if it helps and is stable.
5. If zero-importance features are reported, prune via `COLS_TO_DROP`.

## Contribution Rules
- Do not hardcode hyperparameters or mode logic outside `pipeline/config.py`.
- Use logging only; no `print()` in pipeline code.
- Prefer Polars for data/aggregation work; convert to pandas only for modeling.
- Keep changes modular and small to reduce regression risk.
- Preserve important project docs: `spec.md` and `AGENT_HANDOFF.md`.

## Data/Join Context
Primary key: `SK_ID_CURR`.
- Base: `application_train`, `application_test`
- Direct joins: `bureau`, `previous_application`
- Aggregate-first tables: `bureau_balance`, `POS_CASH_balance`, `installments_payments`, `credit_card_balance`

## Git Hygiene
Ignored artifacts include local data/cache and runtime outputs (`data/`, `cache/`, logs, `catboost_info/`, `__pycache__/`, etc.).  
Before pushing, verify diffs are code/config/docs only.

## Common Commands
```bash
# debug (15% data, 3 folds)
python -m entrypoints.run_fast

# full 5-fold OOF (100% data). Set RUN_OPTUNA=True in config for HPO.
python -m entrypoints.run_full
```
