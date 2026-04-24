# Handoff: automated feature loop (`new_possible_features.md`)

Give the following block to the **next AI coding agent** as its primary instructions. It should read **`new_possible_features.md`** for definitions and **`AGENT_HANDOFF.md`** for repo rules, then execute this loop.

---

## Agent execution contract (paste below)

You are continuing the Home Credit default-risk pipeline in this repo. **Objective:** maximize **5-fold stratified OOF ROC-AUC** using feature families listed in **`new_possible_features.md`**, evaluated only with **`run_full`** (100% train, 5 folds). **Do not** use `run_fast` AUC to keep or drop features (optional: `run_fast` only as a quick crash smoke test if the user allows time).

### Non-negotiables

- **Python:** `.\mlpr\Scripts\python.exe` from the project root (not system `python`). If that path is missing, stop and say so.
- **Logging:** no `print()`. Use the project `logger` only; metrics appear in **`experiment.log`**.
- **Config:** no new hardcoded hyperparameters outside **`pipeline/config.py`**. Keep **`RUN_OPTUNA=False`** unless the user explicitly overrides.
- **KNN:** leave **`USE_KNN=False`** (per project evidence: KNN hurt OOF).
- **Concurrency:** never run two pipeline processes at once (one log file).
- **Polars** for ETL/aggregations; **pandas** only at the modeling boundary, matching existing code.
- **Cache:** if you edit **`pipeline/aggregations.py`**, delete the affected files under **`cache/`** before `run_full` (e.g. `installments_debug0_seed42.parquet` for installment changes, `bureau_*` for bureau changes, `prev_app_*` for previous_application changes, `credit_card_*` for CC changes). **`fe_application` / `fe_bureau_derived` only** → no aggregation cache delete required.
- **Scope:** implement **one checklist unit per iteration** (below). Do not batch multiple unrelated tiers into a single evaluation unless the user explicitly asks (batching breaks attribution).

### Baseline and keep rule

- **Starting baseline OOF:** `0.79048` (no KNN, surgical TE, trinity, current stack).
- After each successful `run_full`, read **`Final OOF AUC`** from **`experiment.log`** for **this run’s** `run_id` (see “Log correlation”).
- **KEEP** the code change only if `Final OOF AUC >= baseline + 0.0001` (i.e. strictly beat the current baseline by at least **0.00001**). Round to 5 decimals for comparison (e.g. 0.79048 → next keep needs ≥ **0.79049**).
- If the change **does not** clear the bar: **revert** that unit’s code (git or manual) so the repo returns to the last kept state before starting the next unit.
- If a run **crashes** (OOM, exception): revert the unit, note the failure, move on or downscope that unit per your judgment.

### How to run and wait (no fixed 10-minute blind sleep)

1. From project root, start **one** background or foreground job:
   `.\mlpr\Scripts\python.exe -m entrypoints.run_full`
2. **Do not** assume 10 minutes is always enough. **Poll `experiment.log`** (tail new lines) until you see a line matching:
   `=== Pipeline Complete | run_id=<uuid> | Final OOF AUC: 0.xxxxx ===`
   for the **same** `run_id` as the `Run session | id=<uuid>` line at the start of that run. Typical wall time is often **~6–15 minutes** on a capable GPU machine; allow **up to ~45 minutes** before declaring a hung run (then investigate).
3. Record: `run_id`, start/end timestamps (from log), `Final OOF AUC`, per-fold AUC line if present, adversarial AUC warning if relevant, cache HIT/WRITE lines for that run.

### Log correlation

Each run begins with:
`Run session | id=<run_id> | ... | RUN_OPTUNA=False`
The final line for that evaluation is:
`=== Pipeline Complete | run_id=<run_id> | Final OOF AUC: 0.xxxxx ===`
Only compare **matching** `run_id` values so you do not read an older run’s score.

### Procedural checklist (process in this order)

Skip the **“Already implemented”** and **“Deprioritized or skip”** sections of `new_possible_features.md`. Implement and evaluate **one unit at a time**:

| Step | Unit | Primary files | Cache delete before `run_full` |
|------|------|---------------|----------------------------------|
| 1 | **A1** — EXT shape + pairwise ratios + `DAYS_*_DIV_EXT3` | `pipeline/features.py` (`fe_application`) | none |
| 2 | **A2** — Installments 1Y / 2Y windows + optional recency gap column | `pipeline/aggregations.py` | `cache/installments_*.parquet` |
| 3 | **A3** — Bureau **Closed** branch + mix / ratio features | `pipeline/aggregations.py` | `cache/bureau_*.parquet` |
| 4 | **B1** — Previous app approved/refused splits + optional 2Y stats | `pipeline/aggregations.py` | `cache/prev_app_*.parquet` |
| 5 | **B2** — Credit card payment vs min regularity + DPD flag rate if useful | `pipeline/aggregations.py` | `cache/credit_card_*.parquet` |
| 6 | **B3** — `SOCIAL_DEF_PER_OBS` | `pipeline/features.py` | none |
| 7 | **B4** — `CREDIT_PER_PERSON` | `pipeline/features.py` | none |
| 8 | **C1** — `AMT_CREDIT - AMT_GOODS_PRICE` (absolute gap) | `pipeline/features.py` | none |
| 9 | **C2** — Region/population vs income feature *only if* columns exist in `schema_sample.txt` | `pipeline/features.py` | none |
| 10 | **C3** — Refused velocity (2Y) on `previous_application` | `pipeline/aggregations.py` | `cache/prev_app_*.parquet` |

After each **KEEP**, set **baseline** to the new `Final OOF AUC` for the **next** comparison.

### If the user leaves the session for ~1 hour

- Work through the checklist **in order** until time runs out.
- It is **normal** not to finish all 10 units in one hour (each `run_full` dominates wall time).
- Prefer **finishing fewer units with clean revert/keep discipline** over half-implemented batches.

### Final report (write before ending the session)

Produce a **single markdown report** (e.g. `FE_RUN_REPORT.md` in the project root, or the chat transcript if the user forbids new files) containing:

1. **Summary table:** for each unit attempted — status (`KEPT` / `REVERTED` / `SKIPPED` / `CRASHED`), **Final OOF AUC**, **Δ vs baseline at start of that step**, `run_id`, and **one sentence why**.
2. **Final baseline** after all kept units.
3. **Code state:** list files touched in the final kept state.
4. **Issues:** OOM, drift warnings, cache problems, or ambiguous log lines.
5. **Next steps:** which checklist row to run next if the loop did not complete.

### Reference docs

- Feature definitions: **`new_possible_features.md`**
- Repo policy, entrypoints, cache keys: **`AGENT_HANDOFF.md`**
- Column names: **`schema_sample.txt`** (do not load huge CSVs into context)

---

## Note for the human operator

- The agent should **poll logs until completion**, not **sleep(600)** once and assume the run finished.
- To avoid merge conflicts, run **one agent / one machine** at a time on this repo.
