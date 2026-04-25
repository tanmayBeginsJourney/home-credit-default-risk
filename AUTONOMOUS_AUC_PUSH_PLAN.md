# Autonomous AUC Push Plan (No Optuna Window)

## Objective

- Starting baseline: `0.79829` (from `experiment.log`)
- Hard constraint: no Optuna runs for the next 12 hours
- Minimum acceptable target: `> 0.80000`
- Ideal target: `> 0.80500`
- Acceptance rule per change:
  - **Keep** if `delta_auc > +0.00010`
  - **Revert** if `delta_auc <= 0`
  - **Borderline** (`0 < delta_auc <= 0.00010`): default revert unless the change significantly improves stability/speed and is a prerequisite for a follow-up experiment
- After each accepted change, update baseline to the newly achieved AUC

---

## Execution Contract for Agent

The agent has full freedom to modify:
- feature engineering
- preprocessing
- model classes and model weights
- stacking logic
- non-Optuna hyperparameters
- validation/calibration strategy

The agent must still follow:
- config-driven behavior (`pipeline/config.py`)
- no `print()` in pipeline code; use project logger
- incremental, one-change-at-a-time loop
- strict accept/revert policy based on OOF AUC deltas

---

## Run Mode Policy (`run_fast.py` vs `run_full.py`)

Use this deterministic rule:

1. **Default decision mode**: run `entrypoints/run_full.py`
   - Full-run AUC is the only source of truth for keep/revert and baseline updates
   - This is mandatory for changes known to be data-volume sensitive

2. **When `run_fast.py` is allowed**
   - Use only for low-risk, low-context edits likely to preserve signal at 15% sample:
     - logging/instrumentation only
     - small blender/stacker code-path sanity checks
     - obvious bugfixes where correctness is binary
   - Never reject a data-heavy feature family solely because it underperforms in fast mode

3. **Force full-run directly (no fast triage) for:**
   - temporal/windowed/recency features
   - table aggregations requiring long-tail history
   - drift gating and adversarial-drop changes
   - any feature requiring stable class-conditional distributions
   - any change touching fold behavior, stacking behavior, or blend calibration

4. **Operational expectation**
   - A single full run of ~12–15 minutes is normal and must not trigger panic/revert by duration alone

---

## Pipeline Completion Detection via `experiment.log`

The agent must poll `experiment.log` and detect completion with:
- line containing: `=== Pipeline Complete | run_id=... | Final OOF AUC: X.XXXXX ===`

Recommended parse regex:
- `Final OOF AUC:\s*([0-9]+\.[0-9]+)`

Also parse context lines for diagnostics:
- `OOF Static Weights AUC:`
- `OOF Stacked Ridge AUC:`
- `Fold i/n Static Weights AUC:`
- `Adversarial Validation AUC:`
- `Zero-importance features`

Polling protocol (must use logs, not fixed sleep):
- Tail `experiment.log` periodically and look for either:
  - completion marker (`=== Pipeline Complete ... Final OOF AUC: ... ===`), or
  - explicit fatal error trace
- During polling, confirm forward progress from heartbeat-like lines such as:
  - fold completion lines
  - model-evaluation stage lines
  - aggregation merge lines
- Runtime of 15+ minutes is expected; do not classify as hang if logs continue progressing

Only classify as failed/hung when:
- no new log activity for a sustained interval well beyond normal stage durations, and
- no completion marker appears, and
- error handling confirms crash/stall likelihood

---

## Single-Experiment Loop (Must Be Followed Exactly)

For each experiment `E_k`:

1. **Record pre-state**
   - current baseline AUC (`baseline_auc`)
   - git commit hash (`git rev-parse HEAD`)
   - experiment ID (timestamp + short title)

2. **Apply one isolated change**
   - one conceptual unit only
   - update config toggles if needed
   - do not batch unrelated ideas

3. **Run mode selection**
   - default to `run_full.py` for score-bearing experiments
   - use `run_fast.py` only for sanity checks where sample-size bias is not material

4. **Poll `experiment.log`**
   - wait for pipeline completion marker
   - extract final OOF AUC

5. **Decision**
   - `delta = auc_new - baseline_auc`
   - if `delta > 0.00010` -> **ACCEPT**
     - keep code
     - set `baseline_auc = auc_new`
     - append accepted change to running changelog
   - if `delta <= 0` -> **REVERT**
     - hard revert only files touched in this experiment
     - verify repo returns to pre-experiment state
   - if `0 < delta <= 0.00010` -> **DEFAULT REVERT**
     - keep only if explicitly marked as enabling infrastructure for next high-value test

6. **Log insight**
   - what changed
   - expected mechanism
   - observed AUC delta
   - side metrics (drift AUC, zero-importance movement, runtime)
   - decision and rationale

---

## Prioritized Experiment Queue (No Optuna)

Order is designed for highest expected gain per compute hour based on top-solution patterns and current pipeline state.

### Phase A — Ensembling/Stacking Gains First

1. **Rank-normalized meta inputs before stacker**
   - Convert each base model OOF prediction to percentile rank before ridge stack
   - Also evaluate static blend on rank-normalized predictions
   - Rationale: reduces scale mismatch; known high-rank Home Credit trick

2. **Meta-learner sweep (non-Optuna, tiny grid)**
   - Compare `Ridge(alpha in {0.1,1,3,10})` and `LogisticRegression(C in {0.5,1,2})`
   - Keep model by full OOF AUC only
   - Tight bounded search only; no long tuning

3. **Constrained blend-weight search**
   - Grid search over simplex for `(LGBM, DART, CatBoost, XGB)` with 0.05 step
   - Evaluate on existing OOF predictions (cheap, no retraining)
   - Adopt best static blend if it beats stacker or improves stacker input quality

### Phase B — High-Impact Feature Additions

4. **OOF-safe KNN target mean feature family**
   - Build `neighbors_target_mean_k` with `k in {100, 300, 500}`
   - Inputs: `EXT_SOURCE_1/2/3`, `AMT_CREDIT/AMT_ANNUITY`
   - Strict fold-safe generation for train OOF; train-neighbor lookup for test
   - Existing implementation is present for one setting; extend to multi-k and robust NaN handling

5. **Recent-window trend/acceleration features**
   - Add delta features across windows (e.g., last 6m vs prior periods) for:
     - installment lateness/severity
     - POS DPD trajectory
     - bureau credit velocity
   - Prefer aggregated trend statistics over raw counts

6. **Interaction features around strongest drift+signal cols**
   - Bounded set only (avoid feature explosion):
     - ratio, difference, clipped ratio with `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `EXT_SOURCE_*`
   - Immediately prune if zero importance across all folds

### Phase C — Robustness and Generalization Controls

7. **Adversarial-drop gating refinement**
   - Current drift AUC is very high (~0.98+). Add stricter auto-drop for top drift features with weak predictive gain
   - Run with and without gated drop list to test net benefit

8. **Seed ensembling without hyperparameter search**
   - Train 2–3 seeds for strongest base model(s), blend predictions
   - Keep only if net full OOF gain justifies runtime

---

## Runtime and Failure Handling

- If run crashes (OOM/timeouts):
  - immediate revert of current experiment
  - reduce memory footprint (downcast/chunk/feature subset)
  - retry once
- If repeated failure on same experiment:
  - mark as infeasible under current hardware window
  - move to next experiment

---

## Baseline Tracking Format (append per full run)

Maintain an internal table (or markdown section) with:
- `experiment_id`
- `change_summary`
- `mode_run` (`fast`, `full`, or `fast->full`)
- `auc_before`
- `auc_after`
- `delta`
- `decision`
- `runtime`
- `notes`

This table is mandatory for deterministic final recommendation quality.

---

## Stop Conditions

Stop immediately when either is met:

1. `baseline_auc > 0.80500` (ideal target reached), or
2. Experiment queue exhausted with no further justified modifications

---

## Final Deliverable if Goal Not Reached

If after exhausting all queued experiments baseline is still `<= 0.80000`:

1. Produce a **work-attempt report** containing:
   - every attempted experiment
   - measured deltas
   - keep/revert decisions
   - what consistently helped/hurt

2. Produce a **new direction proposal** that is significant, justified, and deterministic:
   - include why prior path saturated
   - expected mechanism of gain
   - implementation scope
   - estimated runtime and risk
   - concrete acceptance criteria

3. Proposal quality bar:
   - cannot be generic (“do more feature engineering”)
   - must include exact feature/model/validation changes and why they should improve OOF AUC under this dataset

---

## Immediate Next Experiment (E1)

Start with:
- **Rank-normalized stacking + constrained static blend search**

Why first:
- high likelihood of improvement with limited implementation complexity
- aligns with top-solution blending behavior
- still run full-mode for final decisions because stack behavior is distribution-sensitive

