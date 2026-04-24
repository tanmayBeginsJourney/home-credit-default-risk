# Feature Engineering Run Report

## Summary

Starting baseline from prior work: `0.79163`.

| Unit | Status | Final OOF AUC | Delta vs step baseline | run_id | Reason |
| --- | --- | ---: | ---: | --- | --- |
| Step 0: restore kept A2 in code | KEPT | 0.79128 | -0.00035 vs historical baseline | `d68750a7e8b54c698011e7cdc129f644` | Rebuilt installments cache to 29 columns and confirmed A2 is no longer cache-only; score did not reproduce the historical C3 lift. |
| Tier A: bureau/bureau_balance lifecycle | KEPT | 0.79353 | +0.00190 | `01d70bbe349a41c887698fc05192f7f1` | Broad bureau status, DPD severity, closure timing, debt arithmetic, and date-gap features gave a clear lift. |
| Tier B: installment behavior depth | REVERTED | 0.79362 | +0.00009 | `0db1d1550dfc409db40a2c8faa97db3c` | Missed the `+0.00010` keep threshold by `0.00001`. |
| Tier C: previous-application gaps | REVERTED | 0.79355 | +0.00002 | `cfcd9632a86848dc81a6e41301219c78` | Amount/date gaps and missing count did not add enough over Tier A. |
| Tier D: POS breadth | REVERTED | 0.79343 | -0.00010 | `7a7701b5d25e49a38d16995af90f31eb` | POS status and installment-progress features hurt OOF. |
| Tier D: credit-card breadth | REVERTED | 0.79332 | -0.00021 | `b43381467435441d99f42021cb2b3f8d` | Receivable/drawing/payment-total/mature-count features hurt OOF. |
| Tier E: preprocessing outlier nulls | REVERTED | 0.79303 | -0.00050 | `03ed8e1ca7094ec19fbd349a11e17910` | Nulling extreme income, bureau request, and social-circle values hurt OOF. |
| Final kept-state verification | VERIFIED | 0.79351 | -0.00002 vs kept Tier A run | `ba3763cea9bc4de59986c1dd3d27a4d5` | Clean run with only kept features reproduced Tier A within small run/GPU variation. |

## Final Baseline

- Final kept baseline to use next: `0.79353`.
- Clean verification score: `0.79351`.
- Net gain over prior official baseline `0.79163`: about `+0.00190`.

## Code State

Kept file changes:

- `new_possible_features.md`: replaced stale checklist with Kaggle-grounded backlog and do-not-rerun list.
- `FE_AUTOMATION_HANDOFF.md`: updated baseline, keep rule, cache policy, checklist, and removed missing `AGENT_HANDOFF.md` dependency.
- `pipeline/aggregations.py`: restored kept A2 installment windows in code and added kept Tier A bureau/bureau_balance lifecycle features.

Final full-run cache shapes:

- `bureau_debug0_seed42.parquet`: `(305811, 46)`
- `installments_debug0_seed42.parquet`: `(339587, 29)`
- `credit_card_debug0_seed42.parquet`: `(102445, 10)`
- `pos_cash_debug0_seed42.parquet`: `(337252, 11)`
- `prev_app_debug0_seed42.parquet`: `(338857, 12)`

## Issues

- Rebuilding the A2 installments cache plus existing C3 code produced `0.79128`, not the historical `0.79163`; the later Tier A lift is still strong against both numbers.
- Adversarial validation remained high throughout, around `0.977`, consistent with earlier runs.
- Some reverted units left stale caches during their test runs; affected caches were rebuilt before later comparisons and final verification.

## Next Steps

- Start from final baseline `0.79353`.
- Highest-value next attempt is a smaller Bureau Tier A ablation/refinement: keep the lifecycle family but test pruning high-drift or weak subgroups using feature importance and adversarial-drift signals.
- Secondary path: tune model/ensemble weights on the new Tier A feature space; pure FE is now giving smaller marginal gains than the bureau lifecycle jump.
