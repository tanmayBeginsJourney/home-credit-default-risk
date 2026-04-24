# Kaggle-grounded feature backlog

**Current best 5-fold OOF ROC-AUC:** `0.79163`

This backlog replaces the older checklist. It is based on repo results plus accessible high-scoring or highly reused Home Credit public notebooks and mirrors:

- Kaggle code listing sorted by score: https://www.kaggle.com/competitions/home-credit-default-risk/code?competitionId=9120&sortBy=scoreDescending&excludeNonAccessedDatasources=true
- jsaguiar, LightGBM with Simple Features: https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features
- ogrellier, LightGBM with Selected Features: https://www.kaggle.com/code/ogrellier/lighgbm-with-selected-features
- aantonova, 0.797 LGBM and Bayesian Optimization: https://www.kaggle.com/code/aantonova/797-lgbm-and-bayesian-optimization
- Will Koehrsen manual FE: https://www.kaggle.com/code/willkoehrsen/introduction-to-manual-feature-engineering
- Will Koehrsen manual FE part 2: https://www.kaggle.com/code/willkoehrsen/introduction-to-manual-feature-engineering-p2
- ekrembayar step-by-step notebook: https://www.kaggle.com/code/ekrembayar/homecredit-default-risk-step-by-step-1st-notebook
- oskird top-3% solution repo: https://github.com/oskird/Kaggle-Home-Credit-Default-Risk-Solution

Kaggle pages are JavaScript-rendered; source was inspected through notebook pages and `scriptcontent` downloads where available.

## Current kept stack

Already present or treated as baseline, do not re-add as new:

- Application: `EXT_SRC_PROD`, `EXT_SRC_MEAN`, `EXT_SRC_MIN`, `LOAN_TERM_MONTHS`, `INCOME_PER_FAMILY_MEMBER`, `PHONE_CHANGE_STABILITY`, `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `GOODS_INCOME_RATIO`, `GOODS_CREDIT_RATIO`, `DOWNPAYMENT_INCOME_STRAIN`, `CREDIT_SEEK_VELOCITY`, `ANNUAL_ANNUITY_INCOME_RATIO`, `EMPLOYMENT_DURATION_FRAC`, `NAN_COUNT`.
- Bureau/POS/CC/previous/installments: bureau active split and decay, all-time installments behavior, installment deficit/restructure, POS DPD EMA/acceleration, credit-card utilization, previous-app refusal stats.
- Kept from the last FE loop: **A2 installments 1Y/2Y windows** and **C3 previous-app refused velocity**.

## Do not re-run unchanged

These were already tested against the local stack and reverted for failing the `+0.00010` OOF threshold:

- EXT score dispersion, max-min, pairwise ratios, and DAYS/EXT interactions.
- Bureau closed branch statistics and active/total ratios in the older narrow form.
- Broad previous-app approved/refused aggregations and refused/approved credit difference.
- Credit-card payment-vs-min-regularity ratio and DPD flag rate.
- `SOCIAL_DEF_PER_OBS`, `CREDIT_PER_PERSON`, `CREDIT_GOODS_PRICE_GAP`, `POPULATION_INCOME_RATIO`.

## Tier A - Bureau and bureau_balance lifecycle

**Why:** Top public pipelines repeatedly squeeze signal from bureau status history, DPD severity, closure timing, debt arithmetic, and date deltas. This is broader than the rejected old closed-branch-only test.

Implement one coherent bureau lifecycle unit in `agg_bureau`; delete `cache/bureau_*.parquet` before `run_full`.

- `BB_STATUS_0..5_COUNT`, `BB_STATUS_0..5_FRAC`, `BB_NONZERO_DPD_COUNT`, `BB_NONZERO_DPD_FRAC`, `BB_SEVERE_DPD_COUNT`, `BB_SEVERE_DPD_FRAC`.
- `BB_FIRST_STATUS_*`, `BB_LAST_STATUS_*`, `BB_MONTHS_OBSERVED`, `BB_MONTHS_CLOSED_TO_END` style closure timing where feasible.
- Bureau row helpers before client aggregation: credit minus debt, credit minus limit, credit minus overdue, credit enddate/fact/update gaps, days-credit minus overdue.
- Client aggregations should stay compact: mean/max/sum for counts and fractions; mean/min/max/sum/var only for the strongest numeric debt/date helpers.

## Tier B - Installment behavior depth

**Why:** Current kept features cover all-time behavior and 1Y/2Y windows, but top notebooks commonly include clipped DPD/DBD, payment-ratio variance, and entry-payment recency.

Implement in `agg_installments`; delete `cache/installments_*.parquet`.

- Row helpers: `INST_DPD_POS = max(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0)`, `INST_DBD_POS = max(DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT, 0)`.
- Aggregate `DPD_POS` max/mean/sum, `DBD_POS` max/mean/sum, `PAYMENT_RATIO` max/sum/var, `PAYMENT_DIFF` max/var, `DAYS_ENTRY_PAYMENT` max/mean.
- Optional compact loan-level then client-level block: aggregate by `SK_ID_PREV`, then client-level mean/max/sum over loan summaries. Keep it small to avoid feature explosion.

## Tier C - Previous-application gaps

**Why:** Strong public notebooks use previous-application amount gaps and date gaps more often than only raw amount means.

Implement in `agg_previous_application`; delete `cache/prev_app_*.parquet`.

- Row helpers: `PREV_APP_CREDIT_RATIO`, `PREV_APP_CREDIT_GAP`, `PREV_APP_GOODS_GAP`, `PREV_GOODS_CREDIT_GAP`, `PREV_FIRST_DRAWING_FIRST_DUE_GAP`, `PREV_TERMINATION_LT_500`, `PREV_MISSING_COUNT`.
- Aggregate by client with mean/min/max/sum where sensible; do not re-add broad approved/refused split features unchanged.

## Tier D - POS and credit-card breadth

**Why:** Public kernels frequently aggregate categorical status means and broad numeric monthly-balance summaries. The local stack currently has a very compact POS/CC subset.

Evaluate as separate units:

- POS (`agg_pos_cash`, delete `cache/pos_cash_*.parquet`): status means for common `NAME_CONTRACT_STATUS`, installment-progress ratio `CNT_INSTALMENT_FUTURE / CNT_INSTALMENT`, and flag `CNT_INSTALMENT > CNT_INSTALMENT_FUTURE`.
- Credit card (`agg_credit_card`, delete `cache/credit_card_*.parquet`): receivable totals, drawing ATM/POS/current totals and counts, payment-total-current stats, mature-installment count stats. Do not re-run payment-vs-min-regularity unchanged.

## Tier E - Preprocessing and low-cost application tests

**Why:** Several high-scoring notebooks null obvious outliers before modeling. This may help more than another small ratio at this stage.

Evaluate one small preprocessing unit at a time in `pipeline/data.py`; no aggregation cache delete unless aggregation code changes.

- Null extreme `AMT_REQ_CREDIT_BUREAU_QRT > 10`, social-circle counts above `40`, and extreme `AMT_INCOME_TOTAL > 1e8`.
- Consider `OWN_CAR_AGE` outlier nulling instead of p99 clipping in a separate test.
- Application interactions only if materially different from rejected A1: pairwise EXT products and EXT-by-age/employment products, not ratios or dispersion.

## Execution order

1. Restore/verify kept A2 and C3 are code-defined, not cache-only.
2. Tier A bureau lifecycle.
3. Tier B installment behavior depth.
4. Tier C previous-application gaps.
5. Tier D POS, then Tier D credit-card.
6. Tier E preprocessing tests.

Run **one unit per experiment** with `.\mlpr\Scripts\python.exe -m entrypoints.run_full`.

**Keep rule:** keep only if `Final OOF AUC >= current_baseline + 0.00010`. If the gain is `+0.00010` to `+0.00025`, run a confirmation `run_full`; keep only if the average of the two full runs still clears the threshold.
