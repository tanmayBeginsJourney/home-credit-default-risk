# Feature backlog — ranked for marginal OOF AUC

**Context:** Baseline is **~0.79048** (5-fold OOF, `USE_KNN=False`, surgical TE + trinity). Work **one group per experiment**, then `run_full` and compare. Invalidate `cache/*.parquet` when you change `aggregations.py`.

**Already implemented (do not re-add as “new”):**  
`EXT_SRC_PROD`, `EXT_SRC_MEAN`, `EXT_SRC_MIN`, `LOAN_TERM_MONTHS` (= credit/annuity), `INCOME_PER_FAMILY_MEMBER`, `PHONE_CHANGE_STABILITY`, `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `GOODS_INCOME_RATIO`, `GOODS_CREDIT_RATIO`, `DOWNPAYMENT_INCOME_STRAIN`, `CREDIT_SEEK_VELOCITY`, `ANNUAL_ANNUITY_INCOME_RATIO`, `EMPLOYMENT_DURATION_FRAC` (= employed/age), `NAN_COUNT`, bureau active split + decay, installments all-time aggs + EWMA/late/deficit/restructure, CC util (all + 12M p95), POS DPD EMA/acceleration, prev `PREV_REFUSED_FRAC` and related prev stats.

---

## Tier A — Highest expected lift (missing vs typical 0.805 solutions)

### A1. EXT score shape (application, `fe_application`)

| Feature | Definition | Why |
|--------|------------|-----|
| `EXT_SRC_STD` | Std dev of `EXT_SOURCE_1..3` (row-wise) | Captures *disagreement* between bureaus; trees get product/mean but not dispersion. |
| `EXT_SRC_MAX_MINUS_MIN` | `max(EXT*) - min(EXT*)` | Same idea, robust monotone signal. |
| `EXT_SOURCE_i_DIV_EXT_j` | Pairwise ratios (3 pairs), `+ ε` | Nonlinear mixes of relative standing; orthogonal to product in many regions. |
| `DAYS_EMPL_DIV_EXT3` | `DAYS_EMPLOYED / (EXT_SOURCE_3 + ε)` | Repeatedly cited in 1st-place writeups; links tenure to external score scale. |
| `DAYS_BIRTH_DIV_EXT3` | `DAYS_BIRTH / (EXT_SOURCE_3 + ε)` | Age vs risk score; different curvature than linear terms. |

**Note:** Use the **same cleaned `DAYS_EMPLOYED`** as `data.py` (no 365243 in ratios).

---

### A2. Installments — **1Y / 2Y windows** (`agg_installments`)

**Why:** You already aggregate **all-time** behavior. Recent payment stress is a different signal; top solutions almost always add **recent-only** stats.

Filter on `DAYS_INSTALMENT >= -365` and `>= -730` (separate blocks). Per window, per `SK_ID_CURR`, suggest:

- Mean / min `PAYMENT_RATIO`, mean / max `DAYS_LATE`, mean `IS_LATE`, count rows  
- **Derived:** `INST_1YR_LATE_FRAC - INST_LATE_FRAC` (recency vs history) — one column, high interpretability.

Merge with prefixes `INST_1YR_*`, `INST_2YR_*` to avoid name clashes.

---

### A3. Bureau — **Closed** branch + mix features (`agg_bureau`)

**Why:** You have strong **Active** aggregates. **Closed** history (counts, sum/mean `AMT_CREDIT_SUM`, `DAYS_CREDIT` stats) captures *completed* credit lifecycle and mix.

- Parallel `group_by` for `CREDIT_ACTIVE == "Closed"`.
- Ratios: `BUREAU_ACTIVE_COUNT / (bureau row count)`, `ACTIVE_CREDIT_SUM / (ACTIVE+CLOSED sum + ε)` — adjust to your naming conventions.

---

## Tier B — Solid second wave

### B1. Previous application — **Approved vs Refused** splits (`agg_previous_application`)

**Why:** `PREV_REFUSED_FRAC` exists; many gains come from **conditional** stats (e.g. mean `AMT_CREDIT` / `AMT_ANNUITY` **among refused only** vs **approved only**, counts in last 730 `DAYS_DECISION`).

- `PREV_APPROVAL_RATE` = approved count / total prev rows (explicit; may correlate with refused frac but trees still use both forms).
- Optional: refused-only mean credit, approved-only mean credit, **difference** or ratio.

---

### B2. Credit card — **payment discipline** (`agg_credit_card`)

**Why:** You have utilization and DPD aggregates. Add row-level helpers then aggregate:

- `AMT_PAYMENT_CURRENT / (AMT_INST_MIN_REGULARITY + ε)` → mean / max per `SK_ID_CURR`.
- Mean of `(SK_DPD > 0)` (DPD flag rate) if not redundant with existing `CC_DPD_*`; if importances collapse, drop one.

---

### B3. Application — **social circle** (`fe_application`)

| Feature | Definition | Why |
|--------|------------|-----|
| `SOCIAL_DEF_PER_OBS` | `(DEF_30 + DEF_60) / (OBS_30 + OBS_60 + 1)` | Compresses four columns into one stress ratio; low implementation cost. |

---

### B4. Application — **capacity per mouth** (`fe_application`)

| Feature | Definition | Why |
|--------|------------|-----|
| `CREDIT_PER_PERSON` | `AMT_CREDIT / (CNT_FAM_MEMBERS + 1)` | Not the same as income/family; captures household leverage. |

---

## Tier C — Worth a single experiment each

- **`AMT_CREDIT - AMT_GOODS_PRICE`** (absolute gap, not only normalized strain) — may add a linear component trees fragment less efficiently.  
- **Region × income bin** or **`REGION_POPULATION_RELATIVE` / `AMT_INCOME_TOTAL`** if present in schema — weak priors, can help tail cases.  
- **Refused velocity (2Y):** count of refused prev apps with `DAYS_DECISION >= -730` / total prev in window — complements static refusal rate.

---

## Deprioritized or skip

| Item | Reason |
|------|--------|
| **KNN target feature** (`USE_KNN`, extra `KNN_COLS`) | Empirically **hurt** OOF vs off at ~0.79048; leave off unless you run a deliberate A/B. |
| **`EXT_SRC_NANFLAG` (count of null EXT only)** | Largely subsumed by **`NAN_COUNT`** and per-column null patterns; low marginal value. |
| **`CREDIT_ANNUITY_RATIO`** | Same as **`LOAN_TERM_MONTHS`**. |
| **`EMPLOYED_PERC_AGE` / `INCOME_PER_PERSON`** | Same as **`EMPLOYMENT_DURATION_FRAC`** / **`INCOME_PER_FAMILY_MEMBER`**. |
| **`DOCS_PROVIDED` / `DOCS_RATIO`** | **`DOCUMENT_COUNT`** was zero-importance and dropped; FLAG_DOCUMENT variants are mostly in **`COLS_TO_DROP`**. Unlikely to help. |
| **Huge “cumulative AUC” tables from old Kaggle posts** | Your stack is already strong; treat those numbers as **illustrative**, not promises. |

---

## Suggested order of attack

1. **A1** (EXT shape + ÷EXT3 ratios) — no cache rebuild, fastest iteration.  
2. **A2** (installment 1Y/2Y) — high historical lift in writeups; requires cache delete for installments.  
3. **A3** (bureau closed + mix).  
4. **B1** → **B2** → **B3** / **B4**.  
5. **Tier C** as one-off tries if A/B still short of 0.80.

**Keep rule:** adopt a change only if full OOF improves by **~≥ 0.0001** vs your current best; otherwise revert.
