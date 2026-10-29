# Protocol v1: concentrated associations and the chi-square workflow

Fixed 2026-10-29, before the confirmatory runs. The pilots (pilot1–pilot5) were exploratory and are not reported as evidence.

## Aim
To examine how the three steps of the standard chi-square workflow behave when an association in an I × J table is concentrated in a few cells:
1. the omnibus gate;
2. the residual map;
3. the column-proportion letters.

## Analytical results (numerical checks, theory_checks.py; seed 7)
- **Result 1 (gate):** for a one-cell association, the cell's adjusted residual carries the entire chi-square non-centrality. Check: ratio = 1 to within 1e-6 over 24 configurations.
- **Result 2 (map):** an innocent cell's adjusted residual has mean √N λ_ij. Check: simulated mean z against prediction (4,000 tables per condition).
- **Result 3 (robust fit):** median polish on log probabilities recovers independence exactly when fewer than half of the cells in each row and column deviate (noise-free check).
- **Result 4 (letters):** for a row i ≠ r, P(i|c)/P(i|j) = 1/{1 + (θ − 1)P(r|c)}.

## Experiment 1: gate (exp1_gate.py; seed 101; 1,500 tables per condition)
- **Tables:** 2×3, 3×4, 4×5, 6×8, 8×10.
- **N:** 200, 1,000, 5,000. **Margins:** equal, skewed (rows 0.6^i, columns 0.7^j).
- **Patterns:** null; one cell; two cells (different rows and columns); diffuse (double-centred normal interaction on the log scale). For the non-null patterns, the strength is set so that the chi-square non-centrality gives chi-square power 0.50.
- **Tests (α = 0.05):** Pearson chi-square; G²; maximum |adjusted residual| with a cutoff from 20,000 null tables with the same margins; adjusted residuals with Bonferroni; Fisher cell-versus-rest with Bonferroni.
- **Outcomes:** rejection rate. For the one-cell pattern, also the share of tables in which the true cell is detected by Fisher with Bonferroni, and the share of those detections with a non-significant chi-square.

## Experiment 2: map (exp2_map.py; seed base 2000; 200 tables per cell; B = 99 calibration tables)
- **Tables:** 4×5, 6×8, 8×10. **N:** 200, 1,000, 5,000. **Margins:** equal, skewed.
- **Deviating cells:** K ∈ {0, 1, 3}, θ = 2, scattered; also K = 3 in one row.
- **Methods:**
  - Pearson residuals, no correction.
  - Adjusted residuals: no correction, Bonferroni, Holm, BH.
  - Fisher cell-versus-rest with Bonferroni.
  - Kotze–Hawkins, calibrated.
  - Robust two-stage, calibrated.
  - Backward stepping.
- **Outcomes:** F1, recall, false flags in linked cells (same row or column as a deviating cell) and in other cells; familywise error (K = 0); omnibus significance.

## Experiment 3: letters (exp3_letters.py; seed 303; 5,000 tables per condition)
- **Rows × columns:** 3×3, 4×3, 5×4, 6×4.
- **N:** 300, 1,000, 3,000, 10,000. **θ:** 1, 1.5, 2 (one cell). **Row margins:** equal, skewed.
- **Tests:** two-proportion z tests for every pair of columns within each row, Bonferroni within the row (SPSS default).
- **Outcomes:** share of tables with letters in the true row; mean number of unaffected rows with letters; share of tables with any letter when θ = 1.

## Hypotheses and criteria
A hypothesis is supported when all its criteria pass. One-sided p values against each threshold are computed after the runs: binomial tests for shares, t tests over conditions or tables for means and differences, and bootstrap tests for ratios (2,000 resamples).

| H | Criterion | Threshold |
| --- | --- | --- |
| H1 | One cell, tables with 35 or more df (6×8, 8×10), pooled over N and margins: Fisher-Bonferroni power minus chi-square power | ≥ 0.10 |
| H2 | Diffuse, tables with 35 or more df: chi-square power minus Fisher-Bonferroni power | ≥ 0.15 |
| H3 | One cell, tables with 35 or more df, N = 1,000: share of Fisher-detected true cells whose chi-square is non-significant | ≥ 0.25 |
| H4 | Map, scattered K ≥ 1: ratio of mean linked false flags (adjusted residuals, Bonferroni) at N = 5,000 to N = 1,000 | ≥ 3 |
| H5 | Map, scattered K ≥ 1, N = 5,000: robust two-stage F1 minus adjusted residuals (Bonferroni) F1 | ≥ 0.15 |
| H6 | Map, K = 0: robust two-stage familywise error, mean over conditions, and maximum over conditions | ≤ 0.075 and ≤ 0.12 |
| H7 | Letters, θ = 1.5, N = 10,000: mean share of unaffected rows carrying a letter, pooled | ≥ 0.50 |
| H8 | Letters, θ = 1, tables with 4 or more rows: share of tables with any letter, pooled | ≥ 0.10 |

## Decisions recorded before the runs
- The missing-data factor is dropped. The Python code is final. Data sets with an empty row or column are skipped and counted.
- The theory checks are deterministic and will be reported as computed.

## Amendment 1 (2026-10-29, before Experiment 2 started)
- A bug in the robust two-stage method was found while preparing the real-data illustration. When the candidate cells covered a whole row or column, the quasi-independence refit predicted zero there, and the calibration cutoff became effectively infinite.
- **Fix:** candidates are capped so that every row and column keeps more than half of its cells. This is the breakdown condition of Result 3; the weakest candidates are dropped first.
- Backward stepping uses the same cap.
- The fix was made before any confirmatory Experiment 2 output existed. Experiments 1 and 3 are unaffected.

## Amendment 2 (2026-10-29, after the confirmatory runs; reporting only)
- **Decision labels.** The protocol defined support as all criteria met. Table 4 now also separates hypotheses whose estimate met the threshold but whose one-sided p value was not below 0.05. This applies to H7 (p = 0.062), which is labeled "Threshold met, not significant".
- **Calibration check (calib_check.py; seed base 4040; 150 tables per condition).** This was added after a review. It compared the robust refit's Monte Carlo cutoff with B = 99 and B = 999 on 3,555 new tables.

## Amendment 3 (2026-10-29, after the confirmatory runs; computation fix)
- **H8 counting.** H8 is defined as the share of tables with a letter in any row. The original analysis counted letters only in rows other than the first, which understated the share (0.147). Experiment 3 was rerun with the same seed and an added counter. All earlier outputs were reproduced exactly. The corrected H8 estimate is 0.178 (p < 0.001), and the decision is unchanged.
