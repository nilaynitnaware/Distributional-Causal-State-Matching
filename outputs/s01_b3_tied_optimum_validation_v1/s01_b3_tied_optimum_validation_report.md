# S01-B3 tied-optimum validation report

## Verdict

The implementation now has separate targeted validation of optimizer-generated tied effective optima and simultaneous root-flat physical-command intersection. The claim is accurate with the stated scope: tiny population Gaussian SCMs, bounded shift/scale commands, finite retained mode ties, and direct KL certification. These tests do not establish broad worst-case scalability, support for arbitrary continuous nonroot degeneracy, or a new theorem.

No substantive implementation defect was found. No production source, policy score, tie rule, tolerance, archived result, or manuscript file was changed.

## Frozen validation set and independent reference

`frozen_cases.json` was written before final execution. It contains four realizable cases (n=2 or n=3, B=1) and one cap condition:

- R1: a genuine nonroot shift/scale global tie for DAG `0->1`, with two tied optima also retained for the reverse DAG. Both candidates share `shift_1=1`, which is directly KL-certified.
- R2: a frozen 0.1 mean perturbation breaks R1's forward-DAG tie. The remaining scale command cannot be shared under B=1 because a nonidentity scale on the reverse-DAG root still consumes global commanded support.
- R3: all three path orientations are optimized independently; fixed-coordinate incompatibility proves the simultaneous intersection empty.
- R4: all three path orientations share `shift_0=1`; direct target-to-candidate KL is checked for each candidate.
- C1: the two-candidate family-product upper bound is 4. Cap 3 returns inconclusive with zero charged combinations; cap 4 recovers the expected certificate. An uncached-check cap of zero is also inconclusive, and one-step scoring/fallback does not authorize a stop.

The independent reference enumerates identity and every B=1 shift/scale mode, solves each continuous scalar quadratic analytically under bounds, computes interventional Gaussian moments itself, and evaluates KL with its own Cholesky formula. It never calls the production sparse optimizer or local-state solver. Production optima/families are compared to this complete finite mode enumeration.

## Results

All four SCM cases matched the independent optimum values and complete retained action sets. Both positive cases produced bounded, positive-scale commands within the one-coordinate global budget, and every candidate loss residual was within its declared KL tolerance. Both negative cases are justified by exhaustive mode enumeration plus exact coordinate/budget incompatibility, not numerical failure to find a witness.

The final focused command was:

```text
python -m pytest -q tests/test_s01_tied_optimum_validation.py tests/test_command_space_matching.py tests/test_s01_tree_population.py tests/test_matching_query_complexity_research.py tests/test_small_mec_matching_query.py
```

It passed: `39 passed in 1.42s` (runner-recorded subprocess time 1.76 s). The complete runner, including case construction, took 2.32 s elapsed with measured peak RSS 684,388 KiB under `/usr/bin/time -v`; numerical thread variables were all fixed to one.

Two preflight outcomes are retained for transparency:

- Imposing a 1-GiB *virtual-address* `ulimit` caused Torch's lazy CUDA-library probe to segfault in an existing small-MEC test. This was not an assertion failure; actual RSS was subsequently measured instead.
- An initially over-broad suite included the entire sparse-oracle test file. Its final test expects `outputs/greedy_optimality_oracle_v1` not to exist, but that pre-existing archive exists. The scientifically relevant suite listed above excludes this unrelated archive-absence assertion. Before that assertion, 79 tests passed; the final requested focused suite passed completely.

## Existing versus new coverage

The detailed mapping is in `coverage_matrix.md`. Existing command-space regressions already adequately cover symbolic root scales, canonical disagreement resolved by physical families, root no-op budget charging, bounds, a synthetic three-family pairwise/global failure, and loud rejection of uncertified structural witnesses. S01-B3 therefore did not inflate test counts with duplicates.

New end-to-end evidence is specifically: optimizer-generated nonroot shift/scale ties, an analytic perturbation that removes one, independently enumerated three-DAG positive and negative intersections, and low-cap/increased-cap recovery on a genuinely tied family product.

No new geometry-only fixture was needed. Existing synthetic fixtures remain clearly separate from realizable SCM cases.

## Preservation and provenance

HEAD before and after is `75d1a34ed0f710c4335c6361a2e295faba076ee3`. The worktree was dirty before S01-B3. Relevant pre-existing hashes and diff counts are stored in `implementation_manifest.json`; the only pre-existing relevant tracked diffs were `research/matching_query_complexity.py` (564 insertions, 13 deletions) and `tests/test_small_mec_matching_query.py` (6 insertions, 10 deletions). S01-B3 added only this output directory and `tests/test_s01_tied_optimum_validation.py`. Python bytecode generated by test execution remains unrelated and was not cleaned.

The original 180-instance bank, corrected recomputation, Table 2, S01-B, and S01-B2 outputs were neither rerun nor edited.

## Remaining gaps and claim boundary

- No realizable SCM was established for the stronger Helly-style pattern in which every pair of three candidates is compatible but the triple is not. The existing synthetic family test covers the algorithmic geometry.
- Continuous nonroot flat directions are outside the retained finite-family completeness assumption and remain fail-loud/unsupported.
- Near-tolerance fuzzing, large tied-family products, arbitrary CPDAGs, finite-sample certification, and worst-case scaling are not established.
- The cap's family-product number is a conservative cumulative accounting upper bound, not measured visited search work.

Within those limits, the manuscript may accurately claim separate targeted validation of tied-optimum/root-flat intersections.
