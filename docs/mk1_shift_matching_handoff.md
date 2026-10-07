# Light Tunnel Mk1 shift-only matching handoff

## Executive result

The result is mixed. The source-calibrated original shift model solves the
realizable full-state task S1 perfectly, but the covariance-invariance gate
fails and the harder leave-true-action-out task S2 is only partially successful.
Removing all five directly manipulated actuator coordinates also exposes a
substantial polarizer failure that the empirical finite-mode benchmark avoids.

| experiment / state | model | selected KL median | identity KL median | mean identity improvement | improved | oracle top-1 | mean regret | median Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 full state | original shift | 1.021 | 21.846 | 25.419 | 10/10 | 10/10 | 0.000 | 0.923 |
| S1 full state | empirical finite mode | 1.021 | 21.846 | 25.419 | 10/10 | 10/10 | 0.000 | 0.968 |
| S2 full state | original shift | 10.507 | 21.846 | 13.999 | 5/10 | 5/10 | 1.528 | 0.897 |
| S2 full state | empirical finite mode | 10.397 | 21.846 | 15.351 | 6/10 | 8/10 | 0.177 | 0.958 |
| S1 descendants only | original shift | 0.550 | 9.059 | 6.965 | 8/10 | 7/10 | 4.491 | 0.755 |
| S1 descendants only | empirical finite mode | 0.526 | 9.059 | 11.456 | 10/10 | 10/10 | 0.000 | 0.968 |
| S2 descendants only | original shift | 4.911 | 9.059 | 4.249 | 8/10 | 7/10 | 4.726 | 0.727 |
| S2 descendants only | empirical finite mode | 2.959 | 9.059 | 8.966 | 9/10 | 9/10 | 0.010 | 0.958 |

Here “improved” means strictly lower point-estimate held-out physical KL than
identity. The shift-model improvement interval is wholly positive on 10/10 S1
full-state tasks, 5/10 S2 full-state tasks, 8/10 S1 descendant-only tasks, and
6/10 S2 descendant-only tasks.

The full-state S1 success is real but partly reflects observing the manipulated
root. In the downstream-only diagnostic, S1 joint root+strength accuracy falls
to 7/10 and two medium polarizer targets are made substantially worse than
identity. The original shift model therefore does not support the desired broad
claim that it reliably finds useful leave-out actions, especially from propagated
consequences alone.

## Frozen data protocol

The split exactly preserves the successful Mk1 B=1 matching protocol.

| role | environment | half-open rows | used for |
| --- | --- | --- | --- |
| source calibration | `uniform_reference` | `[0:5000]` | source SCM, total-effect directions, source covariance, standardization |
| action calibration | each root-action file | `[0:400]` | action-specific root shift magnitude; empirical benchmark moments |
| target selection | hidden target file | `[400:600]` | target moments supplied to selector |
| candidate physical evaluation | every eligible candidate file | `[600:800]` | held-out physical KL and empirical oracle |
| target physical evaluation | hidden target file | `[800:1000]` | final target distribution only |
| identity physical evaluation | `uniform_reference` | `[8000:8200]` | no-action physical baseline |

The selector operates on opaque IDs `S000`--`S009` plus `A_ID`. Its callable
interface receives only a dictionary of opaque IDs and predicted scores. S2
constructs its opaque catalog after removing the generating ID. The oracle is
given only reserved physical evaluation arrays.

Uncertainty uses 500 independent resamples of the four non-overlapping
contiguous 50-row blocks in each 200-row target and candidate evaluation block.
These intervals quantify within-session sampling variation only.

## Ten physical root actions

The catalog is the previously accepted mid/strong action pair on each of five
logged roots:

| root | mid environment | fitted shift | strong environment | fitted shift |
| --- | --- | ---: | --- | ---: |
| blue | `uniform_blue_mid` | 85.131 | `uniform_blue_strong` | 170.039 |
| green | `uniform_green_mid` | 84.983 | `uniform_green_strong` | 171.133 |
| pol_1 | `uniform_pol_1_mid` | 40.051 | `uniform_pol_1_strong` | 80.796 |
| pol_2 | `uniform_pol_2_mid` | 39.616 | `uniform_pol_2_strong` | 80.142 |
| red | `uniform_red_mid` | 83.858 | `uniform_red_strong` | 169.821 |

Each magnitude is the action-calibration root mean minus the source-calibration
root mean. No target evaluation row and no independently unavailable physical
dose calibration is used.

## Common states and numerical gate

The primary state is the same predeclared 17-variable causally closed union used
by the reduced gain experiment:

`red, green, blue, pol_1, pol_2, l_11, l_12, l_21, l_22, l_31, l_32, ir_1,
vis_1, ir_2, vis_2, ir_3, vis_3`.

The symmetric diagnostic removes all five directly manipulated coordinates and
retains the 12-variable downstream/remaining state:

`l_11, l_12, l_21, l_22, l_31, l_32, ir_1, vis_1, ir_2, vis_2, ir_3, vis_3`.

Both states were fixed before target results. All 11 reference/action files have
the same ordered 46-column schema. Every source, constrained, empirical, hybrid,
and held-out physical covariance passes the `10^8` numerical stop boundary:
minimum eigenvalue 0.00303 and maximum condition number 1,342.7. No
target-specific state or post-result coordinate selection is used.

## Primary and secondary models

The primary original-shift candidate for action `(i,a)` is

```text
mu_(i,a) = mu_source + a_hat_(i,a) * t_hat_i
Sigma_(i,a) = Sigma_source.
```

The known graph fixes the source parent sets. Source OLS fits on reference
`[0:5000]` determine each root's total-effect direction. Cross-outcome source
residual covariance is retained in the joint source SCM. Action calibration
`[0:400]` estimates only `a_hat` from the manipulated root mean; it does not fit
an unrestricted primary candidate distribution.

The secondary empirical finite-mode benchmark fits a full Gaussian mean and
covariance on the same `[0:400]` action rows. A hybrid sensitivity model keeps
the constrained shift mean but substitutes the action-calibration covariance.
Because every primary shift candidate has the same covariance, its ranking is
exactly the mean-only shift ranking on all 40 target/state/task combinations.

## Covariance/noise invariance gate

The gate fails: 0/10 actions jointly satisfy the frozen tolerances.

The tolerances were fixed before matching output was computed:

- root variance ratio in `[0.8, 1.25]`;
- source-standardized full-state relative covariance Frobenius discrepancy at
  most 0.25;
- corresponding descendant-only discrepancy at most 0.25.

Cached full-file root variance ratios from the prior semantics audit are
0.946--1.049 and support the earlier “approximately location-preserving” label.
On the frozen 200-row physical blocks, root variance ratios are 0.765--1.252 and
8/10 pass. Full-state covariance discrepancies are 0.267--0.579 and 0/10 pass;
descendant-only discrepancies are 0.235--0.606 and 1/10 passes. Corresponding
covariance-only Gaussian KL ranges are 1.815--9.557 for the full state and
0.941--6.678 downstream.

Thus these remain approximate root distribution shifts with stable mean-response
directions, not exact validation of the invariant-covariance original shift
family. This is still closer to the original model than the gain experiment in
one limited respect—the root marginal variance is mostly stable and the primary
mean action is a literal additive root shift—but the required joint covariance
gate does not pass.

## S1: realizable shift matching

### Full state

The original shift model selects the generating root on 10/10, the strength
conditional on the correct root on 10/10, and the joint root+strength on 10/10.
Every choice is the held-out physical oracle, improves over identity with a
strictly positive 95% interval, and has zero regret. Selected physical KL has
mean 0.985 and median 1.021 versus median identity KL 21.846. Spearman agreement
has median 0.923.

The empirical finite-mode benchmark makes the same 10 choices and hence has
identical held-out selected KL and regret, with a higher median Spearman of
0.968. S1 full-state matching therefore loses nothing in selected physical
performance by imposing the original shift mean and source covariance.

### Descendant-only diagnostic

Once all five actuator coordinates are removed, the original shift model's node
accuracy is 8/10 and joint accuracy is 7/10. It finds the physical oracle on
7/10, remains in the physical top three on 8/10, and improves over identity on
8/10. The three disagreements are all polarizer targets:

- `uniform_pol_1_mid` selects `uniform_pol_1_strong`: KL 19.529 versus oracle
  0.447 and identity 5.291; regret 19.082.
- `uniform_pol_2_mid` selects `uniform_pol_1_strong`: KL 25.572 versus oracle
  0.441 and identity 3.905; regret 25.131.
- `uniform_pol_2_strong` selects `uniform_pol_1_strong`: KL 1.224 versus oracle
  0.530; regret 0.694, while still improving strongly over identity.

The empirical catalog selects the generating action and physical oracle on all
10 downstream-only tasks. The two severe medium-polarizer failures are therefore
model failures, not absence of downstream physical information.

## S2: leave-true-action-out projection

### Full state

The original shift model selects the point physical oracle on 5/10 and is in the
physical top three on 9/10. Mean/median/maximum regret are 1.528 / 0.171 / 9.381;
median selected KL is 10.507 versus median oracle KL 10.397 and median identity
KL 21.846. Median Spearman is 0.897.

| target | selected | selected KL | identity KL | physical oracle | oracle KL | regret | rho |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| `blue_mid` | identity | 11.086 | 11.086 | `blue_strong` | 10.745 | 0.341 | 0.988 |
| `blue_strong` | `blue_mid` | 8.424 | 28.096 | `blue_mid` | 8.424 | 0 | 0.891 |
| `green_mid` | identity | 9.139 | 9.139 | `green_strong` | 8.652 | 0.486 | 0.939 |
| `green_strong` | `green_mid` | 7.954 | 29.331 | `green_mid` | 7.954 | 0 | 0.903 |
| `pol_1_mid` | `pol_1_strong` | 24.340 | 14.959 | identity | 14.959 | 9.381 | 0.673 |
| `pol_1_strong` | `pol_1_mid` | 13.032 | 52.082 | `pol_1_mid` | 13.032 | 0 | 0.830 |
| `pol_2_mid` | `pol_2_strong` | 19.388 | 15.595 | identity | 15.595 | 3.793 | 0.794 |
| `pol_2_strong` | `pol_2_mid` | 10.965 | 63.706 | `pol_2_mid` | 10.965 | 0 | 0.697 |
| `red_mid` | `red_strong` | 9.660 | 8.380 | identity | 8.380 | 1.280 | 0.964 |
| `red_strong` | `red_mid` | 10.048 | 31.657 | `red_mid` | 10.048 | 0 | 0.964 |

The model strictly improves over identity on the five strong targets, selects
identity on the blue/green mid targets, and worsens the three remaining mid
targets. The `pol_1_mid` degradation is uncertainty-robust: its improvement
interval is `[-29.657, -1.038]` and data-adaptive regret interval is
`[4.171, 29.869]`. The `pol_2_mid` and `red_mid` improvement intervals include
zero.

The empirical catalog reaches the oracle on 8/10, stays in the physical top
three and uncertainty-aware top set on 10/10, and reduces mean regret to 0.177.
It still strictly improves over identity on only 6/10 because identity is often
the closest available projection for medium targets.

### Descendant-only diagnostic

The original shift model reaches the point oracle on 7/10, remains in the
physical top three on 8/10, and has mean/median/maximum regret
4.726 / 0 / 25.077. It improves over identity on 8/10, but only 6/10 improvement
intervals are wholly positive. The failures again concentrate on polarizers:
`pol_1_mid` regret 18.986, `pol_1_strong` regret 3.201, and `pol_2_mid` regret
25.077. The empirical catalog reaches 9/10 oracles with mean regret 0.010 and
all selections in the physical top three and uncertainty top set.

Performance therefore does not remain uniformly good after direct actuator
coordinates are removed. RGB downstream projection is useful; polarizer action
identity and ranking are not reliably captured by the constrained source shift
model.

## Shift model versus empirical finite modes

| experiment / state | selected-action agreement | mean physical KL cost of shift constraint | source-covariance vs action-covariance selection agreement |
| --- | ---: | ---: | ---: |
| S1 full state | 10/10 | 0.000 | 8/10 |
| S2 full state | 7/10 | 1.352 | 7/10 |
| S1 descendants only | 7/10 | 4.491 | 6/10 |
| S2 descendants only | 6/10 | 4.717 | 6/10 |

The full-state realizable result does not pay a selected-action cost for the
shift restriction. In S2 the empirical covariance/mean catalog avoids the
medium-polarizer over-shoots and lowers mean physical regret by 1.352 full-state
KL units. Downstream-only, the average selected physical KL penalty of the
shift constraint is about 4.5--4.7.

Replacing source covariance with action-calibration covariance changes 2/10 S1
full-state choices, 3/10 S2 full-state choices, and 4/10 choices in each
descendant-only task. This sensitivity agrees with the failed covariance gate.
It does not monotonically help: in S1 full state, enforcing source covariance
actually improves held-out selected physical KL on average because the hybrid
mis-selects the two strong polarizer targets.

## Uncertainty

For every selected action the outputs retain:

- selected physical KL interval;
- improvement-over-identity interval;
- data-adaptive regret interval, subtracting the best candidate separately in
  every bootstrap replicate;
- selected-versus-fixed-point-oracle difference interval; and
- the pairwise uncertainty-aware physical top set.

S1 full-state top sets are singleton for all targets. In S2 full state, the
original shift selection belongs to the uncertainty-aware top set on 9/10;
`pol_1_mid` is the exception. In the descendant-only evaluations, the shift
selection belongs to the top set on 8/10 for S1 and 7/10 for S2. These are
within-session block intervals, not hardware/session replication intervals.

## Direct answers and claim ceiling

1. **Does the physical catalog satisfy original shift covariance/noise
   invariance? No.** Root marginal variance is approximately stable in cached
   full-file diagnostics, but the predeclared joint covariance gate fails 0/10.
2. **Does the original shift model select useful physical actions? Sometimes.**
   It is uniformly useful in full-state realizable S1 and for the five strong S2
   targets, but not uniformly in S2 or downstream-only matching.
3. **Does it select generating root+strength in S1? Yes full-state, 10/10; no
   downstream-only, 7/10 joint.**
4. **Does it find the closest available S2 action? 5/10 full-state and 7/10
   downstream-only.** Mean regrets are 1.528 and 4.726 respectively.
5. **Does performance remain good without actuator coordinates? Not uniformly.**
   The polarizer failures are large and uncertainty-robust for the two medium
   descendant-only targets.
6. **How much is lost versus the empirical finite-mode catalog?** Nothing in S1
   full-state selections, 1.352 mean selected physical KL in S2 full state, and
   about 4.5--4.7 downstream-only.

The precise positive claim ceiling is:

> In one archived Light Tunnel Mk1 hardware session, using a supplied graph,
> calibration-only root shift magnitudes, and disjoint contiguous selection and
> evaluation blocks, a B=1 source shift model selected the generating root and
> strength and the held-out physical oracle for all ten realizable full-state
> targets. When the generating action was removed, it selected the physical
> oracle for 5/10 full-state targets with mean KL regret 1.528, while the
> calibration-matched empirical finite-mode benchmark reached 8/10 with mean
> regret 0.177.

This is not original shift-family validation because the covariance-invariance
gate fails. It does not establish between-session reproducibility, calibrated
physical dose, non-root intervention acquisition, causal discovery, or B>1
composition.

## Files created

- `research/mk1_shift_matching/README.md`
- `research/mk1_shift_matching/run_experiment.py`
- `research/mk1_shift_matching/tests/test_experiment.py`
- `docs/mk1_shift_matching_handoff.md`
- `outputs/mk1_shift_matching/machine_readable/`
  - frozen protocol, action catalog/parameters, state/schema gate and numerical
    diagnostics, covariance-invariance gate/diagnostics, four per-target result
    files, all-candidate rankings, model comparison, bootstrap summaries,
    aggregate results/summary, metadata, and completion record
- `outputs/mk1_shift_matching/tables/`
  - CSV counterparts for all tabular machine-readable artifacts
- `outputs/mk1_shift_matching/full_run.log`

No manuscript or prior output was modified. The root-response direction audit,
gain/common-scale and variance audit, acquisition/probe gates, cross-archive
ranking, and prior gain matching experiments were not rerun.

## GNU screen execution and tests

Before each final launch, `screen -ls` reported no existing session. The final
500-bootstrap run used:

```bash
screen -dmS mk1_shift_matching bash -lc 'cd <historical-worktree> && PYTHONDONTWRITEBYTECODE=1 <historical-python> research/mk1_shift_matching/run_experiment.py --raw-root <historical-raw-data-root> --bootstrap 500 > outputs/mk1_shift_matching/full_run.log 2>&1'
```

The screen session exited normally. `completion.json` records 500 bootstrap
replicates, 120 per-target model/state rows, and 1,260 ranking rows. The log
contains the matching completion record and no warning or error.

Tests were run separately:

- new Mk1 shift-matching integrity/numerical suite: 13 passed;
- reused reduced-matching protocol/regression suite: 12 passed;
- reused heteroskedastic pipeline suite: 3 passed;
- total: 28 passed, zero failures.

`git diff --check` passes.

## Git status

Branch: `experiment-mk1-shift-matching`. No commit was made. Work is confined to
the three new paths listed above; no pre-existing tracked file, manuscript, or
prior output is modified.
