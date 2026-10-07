# Light Tunnel chronological split and selection provenance

This is the canonical record for manuscript migrations B21/E25 and B22/E27.
It documents the shared chronological split and block-bootstrap implementation
records used by the Light Tunnel finite-mode and root-shift boundary analyses.
These analyses are offline replays of archived physical environments: selection
does not execute a newly chosen hardware action.  See [historical execution
provenance](execution-provenance.md#light-tunnel-offline-analyses-b01-p05) for
the distinction between archived-data analysis, later recovered source, and
unavailable launch-time source linkage.

The archived observations are not candidate payload. See [upstream physical-data
retrieval](physical-data-retrieval.md) for the exact upstream input endpoint,
required files, placement, and terms-reference boundary.

## Frozen chronological split (B21-P01)

All intervals below are half-open and use zero-based row indices.  The roles,
offsets, and interval semantics are shared by the finite-mode experiment and
the root-shift boundary experiment.

| Role | Environment | Rows | Use |
| --- | --- | --- | --- |
| Source calibration | reference | `[0,5000)` | mechanisms and moments |
| Candidate calibration | intervention | `[0,400)` | physical-mode model |
| Target selection | intervention | `[400,600)` | matcher input moments |
| Candidate evaluation | intervention | `[600,800)` | held-out candidate KL |
| Target evaluation | intervention | `[800,1000)` | held-out target distribution |
| Identity evaluation | reference | `[8000,8200)` | no-action baseline |

The six blocks have lengths 5000, 400, 200, 200, 200, and 200.  In each
intervention file, calibration, target selection, candidate evaluation, and
target evaluation are in chronological order and have disjoint row ranges;
source and identity use disjoint reference ranges.  `RowBlock` declarations
and `validate_protocol` enforce those non-overlap conditions in the finite-mode
source.  The frozen finite-mode protocol serializes the same six roles and
offsets, including the reference identity block.  The root-shift recovered
source declares the same constants and checks the same relevant separations;
its frozen protocol serializes the same split.

These are contiguous blocks within one recorded environment session, not
independent hardware sessions or physical replications.  The archive replay is
therefore a finite-mode, one-physical-action budget exercise over a recorded
catalog; it does not claim that executing the selected action would collect new
hardware data.  The finite-mode feasibility record likewise says that each
environment has one session and that rows/segments are not independent physical
replications.

## Root-shift boundary protocol (B21-P02)

The root-shift boundary experiment reuses the preceding
[frozen chronological split](#frozen-chronological-split-b21-p01): reference
source calibration is `[0,5000)`, manipulated-root mean estimation is
`[0,400)`, target selection is `[400,600)`, candidate evaluation is
`[600,800)`, target evaluation is `[800,1000)`, and reference identity
evaluation is `[8000,8200)`.  The manuscript transfer refers to that preceding
split rather than repeating its offsets.

Source-fit means and sample standard deviations (divisor `N-1`) define
standardization, and source-only fitting is unchanged.  The primary model uses
the source covariance with its source-calibrated total-effect mean shift.
Selection minimizes predicted target-selection Gaussian KL; action identifier
is the deterministic tie-breaker.  The recovered source keeps the model
equations, covariance regularization/ridge treatment, and numerical thresholds
in their existing implementation; this migration changes no model equation,
covariance/ridge formula, or threshold.

The root-shift source and output live in the separate recovered worktree named
in the historical audit, rather than at a repository-relative path in this
checkout.  Its run metadata and recovered-source hash are later-audit evidence,
not a cryptographic link to the historical launch.  The canonical record does
not infer physical collection provenance from the archive release or retrieval
metadata.

## Retained leakage and physical semantics (B21-P03)

Both transfers retain the following static role boundaries.

- The selector receives opaque action identifiers and predicted scores;
  environment names are decoded only after selection.
- Candidate parameters are fitted only on calibration blocks.  Held-out
  candidate arrays are supplied only to the physical-oracle evaluator.
- Leave-generating-mode-out removes the generating action before minimization.
  The target harness knows the archived target file; the selector does not.
- The physical oracle uses the reserved candidate and target evaluation blocks
  unavailable to selection.  Matching is offline over recorded physical
  environments, so the selected action is not executed on hardware.
- The finite-mode budget is one physical action.  Recorded gain modes are not
  assumed to be invariant-noise theoretical shift--scale interventions.

The finite-mode implementation constructs its catalog from recorded singleton
modes, calls `select_action` on predicted scores keyed by action ID, and only
then decodes the chosen ID for reporting.  Its leave-out variants remove the
target ID before selection; `physical_metrics` receives the reserved candidate
arrays.  The root-shift recovered implementation has the same opaque-ID
selector, target-label exclusion, pre-minimization leave-out, and separate
`bootstrap_physical` evaluation path.  These are static source/serialized
protocol checks, not new experimental results.

## Finite-mode block-bootstrap uncertainty (B22-P01)

The finite-mode implementation fixes `N_BOOTSTRAP = 500`, `SEED = 20260831`,
and `BLOCK_SIZE = 50`. For every held-out 200-row target or candidate evaluation
array, `block_bootstrap` reshapes the array into four contiguous 50-row blocks,
samples four block indices with replacement, and restores the 200-row shape.
In each replicate, `physical_metrics` resamples the target first and then each
candidate in sorted action-ID order. Thus target and candidate blocks are
independently resampled, while the one generator created by `main` is consumed
sequentially across its fixed experiment/action-catalog traversal. Equal literal
seeds here and in root-shift establish neither independent streams nor identical
streams: initialization, calls, shapes, and traversal scope would also matter.

The finite-mode `ci` helper supplies 2.5% and 97.5% endpoints for per-candidate
physical KL, selected physical KL, identity KL, selected-minus-oracle regret,
and identity-minus-selected KL reduction (`delta_identity`); it also supplies
the lower-endpoint top-set comparison. Its `np.quantile` call has no explicit
method. The original finite-mode metadata records NumPy 2.5.1, whose default
quantile method is linear, so **finite-mode** endpoints use linear quantile
interpolation (a 95% interval convention). This is not asserted for root-shift:
although the recovered helper has the same call form, its launch-time
NumPy/method linkage is unavailable.

The serialized finite-mode protocol records the same 500 replicates, 50-row
contiguous-with-replacement method, and scope: within-session sampling
uncertainty, not physical replication. These intervals quantify variation
within recorded sessions; they do not make chronological blocks independent
hardware-session replications.

## Root-shift block-bootstrap uncertainty (B22-P02)

The recovered root-shift implementation also declares 500 replicates, literal
seed `20260831`, and 50-row blocks. Each 200-row target or eligible-candidate
evaluation segment supplies four contiguous 50-row blocks. `bootstrap_physical`
resamples the target and then every candidate independently within each
replicate. `main` creates one generator and passes it through the fixed state,
experiment, model, and target traversal; it is not reset per state, experiment,
or model. The recovered frozen protocol separately records the four-block
independent target/candidate method and its within-session-only scope.

Intervals are attached to candidate physical KL and to selected physical KL,
identity-minus-selected KL reduction, selected-minus-adaptive-oracle regret,
and selected-minus-fixed-oracle comparisons. The recovered source uses
2.5%/97.5% calls, but this record deliberately does not transfer the finite-mode
linear-interpolation qualification to root-shift. Root-shift uncertainty is
within-session sampling variation, not hardware-session replication.

The retained Table 10 rule counts an improvement whenever the held-out
`delta_identity` **point estimate** is positive. It does not require that the
bootstrap interval be wholly positive. This is consistent with the recovered
aggregate's separate `delta_identity > 0` and lower-CI-positive fractions; it
is a manuscript design/definition, not a fresh calculation.

## Retained manuscript design and scope (B22-P03)

The proposed paper text retains the full finite-mode block-resampling design.
Its root-shift paragraph refers to that same 500-replicate block-bootstrap
protocol and retains the within-session qualification and positive-point-
estimate improvement definition. The literal seed, generator-consumption
descriptions, and finite-mode linear convention belong here rather than being
inferred as a claim of independent experiment streams. This record preserves
the B20 distinction between an archived protocol condition and an implementation
guarantee: static source/protocol evidence describes the implementation, not a
newly executed or independently replicated result.

## Migration validation (B21/E25)

| Batch item | Canonical heading | Evidence and scope checked | Provenance class |
| --- | --- | --- | --- |
| B21-P01 | [Frozen chronological split](#frozen-chronological-split-b21-p01) | `research/mk1_reduced_real_matching/run_experiment.py`: `RowBlock`, `SOURCE_FIT`, `MODE_FIT`, `TARGET_SELECTION`, `CANDIDATE_EVALUATION`, `TARGET_EVALUATION`, `IDENTITY_EVALUATION`, and `validate_protocol`; `outputs/mk1_reduced_real_matching/machine_readable/frozen_protocol.json`; recovered root-shift source and protocol: `RowBlock`, constants, and `validate_protocol` | Finite-mode protocol/output: original archived record, with recovered current source for semantics. Root-shift source/protocol: later audit in a separate worktree. |
| B21-P02 | [Root-shift boundary protocol](#root-shift-boundary-protocol-b21-p02) | recovered historical worktree (branch `experiment-mk1-shift-matching`, HEAD `75d1a34ed0f710c4335c6361a2e295faba076ee3`): [`run_experiment.py`](../../research/mk1_shift_matching/run_experiment.py): source `ddof=1` standardization, source-model construction, predicted-score selection, `select_action`, and serialized protocol; [`root-shift handoff`](../mk1_shift_matching_handoff.md) | Later recovered source and output; no launch-time executed-source linkage. |
| B21-P03 | [Retained leakage and physical semantics](#retained-leakage-and-physical-semantics-b21-p03) | `research/mk1_reduced_real_matching/run_experiment.py`: `select_action`, leave-out candidate construction, `physical_metrics`, and `outputs/mk1_reduced_real_matching/machine_readable/frozen_protocol.json`; recovered root-shift source: `candidate_ids_for_experiment`, `run_matching`, `bootstrap_physical`, and `physical_metrics` | Recovered source/protocol semantics; archived output supports recorded role fields, not a new hardware execution. |

## Migration validation (B22/E27)

| Batch item | Canonical heading | Evidence and exact locators | Provenance class and remaining gap |
| --- | --- | --- | --- |
| B22-P01 | [Finite-mode block-bootstrap uncertainty](#finite-mode-block-bootstrap-uncertainty-b22-p01) | `research/mk1_reduced_real_matching/run_experiment.py`: `SEED`/`N_BOOTSTRAP`/`BLOCK_SIZE` (27--29), `block_bootstrap` (283--288), `ci` (291--293), `physical_metrics` (296--355), and `main` generator/traversal (713--728); `outputs/mk1_reduced_real_matching/machine_readable/frozen_protocol.json`: `bootstrap`; `outputs/mk1_reduced_real_matching/machine_readable/run_metadata.json`: NumPy version | Original serialized protocol/metadata supports settings and environment; current recovered source supports behavior. Historical executed-source byte linkage and serialized RNG state/draws are unavailable. |
| B22-P02 | [Root-shift block-bootstrap uncertainty](#root-shift-block-bootstrap-uncertainty-b22-p02) | historical recovered worktree: [`run_experiment.py`](../../research/mk1_shift_matching/run_experiment.py): `SEED`/`N_BOOTSTRAP`/`BLOCK_SIZE` (31--33), `block_bootstrap` (342--347), `ci` (350--352), `bootstrap_physical` (355--379), `physical_metrics` (382--426), `aggregate` (675--680), and `main` generator/traversal (736--769); [`frozen_protocol.json`](../../outputs/mk1_shift_matching/machine_readable/frozen_protocol.json): `bootstrap` | Later recovered source and separate-worktree protocol/output. No historical launch command, executed-source linkage, RNG state, or launch-time quantile-method record. |
| B22-P03 | [Retained manuscript design and scope](#retained-manuscript-design-and-scope-b22-p03) | B22-P01/P02 evidence above; [B01 Light Tunnel boundary](execution-provenance.md#light-tunnel-offline-analyses-b01-p05) | Manuscript-only retained wording is distinguished from source/protocol behavior. No manuscript application or fresh source/PDF verification was performed. |

### Evidence relocation note

The recovered root-shift source, execution records, and handoff were copied byte-exactly from the separate historical worktree. Copying preserves inspected bytes but does not establish launch-time executed-source linkage, RNG state/draws, hardware state, or physical-session records. The absolute worktree locations above remain historical provenance; actionable candidate copies include [`run_experiment.py`](../../research/mk1_shift_matching/run_experiment.py), [`frozen_protocol.json`](../../outputs/mk1_shift_matching/machine_readable/frozen_protocol.json), and the [`root-shift handoff`](../mk1_shift_matching_handoff.md).

Static review checked the local links and all B21/B22 anchors, balanced Markdown
tables and code spans, and trailing whitespace.  The finite-mode research README
contains a repository-root replay command; it is reproduction guidance that
would regenerate outputs, not evidence of the original launch.  The historical
audit records a finite-mode screen command, while the root-shift historical
launch command is not preserved here.  Neither is treated as proof of the
executed source.

Unresolved historical gaps remain: exact root-shift launch-time source linkage,
RNG state/draw sequence, execution date/runtime, host hardware/OS/threading/BLAS
details, and physical collection-session provenance are unavailable. The archive's
release and retrieval metadata do not fill those gaps.
