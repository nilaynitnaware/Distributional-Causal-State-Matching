# Shift–Scale Causal State Matching

This release is tested with Python 3.12 (historical runs used Python 3.12.13).
See the [environment record](docs/reproducibility/ENVIRONMENT.md) for the pinned
installation, platform evidence, thread settings, and provenance limits.
Start with the [a–r reproduction index](REPRODUCIBILITY.md) and the
[verified seed ledger](SEEDS.md).

This repository is a new, independent research implementation of
infrastructure for matching a desired causal state with shift–scale
interventions. It is inspired by:

- Jiaqi Zhang, Chandler Squires, and Caroline Uhler,
  [_Matching a Desired Causal State via Shift Interventions_](https://arxiv.org/abs/2107.01850),
  NeurIPS 2021.
- The accompanying
  [`uhlerlab/causal_mean_matching`](https://github.com/uhlerlab/causal_mean_matching)
  reference repository.

The reference work studies mean matching with shift interventions. This
codebase develops a distinct numerical linear-Gaussian implementation with
explicit shift–scale mechanisms, reproducible modern graph generators, exact
Gaussian moments, and inspectable problem serialization. It does not copy the
legacy problem or policy implementations.

> **Project status:** oracle, exact-moment non-oracle, and sample-only
> non-oracle assemblies are implemented with explicit runtime capability
> checks, scientific-cost accounting, a validated production registry,
> versioned protocols, hard-capped calibration campaigns, and preflight.
> Finite-sample paths are runnable but are scientifically ready only when the
> versioned gates pass for the exact tested regime; see
> [the readiness audit](docs/experiment_readiness.md).

## Environment

The assembly has no recovered portable environment identity. The checked-in
dependency specifications are `requirements.in` and `requirements.txt`; their
pins are specifications, not evidence that the historical environment was
installed or that this candidate has been run. See the
[portable setup and reproduction guide](docs/reproducibility/reproduction-guide.md)
before installing dependencies or launching any computation.

## Mathematical model

Let \(G\) be a directed acyclic graph with nodes \(0,\ldots,p-1\). The
observational mechanism output at node \(i\) is

\[
Z_i = \sum_{j\in\operatorname{Pa}(i)} B[j,i]X_j + \epsilon_i,
\qquad
\epsilon_i\overset{\text{iid}}{\sim}\mathcal N(0,1).
\]

Observationally, \(X_i=Z_i\). The coefficient convention is

```text
B[parent, child] = edge coefficient
```

so the vector equation is

\[
X=B^\top X+\epsilon.
\]

The coefficient matrix belongs to the hidden simulation environment. A
non-oracle matching method must not inspect it.

### Shift–scale intervention

For shift \(a_i\) and positive scale \(s_i\),

\[
X_i^I
  = a_i+s_i\sum_{j\in\operatorname{Pa}(i)}B[j,i]X_j^I+\epsilon_i.
\]

Scale acts only on the endogenous parent contribution. It does not change the
independent disturbance distribution, so the conditional disturbance variance
remains exactly one. Unintervened nodes use \(a_i=0\), \(s_i=1\).

With \(S=\operatorname{diag}(s)\) and \(M=I-SB^\top\),

\[
\mu_I=M^{-1}a,
\qquad
\Sigma_I=M^{-1}M^{-\top}.
\]

A root scale is a no-op because a root has no parent contribution. Non-default
root scales are rejected in targets and omitted from graph-aware search. The
actionable count for a realized graph is \(n+|\{i:\operatorname{Pa}(i)\ne
\varnothing\}|\), not universally \(2n\). The semantic identifier
`mechanism_scale_noise_fixed_v1` is part of configuration fingerprints,
manifests, results, failures, and analysis. See [docs/model.md](docs/model.md).

## Supported graph families

The `--graph-type` option supports:

| Name | Description | Main option |
| --- | --- | --- |
| `path` | Directed path, optionally permuted | `--permute-path` |
| `rooted_tree` | Random labeled tree directed away from a root | `--rooted-tree-root` |
| `randomly_oriented_tree` | Random tree with independently oriented edges | — |
| `erdos_renyi` | Order-consistent Erdős–Rényi DAG | `--edge-probability` |
| `barabasi_albert` | Preferential-attachment skeleton with acyclic orientation | `--attachment-edges` |
| `complete` | Complete DAG under a sampled node order | — |
| `moralized_erdos_renyi` | Chordal DAG with clique parent sets | `--edge-probability` |
| `tree_of_cliques` | Chordal DAG built from overlapping cliques | clique-size and degree options |

Every random API uses an explicit `numpy.random.Generator`. A fixed generation
seed reproduces the graph, coefficients, observational samples, and saved
problem.

## Generate problems

Generation keeps two interventions separate:

1. The **hidden generating intervention** defines the synthetic target and is
   retained only for evaluation.
2. The **current candidate intervention** always starts with shifts equal to
   zero and scales equal to one. The matching runners optimize these values.

No prior over hidden target interventions has been approved. Therefore,
`generate.py` requires explicit target interventions in an NPZ archive rather
than silently sampling shifts or scales.

### 1. Create explicit target interventions

For one five-node problem with two target nodes:

```python
import numpy as np

shift = np.array([0.8, 0.0, -0.4, 0.0, 0.0])
# Identity scales make this example valid for every generated DAG, including
# cases where either shifted node happens to be a root.
scale = np.ones(5)
np.savez("target_interventions.npz", shift=shift, scale=scale)
```

For `--dataset-size N`, `shift` and `scale` must normally both have shape
`(N, n_nodes)`. One-dimensional arrays are accepted when `N=1`. The declared
`--target-support-size` must equal the number of nodes where the shift differs
from zero or the scale differs from one.

### 2. Run generation

```bash
python generate.py \
  --graph-type erdos_renyi \
  --n-nodes 5 \
  --edge-probability 0.35 \
  --target-support-size 2 \
  --target-interventions target_interventions.npz \
  --dataset-size 1 \
  --n-observational-samples 10000 \
  --seed 2026 \
  --output-dir generated/erdos_renyi_example
```

Use `python generate.py --help` for every graph-specific option.

The output directory must be empty. For each problem, generation writes:

- `problem_XXXXXX.npz` containing graph edges, coefficients, observational
  samples, target mean and covariance, hidden generating shift/scale, and
  current shift/scale;
- `problem_XXXXXX.json` containing the generation configuration, instance seed,
  realizability flag, array filename, and format version.

The files use format version `1`. Pickled class instances are not used.

## Main objects

- `LinearGaussianSCM`: immutable DAG and coefficient matrix with observational
  and interventional sampling and exact moments.
- `ShiftScaleIntervention`: immutable node-wise shifts and positive scales.
- `GaussianTarget`: public target mean/covariance, realizability label, public
  metadata, and an evaluation-only generating intervention when one exists.
- `PublicProblemView`: algorithm-facing target moments, observational samples,
  current intervention, and algorithm seeds, with no hidden SCM state.
- `GeneratedProblem`: the public view paired with isolated simulation truth for
  generation and evaluation.
- `ExactMomentBlackBoxEnvironment` and `FiniteSampleBlackBoxEnvironment`:
  capability-declared intervention APIs returning public moments or samples.

## Production execution and readiness

Runnable code and scientific readiness are separate. The production registry
contains nine validated modes: the three oracle modes, supplied/estimated-CPDAG
exact non-oracle modes, supplied/estimated-CPDAG finite modes, and exact/finite
non-realisable modes. Each entry fixes permitted provider roles, target
families, oracle capabilities, required fields, and result schema. Validation
occurs before any environment query.

Four immutable protocols are available:

- `exact_protocol_v1`;
- `exact_noiseless_protocol_v2`;
- `finite_sample_calibration_protocol_v1`;
- `finite_sample_main_protocol_v1`.

`exact_noiseless_protocol_v2` makes exact non-oracle initialization, probes, and
candidate search unbudgeted while retaining their diagnostic counters; the
accepted final intervention remains capped at `2k` coordinates. Finite-sample
protocol budgets are unchanged.

They are reproducibility defaults, not claims of scientific optimality. Every
runner result stores its protocol version and complete resolved protocol. Print
a resolved production configuration and provider graph without executing it:

```bash
python -m experiments.registry \
  --config production_config.json \
  --print-resolved
```

Five deterministic, restartable, hard-capped campaign types cover source
classification, candidate acceptance, local unresolved-state classification,
PC structure recovery, and covariance/reverse-KL estimation. They write
ordinary per-case calibration results and remain distinct from main scientific
experiments. The versioned readiness policy consumes their aggregates plus the
end-to-end diagnostic aggregate and returns one of:

- `ENGINEERING_READY`;
- `CALIBRATION_REQUIRED`;
- `CALIBRATION_FAILED`;
- `SCIENTIFICALLY_READY_UNDER_TESTED_REGIME`.

Finite-sample modes with missing artifacts are always
`CALIBRATION_REQUIRED`. The last status is scoped to the exact protocol and
regime. It does not transfer to another graph size, degree, sample size,
intervention magnitude, or noise regime.

The finite diagnostic decomposition crosses exact versus estimated CPDAG with
exact-diagnostic versus statistical source outcomes, always using empirical
matching. Oracle source outcomes are permitted only in the two explicitly
labeled diagnostic cells. Results expose final reverse KL,
source/orientation/CPDAG errors, optional evaluation-labeled false candidate
decisions, physical cost, and termination.

The local composite remains a proposal surrogate. Global-evaluated proposals
record predicted local improvement, exact or empirical global improvement,
decision, coordinate/background, and local terms. Aggregates report
local/global rank correlation, locally-positive/globally-negative cases, global
rejection, and shift/scale differences. Local loss never overrides global
reverse-KL acceptance.

Non-realisable runs permit positive residual KL. Their standardized termination
vocabulary distinguishes target tolerance, no significant improvement, support,
query and observation exhaustion, patience, no legal action, unlocalized global
mismatch, insufficient evidence, and numerical failure. Optional
evaluation-only best-found comparators report the algorithm gap and observed-gap
closure; they are not proven lower bounds. Hidden target-support metrics are
omitted.

Preflight validates schema, registry/provider resolution, oracle leakage,
target and budget compatibility, dependencies, device, output writability,
deterministic IDs/manifests, protocol, calibration artifacts, and readiness. It
executes no scientific run:

```bash
python -m experiments.preflight \
  --config production_config.json \
  --no-write \
  --explain
```

Use `--json` for machine-readable output. `--strict` returns nonzero for invalid
or unready configurations.

Exact/noiseless one-run execution uses the explicit production executor module.
See [the production execution guide](docs/production_execution.md) for the
checked-in valid configuration and exact dry-run, manifest, one-run, resume, and
aggregation commands.

## Exact and finite query accounting

Environment API activity is distinct from scientific cost:

- `algorithm_environment_calls` counts every environment API call;
- `executed_environment_queries` counts uncached physical executions;
- `cache_hits` counts exact results served without physical execution;
- `failed_environment_queries` counts executed attempts that did not complete;
- `queried_intervention_batches` counts physically queried batches;
- `total_observations` counts finite-sample rows actually returned.

Exact cache hits consume no new physical query or experimental trial. Exact
cache identity includes problem instance, complete shift/scale intervention,
environment semantics, and result-affecting numerical configuration. Failed
executions stay charged and are not cached. Finite baseline reuse creates
neither a query nor fake observations; every additional draw is a new executed
batch. Candidate decisions state whether samples were reused, independently
drawn, or explicitly paired.

## Reproducibility and historical provenance

This repository includes code, reviewed configurations, declared dependencies,
bounded frozen inputs, a diagnostic runner, and reproduction/raw-data retrieval
guidance. It does not promise a complete archive of original outputs, logs, or
internal audit/review evidence. See the [reproducibility index](docs/reproducibility/README.md)
and the [portable setup and reproduction guide](docs/reproducibility/reproduction-guide.md).
Historical execution-provenance and campaign-evidence pages distinguish original
run metadata from dependency specifications, later source recovery, and
audit-host measurements.

For setup boundaries, available frozen inputs, entrypoint interfaces, and
output-safety notes, use the [portable setup and reproduction guide](docs/reproducibility/reproduction-guide.md).

## Testing

Run the complete suite from the repository root:

```bash
conda activate shift-scale
python -m pytest -q
```

The tests cover graph generation and serialization, intervention semantics,
analytic and empirical losses, scheduler/budget behavior, capability boundaries,
source/orientation inference, structure estimation, target families, non-oracle
end-to-end paths, and restartable result handling.

## Current limitations

- Oracle and exact-moment non-oracle modes are runnable under the assumptions
  recorded in `docs/experiment_readiness.md`.
- Finite-sample modes have three-way source, local-state, search, and candidate
  decisions and explicit calibration/readiness machinery, but code completeness
  cannot establish confidence coverage or statistical power. Scoped empirical
  calibration is required before multi-seed experiment grids.
- PC estimation assumes causal sufficiency, Gaussianity, and the configured
  faithfulness/significance regime; estimated graphs are never repaired from
  truth.
- Non-realisable target constructors and a best-found oracle comparator exist,
  but the comparator is not a proven lower bound. Positive residual KL is valid,
  and standardized stopping and comparator-gap fields replace zero-KL failure
  assumptions.
- Hidden target interventions must still be supplied explicitly for the
  realizable problem-generation CLI.
- The hidden DAG, coefficient matrix, and target-generating intervention remain
  simulation/evaluation state and must not be given to non-oracle algorithms.

## Attribution and licensing note

This repository is an independent implementation inspired by the cited paper
and reference repository. The reference repository does not contain a software
license file in its tracked tree, so no substantial reference implementation
code has been copied here. Selected project-authored code, tests, and
configurations are under the [MIT License](LICENSE). Selected project-authored
reproduction documentation, provenance, and required frozen inputs are under
[CC BY 4.0](LICENSE-DATA). See the file-level [license scope](LICENSE-SCOPE.md)
and [third-party/excluded-material notices](THIRD_PARTY_NOTICES.md). Material
outside those explicit scopes remains excluded. Both archived population banks
are excluded; independent contributor/institutional authority is not verified.

Please cite our paper, Distributional Causal State Matching under Partial Causal
Information, when using this work in research. See [CITATION.cff](CITATION.cff)
for the repository citation metadata. This is a scholarly request, not a
mandatory condition of the MIT License or CC BY 4.0 grant.

## Matching semantics and experiment safety

The authoritative matching objective is the reverse-direction Gaussian loss

\[
D_{\mathrm{KL}}(Q_{\text{target}}\Vert P_{\theta})
=\frac12\left[
\operatorname{tr}(\Sigma_\theta^{-1}\Sigma_Q)
+(\mu_\theta-\mu_Q)^\top\Sigma_\theta^{-1}(\mu_\theta-\mu_Q)
-p+\log\det\Sigma_\theta-\log\det\Sigma_Q
\right].
\]

The candidate covariance, not the target covariance, is factored and solved
against in the trace and mean terms. Joint Gaussian KL is the global loss used
for acceptance, rollback, successful termination, oracle optimization, and
final reporting. Successful termination requires only
`global_loss <= global_loss_tolerance`; mean mismatch is diagnostic and equal
means do not imply success when scale interventions leave covariance mismatch.

Actionability is localized with a configurable, normalized composite of
marginal Gaussian KL, selected pairwise Gaussian KL, and conditional Gaussian
KL given parents directed in the current public graph. Unavailable terms are
omitted and remaining weights are renormalized. This score ranks proposals;
only full joint KL accepts or rejects them.

Each certified unresolved source always exposes its shift. Its log-scale is
exposed only when the public graph certifies at least one directed parent. The
greedy scheduler optimizes every legal coordinate with all others frozen,
chooses the largest predicted local reduction, and accepts it only after the
complete candidate improves global KL by the configured tolerance. Updating an
already-selected coordinate consumes no new final-support slot.

Internal analytic oracle evaluations do not consume environment trials.
Black-box accounting follows the API-versus-scientific-cost definitions above.
Exact cache hits add an algorithm call and cache hit but no executed query.
Finite draws and failed executed calculations remain charged.

Final support counts unique nonidentity actionable coordinates. Shift-only and
scale-only targets on `k` nodes require `k`; mixed targets put shift and scale
on the same `k` eligible nodes and require `2k`. The configurable `k` and
`ceil(1.5k)` mixed ablations remain deliberately infeasible and are labelled
as such. The graph-specific ceiling is `n + number_of_nonroots`.

Two oracle baselines are distinct. `known_support_oracle` receives the true DAG,
true coefficients, and exact active coordinate support—but never hidden values.
The third oracle assembly is labelled `idealized_unknown_graph_oracle` because
its CPDAG, graph outcomes, local moments, global loss, and gradients depend on
hidden simulation truth. Non-oracle mode validation rejects true-DAG, true-B,
and oracle-loss providers.
legal/greedy selector. Both recover values through optimization.
The third oracle assembly is labelled `idealized_unknown_graph_oracle` because its CPDAG, graph outcomes,
local moments, global loss, and gradients depend on hidden simulation truth.
Non-oracle mode validation rejects true-DAG, true-B, and oracle-loss providers.

Probe evidence records the accepted shift/scale background under which it was
collected. Temporary probes add alpha to the accepted shift and leave the
accepted scale and candidate unchanged. After any accepted shift or scale
change, graph adjacencies, accepted orientations, and Meek consequences remain,
while moment, local-loss, gradient, and proposal-ranking caches are invalidated.
Numerical evidence from an older background is retained for audit but is not
reported as current-background evidence.

`LinearGaussianSCM` enforces graph support if and only if coefficient support:
every graph edge has magnitude at least `beta_min`, and every nonedge is exactly
zero. This structural invariant does not preclude cancellation among multiple
directed paths. The noiseless orientation path therefore additionally assumes
intervention faithfulness/separation and no relevant total-effect cancellation.

Safe executable slices include the known-support and known-DAG analytic
oracles, the labelled idealized unknown-graph oracle, and the exact-moment
non-oracle mode. Finite-sample exact/estimated-CPDAG assemblies now execute
sample-only decisions and have tiny deterministic integration tests, but they
report `CALIBRATION_REQUIRED` until the scoped calibration gates in
[docs/experiment_readiness.md](docs/experiment_readiness.md). The runner is
implemented for dry runs, manifests, one-run execution, resume, durable
failures, and aggregation; no experiment grid is launched automatically.

Focused scaling, intervention, algorithm, information, and robustness campaigns
are generated declaratively; see
the historical/unresolved locator `docs/ablation_campaigns.md` for supported cells,
exact tier counts, fingerprint safety, grouped scientific aggregation, and
launch commands.
