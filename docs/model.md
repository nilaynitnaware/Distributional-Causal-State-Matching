# Mathematical model

This document fixes the shift-scale semantics used by every simulator,
exact-moment environment, optimizer, and experiment artifact. The semantic
identifier is `mechanism_scale_noise_fixed_v1`.

## Observational linear-Gaussian SCM

For a DAG with coefficient convention `B[parent, child]`,

\[
X_i=\sum_{j\in\operatorname{Pa}(i)}B[j,i]X_j+\varepsilon_i,
\qquad \varepsilon_i\sim\mathcal N(0,\sigma_i^2),\quad
\varepsilon_i\perp\!\!\!\perp\varepsilon_j\ (i\ne j).
\]

or `X = B^T X + epsilon`. The errors are mutually independent and are not
intervention coordinates. The default is `sigma_i^2 = 1`, preserving the
legacy homoskedastic model exactly.

## Corrected shift-scale intervention

For shift `a_i` and positive scale `s_i`,

\[
X_i^I=a_i+s_i\sum_{j\in\operatorname{Pa}(i)}B[j,i]X_j^I+\varepsilon_i.
\]

The identity is `a_i = 0`, `s_i = 1`. Scale multiplies the endogenous parent
contribution only. It does not multiply the exogenous disturbance, so

\[
\operatorname{Var}\!\left(X_i^I\mid X_{\operatorname{Pa}(i)}^I\right)=\sigma_i^2
\]

for every valid scale. The superseded implementation instead used
`a_i + s_i(parent_i + epsilon_i)`, which incorrectly made this conditional
variance `s_i^2 sigma_i^2`.

With `S = diag(s)`, `D = diag(sigma_1, ..., sigma_n)`, and `M = I - S B^T`,
moments are

\[
X^I=a+SB^TX^I+\varepsilon,\qquad
\mu_I=M^{-1}a,\qquad
\Sigma_I=M^{-1}D^2M^{-\top}.
\]

The implementation uses linear solves:

```python
mean = np.linalg.solve(M, shift)
noise_transform = np.linalg.solve(M, np.diag(noise_standard_deviations))
covariance = noise_transform @ noise_transform.T
```

Sampling evaluates the same structural equation in topological order. NumPy
and Torch dense/topological paths use the same independent heteroskedastic
noise factor. The scale intervention never rescales this exogenous noise.

The candidate initialization is

\[
\delta^{(0)}=0,\qquad s^{(0)}=1.
\]

Exact moments use solves and never call `np.linalg.inv(M)`.

## Root scale coordinates and actionability

A root has no parent contribution. Its scale is therefore a mathematical no-op:
changing it leaves both samples and exact moments unchanged. Low-level moment
and sampling methods preserve this fact, while target validation rejects a
non-default root scale. Search policies never enumerate it.

For graph `G`, the actionable coordinates are

\[
\{(i,\text{shift}):i\in V\}\cup
\{(i,\text{log-scale}):\operatorname{Pa}(i)\ne\varnothing\}.
\]

Thus the exact count is

\[
|V|+|\{i:\operatorname{indegree}_G(i)>0\}|=2|V|-|\operatorname{roots}(G)|,
\]

not universally `2n`. This count and the number of scale-eligible nodes are
serialized in production metrics.

## Target composition and budgets

For `k = ceil(rho*n)`:

- shift-only selects `k` arbitrary endogenous nodes and needs `k` coordinates;
- scale-only selects `k` nonroots and needs `k` coordinates;
- mixed selects `k` nonroots and assigns both parameters, needing `2k`.

Graphs with fewer than `k` eligible nodes are regenerated deterministically;
`k` is never reduced. Mixed budgets `k` and `ceil(1.5k)` remain valid ablation
conditions but are recorded as structurally insufficient. Updating an already
selected coordinate consumes no additional support.

## Validity and stability

For a DAG and finite positive scales, a topological permutation makes
`M = I - S B^T` triangular with unit diagonal, so it is mathematically
invertible for every allowed intervention. Implementations nevertheless check
finite systems and moments and convert numerical solve failures to invalid or
infeasible evaluations. Target generation rejects invalid interventions; no
scale is silently clipped outside the documented positive bounded domain.

## Target and candidate boundary

A realizable target exposes only its Gaussian moments. Its hidden generating
intervention is retained for evaluation. Candidate environments reconstruct
moments with the identical equation above. Consequently, installing a valid
hidden target intervention as the candidate gives Gaussian KL zero up to
floating-point precision.
