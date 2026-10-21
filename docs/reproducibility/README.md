# Reproducibility documentation

The repository-level [a–r reproduction index](../../REPRODUCIBILITY.md) is the
single command/status table for the public release.

This directory indexes replication documentation and historical provenance. The
repository includes code, reviewed configurations, dependency specifications,
bounded frozen inputs, the B2 diagnostic runner, and retrieval guidance; it
does not publish a complete output archive, logs, or internal audit/review
records. Inclusion does not establish runtime success or independent ownership
clearance.

- [Portable setup and reproduction guide](reproduction-guide.md) separates
  inspection of assembled evidence, reproduction from present frozen inputs,
  and unvalidated new computation.

- [S01 frozen-input and manifest assembly](s01-frozen-inputs.md) inventories
  the compact frozen records and B2 dependencies, distinguishes selected fresh
  generation from excluded archived population banks, and records runtime
  limits.

- [Release source manifest](../../provenance/release_source_manifest.md) records the current source snapshots and their static dependency-review qualification.

- [Supporting validation evidence](supporting-validation-evidence.md) records
  the bounded Appendix-D audit, scalability-feasibility, and tied-optimum
  validation exports. It distinguishes original records,
  later audits, and recovered source; it is not a claim of complete public
  reproducibility or worst-case scalability.

- [Historical execution provenance](execution-provenance.md) is the canonical
  record for manuscript migration batch B01 (population acquisition,
  calibrated and calibration-free acquisition, robustness, Light Tunnel, and
  Public Greedy). Its [migration-validation section](execution-provenance.md#migration-validation)
  records evidence coverage, conflicts, unavailable fields, and static command
  checks.

- [Upstream physical-data retrieval](physical-data-retrieval.md) records the
  exact Mk1/Mk2 endpoints, required filenames, placements, terms references,
  and the distinction between publisher and historical-local hash evidence.
- [Small-MEC generation and runtime provenance](small-mec-generation-provenance.md)
  is the canonical record for manuscript migration batches B02 and B03. Its
  [B02 validation map](small-mec-generation-provenance.md#b02-coverage-map)
  distinguishes the original campaign, dependency lock, later deterministic
  reconstruction, corrected-recomputation runtime, and later audit host. Its
  [B03 validation map](small-mec-generation-provenance.md#b03-coverage-map)
  preserves source and target seed/RNG construction, proposal-denominator
  reconstruction, and the retained scientific selection account. The
  campaign-specific historical locators
  `docs/small_mec_matching_query_handoff.md` and
  excluded historical locator
  `outputs/small_mec_command_space_paired_v2/corrected_audit/README.md`
  remain supporting records; neither is a selected public dependency here.
- [Small-MEC numerical command validation](small-mec-command-validation.md) is
  the canonical record for manuscript migration batch B04. It preserves the
  [paired-recomputation assurance](small-mec-command-validation.md#paired-recomputation-assurance-b04-p01),
  [full matching-terminal catalog](small-mec-command-validation.md#full-matching-terminal-catalog-b04-p02),
  [per-candidate residual audit](small-mec-command-validation.md#per-candidate-certification-and-residual-b04-p03),
  and [export inventory](small-mec-command-validation.md#export-inventory-b04-p04).
  Its [migration-validation section](small-mec-command-validation.md#migration-validation)
  records evidence coverage, attribution, gaps, and static command checks.
- [Small-MEC stopping--selection and integrity validation](small-mec-stopping-selection-validation.md)
  is the canonical record for manuscript migration batch B06. It preserves the
  [540 policy--instance decompositions](small-mec-stopping-selection-validation.md#policy-instance-decomposition-checks-b06-p01),
  [2,958 three-policy candidate paths](small-mec-stopping-selection-validation.md#three-policy-trajectory-validation-b06-p02),
  [1,972 two-rule recovery checks](small-mec-stopping-selection-validation.md#two-constructive-rule-recovery-checks-b06-p03),
  and [consolidated command-intersection assurance](small-mec-stopping-selection-validation.md#consolidated-command-intersection-assurance-b06-p04).
  Its [migration-validation section](small-mec-stopping-selection-validation.md#migration-validation)
  maps every transferred item to the inspected evidence and retained gaps.
- [Population tree pilot provenance](population-tree-pilot.md) is the canonical
  record for manuscript migration batch B07. It preserves the n=20 pilot's
  [seed schedule](population-tree-pilot.md#seed-provenance-b07-p01),
  [shared-cache protocol and policy order](population-tree-pilot.md#shared-certificate-cache-and-policy-order-b07-p02),
  [marginal uncached-search counts](population-tree-pilot.md#marginal-uncached-search-counts-b07-p03),
  and [peak worker RSS](population-tree-pilot.md#peak-worker-memory-b07-p04),
  while keeping the later n=40 isolated-cache protocol distinct. Supporting
  records include historical locators
  `outputs/s01_tree_population_pilot_v1/s01_tree_population_pilot_report.md`,
  `outputs/s01_c_tree_population_n40_v1/s01_c_tree_population_n40_report.md`,
  and `outputs/s08_tree_stratification_audit_v1/s08_tree_stratification_report.md`.
  They are historical locators rather than selected public dependencies; S08 is
  absent. The [n=20 worker-resource and shared-cache clarification](population-tree-pilot.md#worker-resource-and-shared-cache-clarification-b29-p01)
  explains the configured address-space limit, measured RSS, and shared-cache
  scope.
- [Population tree scale-extension provenance](population-tree-scale-extension.md)
  is the canonical record for manuscript migration batch B08. It preserves the
  n=40 seed/freezing, 12,800-command validation, KL residual, and peak-worker
  RSS records while keeping isolated-cache timing distinct from n=20. Its B29
  continuation preserves [per-task cache/worker/completion scope](population-tree-scale-extension.md#task-isolation-worker-limits-and-completion-clarification-b29-p02)
  and the full-precision [per-task timing scope](population-tree-scale-extension.md#per-task-computation-time-and-bottleneck-scope-b29-p03).
- [Tree exact-oracle terminal-command validation](tree-exact-oracle-validation.md)
  is the canonical record for manuscript migration batch B09. It preserves the
  [1,440 returned terminal-command checks](tree-exact-oracle-validation.md#terminal-command-validation-inventory-b09-p01)
  and the separately defined
  [optimality-residual and true-root excess-KL maxima](tree-exact-oracle-validation.md#recorded-command-loss-statistics-b09-p02),
  while keeping them distinct from B04, the n=20 pilot, and B08's n=40
  validation.
- [Population tree budget/target stratification](population-tree-stratification.md)
  is the canonical record for manuscript migration batch B10. It preserves the
  [bootstrap seed and source-cluster schedule](population-tree-stratification.md#bootstrap-seed-and-execution-protocol-b10-p01)
  and the retained
  [paired advantages, intervals, and uncertainty scope](population-tree-stratification.md#retained-results-and-uncertainty-scope-b10-p02),
  while keeping within-bank paired policy contrasts distinct from the unpaired
  size banks.
- [Exact-population campaign provenance and audit](exact-population-campaign.md)
  is the canonical record for manuscript migration batches B11--B13. It
  preserves the
  [master-seed and component-stream recipe](exact-population-campaign.md#seed-and-component-stream-recipe-b11-p01)
  and the retained
  [450-problem source-bank context](exact-population-campaign.md#retained-source-bank-context-b11-p02),
  including the verified zero-based graph-family/replicate conventions and
  spawned-child assignment order. It also preserves the
  [campaign coverage and validity audit](exact-population-campaign.md#campaign-coverage-and-validity-b12-p01)
  and the recorded
  [absolute KL-gap statistics and threshold](exact-population-campaign.md#numerical-audit-statistics-b12-p02),
  while distinguishing 450 independent source problems from repeated rows and
  the later audit summary from original execution records. For B13 it preserves
  the later analysis's
  [paired policy bootstrap protocol](exact-population-campaign.md#paired-policy-bootstrap-protocol-b13-p01)
  and
  [linear percentile-interval convention](exact-population-campaign.md#percentile-interval-convention-b13-p02),
  including the 20260811 seed, 10,000 draws, source--information-instance
  resampling unit, per-scope RNG reset, and distinction from B10's
  source-cluster bootstrap.
- [Calibrated finite-sample acquisition and coupling](finite-sample-acquisition.md)
  is the canonical record for manuscript migration batches B14--B16. It preserves the
  [master seed and labelled SHA-256 stream construction](finite-sample-acquisition.md#master-seed-and-sample-stream-construction-b14-p01)
  and the retained
  [sample/seed coupling protocol](finite-sample-acquisition.md#retained-dependence-and-coupling-protocol-b14-p02),
  including fixed source/target realizations across intervention sizes, nested
  node-probe prefixes, same-realised-data method pairing, and shared
  within-condition bootstrap roots. It also cross-links the
  [per-node intervention-bootstrap XOR transformation](finite-sample-acquisition.md#intervention-bootstrap-seed-transformation-b15-p01)
  and the retained
  [support-recovery and orientation-gating context](finite-sample-acquisition.md#retained-support-recovery-and-orientation-gating-b15-p02).
  For B16 it additionally preserves the later recovered
  [reporting-bootstrap seed construction](finite-sample-acquisition.md#reporting-bootstrap-seed-construction-b16-p01)
  and [reporting interval/standard-deviation conventions](finite-sample-acquisition.md#reporting-intervals-and-standard-deviations-b16-p02),
  describing 5,000-draw method-group summaries that resample source-problem
  units and report percentile intervals, distinct from the 256 internal decision
  bootstraps.
  It keeps recovered-source evidence and missing launch linkage explicit and
  distinguishes this calibrated campaign from the calibration-free diagnostic
  and robustness experiments.
- [Calibration-free all-node diagnostic](calibration-free-diagnostic.md) is the
  canonical record for Appendix D.3.1 migration batches B18/E21. It preserves
  the frozen sign/stream recipe, design coupling, estimator and support rules,
  result inventory, and source-unit reporting scope, while distinguishing
  frozen original-diagnostic source from current dependency locators and
  preserving missing launch and raw-array records.
- [Learned public graphs in the finite-sample robustness campaign](learned-public-graph-robustness.md)
  is the canonical record for Appendix D.4 migration B19/E22. It preserves the
  recovered PC traversal, covariance and conflict-handling semantics alongside
  original robustness manifest/result evidence, while keeping missing
  historical executed-source linkage explicit.  The same record also carries
  the Evaluation and Aggregation protocol for B20/E23: its
  [seed and retained-dependence protocol](learned-public-graph-robustness.md#seed-and-retained-dependence-protocol-b20-p01),
  [reporting aggregation protocol](learned-public-graph-robustness.md#reporting-aggregation-protocol-b20-p02),
  and [exclusion and accounting rules](learned-public-graph-robustness.md#exclusion-and-accounting-rules-b20-p03)
  distinguish original artifacts from recovered source and preserve the
  historical-evidence gaps.
- [Light Tunnel chronological split and selection provenance](light-tunnel-chronological-splits.md)
  is the canonical record for B21/E25 and B22/E27. It preserves the shared
  six-role, half-open chronological split; the finite-mode and recovered
  root-shift 500-replicate, four-by-50-row block-bootstrap records; the
  finite-mode linear quantile convention; and Table 10's positive-point-
  estimate improvement definition. It keeps within-session uncertainty,
  archived offline replay, and later recovered source distinct from physical
  collection, independent-session replication, and launch-time evidence.

The evidence labels used here are deliberate: an original execution record can
support a historical run fact; a dependency lock specifies requirements but
does not prove what was installed; recovered source or hashes are later
evidence unless launch-time linkage was recorded; and measurements made on a
later audit host are not original-run provenance.
