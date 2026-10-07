# Supporting validation evidence

This page indexes the selected evidence export. It does not modify the copied
records or turn the candidate into a complete reproduction release. Historical
absolute paths and commands inside those immutable records are retained as
historical records, not rewritten as public-candidate locations.

These small checking records are selected evidence but remain subject to
ownership review. Their links identify bytes currently on disk; they do not
settle licensing, a complete archive, or successful execution.

## Evidence groups and purpose

### Appendix-D reproducibility audit

The two machine-readable fact manifests,
[`facts_manifest.json`](../../outputs/appendix_d_reproducibility_audit/facts_manifest.json)
and [`facts_manifest_v2.json`](../../outputs/appendix_d_reproducibility_audit/facts_manifest_v2.json),
are later audit/reconstruction records. `facts_manifest.json` is the original
audit facts export; `facts_manifest_v2.json` records the corrected audit facts
and supersedes v1 where its corrections apply. They support numerical-audit
and provenance-gap accounting, but do not make every historical dirty or
untracked execution state byte-exactly recoverable.

### Scalability feasibility probes

The excluded historical locators
`outputs/s01_scalable_matching_feasibility_v1/s01_feasibility_audit.md`,
`outputs/s01_scalable_matching_feasibility_v1/s01_probe_results.json` (SHA-256
`7d4f4ed56d86fefa28fe19cea4f39b7a79952484b4af443448c41bfc7ac7f0d1`), and
`outputs/s01_scalable_matching_feasibility_v1/run_s01_probes.py` are a later
bounded diagnostic. The audit explicitly limits its positive
result to population tree-MEC representation and its six specified probes. It
does not establish a general finite-sample policy, complete large-MEC coverage,
or a worst-case scalability guarantee.

### Tied-optimum validation

The [tied-optimum report](../../outputs/s01_b3_tied_optimum_validation_v1/s01_b3_tied_optimum_validation_report.md),
[frozen cases](../../outputs/s01_b3_tied_optimum_validation_v1/frozen_cases.json),
[case results](../../outputs/s01_b3_tied_optimum_validation_v1/case_results.json),
[validation results](../../outputs/s01_b3_tied_optimum_validation_v1/validation_results.json),
[coverage matrix](../../outputs/s01_b3_tied_optimum_validation_v1/coverage_matrix.md),
[runner](../../outputs/s01_b3_tied_optimum_validation_v1/run_s01_b3_tied_optimum_validation.py),
[independent reference](../../outputs/s01_b3_tied_optimum_validation_v1/tied_optimum_reference.py),
and [implementation manifest](../../outputs/s01_b3_tied_optimum_validation_v1/implementation_manifest.json)
are later validation evidence for four small population Gaussian SCM cases.
The report retains negative cases and cap-inconclusive outcomes. Its scope is
bounded shift/scale commands, finite retained ties, and direct KL checks; it
does not establish arbitrary continuous degeneracy, finite-sample
certification, arbitrary CPDAG coverage, or worst-case scaling.

## Dependencies and retained gaps

The scalability runner is an excluded historical source locator retained only
to identify the bounded diagnostic above; it is not a selected dependency or a
runnable fresh route. Its default Small-MEC frozen input is absent from public
membership, and no sufficient archived replay recipe is supplied. The
tied-optimum runner is a selected source record, but its dependencies are not
yet assembled here. In particular, the tied-optimum manifest names `graphs/`,
`oracle/`, `research/`, and `tests/` inputs; the feasibility records name
research/oracle modules and older output groups. Those source runners and
helpers remain necessary only for the bounded evidence described here.
The tied-optimum manifest also records a historical absolute interpreter path;
this candidate does not claim that path resolves.

The Appendix-D v2 narrative audit itself remains a private internal document;
only its two serialized fact manifests are exported in this batch. The
canonical `docs/paper_support/` directory was absent at the inspected source
location, so it contributes no export. No license conclusion is made
for these records or their transitive dependencies.

## Internal-document correspondence

| Original locator | Role | Public representation | Remaining gap |
| --- | --- | --- | --- |
| `appendix_d_reproducibility_audit_v2.md` | Later read-only Appendix-D audit and correction narrative | `facts_manifest.json` and `facts_manifest_v2.json` above | The private prose audit is not copied wholesale; its historical-source recoverability discussion remains a separately controlled record. |
| `docs/paper_support/<referenced files>` | Potential supporting-paper locator | None; no such directory was present at the inspected canonical source path | A later scoped export requires an actual referenced file and its license/access decision. |
| `outputs/s01_scalable_matching_feasibility_v1/` | Bounded feasibility analysis and six diagnostic probes | No selected payload; the audit, probe output, and runner are excluded historical locators | The runner's default Small-MEC frozen input is absent, no sufficient archived replay recipe is supplied, and transitive dependencies/environment remain unresolved. |
| `outputs/s01_b3_tied_optimum_validation_v1/` | Tied-optimum certificates, negative cases, and implementation provenance | Eight scientific/source/validation records above | `exploratory_constructions.md` is withheld as internal planning, generated bytecode is excluded, and source/test dependencies remain unresolved. |

Internal review recommendations are not scientific findings. The findings and
limits stated on this page are limited to the linked copied records; release
planning, licenses, and public-access decisions remain private-control work.
