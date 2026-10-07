# Upstream physical-data retrieval

This page describes how readers obtain physical data directly from its upstream
publisher. The candidate includes **no raw archive or CSV**, and this is an
unexecuted plan—not a download, extraction, checksum verification, successful
run, redistribution grant, or public-release decision.

## Before retrieving

Use a new empty scratch/download directory outside existing data. Do not
overwrite an archive or extracted tree. Retain the final URL, time, byte size,
tool version, MD5, and SHA-256 in a fresh local log; inspect archive members
before final placement. The steps are deliberately descriptive because this
batch did not download or inspect either archive.

Publisher MD5 is an upstream integrity value. SHA-256 values called *historical
local* describe pre-existing local files. Neither is freshly verified here; a
fresh match is new evidence and a mismatch must be reported.

## Mk1 root-shift offline replay

Retrieve only `lt_interventions_standard_v1` **v1.0** (2024-04-14) from
`https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/lt_interventions_standard_v1.zip`.
The publisher MD5 is `476664d024f88e8b7640998bb5e9ee33`; historical local
evidence is 3,909,455 bytes and SHA-256
`8781960b5fff5d752f57566393d4b0f8706cac9b2cd2a3d3226442f17c7bab60`.

After layout inspection, the recovered reader expects these exact paths:

```
<raw-root>/lt_interventions_standard_v1/uniform_reference.csv
<raw-root>/lt_interventions_standard_v1/uniform_red_mid.csv
<raw-root>/lt_interventions_standard_v1/uniform_red_strong.csv
<raw-root>/lt_interventions_standard_v1/uniform_green_mid.csv
<raw-root>/lt_interventions_standard_v1/uniform_green_strong.csv
<raw-root>/lt_interventions_standard_v1/uniform_blue_mid.csv
<raw-root>/lt_interventions_standard_v1/uniform_blue_strong.csv
<raw-root>/lt_interventions_standard_v1/uniform_pol_1_mid.csv
<raw-root>/lt_interventions_standard_v1/uniform_pol_1_strong.csv
<raw-root>/lt_interventions_standard_v1/uniform_pol_2_mid.csv
<raw-root>/lt_interventions_standard_v1/uniform_pol_2_strong.csv
```

Its static interface is `python research/mk1_shift_matching/run_experiment.py
--raw-root <raw-root>`, but that recovered runner writes a historical output
location and is not a safe execution recommendation. It reads CSV with pandas;
each action must have the reference columns and ordered `COMMON_STATE`:
`red, green, blue, pol_1, pol_2, l_11, l_12, l_21, l_22, l_31, l_32, ir_1,
vis_1, ir_2, vis_2, ir_3, vis_3`. Fitting/slices are generated in memory.

`lt_test_v1` is optional for a separate feasibility audit only—not a
root-shift substitute or selected input.

## Mk2 linked-LEDs feasibility audit

Retrieve only `lt_spice_v1` **v1.0** (2026-04-28) from
`https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/lt_spice_v1.zip`.
The publisher MD5 is `47a2be5bd43ec5e098a15ea91bd2e60d`. Historical local
evidence is 40,572,526 bytes and ZIP SHA-256
`5e9f37c29b4823df34798e3ca9a39bf6f3a4e0ef71c8a92c45e6ed52ae99a118`, not a
publisher SHA-256 or fresh verification.

After layout inspection, extract to
`research/chambers_mk2_linked_leds/data/lt_spice_v1/` with exactly:
`exp_1_low_noise.csv`, `exp_2_noisy.csv`, `exp_3_spice_net_noise_1.csv`, and
`exp_3_spice_net_noise_2.csv`. The static reader treats each as a 104-column
table, requires `timestamp`, models `current_ls_raw <- green`, `ir_1 <- green`,
`ir_2 <- green,ir_1`, and `ir_3 <- green,ir_2`, and inventories present
control fields. `_batch`, residuals, folds, tables, and figures are generated,
not upstream files. The old package helper is not a provenance replacement:
package version, transaction, source payload, and output-safe interface remain
unavailable.

## Terms, credit, and unresolved pins

The inspected official dataset READMEs state CC BY 4.0 for their images/CSV
files and distinguish other software as MIT. This is an upstream reference,
not data relicensing by the candidate. Preserve upstream ownership. The Mk1
page requests Gamella, Peters and Buehlmann (2025), DOI
`10.1038/s42256-024-00964-x`; Mk2 additionally requests Vollmer, Pfister and
Weichwald (2026), arXiv `2604.09135`.

The retained provenance record pins the Mk1 README at
`6da4e646cf62bc902f87911c866298a8c98ad2eb` ([text](https://raw.githubusercontent.com/juangamella/causal-chamber/6da4e646cf62bc902f87911c866298a8c98ad2eb/datasets/lt_interventions_standard_v1/README.md), SHA-256
`b2440ead68c2c9d49cf3d690c1ef607e729616650439b0e6f38e92f6b183d3b3`) and
the Mk2 README at `2523f2584a6d10e7ef46fa6c614d2a7f7bd15744`
([text](https://raw.githubusercontent.com/juangamella/causal-chamber/2523f2584a6d10e7ef46fa6c614d2a7f7bd15744/datasets/lt_spice_v1/README.md), SHA-256
`bd228d0f081f33e67ccd32d80e27f4089a0d0792a30b314a3b8b0259023b12b2`). These
texts support the stated dataset terms and requested credit at those revisions,
not archive bytes, layout, checksums, archive-to-license linkage, historical
retrieval identity, ownership, or a redistribution grant. Mk1's pinned hash
matches an earlier mutable capture, but does not establish that the
earlier mutable retrieval came from that revision. Fresh retrieval, the original
Mk2 transaction, environment/runtime validation, output safety, and distribution
approval remain open.

Mk1 uses fixed contiguous within-session blocks. Its bootstrap expresses
within-session uncertainty; offline selection does not execute hardware or show
independent-session replication. See [Light Tunnel chronological splits](light-tunnel-chronological-splits.md).
