# Historical Git blob exports

Each file below this directory is the exact byte export of an identified
`git:<commit>:<path>` locator cited by a candidate reproducibility page. The
directory name is the full resolved canonical commit, and `manifest.tsv`
records the Git blob ID and SHA-256 for each export.

These exports do not establish the original installed environment, launch-time
source linkage, or a complete standalone historical checkout. Transitive
dependencies that were not explicitly cited remain unresolved. The exported
files do not imply that a future public Git history contains the referenced
commits. License review remains pending.
