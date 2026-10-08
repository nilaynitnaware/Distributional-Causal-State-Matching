# Historical Git blob exports

This directory currently contains no historical Git-blob exports. If exports
are added, each must be an exact byte export of an identified
`git:<commit>:<path>` locator and be accompanied by a `manifest.tsv` recording
its Git blob ID and SHA-256.

Historical exports would not establish the original installed environment,
launch-time source linkage, or a complete standalone historical checkout.
They would not imply that a future public Git history contains the referenced
commits. Independent ownership/permission review remains unverified.
