# Mk1 shift matching

This module evaluates the ten accepted Light Tunnel Mk1 RGB/polarizer root
actions as a leakage-safe B=1 shift-only matching catalog.

The primary model estimates only each action's root shift magnitude from rows
`[0:400]`; its total-effect direction and invariant covariance come from the
source SCM fitted on `uniform_reference[0:5000]`. A fully empirical Gaussian
finite-mode catalog fit on the same action rows is reported only as a secondary
benchmark. S1 retains the generating action and S2 removes it. Both are evaluated
on the predeclared 17-variable full state and the symmetric 12-variable state
that removes all five directly manipulated actuator coordinates.

Final runs use 500 contiguous-block bootstrap replicates and must be launched in
GNU screen. See `docs/mk1_shift_matching_handoff.md` for the exact command and
results.

