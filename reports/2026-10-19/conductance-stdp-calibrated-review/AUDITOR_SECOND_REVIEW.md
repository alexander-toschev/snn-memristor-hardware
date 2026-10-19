# Independent auditor review

19 October2026. Review and synthetic tests only; no native Connector, native
SIM, hardware or network experiment was executed by this reviewer.

Final reviewed auditor SHA256:
`f1fab366818f554e90dea185c0a23073074f0e2984bd5af3c15ff415cac8bcfa`.

The initial review reproduced three accepted corruptions after recalculating
raw/result hashes and updating compact observations: incorrect training
eligibility, incorrect sham resistance, and two updates referring to the same
physical pulse. The auditor author corrected these before native acquisition.

The final source binds decision/complete records to exact native commands,
addresses, signs, amplitude, phase, guard/sham measurements and eligibility;
it requires unique exhaustive ownership of write/guard/sham calls. Physical
non-fit requests must occur after gain freeze. Original acquisition files,
physical parameters and budgets were not changed.

The independent numerical replay uses fresh WL5 fit measurements only, all nine
preregistered candidates, the fixed eligibility/tie-break rule and one common
frozen gain for both banks. It verifies all LIF/eligibility trajectories,
checkpoint/bracket ownership, control/update decisions, completed coverage and
the no-eligible prefix. No-eligible is an informative stop status, not a
successful learning result.

An independent run of `tests.test_audit_conductance_stdp_calibrated` passed:
2 unittest methods,29.207s, covering complete and no-eligible synthetic bundles
and12 rehashed semantic corruptions. Fake data are unit tests only and do not
establish nativeSIM activity, hardware behavior or learning efficacy.

Verdict: no remaining blocker found for the registered nativeSIM run followed
by independent actual-data audit. Hardware registration remains separate and
closed. Every real read/write/diagnostic continues through public MemriBoard
Connector, native converters and INI; no interface/driver bypass is allowed.
