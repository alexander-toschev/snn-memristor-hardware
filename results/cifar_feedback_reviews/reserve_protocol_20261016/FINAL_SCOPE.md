# Finite reserve coverage: A/B/C design

2026-10-16. Offline goal-control design. This does not authorize a hardware launch and does not change the frozen A implementation.

The baseline has complete300/2400 acquisition but does not yet resolve all already known plausible reserves. Freezing a smaller pool cannot by itself establish maximum justified use. The finite set below is fixed before new measurements; no extra candidates are added after observing programming or CIFAR errors.

## Fixed evidence set

Known quiet but unassigned: WL2/3, WL4/7, WL5/6, WL6/2, WL6/11. Historically singleton-supported outside118: WL3/12, WL3/13, WL3/31, WL5/15, WL7/15. Historical support does not establish fresh RESET eligibility or accurate programming.

| Experiment | Actual trained taps | RESET pool | Special allocation | Max calls / writes |
|---|---|---|---|---:|
| A, existing frozen protocol | f3,parent4,taps52–64 | WL6 original16; WL7 original16 +BL15 | WL7/15 required at negative tap64; WL6/2,11 priority if freshly quiet |24323 /3232|
| B, separate proposed protocol | f2,parent0,taps0–10 | WL4 original14; WL5 original14 +BL15 | If freshly quiet, WL4/7 positive and WL5/6 negative at tap0; WL5/15 negative at tap1 |21421 /2794|
| C, separate proposed protocol | f1,parent0,taps0–8 | WL2 original12; WL3 original12 +BL12,13,31 | If freshly quiet, WL2/3 positive at tap0; WL3/12,13,31 negative at taps0,1,2 |18957 /2466|

B/C totals40378 calls/5260 writes,20 new coefficients and1200 new partials. All A/B/C totals64701/8492,33 new coefficients and1800 new partials. On complete success267 original coefficients remain imported. Counts independently recomputed from immutable fixture/routing JSON and command structure without importing collector budget functions.

B uses29 candidates:7076 RESET +8921 pair +5424 measurement calls. C uses27:6588 RESET +7299 pair +5070 measurement. Each candidate has at most55 RESET writes and244 total calls; each assigned pair at most109 SET writes and811 total calls. No larger voltage, new mode, reduced quiet threshold or weakened current/native-R/ADC guard is introduced.

## Allocation and failure policy

For B/C, reserve every freshly eligible preferred address for its designated slot first. Fill remaining slots from ordinary fresh-quiet order excluding already reserved addresses. A failed preferred candidate gets a preserved negative screening disposition and does not alone abort a viable11/9-coefficient route. A remains unchanged: its purpose specifically requires the new WL7/15 for the13-coefficient capacity check.

Every candidate gets one bounded sweep and independent confirmation. No rescreen-until-pass, remap based on CIFAR accuracy, SET solely to increase counters or digital per-weight replacement. Whole-route insufficient capacity stops that experiment before SET; this does not label all quiet candidates unsuitable. Safety/transport aborts stop the run and remain distinct from scientific negative qualification. Successful routes program all weights before fresh C/anchors/CIFAR and final validation.

The terminal per-reserve dispositions are: (1) freshly quiet and actually assigned/measured in CIFAR, with separate weight fidelity; (2) not freshly quiet under this bounded protocol; (3) freshly quiet but unable to form a complete permitted route; or (4) not assessed because a technical stop occurred. Only the first two settle an individual cell's intended finite test. The latter two are incomplete evidence, never dead-cell claims or successful use.

## Immutable composition

A imports287 exact taps; B/C replace only their original20 taps. New derived sessions bind exact source-specific parent manifests, audits, raw/result hashes, and nested42-prefix ancestry. If A fails, any baseline import and failed-A command accounting need explicit separate reviewed bindings; success cannot be inferred. Cumulative command history counts each ancestor once, including failed attempts; contributing data counters exclude failed attempts. No excluded contribution is quietly restored after a failed replacement. Baseline remains independently valid.

## Finite completion criterion

After this fixed package: retain audited full300/2400 evidence and native SIM comparison; show a table for all ten known reserves with fresh disposition, assigned/SET/accurate counts separately, and recompute the unique contributing address union after replacements. Do not promise105+10: remapping can displace another uniquely used address. Explain any displacement from the actual assignment ledger rather than hiding it in a historical union. State measured simultaneous capacity per tile separately from union across time.

Other addresses with mode9 conductance but no resolved singleton response, or nonlinear joint-input responses, are outside the currently qualified one-hot operator. Existing evidence justifies retaining them as unresolved for this experiment, not as dead. A further all256 or nonlinear-method search is not required to finish this finite experiment. Absolute optimality over all possible states, mappings and measurement modes is not asserted.

The goal is not to force the20% fidelity threshold by endless cycling. Complete measurement may reveal poor weight fidelity. Nevertheless a known usable reserve cannot be dismissed merely by restating the frozen pool limit; the explicit A/B/C dispositions resolve that identified gap. Any technical or capacity failure needs a candid remaining-gap decision, not automatic extra scope or false completion.
