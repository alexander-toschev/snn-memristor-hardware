# B/C prebinding review

2026-10-16. **GO to finalize actual parent bindings and the independent auditor; NOT a native SIM or hardware GO yet.** No blocking implementation issue found in this draft. Reviewer performed offline reads, exact snapshot hashing and independent scope/count arithmetic; no backend or Git action. Fourteen focused tests passed as reported by the programmer; their source/assertions were reviewed rather than rerunning the same suite.

Reviewed snapshot: runtime `89764591d65e3ffb082fe7094f353cd2b2ccd5fff362022d78fc31e5e1f95401`; routing plan `87d2656587a5f1484243019d82434b8b89e25b81ebfd92818883a0188d59461b`; config `38ad746440e367ad863e3d6c5f742692749c9a292078132af025136d96bf254c`. All seven hashes in DRAFT_SNAPSHOT.json matched actual files, including both test modules and DESIGN.md.

## Scope and safety

B preserves filter2,parent0,taps0–10. Its pools are exactly original WL4/14 and WL5/14 plus BL15. Preferred-if-quiet assignments are WL4/7 and WL5/6 at tap0 and WL5/15 at tap1. C preserves filter1,parent0,taps0–8, original WL2/12 and WL3/12 plus BL12,13,31. Preferred assignments are WL2/3 at tap0 and negative WL3/12,13,31 at taps0,1,2. Thus all seven remaining predetermined addresses are screened; no additional posthoc addresses appear.

Fresh screening precedes route/SET. All eligible preferred cells are reserved for their designated slots before any fallback fills an empty slot. Unknown/duplicate quiet evidence is rejected. A nonquiet reserve is reported and replaced only by another freshly quiet pool address; it does not independently abort a complete route. Physical assignments are unique. Historical quiet evidence never substitutes for current measurements.

Independent recomputation used only immutable fixture/routing JSON and the pure routing plan: B29 candidates/11 weights/5424 tail =21421 calls and2794 writes; C27/9/5070 =18957 and2466. Combined40378/5260 covers20 fresh weights and1200 partials. Per-block and remaining aggregate impact/write/decision ceilings are applied before acquisition. All inherited native-R/current/ADC/response guards, bounded sweeps, source-specific fixed gains and public Connector calls remain unchanged.

## Failure conservation

Only explicit RouteCapacityError allows B scientific-capacity failure to be preserved and C to execute once. It is raised after complete candidate screening and before SET. Other ValueError, transport, guard, budget or invalid-response failures propagate and stop all blocks. The finally branch records actual counters even after a capacity failure. No resweep/retry or old-B fallback occurs.

If either block fails, the collector preserves successful partial chunks but raises before the full aggregate/summary. A C-only result is not mislabelled full300. Dispositions distinguish nonquiet, quiet-but-no-route, selected-but-incomplete, unassessed and complete actual CIFAR assignment. No SET is forced merely to improve address counters. Failed-address cause is not inferred as dead-cell evidence.

## Composition, calibration and provenance

Exact B11+C9 are removed from the complete parent;280 remaining coefficients retain source/origin hashes. The old aggregate checks exact300/75-per-filter and measured600-partial ledgers; new aggregation cannot fill omitted contributions with zeros or CPU weights. All writes precede fresh anchors/C2, frozen before new CIFAR; fixed decoder uses actual two-channel sums only. Sum75 precedes trained BN and one LIF step per logical time.

Parent choice is explicit: completed-A, or audited baseline plus separately bound failed-A evidence. Manifest/audit identity, source, acquisition status, complete2400 coverage, counters and all result hashes are checked, then ancestor42/baseline proofs are revalidated. Failed-A accounting adds only its new attempts to the baseline cumulative total, never its cumulative ancestry twice. The old data are not mutated.

The entrypoint rejects draft=True/native_execution_enabled=False before filesystem/backend work. Both bindings currently None. This is intentional and must not be bypassed merely for a dry-run or to manufacture a passing test.

## Required before execution review

Freeze the actual source-specific completed-A or baseline+failed-A bindings after their independent audits. Exercise those actual import paths and mutation/cross-source rejection, including the failed-A choice if used. Current draft tests prove arithmetic/branch behavior, but do not establish nonexistent hardware parent bindings. Freeze final config/dispatcher/source hashes and review the independent raw auditor. Publish a clean snapshot, run native SIM once, then assess the real audit before any hardware GO.

The final report must combine A/B/C dispositions for all ten known reserves and recalculate the contributing address union after excluding old chunks. An earlier unique address may be displaced: a count like105+10 is not assumed. Complete acquisition, measured simultaneous tile capacity, union across time, SET-attempted cells and accurately programmed weights remain separate. This finite scope does not expand in response to poor accuracy and does not claim global optimality over every mode/state of all256 cells.
