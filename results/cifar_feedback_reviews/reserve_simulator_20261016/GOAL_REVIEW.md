# Reserve A: goal-controller hardware gate

2026-10-16. **GO for one bounded hardware A execution** through the published repository and `experiment.py --cifar-feedback-reserve --run-hardware`, after Raspberry preflight verifies the frozen source/INI and exact parent bindings. This is neither a weight-quality pass nor completion of the A/B/C coverage goal. Reviewer performed offline reads/hash/compact arithmetic checks only; no duplicate raw replay or backend call.

## Verified evidence

- Native SIM session `55134577-a813-4d1b-a3de-67c81ab285ca`; acquisition source `43d2e7738754e85a6a20b8a802adad9f956603ec`, tracked clean. Unrelated untracked results are recorded, not silently declared absent.
- Independent audit PASS355033 checks/0 errors; audit SHA `6691aa35788b1d06f281b44ec0c5c70d84c42bfb9a1265e30f76d9d5f2e30fe8`; manifest SHA `8d728c0b8a94c1282f5f4120ffb7abf90b755c7bdcd652fd48eec334343debbc`; auditor SHA `372f48a05abc652cda0f2c9d5891660a9c23429f53d18ea84749c587f5e10459`. All match actual local bytes.
- Runtime `a4e3a822ede1460c905435000912dc91b851b26c8d0608006ceba12708d3f7c9` and config `d042e27faf56a0334f7669689255ea45ad1d43cface8a3fc93ffdf42f4415b9c` match the reviewed snapshot and manifest. Dispatcher bytes also match the launch inventory.
- Compact result, summary, completed wrappers, frozen route/matrix/calibration SHA match manifest. Independently checked exact300 unique `(filter,tap)`,75 each;200 trajectories/2400 outputs;13 fresh taps52–64 and600 fresh partials. The other287 retain historical provenance, including nested42-prefix evidence.
- Fresh acquisition13852 impacts/1245 nonzero writes/1245 decisions, zero current skips, below24323/3232. All33 candidates were quiet in this SIM state. Actual route uses WL6/2,11 first and WL7/15 as negative tap64. Matrix identity, fixed SIM gain0.03. The independent raw audit verifies route/calibration freezing before subsequent SET/CIFAR respectively.

## Quality and scientific limits

Fresh weight fidelity remains failed: NRMSE0.2866001168 and uncertaintyNRMS0.2255898848 exceed their reported20% criteria. No resolved wrong signs or unresolved significant signs; three small weights remain quantization-unresolved. This is a diagnostic full-measurement protocol: accuracy failure is preserved and does not relax or replace safety/route gates. Hardware fixed gain remains1; fitted alpha is diagnostic only. Source-specific gains and physical pulse-duration uncertainty are disclosed.

WL7/15 was assigned and measured without SET in SIM. Residual-selected polarity is correct; no write is required solely to increase the address counter. Thirteen assignments coexist in this one microtile, but the measurement is sequential native one-hot, not a parallel32-input analog MVM. Imported287 data are not evidence of simultaneous current array state.

The imported SIM parent retains the previously audited narrow provenance warning about two untracked validation files at its historical launch; their launch bytes were unknown. That disclosed exception is not a new runtime exception and does not make the entire ancestry fully clean. Current A runtime/source and result pins are verified.

## Hardware conditions and stopping rules

One candidate RESET sweep and one SET sweep per assigned pair; maximum24323 native calls and3232 nonzero attempts/decisions. Keep fresh quiet confirmation, required WL7/15 plus13 positive/12 original negative capacity, native-R current prediction, fresh pair guard reads, ADC/headroom and response validation unchanged. Route failure stops before SET; safety/transport failure stops and preserves partial evidence. No automatic retry, increased budget, threshold relaxation, CPU substitution or old13 fallback.

After hardware acquisition: run the independent audit, compare actual fresh response/derived300 outputs with SIM, report route/quality/cell dispositions and exact command history. B/C remains a separately reviewed finite package for other known reserves. This GO does not approve an unreviewed B/C implementation or declare maximum cell use achieved.
