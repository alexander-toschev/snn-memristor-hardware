# Runbook for Claude: Two-Crossbar Real-Hardware Experiment (RPi5 + RRAM)

**Audience:** this document is written for an AI agent (Claude) operating a shell session on
the physical Raspberry Pi 5 that controls the NNGU memristor crossbar board, with a human
(Alex or Grigory) present/supervising. Read this whole document before running anything.

**Scope: READ-ONLY / CHARACTERIZATION FIRST.** Do not run write/program/erase operations on
the crossbar until a human operator (Alex) explicitly confirms it in this session. Default to
read-only (`mode_7` with `v_dac=0`, i.e. measurement pulses only) until told otherwise.

---

## 0. Who to ask, and what not to do

- Hardware owner / contact: **Grigory Zharkov** (NNGU), email grigoriyzharkov@gmail.com,
  Telegram @Giiiul. If anything is ambiguous about the board, ask before acting — this is
  shared physical hardware, not a simulator.
- **Never exceed 3.3 V on any cell.** This is the transistor gate voltage limit on this
  specific board. Exceeding it risks permanently damaging the board. This is a hard ceiling,
  not a soft guideline.
- **Do not touch row WL=1.** One cell's transistor gate is already blown there; avoid
  addressing any cell with `wl=1`.
- Known-good cells to use for initial testing (don't need to search blindly):
  `(wl=0, bl=2)`, `(wl=2, bl=6)`, `(wl=2, bl=4)`. Well-behaved resistance range is roughly
  **400–2000 Ω** — if a cell reads far outside that (e.g. <200 Ω or >200,000 Ω), it's likely
  stuck/unswitchable; don't force it.
- The Pi's keyboard has **no Russian layout** — use Latin-only names for any files/folders you
  create.
- When finished for the session: do not shut down the Pi yourself unless Alex/Grigory tells
  you to — Grigory needs to be told separately to power off the board.

---

## 1. Context — what this experiment is and isn't

We (the `snn-mnist` / CSNN-CIFAR-100:20 dissertation project) trained a spiking CNN in
simulation. We want to validate that our software memristor simulator (`MemriBoard`
`BoardSimulator`) accurately predicts real write-noise and differential-mapping fidelity, by
programming a few already-trained Conv1 filter weights onto 2 real 32×8 crossbars and reading
them back.

**This is inference-only.** We are not training anything on the chip — we're writing
pre-computed weights and checking how faithfully the hardware reproduces them vs. the
simulator's prediction. See `docs/two-crossbar-hardware-plan.md` for the full experimental
plan (Phases 1–4) and `docs/HARDWARE_DRIVER_INTERFACE.md` for the software interface contract
our test code expects from the hardware driver.

**Current status (as of 2026-10-14):** we have NOT yet confirmed which driver class is wired
up in the MemriBoard app on this specific Pi (expected to be something like
`rp5_fpga_python` / `RPI_modes_FPGAed`, referenced in `MemriBoard/manager/board.py` but not
found in any public repo we could locate). **Your first job is reconnaissance, not
experimentation** — see Phase A below.

---

## 2. Reference code (clone this first)

Our test/protocol code and the MemriBoard simulator (as a submodule) are here:

```bash
git clone --recurse-submodules https://github.com/alexander-toschev/snn-memristor-hardware.git
cd snn-memristor-hardware
```

Key files to read before touching the real board:
- `DRIVER_INTERFACE.md` — the exact `conn` object interface (`connect`, `mode_7`, `mode_mvm`)
  that our protocol code expects from any driver, with safety-critical constants table.
- `TWO_CROSSBAR_PLAN.md` — the experiment plan phases.
- `experiments/seq_reprogram_dryrun_20261013/` — our rehearsed protocol scripts, currently
  tested only against the software `BoardSimulator`. These are what we eventually want running
  against the real driver with zero/minimal code changes.
- `memristor_hw/MemriBoard/manager/board.py` and `manager/service/drivers.py` (inside the
  submodule) — this is where `rp5_fpga_python` is referenced; this is your starting point for
  Phase A reconnaissance below.

---

## 3. Phase A — Reconnaissance (do this first, read-only, no hardware I/O needed)

Goal: find out what's actually configured on this Pi, without touching the crossbar yet.

1. Locate the MemriBoard installation on this Pi (per Grigory's instructions, likely
   `~/Documents/MemriBoard/`).
2. Find its driver/board-type configuration — look for something equivalent to
   `simulator_settings.ini`, or search the app's config/state for `board_type`. Report what
   value is set (we expect `rp5_fpga_python` but need to confirm).
3. Search for the actual driver implementation:
   ```bash
   find ~/Documents/MemriBoard -iname "*rp5_fpga*" -o -iname "*rpi_FPGAed*"
   python3 -c "import MemriCORE" 2>&1   # check if the MemriCORE package is importable
   pip show MemriCORE 2>&1
   find / -iname "MemriCORE" -type d 2>/dev/null
   ```
4. If found, read the driver class source (likely something like `RPI_modes_FPGAed`) and
   report its actual method signatures back — we need to confirm it matches (or adapt)
   `DRIVER_INTERFACE.md`'s expected `connect()` / `mode_7()` / `mode_mvm()` contract.
5. Check what `connect()` requires as an argument on this real setup (a device path? an IP?
   nothing, since it's local?).

**Report back before proceeding to Phase B.** Summarize: driver class name + location,
constructor/connect signature, and whether method names match our expected interface.

---

## 4. Phase B — Single-cell read-only characterization (after Phase A, and after explicit
   human go-ahead for hardware I/O)

Goal: read a few known-good cells without writing anything, to sanity-check the driver and
confirm resistance values are plausible before any write/erase.

1. Using whatever driver object Phase A identified, perform a **read-only** measurement
   (equivalent to `mode_7(v_dac=0, tms=0, tus=0, rev=0, task_id=0, wl, bl)`) on the three
   known-good cells: `(wl=0,bl=2)`, `(wl=2,bl=6)`, `(wl=2,bl=4)`.
2. Confirm the returned resistance is in a sane range (expect roughly 400–2000 Ω per
   Grigory's notes, though these specific cells' current state is unknown until you read them —
   any stable finite value is fine, just sanity-check it's not nonsense/an error code).
3. Do **not** issue any write/program/erase pulse in this phase. If the driver's API doesn't
   cleanly separate read from write, stop and ask before proceeding — don't guess.
4. Report the 3 readback values and timing (how long each read took) back to the human
   operator.

---

## 5. Phase C — Write/erase (ONLY with explicit human confirmation, not before)

Do not proceed to this phase without an explicit go-ahead in this session from Alex. When
cleared:

- Follow the per-cycle protocol in `TWO_CROSSBAR_PLAN.md`: erase (RESET) → program
  differential G+/G− → readback → compute RMSE/Pearson vs the ideal target → log → next.
- **Never exceed 3.3 V** on any pulse (hard ceiling, see Section 0).
- Start with a single cell, single pulse, smallest reasonable amplitude, and verify the
  resistance moves in the expected direction before doing anything more ambitious.
- Save every programming/readback history to a file (Latin filenames) before erasing the next
  cell — these logs are the actual scientific deliverable.

---

## 6. What to report back after each phase

For Phase A: driver class/location, signature, interface match/mismatch vs
`DRIVER_INTERFACE.md`.

For Phase B: the 3 readback resistance values, read latency, any errors encountered.

For Phase C (only once reached): per-cycle RMSE/Pearson vs ideal, full programming history
file paths, any anomalies (unexpected resistance jumps, cells that refuse to switch, etc).

If anything is unclear, unexpected, or looks risky (voltage computation doesn't obviously
respect the 3.3 V ceiling, driver behaves differently from the interface contract, etc.) —
**stop and ask a human**, don't improvise on live hardware.
