# Changelog

## Unreleased

### Added
- `f_star_audit.py` and `data/cfmr_observed.json`: measured `f(x0)` and the best
  objective value reached from `x0`, with the toolchain that produced them.
- `cfmr.f_L()` and `cfmr.load_observed()`. Benchmarks should take `f_L` from
  `f_L()` rather than reading `f_star_expected` directly — see *Reference values are
  not certified minima* in the README for the two problems where it matters.
- `cfmr_profile.py` now scores against `cfmr.f_L()` and warns when a run reaches
  materially below the floor it was scored against.
- `data/cfmr_observed.json` folds in a full 90-problem Py-BOBYQA sweep at
  `100(n+1)` evaluations, `rhoend=1e-12` (2026-09-10/11, 90/90, no failures) on
  top of L-BFGS-B from `x0`. The wider sweep found no problem beyond the two
  already known where a solver goes below the tabulated `f*`.

### Changed
- README: the verification section states the result (90/90 on CUTEst 2.7.1 /
  SIFDecode 3.1.1 / MASTSIF `29adac9f`) instead of disclaiming that no check had
  been run. It also notes that `cfmr_profile.py` uses `BUDGET_SCALE = 100` while
  the dataset's own `budget_scale` field is 50, and why.
- `.gitignore`: benchmark outputs are matched by pattern, since the configuration
  is part of the file name.

### Fixed
- `cfmr_profile.py`'s stale-floor warning compared against zero, so a solver that
  converged a little further into the *same* local minimum than the earlier run
  that set the floor tripped it. It now needs an undercut worth 1% of the narrowest
  plotted tau's band, which is the point below which nothing on any curve can move.
  (Seen on `EXT::BROYDN7D`: the 100(n+1) run undercut the 50(n+1) floor by 1.6e-8,
  a relative 4.5e-10 -- the same point, converged tighter.)
- `EXT::NONDQUAR`, `f_x0_expected`: `1000000.0` -> `106.0`. The source paper prints
  `f(x0) = 106`, `f(x*) = 0` at `N = 100`; the old value was a transcription slip.
  Confirmed against the printed table, and CUTEst reports `106.0` at `x0` on the
  installation named above. `f_star_expected` (0) and `n` (100, `{"N": 100}`) were
  already correct and are unchanged.

  With this row settled, every value in `cfmr_problems.json` is again a number read
  off a printed table, which is what makes the README's claim that any row can be
  checked against the source by eye true of all 90.
