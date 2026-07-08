# HELM benchmark history

Run `python benchmark/benchmark_helm.py` (times only the `helm()` call, best of 3,
xlsx loading excluded) and append a section here after every optimization phase.
The regression gate is `pytest` (all 10 method combinations × 4 cases must pass,
see `test/test_helmpy.py`).

## Baseline — Phase 0 (2026-07-07)

State: loader regressions fixed (`Number_bus`/`phase_dict` reverted to dicts),
no performance changes yet. Machine: Windows 11, Python 3.13.14, numpy 2.3.5,
scipy 1.18.0 (conda env pp13). `mismatch=1e-8`, `enforce_Q_limits=True`.

| case | method | time [s] | coefficients | restarts |
|---|---|---|---|---|
| case9 | PV2 | 0.002 | 15 | 1 |
| case9 | DS-M2-PV2 | 0.002 | 15 | 1 |
| case118 | PV2 | 0.058 | 15 | 2 |
| case118 | DS-M2-PV2 | 0.058 | 15 | 2 |
| case1354pegase | PV2 | 1.733 | 25 | 3 |
| case1354pegase | DS-M2-PV2 | 2.352 | 25 | 4 |
| case2869pegase | PV2 | 6.326 | 29 | 4 |
| case2869pegase | DS-M2-PV2 | 4.886 | 25 | 4 |

Profile shares at this baseline (case2869pegase, PV2): analytic continuation
(`Pade`) ~30 %, per-bus Python recurrence loops ~60 %, dense `Ytrans_mod`
construction ~12 %, sparse LU solves < 1 %. The "restarts" column counts full
recomputations caused by PVLIM→PQ switching — the Phase 1 two-phase tolerance
targets exactly this (restarts × full-accuracy runs).

## Phase 1 — quick wins (2026-07-08)

Changes: two-phase Q-limit enforcement (switch decisions at 1e-4, one final
pass at the requested mismatch, which *resumes* the converged coarse series
instead of restarting); `Ytrans_mod` built as `lil_matrix` (no dense (2N+1)²
allocation); convergence checks every 4 coefficients after the first two
failed checks; direct summation instead of Padé for buses whose coefficient
tail is far below the tolerance. Same machine/settings as baseline.

| case | method | time [s] | vs baseline | coefficients | runs |
|---|---|---|---|---|---|
| case9 | PV2 | 0.003 | — | 15 | 2 |
| case9 | DS-M2-PV2 | 0.003 | — | 15 | 2 |
| case118 | PV2 | 0.063 | 0.92× | 15 | 3 |
| case118 | DS-M2-PV2 | 0.060 | 0.97× | 15 | 3 |
| case1354pegase | PV2 | 1.476 | 1.17× | 27 | 4 |
| case1354pegase | DS-M2-PV2 | 1.680 | 1.40× | 27 | 5 |
| case2869pegase | PV2 | 4.818 | 1.31× | 31 | 5 |
| case2869pegase | DS-M2-PV2 | 3.875 | 1.26× | 27 | 5 |

**Result vs plan:** measured ~1.2–1.4× on the large cases, well below the ~5×
the plan estimated. The estimate wrongly extrapolated from "Q-limits off =
1.26 s": disabling limits removes the restart runs entirely, while the
two-phase scheme can only *shorten* them (15–17 instead of 23–29 coefficients
each) — the 3–4 switch-triggered recomputations themselves are algorithmically
unavoidable and their per-coefficient cost is untouched until the Phase 2
vectorization. The two-phase change stays because its benefit multiplies with
Phase 2 (cheap coefficients make the restart count the dominant factor).
On grids without Q-limit switching the "runs" column is 2 (tiny cost: the
final pass resumes and only verifies), and small grids see a slight overhead
(case118: −8 %). Accuracy vs stored references after Phase 1: worst
3.9e-9 p.u. / 4.5e-7 deg over all 40 method×case combinations.

Solution deviation is bounded by the convergence tolerance, not bit-identical:
stopping points changed (check cadence, resume). Test gates relaxed
accordingly (1e-7 p.u. / 5e-6 deg, >10× headroom over observed).
