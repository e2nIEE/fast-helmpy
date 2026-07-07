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
