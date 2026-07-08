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

## Phase 2 — vectorized recurrence (2026-07-08)

Changes: the per-bus Python loops of the coefficient recurrence are replaced
by whole-bus-axis array operations — neighbor sums are sparse mat-vecs
(`Ytrans_csr @ V[:, n-1]`, plus a `Yphase_csr` matrix for phase-shifter
corrections), the coefficient convolutions (`VV`, `S1`, `Fconv`, PV1 `aux`,
the `W` inverse-voltage series) are `einsum` reductions along the stored
series history, and the PV1 right-hand-side column correction is one sparse
mat-vec (`Y_Vsp_cols`). The seven per-bus `evaluate_bus_eq_*` functions are
gone; `evaluate_rhs` handles all bus groups, both PV models and both DSB
methods. Same machine/settings as baseline.

| case | method | time [s] | vs Phase 1 | vs baseline | coefficients | runs |
|---|---|---|---|---|---|---|
| case9 | PV2 | 0.002 | — | — | 15 | 2 |
| case9 | DS-M2-PV2 | 0.002 | — | — | 15 | 2 |
| case118 | PV2 | 0.027 | 2.3× | 2.1× | 15 | 3 |
| case118 | DS-M2-PV2 | 0.026 | 2.3× | 2.2× | 15 | 3 |
| case1354pegase | PV2 | 0.620 | 2.4× | 2.8× | 27 | 4 |
| case1354pegase | DS-M2-PV2 | 0.733 | 2.3× | 3.2× | 27 | 5 |
| case2869pegase | PV2 | 2.038 | 2.4× | 3.1× | 31 | 5 |
| case2869pegase | DS-M2-PV2 | 1.653 | 2.3× | 3.0× | 27 | 5 |

The recurrence itself is essentially free now: `evaluate_rhs` + the LU
back-substitutions account for ~0.1 s of the 2.0 s on case2869pegase.
Remaining profile: per-bus scalar `Pade()` calls ~70 % (Phase 3 batches
these), `modif_Ytrans` LIL rebuilds after Q-limit switches ~25 %
(4 × 160k Python-level element assignments — worth vectorizing into a COO
build alongside Phase 3). Worst deviation vs stored references is unchanged
from Phase 1 (3.9e-9 p.u. / 4.5e-7 deg over all 40 combinations): stopping
points are identical, only the summation order changed. Side effect: the
`ComplexWarning`s from implicit complex→float casts are gone (explicit
`.real` everywhere), and the full test suite runs in 28 s instead of 85 s.
