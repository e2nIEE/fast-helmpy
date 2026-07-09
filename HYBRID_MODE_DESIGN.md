# Phase 5 design: hybrid HELM → Newton-Raphson mode

**Status:** design only — not implemented. Written 2026-07-09 for a later
session; see `IMPROVEMENT_PLAN.md` Phase 5 for context.

## Why

HELM's unique value is *robustness*: no starting-point dependence, guaranteed
convergence to the operable branch when a solution exists (Stahl), and a
definite divergence signal when none does. Its cost is the series/continuation
machinery. Newton-Raphson is the mirror image: quadratic convergence when the
start is good, unreliable when it isn't. The hybrid uses each where it is
strong: **HELM to a loose tolerance for a correct-branch start vector, NR for
the last digits.**

Measured context (case2869pegase, PV2, Q-limits on, after Phase 3):
full-accuracy HELM (1e-8) ≈ 0.39 s, of which the tail coefficients and final
checks dominate; HELM to 1e-3 needs only ~15–17 coefficients and Q-limit
switching is already settled there (the two-phase enforcement decides switches
at 1e-4). A numba-NR (pandapower) polish from that start typically needs 1–3
iterations ≈ tens of ms. Expected hybrid total: **~0.2 s**, vs 0.39 s full
HELM — and the gap widens with tighter tolerances and heavier loading.

## Mode A — external polish (recommended first; pandapower's use case)

No new solver inside HELMpy. The caller runs the polish with its own NR:

```python
result = helmpy.solve_helm(Ybus, Sbus, bus_types, V_specified,
                           Qmin=Qmin, Qmax=Qmax, mismatch=1e-3)
# hand result.V to pandapower as start vector:
# pp.runpp(net, init_vm_pu=np.abs(result.V), init_va_degree=np.angle(result.V, deg=True))
```

Everything needed already exists after Phase 4. Remaining work is
documentation plus one convenience: `HelmResults` already reports
`switched_buses` so the caller can pre-apply the PV→PQ switches HELM decided
(otherwise pandapower's own Q-limit loop re-derives them, costing extra NR
iterations).

Recommended fallback ladder for a `LoadflowNotConverged` handler:
1. plain NR (flat or previous-solution start) — the fast path;
2. HELM `mismatch=1e-3` → NR from `result.V`;
3. full-accuracy HELM (`mismatch=1e-8`) — also the arbiter: if HELM diverges,
   the case has no physical solution and no solver will fix that.

## Mode B — internal hybrid (`solve_helm(..., mode="hybrid")`)

Self-contained variant for callers without their own NR. Needs a compact
array-based NR inside HELMpy — the existing `nr.py`/`nr_ds.py` are
xlsx-bound, global-state-heavy and unsuitable; a fresh ~100-line polar NR on
`run.Ytrans_csr`/`Yshunt` (Jacobian in CSR, `scipy.sparse.linalg.spsolve`,
bus types frozen to HELM's post-switching state) is the cleaner path.

Design decisions fixed here so implementation is mechanical:
- **Handoff tolerance:** default `helm_mismatch=1e-3` (Q-switching settles at
  1e-4 already; below ~1e-2 NR from the HELM point converges reliably).
- **Q-limits in the NR phase:** frozen — bus types are taken from the HELM
  phase verbatim. Rationale: re-enabling switching inside NR reintroduces the
  oscillation problems HELM avoids; marginal-limit cases differ from the
  full-HELM solution by at most the limit tolerance and the residual check
  reports it honestly.
- **Distributed slack:** NR polish solves the augmented system with the same
  K factors and a Ploss variable (one extra row/column, mirroring row 2N);
  start value from the HELM Ploss series.
- **Convergence/failure:** NR gets `max_iter=10`; if it fails (pathological,
  since the start is near-exact), fall back to continuing HELM to full
  accuracy — never return an unconverged result silently. `HelmResults` gains
  `mode_used` and `nr_iterations` fields.

## Validation plan

- Hybrid vs full HELM on the 4 shipped grids × PV1/PV2 × DS on/off:
  `max|ΔV| < 10·mismatch`, residual ≤ mismatch.
- Stress: scale case118/case2869 loading up to near the nose point; assert
  the hybrid either matches full HELM or falls back, never diverges silently.
- Benchmark entries in `BENCHMARKS.md` (expect ~2× on the pegase cases at
  1e-8, more at tighter tolerances).

## Effort estimate

Mode A: ~0.5 day (docs + `switched_buses` usage example + benchmark row).
Mode B: 1–2 days (compact NR + DS augmentation + tests).
