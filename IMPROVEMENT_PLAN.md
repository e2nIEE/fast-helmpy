# HELMpy improvement plan

**Date:** 2026-07-07
**Basis:** profiling analysis in `c:\git\pandapower_ai\ANALYSIS_helmpy_speedup.md`
**Deployment model:** this fork will become a standalone repository; pandapower will
consume it **as a library** (AGPL stays untouched, no code moves into pandapower).

**Baseline (case2869pegase, `mismatch=1e-8`, `pv_bus_model=2`, Q-limits on):** 6.3 s,
29 coefficients, 4 full restarts. Reference for all speedup claims below.

Phases are ordered by (ease × impact). Each phase leaves the package in a working,
validated state, so work can stop after any phase.

---

## Phase 0 — Groundwork: correctness + measurement (prerequisite, ~1 day)

> **Status: DONE (2026-07-07).** `Number_bus`/`phase_dict` reverted to dicts;
> pytest suite in `test/test_helmpy.py` (41 tests, all green; pegase cases marked
> `slow`); benchmark harness in `benchmark/benchmark_helm.py`; baseline recorded
> in `BENCHMARKS.md`.

Nothing can be safely optimized without this.

**Tasks**
1. Fix the two dict→ndarray regressions in `helmpy/core/classes.py`:
   - `Number_bus` (line ~191): must handle non-consecutive bus numbers → revert to dict
     or index via a mapping array sized by `max(bus_number)`.
   - `phase_dict` (line ~212): stores `[[buses], [admittances]]` lists → revert to dict.
   Without these, case1354pegase / case2869pegase (and any real-world grid with
   non-consecutive numbering or phase shifters) crash on load.
2. Turn `test/test_helmpy.py` into a proper pytest suite: parametrize over the
   10 pv/DSB method combinations × 4 cases, assert max |ΔV|, |Δθ| against the stored
   MATPOWER/HELM reference results (`data/results/`). This is the regression gate for
   every later phase.
3. Add a benchmark script (repeatable timing per case + coefficient count + restart
   count), commit baseline numbers.

**Acceptance:** all 4 shipped cases pass on all 10 method combinations; baseline
timings recorded.

---

## Phase 1 — Quick wins, no structural change (~1–2 days, est. 6.3 s → ~1.2 s)

> **Status: DONE (2026-07-08), delivered less than estimated.** Measured
> 1.2–1.4× on the pegase cases (case2869 PV2: 6.33 s → 4.82 s), not ~5×: the
> two-phase scheme shortens the Q-limit restart runs but cannot remove them,
> and their per-coefficient cost stays until Phase 2. Includes a series
> *resume* for the final full-accuracy pass (extends the coarse series when no
> bus switched). Details and revised reasoning in `BENCHMARKS.md`. The 41-test
> regression suite passes; gates relaxed to 1e-7 p.u. / 5e-6 deg because
> stopping points changed (worst observed deviation 3.9e-9 / 4.5e-7).

Three independent changes, each small and separately verifiable.

### 1.1 Two-phase Q-limit enforcement  *(measured ~5× on case2869)*
Today every PVLIM→PQ switch discards the series and re-converges to full `mismatch`
(case2869: 4 restarts to 1e-8). Change `computing_voltages_mismatch` /
`check_PVLIM_violation` so that:
- while any Q-limit switching is still possible, converge only to
  `max(mismatch, q_switch_tol)` (default `1e-4`) before checking limits;
- once a pass completes with no switch, do one final run to the user's `mismatch`.

Switching decisions depend on Q values, which are settled far above 1e-8 accuracy —
verify on all test cases that the *sequence of bus-type switches is identical* to the
current behavior, and final voltages match within `mismatch`.

### 1.2 Sparse construction of `Ytrans_mod`  *(−0.85 s, removes O(N²) memory)*
`modif_Ytrans` currently allocates a dense `(2N+1)²` float array (~264 MB at 2869
buses) and converts via `csc_matrix(dense)` (the `nonzero` scan alone is 0.7 s).
Build COO triplets (`rows`, `cols`, `vals` lists / preallocated arrays) directly from
`branches_buses` and pass them to `csc_matrix((vals, (rows, cols)), shape=...)`.
Also make `CaseData.Ytrans` / `Y` sparse (CSR) — prerequisite for Phase 2's mat-vecs
and for grids > 3000 buses at all.

### 1.3 Cheaper convergence checking cadence
- After the first two checks, check every 4th coefficient instead of every 2nd
  (Padé order grows by 1 per 2 coefficients; the mismatch typically drops ~1 order of
  magnitude per check, so the overshoot cost is small and the check cost halves).
- Skip the per-bus Padé comparison for buses whose series tail
  `|c_i[n]| + |c_i[n−1]|` is already far below `mismatch` (direct summation provably
  converged there); Padé-evaluate those buses once at the end.

**Acceptance:** regression suite green; benchmark shows ≥4× on case2869; memory
high-water mark reduced.

---

## Phase 2 — Vectorize the coefficient recurrence (~3–5 days, est. → ~0.4 s)

> **Status: DONE (2026-07-08).** case2869pegase PV2: 4.82 s → 2.04 s (3.1×
> cumulative vs baseline). All bus groups, both PV models and both DSB methods
> are handled by one vectorized `evaluate_rhs`; the recurrence now costs ~0.1 s
> — the remainder is Padé (~70 %, Phase 3) and the LIL matrix rebuilds after
> Q-limit switches (~25 %). Deviations vs references unchanged from Phase 1;
> all 41 tests pass (suite now 28 s instead of 85 s). One fidelity note: the
> original DS-M2 slack-row shunt convolution pairs orders summing to n while
> the PV rows pair n−1; replicated exactly (see comment in `evaluate_rhs`) —
> worth clarifying against the paper in a later phase.

This attacks the ~60 % of runtime spent in per-bus Python loops
(`evaluate_bus_eq_*`, `calculate_inverse_voltages_w_array`,
`compute_complex_voltages`, `Calculo_Vre_PV`). Highest effort, highest payoff;
results must stay bit-identical (same arithmetic, reordered).

**Design**
- Precompute once per run: index arrays `pq_idx`, `pv_idx`, `slack_idx`, and a sparse
  CSR `Ytrans` (off-diagonal transfer admittances).
- Replace per-bus neighbor sums `PP = Σ_k Ytrans[i,k]·V[k,n−1]` with one sparse
  mat-vec for *all* buses: `PP = Ytrans_csr @ V[:, n-1]`.
- Replace per-bus convolutions with axis-1 reductions over the coefficient history,
  e.g.:
  - `W[:, n] = -np.einsum('ij,ij->i', W[:, :n], V_complex[:, n:0:-1])`
  - `VV = np.einsum('ij,ij->i', V_complex[:, 1:n], np.conj(V_complex[:, n-1:0:-1]))`
  - the `barras_CC` / `slack_CC` accumulations become rows of an `(N, max_coef)`
    array updated with the same pattern.
- Assemble `Soluc_eval[:, n]` for each bus group with fancy indexing
  (`Soluc_eval[2*pq_idx, n] = result.real; Soluc_eval[2*pq_idx+1, n] = result.imag`).
- `compute_complex_voltages` collapses to
  `V_complex[:, n] = coefficients[0::2, n] + 1j*coefficients[1::2, n]` (+ PV1 variant).
- Keep the existing per-bus functions behind a debug flag for differential testing
  (`assert np.allclose(vectorized, loopy, rtol=0, atol=1e-15)` on small cases), then
  remove them.

Do `pv_bus_model=2` first (simpler, default), then PV1, then the two DSB methods.

**Explicit non-goal:** numba. Pure numpy/scipy keeps the library dependency-light for
pandapower consumption; revisit numba only if Phase 2+3 measurements still disappoint.

**Acceptance:** regression suite green for all 10 method combinations; ≥5× over
Phase 1 on the recurrence portion of the profile.

---

## Phase 3 — Analytic continuation overhaul (~2–3 days, est. → ~0.2 s)

Padé is ~30 % of the baseline and is currently 24 759 scalar calls per run on
case2869. Two sub-steps:

### 3.1 Batched Padé (same values, one LAPACK call)
- The denominator matrix in `Pade()` is Hankel (`mat_c[r, c] = serie[r+1+c]`): build
  it for *all* buses at once with `numpy.lib.stride_tricks.sliding_window_view` on the
  `(N, n)` coefficient array → `(N, L, L)` stack, one call to `np.linalg.solve`
  (LAPACK handles stacked systems), numerator via vectorized convolution
  (`einsum`). Replaces N per-bus calls per check with ~3 array ops.
- Optional micro-variant if L grows large: `scipy.linalg.solve_toeplitz` (Levinson,
  O(L²)) per bus — only if profiling ever shows the batched O(L³) solve dominating
  (unlikely at L ≈ 15).

### 3.2 Residual-based convergence criterion
Replace "compare two consecutive Padé evaluations per bus" with a physical check:
after each batched Padé evaluation, compute the actual power mismatch
`ΔS = S_spec − V ∘ conj(Y_full · V)` (one sparse mat-vec, O(nnz)) and stop when
`‖ΔS‖∞ ≤ mismatch`. Benefits:
- one cheap global check instead of N approximant comparisons;
- it is the *correct* criterion — approximant stagnation can occur without physical
  convergence (and vice versa);
- it makes the tolerance directly comparable with pandapower/MATPOWER (`tol` on power
  mismatch), which matters for the library API in Phase 4.
Keep the old criterion available behind a flag for one release for comparability.

**Deliberately deferred (research options, Phase 6):** Wynn epsilon / Viskovatov as
incremental evaluators — they compute the identical diagonal-Padé value and only pay
off if checks are much more frequent than after 3.2; Levin/Weniger/Shafer transforms —
possible coefficient savings but give up Stahl's convergence guarantee.

**Acceptance:** voltages match Phase 2 results within `mismatch`; total case2869
runtime ≤ ~0.3 s; convergence decisions consistent across the test matrix.

---

## Phase 4 — Library API + packaging for pandapower (~2–4 days, no speedup goal)

Turns the optimized solver into the dependency pandapower will actually call.

**Tasks**
1. **Array-based entry point** decoupled from xlsx/pandas:
   `solve_helm(Ybus, Sbus, V0_mag, bus_types, Qmin, Qmax, slack_weights=None, ...) -> (V, converged, meta)`
   taking the same per-unit arrays a ppc/pandapower net already has. The existing
   xlsx loader becomes one optional frontend (`pandas` → optional dependency,
   `openpyxl` only for the loader extra).
2. Return a results object (V complex, coefficient count, restarts, Q at generators,
   convergence flag + reason) instead of printing; route prints through `logging`.
3. Remove `warnings.filterwarnings("ignore")` and the global pandas display options
   at import time (library hygiene).
4. Packaging: `pyproject.toml`, wheels, semver, CI running the Phase 0 test suite +
   benchmark tracking; README section "using HELMpy from pandapower" with a minimal
   ppc→arrays example.
5. Keep AGPL headers; document the library-call usage (AGPL is unaffected by
   *using* the package as a dependency; pandapower itself stays BSD).

**Acceptance:** `pip install helmpy && solve_helm(...)` works from a bare env with
only numpy+scipy; a pandapower net can be solved end-to-end via a ~20-line adapter.

---

## Phase 5 — Hybrid mode: HELM start + Newton polish (~1–2 days, opt-in)

For the pandapower fallback use case (`LoadflowNotConverged`), the fastest robust
combination is: HELM to loose tolerance → NR to tight tolerance.

**Tasks**
- Add `mode="hybrid"`: run HELM to `1e-3` (case2869: 15 coefficients, Q-limit
  switching already settled), return V as start vector; the caller (pandapower)
  runs its own numba-NR from that point (typically 1–3 iterations to 1e-8).
- Alternatively/additionally polish internally with HELMpy's own `nr.py` so the
  library is self-contained.
- Fallback ladder documented: NR flat-start → NR from HELM start → full-accuracy HELM.

**Acceptance:** on the test grids, hybrid result equals full-HELM result within
tolerance; wall clock ≤ half of full-accuracy HELM.

---

## Phase 6 — Algorithmic extensions for hard cases (research, as needed)

Only relevant when the motivating slow cases are *heavily loaded* grids (coefficient
blow-up near the nose point) rather than merely large ones. Decide after collecting a
coefficient-count histogram from real pandapower cases via the Phase 4 API.

1. **Multistage / restarted embedding:** evaluate the series at `s = σ < 1` where it
   converges fast, re-embed around that partial solution, repeat until `s = 1`.
   Bounds the per-stage coefficient count; the main defense against near-collapse
   slowdown. Moderate math effort (re-derivation of the recurrence around a non-flat
   base point), high value for stressed grids.
2. **Embedding comparison:** measure PV model 1 vs 2 coefficient counts on stressed
   cases; consult Rao/Feng/Tylavsky 2016 for alternative embeddings before inventing
   new ones.
3. **Alternative sequence transformations** (Levin-u/t, Weniger δ, Shafer quadratic
   approximants) as an A/B experiment behind the Phase 3 continuation interface —
   accept only with a case study showing fewer coefficients *and* no robustness loss.
4. **Incremental continuation** (vectorized Wynn epsilon extended per new coefficient)
   if Phase 3 profiling shows the batched Padé still matters.

---

## Summary: expected trajectory (case2869pegase, 1e-8)

| after phase | est. wall clock | cumulative speedup | effort |
|---|---|---|---|
| baseline | 6.3 s | 1× | — |
| 0 (groundwork) | 6.3 s | 1× | ~1 d |
| 1 (quick wins) | ~1.2 s | ~5× | 1–2 d |
| 2 (vectorized recurrence) | ~0.4 s | ~15× | 3–5 d |
| 3 (continuation overhaul) | ~0.2 s | ~30× | 2–3 d |
| 4 (library API) | ~0.2 s | — | 2–4 d |
| 5 (hybrid, for fallback use) | ~0.1 s effective | ~60× | 1–2 d |

Estimates for phases 2–3 are extrapolated from the profile shares (recurrences ~60 %,
continuation ~30 %, both reduced to array-op cost); phase 1's 5× is measured. Re-run
the benchmark after each phase and adjust the plan if a step underdelivers.

## Risks / watch items

- **Bit-compatibility during vectorization:** float summation order changes results at
  ~1e-16; the regression gate must use tolerances (`atol≈1e-12`), not exact equality.
- **Padé conditioning:** the Hankel systems are ill-conditioned; batched solves must
  handle singular groups (fall back to lower order for affected buses). Double
  precision caps usable `mismatch` at ~1e-10…1e-12 — document this.
- **PV1 / DSB code paths:** four model combinations share the hot loops; vectorize
  behind a common structure or the matrix of variants becomes unmaintainable. If
  needed, deprecate one PV model after benchmarking (thesis: PV2 generally better
  conditioned).
- **Windows/pp13 quirk:** threaded BLAS crash constraint noted in the pandapower work
  applies to that env, not to this package generally; CI should test Linux + Windows.
