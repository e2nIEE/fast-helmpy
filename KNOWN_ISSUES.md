# Known issues

## 1. DS-M2 slack-row shunt convolution pairs the wrong orders (suspected upstream bug)

**Status:** open — behavior faithfully preserved during the Phase 2 vectorization;
fix deliberately deferred.
**Where:** `helmpy/core/helm.py`, `evaluate_rhs()`, branch `DSB_model_method == 2`,
`n >= 2`, inside `if case.conduc_buses[slack]:` (marked with a "Faithful to the
original slack formula" comment). Original loop-based code: `evaluate_bus_eq_dsb_method2`
in commit `77b20f5` and earlier (`for k in range(1, n-1): VV += V[k]*conj(V[n-k])`).

**The discrepancy.** The order-n equation of a bus with shunt conductance G
contains the term `G * sum_{x+y=n-1} V[x]*conj(V[y])`, i.e. products of
coefficient orders summing to **n−1**. Splitting off the x=0 / y=0 terms
(V[0] = 1) gives `G * (VV_{n-1} + 2*Re(V[n-1]))` with
`VV_{n-1} = sum_{k=1..n-2} V[k]*conj(V[n-1-k])`.

- The **PV-bus rows** (former `evaluate_bus_eq_dsb_generator_pv2`) implement
  exactly this: they use `VVanterior`, the VV of the *previous* order
  (today: `run.VV_prev`).
- The **DS-M2 slack row** (former `evaluate_bus_eq_dsb_method2`) instead
  computes `sum_{k=1..n-2} V[k]*conj(V[n-k])` — products of orders summing to
  **n**, not n−1 — and adds `2*Re(V[n-1])`, which belongs to the n−1 pairing.
  The two disagree from n = 3 onward (at n = 2 both give 0).

There is no obvious derivation in which mixing an order-n product sum with the
order-(n−1) boundary terms is correct, so this looks like an off-by-one in the
original HELMpy implementation rather than a deliberate formulation choice.

**Impact.** Only distributed-slack method 2 (`DSB_model_method=2`) **and** a
slack bus with nonzero shunt *conductance* (`Re(Yshunt[slack]) != 0`,
`conduc_buses[slack]`). None of the four shipped cases has a conductive slack
bus (verified 2026-07-08: `conduc_buses[slack]` is False for case9, case118,
case1354pegase, case2869pegase), so:
- the entire regression suite never executes this branch, and
- the stored reference results can neither confirm nor refute a fix.

If the term is wrong, symptoms would be slower/failed coefficient convergence
or a small power-balance error at the slack bus in affected grids — not a
crash.

**How to fix (later session).**
1. Derive the slack-row equation from the DS-M2 formulation in Ortega/Molina/
   Muñoz/Oliva, "Distributed slack bus model formulation for the holomorphic
   embedding load flow method" (ITEES 2019, DOI 10.1002/2050-7038.12253) and
   check which pairing the paper prescribes. The thesis PDFs in `papers/` do
   not cover the DS extension.
2. Build a test grid whose slack bus has a shunt conductance (add a bus-level
   G in the Buses sheet), solve with `DSB_model=True, DSB_model_method=2`, and
   compare against `DSB_model_method=1` (independent formulation of the same
   physics — method 1 has no such shunt convolution on its slack row) and
   against the NR distributed-slack solver (`helmpy.nr_ds`). Today's residual-
   based convergence check (Phase 3) also flags the error directly: with the
   wrong term the converged solution shows a persistent P residual at the
   slack bus.
3. If confirmed, the fix in `evaluate_rhs` is one line — use
   `run.VV_prev[slack]` (the n−1 pairing, like the PV rows) instead of the
   reconstructed `slack_VV = VV[slack] - Re(V[slack,n-1]*conj(V[slack,1]))` —
   plus a new regression case as in step 2. No stored references change.
