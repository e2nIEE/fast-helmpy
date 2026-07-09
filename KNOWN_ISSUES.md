# Known issues

## 1. DS-M2 slack-row shunt convolution pairs the wrong orders (upstream bug)

**Status: FIXED (2026-07-09).** Confirmed by derivation and numerically, then
corrected in `evaluate_rhs()` — the slack row now uses `run.VV_prev` (orders
summing to n−1), identical to the PV-bus rows. Regression test:
`test/test_api.py::test_ds_m2_conductive_slack_matches_m1`.

**Numerical confirmation (case9 + G=0.5 shunt conductance at the slack,
V_sp=1.05):** before the fix, DS-M2 under the legacy criterion "converged" to
a solution with physical power residual **1.250e-3 = G·(V_sp−1)²** (exactly
the missing term) and deviated from DS-M1 by 2.4e-4 p.u.; under the residual
criterion it correctly never converged. After the fix, DS-M1 and DS-M2 agree
to 3e-16 with residual ≤ 5e-10 under both criteria.

**Sharpened impact analysis.** The slack voltage series is linear in s
(V_s = 1 + (V_sp−1)·s, all higher coefficients zero). Consequently the buggy
convolution `sum_{k=1..n-2} V[k]*conj(V[n-k])` is *identically zero* at the
slack bus, while the correct one contributes exactly `G·(V_sp−1)²` at order
n=3 (and nothing at other orders). So the bug is equivalent to dropping that
single term: it requires `Re(Yshunt[slack]) != 0` **and** `V_sp != 1.0`, and
its magnitude scales with `G·(V_sp−1)²`.

**Where it was:** `helmpy/core/helm.py`, `evaluate_rhs()`, branch
`DSB_model_method == 2`, `n >= 2`, inside `if case.conduc_buses[slack]:`.
Original loop-based code: `evaluate_bus_eq_dsb_method2`
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

**Derivation (why n−1 is correct).** The embedded slack-row equation is the
real power balance `Re{ Σ_{x+y=n} conj(V[x])·I[y] } = [s·P + K·Ploss(s)]_n`
with the shunt current s-scaled: `I[y] = (Ytrans·V)[y] + Yshunt·V[y−1] + …`.
The s-scaling shifts the shunt term one order down, so its convolution runs
over products `conj(V[x])·V[y]` with `x + y = n − 1`. Splitting off the
x=0 / y=0 terms (V[0] = 1) yields `Yshunt.real·(VV_{n−1} + 2·Re(V[n−1]))`
with `VV_{n−1} = Σ_{k=1..n−2} V[k]·conj(V[n−1−k])` — precisely what the
PV-bus rows implement via `VV_prev`. The original slack row paired
`x + y = n`, i.e. it treated the shunt as if it were *not* s-embedded, which
contradicts its own boundary term `2·Re(V[n−1])` (an x+y=n−1 term). This
matches the DS formulation paper's uniform treatment (Ortega/Molina/Muñoz/
Oliva, ITEES 2019, DOI 10.1002/2050-7038.12253); the fix makes slack and PV
rows consistent. Stored reference results are unaffected (no shipped case
has a conductive slack bus).

## 2. Crash (or silent truncation) past 40 coefficients (upstream bug)

**Status: FIXED (2026-07-09).** `RunVariables.expand_coef_arrays()` replaces
the coefficient arrays with larger ones when order 40 is reached, but
`computing_voltages_mismatch` had already bound local aliases
(`Soluc_eval`, `coefficients`, `V_complex`, `Vre_PV`) to the *old* 40-column
arrays — any run needing more than 40 coefficients crashed with an
IndexError (this pattern predates the vectorization; upstream HELMpy has the
same latent bug). Never triggered before because every shipped case
converges well below 40 coefficients; found while reproducing issue 1, whose
residual-criterion run correctly refuses to converge and runs long. Fixed by
re-binding the locals after expansion. Regression test:
`test/test_api.py::test_more_than_40_coefficients` (case9 at 2.4× loading,
~95 coefficients). A related edge in `solve_helm` post-processing (diagonal
Padé needs an odd series length; diverged runs stop at an even one) is fixed
alongside and covered by `test_divergence_reported_beyond_collapse`.
