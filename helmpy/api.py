"""
Array-based library entry point for the HELM solver.

HELMpy, open source package of power flow solvers developed on Python 3
Copyright (C) 2019 Tulio Molina tuliojose8@gmail.com and Juan José Ortega juanjoseop10@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, diags

from helmpy.core.classes import CaseData
from helmpy.core.helm import helm, power_residual
from helmpy.core.analytic_continuation import Pade

# ppc/pypower-style bus type codes
PQ = 1
PV = 2
SLACK = 3


@dataclass
class HelmResults:
    """Result of a solve_helm() call. All quantities are per unit.

    converged            solution found within max_coefficients
    V                    complex bus voltages (rotated by slack_angle_degrees)
    n_coefficients       series length of the final run
    coefficients_per_run series lengths of every run (Q-limit switches restart)
    residual             final infinity-norm power/voltage defect (p.u.)
    S_injection          complex power injection V * conj(Ybus @ V) at every bus
    switched_buses       PV buses converted to PQ by Q-limit enforcement
    """
    converged: bool
    V: np.ndarray
    n_coefficients: int
    coefficients_per_run: list
    residual: float
    S_injection: np.ndarray
    switched_buses: np.ndarray


def create_case_from_arrays(Ybus, Sbus, bus_types, V_specified,
                            Qmin=None, Qmax=None, name='from_arrays'):
    """Build a CaseData object from per-unit arrays (no xlsx, no pandas).

    The full bus admittance matrix is split internally into a zero-row-sum
    transfer matrix (the HELM embedding requires the no-load solution V=1)
    and per-bus shunt admittances Yshunt_i = sum_j Ybus[i, j]. Asymmetric
    entries from phase-shifting transformers stay in the transfer matrix;
    this is an equally valid embedding with the same solution at s=1.

    :param Ybus: (N, N) complex bus admittance matrix, dense or scipy sparse
    :param Sbus: (N,) complex specified net power injection per bus.
        Real part: net active injection (generation minus load), used at PQ
        and PV buses. Imaginary part: net reactive injection at PQ buses; at
        PV buses it is treated as the (negative) reactive demand offset that
        relates the bus's reactive injection to the generator limits.
    :param bus_types: (N,) ints, ppc convention: 1 = PQ, 2 = PV, 3 = slack
        (exactly one slack)
    :param V_specified: (N,) voltage magnitude setpoints; used at PV and
        slack buses only
    :param Qmin, Qmax: (N,) generator reactive limits (Qg, i.e. injection
        plus reactive demand) per bus; None disables limit enforcement for
        that side
    :return: CaseData ready for helm()
    """
    Ybus = csr_matrix(Ybus, dtype=np.complex128)
    if Ybus.shape[0] != Ybus.shape[1]:
        raise ValueError("Ybus must be square.")
    N = Ybus.shape[0]

    bus_types = np.asarray(bus_types, dtype=np.int64)
    Sbus = np.asarray(Sbus, dtype=np.complex128)
    V_specified = np.asarray(V_specified, dtype=np.float64)
    for arr, label in ((bus_types, 'bus_types'), (Sbus, 'Sbus'),
                       (V_specified, 'V_specified')):
        if arr.shape != (N,):
            raise ValueError(f"{label} must have shape ({N},).")

    slack_buses = np.flatnonzero(bus_types == SLACK)
    if slack_buses.size != 1:
        raise ValueError("bus_types must contain exactly one slack bus (3).")
    slack = int(slack_buses[0])
    pv_buses = np.flatnonzero(bus_types == PV)
    unknown = ~np.isin(bus_types, (PQ, PV, SLACK))
    if unknown.any():
        raise ValueError("bus_types entries must be 1 (PQ), 2 (PV) or 3 (slack).")

    # Split Ybus: Yshunt = row sums, transfer matrix rows sum to zero
    Yshunt = np.asarray(Ybus.sum(axis=1)).ravel()
    Ytrans_csr = (Ybus - diags(Yshunt, format='csr', dtype=np.complex128)).tocsr()

    case = CaseData(name, N, pv_buses.size + 1, dense_matrices=False)
    case.slack = slack
    case.slack_bus = slack
    case.list_gen = pv_buses.astype(np.int64)
    case.Buses_type[:] = 'PQ'
    case.Buses_type[pv_buses] = 'PVLIM'
    case.Buses_type[slack] = 'Slack'

    case.V = np.ones(N, dtype=np.float64)
    case.V[pv_buses] = V_specified[pv_buses]
    case.V[slack] = V_specified[slack]

    # Net injections: Pg carries the full net active injection (Pd = 0);
    # the reactive part enters as demand so that run.Qg starts at zero.
    case.Pg = Sbus.real.copy()
    case.Pd = np.zeros(N, dtype=np.float64)
    case.Qd = -Sbus.imag.copy()

    case.Qgmax = np.full(N, np.inf) if Qmax is None \
        else np.asarray(Qmax, dtype=np.float64).copy()
    case.Qgmin = np.full(N, -np.inf) if Qmin is None \
        else np.asarray(Qmin, dtype=np.float64).copy()

    case.Yshunt = Yshunt
    case.Shunt = np.zeros(N, dtype=np.complex128)
    case.conduc_buses = Yshunt.real != 0
    case.phase_barras = np.full(N, False)
    case.phase_dict = dict()
    case.branches_buses = None  # only needed by the xlsx reporting path
    case.Ybr_list = []
    case.N_branches = 0

    # Precomputed sparse matrices for the vectorized recurrence
    case.Ytrans_csr = Ytrans_csr
    case.Yphase_csr = None
    return case


def solve_helm(Ybus, Sbus, bus_types, V_specified, Qmin=None, Qmax=None,
               slack_weights=None, distributed_slack=False,
               mismatch=1e-8, max_coefficients=100, enforce_q_limits=True,
               pv_bus_model=2, dsb_model_method=None, convergence='residual',
               slack_angle_degrees=0.0, name='from_arrays') -> HelmResults:
    """Solve a power flow with HELM from per-unit arrays.

    See create_case_from_arrays() for the array conventions. Additional
    parameters mirror helm(): distributed_slack activates the distributed
    slack bus model (participation factors from slack_weights when given,
    otherwise proportional to positive net injections), enforce_q_limits
    switches PV buses to PQ at their limits, convergence selects the
    'residual' (physical power mismatch) or legacy 'pade' criterion.

    The slack bus is solved at angle 0 and the returned voltages are rotated
    by slack_angle_degrees afterwards (exact: a global rotation leaves all
    power flows unchanged).
    """
    case = create_case_from_arrays(Ybus, Sbus, bus_types, V_specified,
                                   Qmin=Qmin, Qmax=Qmax, name=name)
    if enforce_q_limits and Qmin is None and Qmax is None:
        enforce_q_limits = False  # nothing to enforce

    run, n_coefficients, diverged = helm(
        case,
        mismatch=float(mismatch),
        max_coefficients=int(max_coefficients),
        enforce_Q_limits=bool(enforce_q_limits),
        pv_bus_model=pv_bus_model,
        DSB_model=bool(distributed_slack),
        DSB_model_method=dsb_model_method,
        K_factors=slack_weights,
        convergence=convergence,
    )

    V = run.V_complex_profile.copy()
    S_injection = V * np.conj(run.Ytrans_csr @ V + case.Yshunt * V)

    # Final defect of the returned solution (same measure as the 'residual'
    # convergence criterion)
    effective_method = dsb_model_method if dsb_model_method is not None \
        else (2 if distributed_slack else None)
    Pi = run.Pg - case.Pd
    Si = Pi + run.Qg * 1j - case.Qd * 1j
    Ploss = Pade(run.coefficients[2 * case.N], n_coefficients) \
        if effective_method is not None else 0.0
    residual = float(power_residual(V, Ploss, Si, Pi, effective_method,
                                    case, run))

    if slack_angle_degrees != 0.0:
        V = V * np.exp(1j * np.deg2rad(slack_angle_degrees))

    return HelmResults(
        converged=not diverged,
        V=V,
        n_coefficients=int(n_coefficients),
        coefficients_per_run=list(run.list_coef),
        residual=residual,
        S_injection=S_injection,
        switched_buses=np.array(sorted(run.list_gen_remove), dtype=np.int64),
    )
