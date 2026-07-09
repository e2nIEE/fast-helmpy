"""
HELMpy, open source package of power flow solvers developed on Python 3 
Copyright (C) 2019 Tulio Molina tuliojose8@gmail.com and Juan José Ortega juanjoseop10@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import factorized

from helmpy.core.classes import RunVariables, CaseData
from helmpy.core.analytic_continuation import Pade, pade_batched

logger = logging.getLogger(__name__)

def modif_Ytrans(DSB_model_method, pv_bus_model, case, run):
    """Create the modified Y matrix (CSC, pre-factorized into run.solve) and,
    for pv_bus_model 1, the extracted PV-bus columns (run.Y_Vsp_cols).

    Assembled as COO triplets derived array-wise from the sparse Ytrans
    pattern — no per-element Python assignments, O(nnz) memory.
    """
    # Assign local variables for faster access
    N = case.N
    slack = case.slack
    Buses_type = run.Buses_type
    K = run.K
    length = run.length

    coo = run.Ytrans_csr.tocoo()
    r = coo.row.astype(np.int64)
    c = coo.col.astype(np.int64)
    v = coo.data

    pv_bus = (Buses_type == 'PV') | (Buses_type == 'PVLIM')
    pq_bus = Buses_type == 'PQ'

    # Rows expanded to the full 2x2 real block (real and imaginary equation):
    # PQ buses always; with PV model 1 also PV buses. PV model 2 keeps only
    # the real-power row 2i for PV buses.
    if pv_bus_model == 1:
        full_mask = pq_bus[r] | pv_bus[r]
    else:
        full_mask = pq_bus[r]
    r4 = r[full_mask]
    c4 = c[full_mask]
    v4 = v[full_mask]
    rows = [2*r4, 2*r4, 2*r4 + 1, 2*r4 + 1]
    cols = [2*c4, 2*c4 + 1, 2*c4, 2*c4 + 1]
    vals = [v4.real, -v4.imag, v4.imag, v4.real]
    if pv_bus_model == 2:
        pv2_mask = pv_bus[r]
        r2 = r[pv2_mask]
        c2 = c[pv2_mask]
        v2 = v[pv2_mask]
        rows += [2*r2, 2*r2]
        cols += [2*c2, 2*c2 + 1]
        vals += [v2.real, -v2.imag]

    if DSB_model_method is not None:
        # Last row: real power balance of the slack bus
        slack_mask = r == slack
        cs = c[slack_mask]
        vs = v[slack_mask]
        rows += [np.full(cs.size, 2*N, dtype=np.int64)] * 2
        cols += [2*cs, 2*cs + 1]
        vals += [vs.real, -vs.imag]

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    if pv_bus_model == 1:
        # Extract the real-voltage columns of the PV buses: those entries move
        # to Y_Vsp_cols (column g holds the matrix column of PV bus
        # run.list_gen[g]), making the per-coefficient right-hand-side
        # correction a single sparse mat-vec.
        col_bus = cols >> 1
        extract = ((cols & 1) == 0) & pv_bus[col_bus]
        gpos = np.empty(N, dtype=np.int64)
        gpos[run.list_gen] = np.arange(len(run.list_gen))
        run.Y_Vsp_cols = csc_matrix(
            (vals[extract], (rows[extract], gpos[col_bus[extract]])),
            shape=(length, len(run.list_gen)), dtype=np.float64,
        )
        keep = ~extract
        rows = rows[keep]
        cols = cols[keep]
        vals = vals[keep]

    # Definition rows and the distributed-slack column:
    # slack identity rows; PV rows 2i+1 (magnitude equation for PV model 2,
    # real-voltage definition for PV model 1); loss-participation column.
    extra_rows = [np.array([2*slack, 2*slack + 1], dtype=np.int64)]
    extra_cols = [np.array([2*slack, 2*slack + 1], dtype=np.int64)]
    extra_vals = [np.ones(2)]
    pv_buses = np.flatnonzero(pv_bus)
    if pv_buses.size:
        extra_rows.append(2*pv_buses + 1)
        extra_cols.append(2*pv_buses)
        extra_vals.append(np.ones(pv_buses.size))
    if DSB_model_method is not None:
        list_gen = np.asarray(run.list_gen, dtype=np.int64)
        extra_rows.append(np.concatenate((2*list_gen, [2*N])))
        extra_cols.append(np.full(list_gen.size + 1, 2*N, dtype=np.int64))
        extra_vals.append(np.concatenate((-K[list_gen], [-K[slack]])))

    rows = np.concatenate([rows] + extra_rows)
    cols = np.concatenate([cols] + extra_cols)
    vals = np.concatenate([vals] + extra_vals)

    Ytrans_mod = csc_matrix((vals, (rows, cols)), shape=(length, length),
                            dtype=np.float64)

    # Return a function for solving a sparse linear system, with Ytrans_mod pre-factorized.
    run.solve = factorized(Ytrans_mod)

def Unknowns_soluc(DSB_model_method, pv_bus_model, N, run):
    """Initialize the order-0 coefficients and the bus-group index arrays used
    by the vectorized right-hand-side evaluation."""
    # Assign local variables for faster access
    coefficients = run.coefficients
    Buses_type = run.Buses_type

    # Bus groups. pv_idx is sorted and therefore identical to run.list_gen.
    run.pq_idx = np.flatnonzero(Buses_type == 'PQ')
    run.pv_idx = np.flatnonzero((Buses_type == 'PV') | (Buses_type == 'PVLIM'))

    # Assign 0 to the first coefficients and evaluated solutions.
    coefficients[:,0].fill(0)
    run.Soluc_eval[:,0].fill(0)
    # Order-0 solution: V = 1 at every bus. In PV model 1 the real part of the
    # PV bus voltages lives in Vre_PV instead of the coefficients vector.
    coefficients[0:2*N:2, 0] = 1
    if pv_bus_model == 1:
        coefficients[2*run.pv_idx, 0] = 0

def build_case_sparse_matrices(case, run):
    """CSR matrices used by the vectorized recurrence.

    Ytrans_csr holds the transfer admittances (row i restricted to
    branches_buses[i], exactly the terms the former per-bus loops summed).
    Yphase_csr holds the phase-shifter admittance corrections from
    case.phase_dict, or None when the case has no phase-shifting branches.

    Cases built from arrays (helmpy.api) carry precomputed sparse matrices;
    those are used directly.
    """
    if case.Ytrans_csr is not None:
        run.Ytrans_csr = case.Ytrans_csr
        run.Yphase_csr = case.Yphase_csr
        return

    N = case.N
    branches_buses = case.branches_buses
    Ytrans = case.Ytrans

    counts = [len(branches_buses[i]) for i in range(N)]
    indptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.fromiter(
        (j for i in range(N) for j in branches_buses[i]), np.int64, nnz)
    data = np.fromiter(
        (Ytrans[i][j] for i in range(N) for j in branches_buses[i]),
        np.complex128, nnz)
    run.Ytrans_csr = csr_matrix((data, indices, indptr), shape=(N, N))

    if case.phase_barras.any():
        rows = []
        cols = []
        vals = []
        for i in np.flatnonzero(case.phase_barras):
            for k in range(len(case.phase_dict[i][0])):
                rows.append(i)
                cols.append(case.phase_dict[i][0][k])
                vals.append(case.phase_dict[i][1][k])
        run.Yphase_csr = csr_matrix(
            (vals, (rows, cols)), shape=(N, N), dtype=np.complex128)
    else:
        run.Yphase_csr = None

#---------------------------------------------------------------------------------------
def evaluate_rhs(n, Si, Pi, pv_bus_model, DSB_model_method, case, run):
    """Vectorized evaluation of the order-n right hand side (Soluc_eval[:, n]).

    Replaces the former per-bus evaluate_bus_eq_* functions: the neighbor sums
    are sparse mat-vecs over all buses at once and the coefficient
    convolutions are einsum reductions along the stored series history.
    For pv_bus_model 1 this also fills Vre_PV[:, n].
    """
    N = case.N
    slack = case.slack
    V = run.V_complex
    W = run.W
    YV = run.YV
    F = run.F
    Soluc_eval = run.Soluc_eval
    Yshunt = case.Yshunt
    pq = run.pq_idx
    pv = run.pv_idx

    Vn1 = V[:, n-1]
    # Neighbor sums of the newest coefficient, for every bus at once:
    # YV[:, m] = sum_k Ytrans[i,k] * V[k, m]
    YV[:, n-1] = run.Ytrans_csr @ Vn1
    Fn1 = None
    if F is not None:
        F[:, n-1] = run.Yphase_csr @ Vn1
        Fn1 = F[:, n-1]

    Soluc_eval[:, n] = 0

    # Convolutions along the coefficient history (orders 1 .. n-1), all buses
    S1 = VV = Fconv = None
    if n >= 2:
        conjV_rev = np.conj(V[:, n-1:0:-1])
        # VV_n = sum_{k=1..n-1} V[k]*conj(V[n-k]); real by conjugate symmetry
        VV = np.einsum('ij,ij->i', V[:, 1:n], conjV_rev).real
        if pv_bus_model == 2 or DSB_model_method == 2:
            # S1 = sum_{x=1..n-1} conj(V[n-x]) * YV[x]
            S1 = np.einsum('ij,ij->i', conjV_rev, YV[:, 1:n])
            if F is not None:
                # Fconv = sum_{x=0..n-1} conj(V[x]) * F[n-1-x]
                Fconv = np.einsum('ij,ij->i', np.conj(V[:, :n]), F[:, n-1::-1])

    # PQ buses: conj(S)*conj(W[n-1]) - Yshunt*V[n-1] - phase corrections
    if pq.size:
        result = np.conj(Si[pq]) * np.conj(W[pq, n-1]) - Yshunt[pq] * Vn1[pq]
        if Fn1 is not None:
            result = result - Fn1[pq]
        Soluc_eval[2*pq, n] = result.real
        Soluc_eval[2*pq + 1, n] = result.imag

    # PV buses
    if pv.size:
        if pv_bus_model == 2:
            # Row 2i: real power balance; row 2i+1: voltage magnitude equation
            if n == 1:
                CC = Pi[pv] - Yshunt[pv].real
                if Fn1 is not None:
                    CC = CC - F[pv, 0].real
                mag_rhs = (case.V[pv]**2 - 1) / 2
            else:
                CC = -S1[pv].real \
                    - Yshunt[pv].real * (run.VV_prev[pv] + 2*Vn1[pv].real)
                if Fconv is not None:
                    CC = CC - Fconv[pv].real
                mag_rhs = -VV[pv] / 2
            Soluc_eval[2*pv, n] = CC
            Soluc_eval[2*pv + 1, n] = mag_rhs
        else:  # pv_bus_model == 1
            if n == 1:
                run.Vre_PV[pv, 1] = (case.V[pv]**2 - 1) / 2
                result = Pi[pv] * np.conj(W[pv, 0]) - Yshunt[pv] * Vn1[pv]
            else:
                run.Vre_PV[pv, n] = -VV[pv] / 2
                conjW_rev = np.conj(W[pv, n-1:0:-1])
                # aux = sum_{k=1..n-1} coeff_re[k] * conj(W[n-k])
                aux = np.einsum('ij,ij->i', run.coefficients[2*pv, 1:n], conjW_rev)
                result = Pi[pv] * np.conj(W[pv, n-1]) - Yshunt[pv] * Vn1[pv] \
                    - 1j*aux
                if DSB_model_method is not None:
                    # aux_Ploss = sum_{k=1..n-1} Ploss[k] * conj(W[n-k])
                    aux_Ploss = conjW_rev @ run.coefficients[2*N, 1:n]
                    result = result + run.K[pv] * aux_Ploss
            if Fn1 is not None:
                result = result - Fn1[pv]
            Soluc_eval[2*pv, n] = result.real
            Soluc_eval[2*pv + 1, n] = result.imag

    # Slack bus rows: V = V_specified at order 1, zero afterwards
    if n == 1:
        Soluc_eval[2*slack, 1] = case.V[slack] - 1
        Soluc_eval[2*slack + 1, 1] = 0

    # Distributed-slack row (2N): real power balance of the slack bus
    if DSB_model_method == 1:
        if n == 1:
            aux_Ploss = 0
        else:
            aux_Ploss = np.dot(run.coefficients[2*N, 1:n],
                               np.conj(W[slack, n-1:0:-1]))
        result = Pi[slack] * np.conj(W[slack, n-1]) \
            - Yshunt[slack] * Vn1[slack] + run.K[slack] * aux_Ploss
        if Fn1 is not None:
            result = result - Fn1[slack]
        Soluc_eval[2*N, n] = result.real
    elif DSB_model_method == 2:
        if n == 1:
            CC = Pi[slack] - Yshunt[slack].real
            if Fn1 is not None:
                CC = CC - F[slack, 0].real
        else:
            CC = -S1[slack].real
            if case.conduc_buses[slack]:
                # Shunt convolution pairs orders summing to n-1, like the
                # PV-bus rows. The original HELMpy paired orders summing to n
                # here (off-by-one); see KNOWN_ISSUES.md issue 1 for the
                # derivation and the conductive-slack regression test.
                CC = CC - Yshunt[slack].real * (run.VV_prev[slack] + 2*Vn1[slack].real)
            if Fconv is not None:
                CC = CC - Fconv[slack].real
        Soluc_eval[2*N, n] = CC

    # History term for the next order's shunt convolution
    if VV is not None:
        run.VV_prev[:] = VV

#---------------------------------------------------------------------------------------
def compute_complex_voltages(n, pv_bus_model, case, run):
    """Complex voltages of coefficient n from the solved real/imag pairs."""
    # Assign local variables for faster access
    coefficients = run.coefficients
    N = case.N

    run.V_complex[:, n] = coefficients[0:2*N:2, n] + 1j*coefficients[1:2*N:2, n]
    if pv_bus_model == 1 and run.pv_idx.size:
        pv = run.pv_idx
        run.V_complex[pv, n] = run.Vre_PV[pv, n] + 1j*coefficients[2*pv + 1, n]

def calculate_inverse_voltages_w_array(n, case, run):
    """W computing - Inverse voltages "W" array.

    W[:, n] = -sum_{k=0..n-1} W[:, k] * V[:, n-k] for every bus at once.
    """
    run.W[:, n] = -np.einsum(
        'ij,ij->i', run.W[:, :n], run.V_complex[:, n:0:-1])

def P_iny(i, case, run):
    """Computing P injection at bus i. Must be used after Voltages_profile()"""
    # Assign local variables for faster access
    Yre = case.Yre
    Yimag = case.Yimag
    branches_buses = case.branches_buses
    Vre = run.Vre
    Vimag = run.Vimag

    Piny = 0
    for k in branches_buses[i]:
        Piny += Vre[i] * (Yre[i][k]*Vre[k] - Yimag[i][k]*Vimag[k]) \
                + Vimag[i] * (Yre[i][k]*Vimag[k] + Yimag[i][k]*Vre[k])
    return Piny

def Q_iny(i, case, run):
    """Computing Q injection at bus i. Must be used after Voltages_profile()"""
    # Assign local variables for faster access
    Yre = case.Yre
    Yimag = case.Yimag
    branches_buses = case.branches_buses
    Vre = run.Vre
    Vimag = run.Vimag

    Qiny = 0
    for k in branches_buses[i]:
        Qiny += Vimag[i] * (Yre[i][k]*Vre[k] - Yimag[i][k]*Vimag[k]) \
                - Vre[i] * (Yre[i][k]*Vimag[k] + Yimag[i][k]*Vre[k])
    return Qiny

def check_PVLIM_violation(detailed_run_print, case, run):
    """Verification of Qgen limits for PVLIM buses"""
    # Assign local variables for faster access
    Qd = case.Qd
    Qgmax = case.Qgmax
    Qgmin = case.Qgmin
    Qg = run.Qg
    list_gen = run.list_gen
    list_gen_remove = run.list_gen_remove
    Buses_type = run.Buses_type

    # Reactive injection at every bus from the converged voltages: one sparse
    # mat-vec (equivalent to the per-bus Q_iny sums, but without needing the
    # dense Y matrix, which array-built cases do not carry)
    V = run.V_complex_profile
    Ybus_V = run.Ytrans_csr @ V + case.Yshunt * V
    if run.Yphase_csr is not None:
        Ybus_V += run.Yphase_csr @ V
    Q_injection = (V * np.conj(Ybus_V)).imag

    flag_violacion = False
    for i in list_gen:
        Qg_incog = Q_injection[i] + Qd[i]
        Qg[i] = Qg_incog
        if Qg_incog > Qgmax[i] or Qg_incog < Qgmin[i]:
            flag_violacion = True
            Buses_type[i] = 'PQ'
            list_gen_remove.append(i)
            Qg[i] = Qgmax[i] if Qg_incog > Qgmax[i] else Qgmin[i]
            if detailed_run_print:
                logger.info(
                    'Bus %d exceeded its Qgen limit with %f. The exceeded '
                    'limit %f will be assigned to the bus', i+1, Qg_incog, Qg[i])
    return flag_violacion

def compute_k_factor(case, run):
    """Computing of the K factor for each PV bus and the slack bus.
    
    Only the PV buses are considered to calculate Pgen_total. The PV buses 
    that were converted to PQ buses are NOT considered.
    """
    # Assign local variables for faster access
    K = run.K
    Pg = run.Pg

    K.fill(0)

    # If external participation factors were supplied (e.g. pandapower's slack_weight),
    # use them directly, normalized over the buses with a non-zero weight.
    external_K = getattr(run, "external_K", None)
    if external_K is not None:
        weights = np.asarray(external_K, dtype=np.float64)
        total = weights.sum()
        if total > 0:
            K[:] = weights / total
            return

    Pgen_total = 0
    Distrib = []
    # Active power that the slack must generate to compensate the system
    Pg[case.slack] = run.Pg_imbalance
    for i in run.list_gen:
        if Pg[i] > 0:
            Pgen_total += Pg[i]
            Distrib.append(i)
    if Pg[case.slack] > 0:
        Pgen_total += Pg[case.slack]
        Distrib.append(case.slack)
    for i in Distrib:
        K[i] = Pg[i]/Pgen_total

def K_slack_1(case, run):
    """Set the slack's participation factor to 1 and the rest to 0. 
    
    Classic slack bus model.
    """
    run.K.fill(0)
    run.K[case.slack] = 1


def power_residual(V, Ploss, Si, Pi, DSB_model_method, case, run):
    """Infinity norm of the solution defect of the continued voltages V:
    power mismatch at PQ buses (P and Q), P mismatch and voltage-magnitude
    defect at PV buses, and the slack voltage defect. With a distributed
    slack, the specified P includes each bus's loss participation K*Ploss and
    the slack bus's own P balance is checked as well.

    One sparse mat-vec, O(nnz). This is the physical convergence criterion,
    directly comparable to a Newton-Raphson power-mismatch tolerance — unlike
    the legacy criterion (stagnation of consecutive Padé evaluations), which
    can stall without the power flow equations actually being satisfied.
    """
    slack = case.slack
    Ybus_V = run.Ytrans_csr @ V + case.Yshunt * V
    if run.Yphase_csr is not None:
        Ybus_V += run.Yphase_csr @ V
    Scalc = V * np.conj(Ybus_V)

    if DSB_model_method is not None:
        Pspec = Pi + run.K * Ploss
    else:
        Pspec = Pi

    residual = abs(V[slack] - case.V[slack])
    pq = run.pq_idx
    pv = run.pv_idx
    if pq.size:
        residual = max(residual,
                       np.abs(Pspec[pq] - Scalc[pq].real).max(),
                       np.abs(Si[pq].imag - Scalc[pq].imag).max())
    if pv.size:
        residual = max(residual,
                       np.abs(Pspec[pv] - Scalc[pv].real).max(),
                       np.abs(np.abs(V[pv]) - case.V[pv]).max())
    if DSB_model_method is not None:
        residual = max(residual, abs(Pspec[slack] - Scalc[slack].real))
    return residual


def computing_voltages_mismatch(
    detailed_run_print, mismatch, max_coef, enforce_Q_limits,
    pv_bus_model, DSB_model_method, case, run, start_coef=0,
    convergence='residual'
):
    """Loop of coefficients computing until the mismatch is reached.

    start_coef > 0 resumes a converged series: coefficients 0..start_coef are
    still valid in the run arrays (only allowed when bus types and therefore
    the matrix did not change) and computation continues from there. Used for
    the final full-accuracy pass after the coarse Q-limit phase.

    convergence='residual' stops when the power residual of the analytically
    continued voltages is below mismatch; 'pade' restores the legacy
    criterion (change of consecutive Padé evaluations below mismatch).
    """
    # Assign local variables for faster access
    N = case.N
    list_coef = run.list_coef
    solve = run.solve
    V_complex_profile = run.V_complex_profile
    Soluc_eval = run.Soluc_eval
    coefficients = run.coefficients
    Vre_PV = run.Vre_PV
    V_complex = run.V_complex
    
    # Variables initialization
    coef_actual = start_coef
    series_large = start_coef + 1
    # Convergence-check schedule: first check at 5 coefficients, then every 2;
    # after two failed checks every 4. Each check Pade-evaluates many buses,
    # which costs far more than computing two additional coefficients.
    # When resuming, the first check happens at the next odd series length
    # (Pade uses diagonal approximants, which need an odd number of terms).
    if start_coef == 0:
        next_check = 5
    else:
        next_check = series_large + (2 if series_large % 2 == 1 else 1)
    checks_failed = 0
    if start_coef == 0:
        run.W[:,0] = 1 # Assign 1 to the inverse voltages of coefficients 0

    # Buses whose coefficient tail decayed far below the tolerance are summed
    # directly inside pade_batched (their series already converged at s=1)
    tail_tol = 1e-3 * mismatch

    # Compute Vre_PV and V_complex for coefficient 0
    if start_coef == 0:
        if pv_bus_model == 1:
            Vre_PV[:,0] = 1
        compute_complex_voltages(0, pv_bus_model, case, run)
        run.VV_prev[:] = 0

    # Compute active and complex power injection
    Pi = run.Pg - case.Pd
    Si = Pi + run.Qg*1j - case.Qd*1j

    # Flags
    first_check = True
    flag_recalculate = False
    flag_divergence = False

    while True:
        coef_actual += 1
        if coef_actual == 40:
            # Expand the coefficient arrays to the maximum. They were originally set to 40
            run.expand_coef_arrays()
            # Re-bind the local aliases: expansion replaces the arrays on run,
            # and the stale 40-column references would otherwise crash (or
            # silently truncate) any case needing more than 40 coefficients.
            Soluc_eval = run.Soluc_eval
            coefficients = run.coefficients
            V_complex = run.V_complex
            Vre_PV = run.Vre_PV
        if detailed_run_print:
            logger.debug("Computing coefficient: %d", coef_actual)

        # Compute the right hand side of the matrix equation (fills
        # Soluc_eval[:, n] for all buses; also Vre_PV[:, n] for pv_bus_model 1)
        evaluate_rhs(coef_actual, Si, Pi, pv_bus_model, DSB_model_method, case, run)

        # Determine right_hand_side of matrix equation
        if pv_bus_model == 1:
            # Subtract the extracted PV-bus matrix columns times the PV real
            # voltage coefficients (single sparse mat-vec)
            right_hand_side = Soluc_eval[:,coef_actual] \
                - run.Y_Vsp_cols @ Vre_PV[run.list_gen, coef_actual]
        else: # pv_bus_model == 2:
            right_hand_side = Soluc_eval[:,coef_actual]

        # New column of coefficients
        coefficients[:,coef_actual] = solve(right_hand_side)

        # Compute V_complex and inverse voltages for current coefficient 
        compute_complex_voltages(coef_actual, pv_bus_model, case, run)
        calculate_inverse_voltages_w_array(coef_actual, case, run)
        
        # Mismatch check
        flag_mismatch = False
        series_large += 1
        if series_large == next_check:
            if convergence == 'residual':
                # Analytic continuation of every bus, then the physical check
                V_complex_profile[:] = pade_batched(V_complex, series_large, tail_tol)
                Ploss = Pade(coefficients[2*N], series_large) \
                    if DSB_model_method is not None else 0.0
                residual = power_residual(V_complex_profile, Ploss, Si, Pi,
                                          DSB_model_method, case, run)
                flag_mismatch = residual > mismatch
                if detailed_run_print:
                    logger.debug("Power residual at %d coefficients: %.3e",
                                 series_large, residual)
            else:  # convergence == 'pade' (legacy criterion)
                if first_check:
                    first_check = False
                    previous_profile = pade_batched(V_complex, series_large - 2, tail_tol)
                else:
                    previous_profile = V_complex_profile.copy()
                V_complex_profile[:] = pade_batched(V_complex, series_large, tail_tol)
                magn_delta = np.abs(np.abs(previous_profile) - np.abs(V_complex_profile))
                angle_delta = np.abs(np.angle(previous_profile) - np.angle(V_complex_profile))
                flag_mismatch = bool((magn_delta > mismatch).any()
                                     or (angle_delta > mismatch).any())
            if flag_mismatch:
                checks_failed += 1
                next_check = series_large + (2 if checks_failed < 2 else 4)
            if not flag_mismatch:
                # Qgen check or ignore limits
                if enforce_Q_limits:
                    if check_PVLIM_violation(detailed_run_print, case, run):
                        logger.info("At coefficient %d the system is to be resolved due to PVLIM to PQ switches", series_large)
                        list_coef.append(series_large)
                        flag_recalculate = True
                        break
                logger.info('Convergence has been reached. %d coefficients were calculated', series_large)
                list_coef.append(series_large)
                break
        if series_large > max_coef-1:
            logger.warning('Maximum number of coefficients has been reached. The problem has no physical solution')
            flag_divergence = True
            break
    
    return flag_recalculate, flag_divergence, series_large

def convert_complex_to_polar_voltages(complex_voltage, N):
    """Separate each voltage value in magnitude and phase angle (degrees)"""
    polar_voltage = np.empty((N,2), dtype=np.float64)
    polar_voltage[:,0] = np.absolute(complex_voltage)
    polar_voltage[:,1] = np.angle(complex_voltage, deg=True)
    return polar_voltage


def power_balance(enforce_Q_limits, algorithm, case, run):
    """Computation of power flow through branches and power balance"""
    # Save for later: Pi=None, Qi=None, K=None 

    # Assign local variables for faster access
    Ybr_list = case.Ybr_list
    Shunt = case.Shunt
    slack = case.slack
    Pd = case.Pd
    Qd = case.Qd
    K = run.K
    V_complex_profile = run.V_complex_profile
    Pg = run.Pg
    Qg = run.Qg
    list_gen = run.list_gen

    # Define array to power flow through branches data
    Power_branches = np.zeros((case.N_branches,8), dtype=np.float64)

    for branch in range(case.N_branches):

        Bus_from =  Power_branches[branch][0] = int(Ybr_list[branch][0])
        Bus_to = Power_branches[branch][1] = int(Ybr_list[branch][1])
        Ybr = Ybr_list[branch][2]

        V_from = V_complex_profile[Bus_from]
        V_to = V_complex_profile[Bus_to]
        V_vector = np.array([V_from,V_to])
        
        I =  np.matmul(Ybr,V_vector)

        S_ft = V_from * np.conj(I[0]) * 100
        S_tf = V_to * np.conj(I[1]) * 100
        S_branch_elements = S_ft + S_tf

        Power_branches[branch][2] = np.real(S_ft)
        Power_branches[branch][3] = np.imag(S_ft)

        Power_branches[branch][4] = np.real(S_tf)
        Power_branches[branch][5] = np.imag(S_tf)

        Power_branches[branch][6] = np.real(S_branch_elements)
        Power_branches[branch][7] = np.imag(S_branch_elements)

    P_losses_line = np.sum(Power_branches[:,6])/100
    Q_losses_line = np.sum(Power_branches[:,7]) * 1j /100

    # Computation of power through shunt capacitors, reactors or conductantes, Power balanca
    S_shunt = 0
    for i in range(case.N):
        if Shunt[i] != 0:
            S_shunt += V_complex_profile[i] * np.conj(V_complex_profile[i]*Shunt[i])

    Qload = np.sum(Qd) * 1j
    Pload = np.sum(Pd)

    if 'HELM' in algorithm:

        if not enforce_Q_limits:
            for i in list_gen:
                Qg[i] = Q_iny(i, case, run) + Qd[i]
        Qgen = (np.sum(Qg) + Q_iny(slack, case, run) + Qd[slack]) * 1j

        if 'DS' in algorithm: # algorithm models the distributed slack
            Pmismatch = P_losses_line + np.real(S_shunt)
            Pgen = np.sum(Pg + K*Pmismatch)
        else:
            Pgen = np.sum(Pg) + P_iny(slack, case, run) + Pd[slack]

    # elif 'NR' in algorithm:

    #     if not enforce_Q_limits:
    #         for i in list_gen:
    #             Qg[i] = Qi[i] + Qd[i]
    #     Qgen = (np.sum(Qg) + Qi[slack] + Qd[slack]) * 1j

    #     if 'DS' in algorithm: # algorithm models the distributed slack
    #         Pmismatch = P_losses_line + np.real(S_shunt)
    #         Pgen = np.sum(Pg + K*Pmismatch)
    #     else:
    #         Pgen = np.sum(Pg) + Pi[slack] + Pd[slack]

    S_gen = (Pgen + Qgen) * 100
    S_load = (Pload + Qload) * 100
    S_mismatch = (P_losses_line + Q_losses_line + S_shunt) * 100

    if 'DS' in algorithm:
        return (Power_branches, S_gen, S_load, S_mismatch, Pmismatch)
    else:
        return (Power_branches, S_gen, S_load, S_mismatch, None)

def print_voltage_profile(V_polar_final, N):
    """Print voltage profile."""
    print("\n\tVoltage profile:")
    print("   Bus    Magnitude (p.u.)    Phase Angle (degrees)")
    if N <= 31:
        print(*("{:>6d}\t     {:1.6f}\t\t{:11.6f}" \
            .format(i,mag,ang) for i,(mag,ang) in enumerate(V_polar_final)), sep='\n')
    else:
        print(*("{:>6d}\t     {:1.6f}\t\t{:11.6f}" \
            .format(i,mag,ang) for i,(mag,ang) in enumerate(V_polar_final[0:14])), sep='\n')
        print(* 3*("     .\t         .\t\t      .",), sep='\n')
        print(*("{:>6d}\t     {:1.6f}\t\t{:11.6f}" \
            .format(i,mag,ang) for i,(mag,ang) in enumerate(V_polar_final[N-14:N],N-14)), sep='\n')
    print()

def create_power_balance_string(
    mismatch, scale, algorithm,
    list_coef_or_iterations, S_gen, S_load, S_mismatch,
    Ploss=None, Pmismatch=None
):
    coef_or_iterations = 'Coefficients' if algorithm[0:2] == 'HE' else 'Iterations'
    output = \
        'Scale: {}   Mismatch: {}'.format(scale, mismatch) + \
        '   {:s} per PVLIM-PQ switches: {:s}' \
            .format(coef_or_iterations, str(list_coef_or_iterations)) + \
        "\n\n  *  Power Balance:  *" + \
        "\n\nTotal generated power (MVA):  ----------------> {:< 22.15f} {:=+23.15f} j" \
            .format(np.real(S_gen),np.imag(S_gen)) + \
        "\nTotal demanded power (MVA):  -----------------> {:< 22.15f} {:=+23.15f} j" \
            .format(np.real(S_load),np.imag(S_load)) + \
        "\nTotal power through branches and shunt" + \
        "\nelements (mismatch) (MVA):  ------------------> {:< 22.15f} {:=+23.15f} j" \
            .format(np.real(S_mismatch),np.imag(S_mismatch)) + \
        "\n\nComparison: Generated power (MVA):  ----------> {:< 22.15f} {:=+23.15f} j" \
            .format(np.real(S_gen),np.imag(S_gen)) + \
        "\n            Demanded plus mismatch power (MVA): {:< 22.15f} {:=+23.15f} j" \
            .format(np.real(S_load+S_mismatch),np.imag(S_load+S_mismatch))
    if Ploss is not None:
        output = output + \
        "\n\nComparison: Active power losses 'Ploss' variable (MW):  ---------------------> {:< 22.15f}" \
            .format(np.real(Ploss*100)) + \
        "\n            Active power through branches and shunt elements 'Pmismatch' (MW): {:< 22.15f}" \
            .format(np.real(Pmismatch*100))

    return output

def write_results_on_files(
    mismatch, scale, algorithm,
    V_polar_final, Power_branches,
    results_file_name, run,
    power_balance_string,
):
    import pandas as pd  # optional dependency, only needed for xlsx reports

    files_name = \
        'Results' + ' ' + \
        algorithm + ' ' + \
        str(results_file_name) + ' ' + \
        str(scale) + ' ' + \
        str(mismatch)

    # Write voltage profile and branch data to .xlsx file
    voltages_dataframe = pd.DataFrame()
    voltages_dataframe["Complex Voltages"] = run.V_complex_profile
    voltages_dataframe["Voltages Magnitude"] = V_polar_final[:,0]
    voltages_dataframe["Voltages Phase Angle"] = V_polar_final[:,1]
    power_flow_dataframe = pd.DataFrame()
    power_flow_dataframe["From Bus"] = Power_branches[:,0]
    power_flow_dataframe["To Bus"] = Power_branches[:,1]
    power_flow_dataframe['From-To P injection (MW)'] = Power_branches[:,2]
    power_flow_dataframe['From-To Q injection (MVAR)'] = Power_branches[:,3]
    power_flow_dataframe['To-From P injection (MW)'] = Power_branches[:,4]
    power_flow_dataframe['To-From Q injection (MVAR)'] = Power_branches[:,5]
    power_flow_dataframe['P flow through branch and elements (MW)'] = Power_branches[:,6]
    power_flow_dataframe['Q flow through branch and elements (MVAR)'] = Power_branches[:,7]
    xlsx_name = files_name + '.xlsx'
    xlsx_file = pd.ExcelWriter(xlsx_name) # pylint: disable=abstract-class-instantiated
    voltages_dataframe.to_excel(xlsx_file, sheet_name="Buses")
    power_flow_dataframe.to_excel(xlsx_file, sheet_name="Branches")
    xlsx_file.save()

    # Write power balance and other data to .txt file
    # Coefficients/Iterations per PVLIM-PQ switches are written
    txt_name = files_name + ".txt"
    txt_file = open(txt_name,"w")
    txt_file.write(power_balance_string)
    txt_file.close()

    print("\nResults have been written on the files:\n\t%s \n\t%s"%(xlsx_name,txt_name))

def validate_arguments(
        case,
        detailed_run_print, mismatch, scale,
        max_coefficients, enforce_Q_limits,
        results_file_name, save_results,
        pv_bus_model, DSB_model, DSB_model_method, 
):
    if (type(detailed_run_print) is not bool or \
        type(mismatch) is not float or \
        not(
            type(scale) is float or
            type(scale) is int
        ) or \
        type(max_coefficients) is not int or \
        type(enforce_Q_limits) is not bool or \
        not(
            results_file_name is None or
            type(results_file_name) is str
        ) or \
        type(save_results) is not bool or \
        type(pv_bus_model) is not int or \
        type(DSB_model) is not bool or \
        not(
            DSB_model_method is None or
            type(DSB_model_method) is int
        )
    ):
        raise ValueError("Erroneous argument type.")
    if max_coefficients < 5:
        raise ValueError("'max_coefficients' must be equal or greater than five (5).")
    if pv_bus_model not in (1, 2):
        raise ValueError("'pv_bus_model' must be the integer 1 or 2.")
    if DSB_model_method is not None and DSB_model_method not in (1, 2):
        raise ValueError("'DSB_model_method' must be the integer 1 or 2.")


# Main loop
def helm(case, detailed_run_print=False, mismatch=1e-4, scale=1, max_coefficients=100, enforce_Q_limits=True,
         results_file_name=None, save_results=False, pv_bus_model=2, DSB_model=False, DSB_model_method=None,
         K_factors=None, convergence='residual',
         ) -> Tuple[RunVariables, int, bool]:

    # Arguments validation (raises ValueError on bad input)
    validate_arguments(case, detailed_run_print, mismatch, scale, max_coefficients, enforce_Q_limits,
                       results_file_name, save_results, pv_bus_model, DSB_model, DSB_model_method)
    if convergence not in ('residual', 'pade'):
        raise ValueError("'convergence' must be 'residual' or 'pade'.")
    if (detailed_run_print or save_results) and case.Y is None:
        raise ValueError("detailed_run_print/save_results need a case with "
                         "branch-level data (xlsx frontend); array-built "
                         "cases do not carry it.")

    if DSB_model and DSB_model_method is None:
        DSB_model_method = 2

    #  Construct algorithm string
    pv_bus_model_str = 'PV1' if pv_bus_model == 1 else 'PV2'
    if DSB_model_method is not None:
        DSB_model_method_str = 'M1' if DSB_model_method == 1 else 'M2'
        algorithm = 'HELM DS ' + DSB_model_method_str + ' ' + pv_bus_model_str
    else:
        algorithm = 'HELM ' + pv_bus_model_str

    if results_file_name is None:
        results_file_name = case.name

    max_coef = max_coefficients

    # set case at the scale
    if scale != 1:
        case.set_scale(scale)

    # Declare run_variables_class objects.
    # Variables/arrays initialization are inside it
    run = RunVariables(case, pv_bus_model, DSB_model, DSB_model_method, max_coef)

    # Optional externally-supplied distributed-slack participation factors (one weight per
    # bus). When given, they override the default generation-proportional K factors.
    run.external_K = K_factors

    # Sparse case matrices for the vectorized recurrence (independent of bus
    # types, so they survive PVLIM->PQ switches)
    build_case_sparse_matrices(case, run)

    # Two-phase Q-limit enforcement: PVLIM->PQ switching decisions only need a
    # coarse solution, so while bus types are still settling the series is only
    # converged to q_switch_mismatch. One final pass then runs at the requested
    # mismatch. (Previously every switch restarted a full-accuracy run; on
    # case2869pegase that meant 4 complete 1e-8 computations.)
    q_switch_mismatch = max(mismatch, 1e-4)
    current_mismatch = q_switch_mismatch if enforce_Q_limits else mismatch
    structure_changed = True
    resume_from = 0

    while True:
        # The matrix and bus-type dependent setup only has to be redone after a
        # PVLIM->PQ switch, not for the final full-accuracy pass.
        if structure_changed:
            # Re-construct list_gen. List of generators (PV buses)
            run.list_gen = np.setdiff1d(run.list_gen, run.list_gen_remove, assume_unique=True)

            # Define K factors
            if DSB_model:
                # Computing the K factor for each PV bus and the slack bus.
                compute_k_factor(case, run)
            elif DSB_model_method is not None:
                # Set the slack's participation factor to 1 and the rest to 0. Classic slack bus model.
                K_slack_1(case, run)

            # Create modified Y matrix and list that contains the respective column to its voltage on PV and PVLIM buses
            modif_Ytrans(DSB_model_method, pv_bus_model, case, run)

            # Arrays and lists creation
            Unknowns_soluc(DSB_model_method, pv_bus_model, case.N, run)
            structure_changed = False

        # Loop of coefficients computing until the mismatch is reached
        flag_recalculate, flag_divergence, series_large = computing_voltages_mismatch(detailed_run_print,
                                                                                      current_mismatch,
                                                                                      max_coef, enforce_Q_limits,
                                                                                      pv_bus_model, DSB_model_method,
                                                                                      case, run,
                                                                                      start_coef=resume_from,
                                                                                      convergence=convergence)

        if flag_recalculate:
            # A PVLIM bus switched to PQ; rebuild and stay at the coarse tolerance.
            structure_changed = True
            resume_from = 0
            continue
        if flag_divergence or current_mismatch <= mismatch:
            break
        # Bus types are settled; the final pass at the requested accuracy
        # extends the already-computed series instead of restarting it.
        current_mismatch = mismatch
        resume_from = series_large - 1
    # reset scale case
    if scale != 1:
        case.reset_scale()

    if not flag_divergence:
        if detailed_run_print or save_results:
            Ploss = None

            if DSB_model_method is not None:
                Ploss = Pade(run.coefficients[2*case.N], series_large)

            Power_branches, S_gen, S_load, S_mismatch, Pmismatch = power_balance(enforce_Q_limits, algorithm, case, run)

            if detailed_run_print or save_results:
                V_polar_final = convert_complex_to_polar_voltages(run.V_complex_profile, case.N)

                power_balance_string = create_power_balance_string(mismatch, scale, algorithm, run.list_coef, S_gen,
                                                                   S_load, S_mismatch, Ploss, Pmismatch)
                if detailed_run_print:
                    print_voltage_profile(V_polar_final, case.N)
                    print(power_balance_string)
                if save_results:
                    write_results_on_files(mismatch, scale, algorithm, V_polar_final, Power_branches, results_file_name,
                                           run, power_balance_string)

    return run, series_large, flag_divergence
