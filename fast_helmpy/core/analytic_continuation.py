"""
Algorithm for the analytic continuation of the power series.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view



def Epsilon(serie_completa, largo_actual):
    """
    # Epsilon Algorithm for the analytic continuation of the power series through Padé approximants

    :param serie_completa:
    :param largo_actual:
    :return:
    """
    serie = serie_completa[:largo_actual]
    continuation = np.zeros((len(serie),len(serie)+1), dtype=complex)
    continuation[0][1] = serie[0]
    for i in range(1,len(serie)):
        continuation[i][1] = continuation[i-1][1] + serie[i]
    for col in range(2,len(serie)+1):
        for row in range(0,len(serie)+1-col):
            continuation[row][col] = continuation[row+1][col-2] + 1/(continuation[row+1][col-1]-continuation[row][col-1])

    return continuation[0][len(serie)]


def Pade(serie_completa,largo_actual):
    """
    Matrix method for the analytic continuation of the power series through Padé approximants
    """
    serie = serie_completa[:largo_actual]
    L = int((len(serie)-1)/2)
    mat_c = np.zeros((L,L), dtype=complex)
    for i in range(1,L+1):
        for j in range(i,L+i):
            mat_c[i-1][j-i] = serie[j]
    vec_b = -np.linalg.solve(mat_c,serie[L+1:len(serie)])
    b = np.ones(L+1, dtype=complex)
    for i in range(1,L+1):
        b[i] = vec_b[L-i]
    a = np.zeros(L+1, dtype=complex)
    a[0] = serie[0]
    for i in range(1,L+1):
        aux = 0
        for k in range(i+1):
            aux += serie[k]*b[i-k]
        a[i] = aux
    return np.sum(a)/np.sum(b)


def pade_batched(series_matrix, largo_actual, tail_tol=0.0):
    """Diagonal Padé approximant at s=1 for many series at once.

    Same values as Pade() per row, but the denominator systems of all rows are
    solved in a single stacked LAPACK call: the coefficient matrix is Hankel
    (entry (r, c) = serie[1+r+c]), so the whole (rows, L, L) stack is a
    stride-tricks view of the series array — no per-row matrix assembly.

    Rows whose last two coefficients sum to less than tail_tol are summed
    directly instead: their series already converged at s=1 and their Hankel
    systems would be near-singular noise.

    :param series_matrix: (rows, >=largo_actual) complex array, one series per row
    :param largo_actual: number of series terms to use (odd, >= 3)
    :param tail_tol: threshold for the direct-summation shortcut (0 disables)
    :return: (rows,) complex array of approximant values at s=1
    """
    S = np.ascontiguousarray(series_matrix[:, :largo_actual])
    n_rows = S.shape[0]
    L = (largo_actual - 1) // 2

    result = np.empty(n_rows, dtype=np.complex128)
    if tail_tol > 0.0:
        direct = (np.abs(S[:, -1]) + np.abs(S[:, -2])) < tail_tol
    else:
        direct = np.zeros(n_rows, dtype=bool)
    if direct.any():
        result[direct] = S[direct].sum(axis=1)
    todo = ~direct
    if not todo.any():
        return result
    T = S[todo]

    # Hankel stack: A[b, r, c] = T[b, 1+r+c]
    A = sliding_window_view(T[:, 1:2*L], L, axis=1)
    rhs = T[:, L+1:2*L+1]
    try:
        vec_b = -np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        # An exactly singular row poisons the whole batch; recover the rest
        # with the scalar routine (which raises only for the singular row,
        # matching the former per-bus behavior).
        result[todo] = [Pade(row, largo_actual) for row in T]
        return result

    # b = [1, vec_b[L-1], ..., vec_b[0]]; partial sums B_j = sum_{m<=j} b_m
    b_full = np.concatenate(
        (np.ones((T.shape[0], 1), dtype=np.complex128), vec_b[:, ::-1]), axis=1)
    B = np.cumsum(b_full, axis=1)
    # sum(a) = sum_{k=0..L} serie[k] * B_{L-k}
    sum_a = np.einsum('ij,ij->i', T[:, :L+1], B[:, ::-1])
    result[todo] = sum_a / B[:, -1]
    return result

