"""
Tests for the array-based library API (helmpy.api.solve_helm).

The arrays are derived from the xlsx cases' full Ybus, so the pegase cases
also exercise the internal Ybus decomposition with asymmetric entries
(phase-shifting transformers).
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from conftest import get_case_bundle, HELMPY_PATH
import helmpy

# Same reasoning as in test_helmpy.py: agreement with the references is
# bounded by the convergence tolerance. The array path additionally uses a
# different (equally valid) embedding for phase-shifter grids, so its
# stopping points differ; observed worst deviation 2026-07-09: 4e-9 p.u.
MAGNITUDE_ATOL = 1e-7  # p.u.
ANGLE_ATOL = 5e-6  # degrees

CASES = [
    pytest.param("case9"),
    pytest.param("case118"),
    pytest.param("case1354pegase", marks=pytest.mark.slow),
    pytest.param("case2869pegase", marks=pytest.mark.slow),
]


def arrays_from_case(case):
    """ppc-style arrays from a loaded xlsx CaseData object."""
    N = case.N
    Ybus = csr_matrix(case.Y)
    Sbus = (case.Pg - case.Pd) - 1j * case.Qd
    bus_types = np.ones(N, dtype=int)
    bus_types[case.list_gen] = 2
    bus_types[case.slack] = 3
    V_specified = np.ones(N)
    V_specified[case.list_gen] = case.V[case.list_gen]
    V_specified[case.slack] = case.V[case.slack]
    Qmax = np.full(N, np.inf)
    Qmin = np.full(N, -np.inf)
    Qmax[case.list_gen] = case.Qgmax[case.list_gen]
    Qmin[case.list_gen] = case.Qgmin[case.list_gen]
    return Ybus, Sbus, bus_types, V_specified, Qmin, Qmax


@pytest.mark.parametrize("case_name", CASES)
def test_solve_helm_matches_reference(case_name):
    bundle = get_case_bundle(case_name)
    Ybus, Sbus, bus_types, V_specified, Qmin, Qmax = arrays_from_case(bundle.case)

    result = helmpy.solve_helm(Ybus, Sbus, bus_types, V_specified,
                               Qmin=Qmin, Qmax=Qmax, mismatch=1e-8)

    assert result.converged
    assert result.residual <= 1e-8
    np.testing.assert_allclose(np.abs(result.V), bundle.classic_magnitude,
                               rtol=0, atol=MAGNITUDE_ATOL)
    np.testing.assert_allclose(np.angle(result.V, deg=True),
                               bundle.classic_angle,
                               rtol=0, atol=ANGLE_ATOL)


def test_solve_helm_distributed_slack_self_consistent():
    """DS mode: no reference comparison (participation factors are derived
    from net instead of gross generation here), but the converged solution
    must satisfy the distributed-slack power balance."""
    bundle = get_case_bundle("case118")
    Ybus, Sbus, bus_types, V_specified, Qmin, Qmax = arrays_from_case(bundle.case)

    result = helmpy.solve_helm(Ybus, Sbus, bus_types, V_specified,
                               Qmin=Qmin, Qmax=Qmax, mismatch=1e-8,
                               distributed_slack=True)
    assert result.converged
    assert result.residual <= 1e-8


def test_solve_helm_slack_angle_rotation():
    bundle = get_case_bundle("case9")
    Ybus, Sbus, bus_types, V_specified, Qmin, Qmax = arrays_from_case(bundle.case)

    base = helmpy.solve_helm(Ybus, Sbus, bus_types, V_specified)
    rotated = helmpy.solve_helm(Ybus, Sbus, bus_types, V_specified,
                                slack_angle_degrees=30.0)
    np.testing.assert_allclose(
        rotated.V, base.V * np.exp(1j * np.deg2rad(30.0)), rtol=0, atol=1e-12)
    # power flows are rotation invariant
    np.testing.assert_allclose(rotated.S_injection, base.S_injection,
                               rtol=0, atol=1e-12)


def test_solve_helm_input_validation():
    bundle = get_case_bundle("case9")
    Ybus, Sbus, bus_types, V_specified, _, _ = arrays_from_case(bundle.case)

    with pytest.raises(ValueError, match="exactly one slack"):
        helmpy.solve_helm(Ybus, Sbus, np.ones(len(Sbus), dtype=int), V_specified)
    with pytest.raises(ValueError, match="shape"):
        helmpy.solve_helm(Ybus, Sbus[:-1], bus_types, V_specified)
    with pytest.raises(ValueError, match="convergence"):
        helmpy.solve_helm(Ybus, Sbus, bus_types, V_specified, convergence="bogus")


def test_import_without_pandas():
    """Importing helmpy and solving from arrays must not require pandas."""
    import subprocess
    import sys
    code = (
        "import sys\n"
        "sys.modules['pandas'] = None\n"  # poison: any 'import pandas' fails
        f"sys.path.insert(0, r'{HELMPY_PATH}')\n"
        "import helmpy\n"
        "import numpy as np\n"
        "Y = np.array([[1/0.1j + 1, -1/0.1j], [-1/0.1j, 1/0.1j]])\n"
        "r = helmpy.solve_helm(Y, np.array([0, -0.5 - 0.2j]),\n"
        "                      np.array([3, 1]), np.array([1.0, 1.0]))\n"
        "assert r.converged and r.residual < 1e-8\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
