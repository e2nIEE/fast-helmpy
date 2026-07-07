"""
Regression tests for the HELM solver: every PV-bus/distributed-slack method
combination is solved on every shipped case and compared against the stored
reference voltage profiles in data/results/.

Run the fast subset (case9, case118) with:      pytest -m "not slow"
Run everything (adds the pegase cases, ~2 min): pytest
"""

import numpy as np
import pytest

from conftest import get_case_bundle
import helmpy

# Tolerances against the stored references. Worst observed deviation on
# 2026-07-07 (numpy 2.x, all 40 combinations): 4.2e-13 p.u. / 4.5e-11 deg.
# Kept ~100x above that so legitimate refactoring (e.g. reordered float
# summation) passes while real logic errors (typically > 1e-6) fail.
MAGNITUDE_ATOL = 1e-10  # p.u.
ANGLE_ATOL = 1e-8  # degrees

MISMATCH = 1e-8

CASES = [
    pytest.param("case9"),
    pytest.param("case118"),
    pytest.param("case1354pegase", marks=pytest.mark.slow),
    pytest.param("case2869pegase", marks=pytest.mark.slow),
]

# (pv_bus_model, DSB_model, DSB_model_method) - the 10 supported combinations.
# DSB_model=False with a DSB_model_method set uses the DS formulation with the
# classic slack (K_slack=1), so it must reproduce the classic-slack references.
PV_DSB_METHODS = [
    pytest.param(1, False, None, id="PV1"),
    pytest.param(2, False, None, id="PV2"),
    pytest.param(1, False, 1, id="DS-M1-PV1-classicK"),
    pytest.param(2, False, 1, id="DS-M1-PV2-classicK"),
    pytest.param(1, True, 1, id="DS-M1-PV1"),
    pytest.param(2, True, 1, id="DS-M1-PV2"),
    pytest.param(1, False, 2, id="DS-M2-PV1-classicK"),
    pytest.param(2, False, 2, id="DS-M2-PV2-classicK"),
    pytest.param(1, True, 2, id="DS-M2-PV1"),
    pytest.param(2, True, 2, id="DS-M2-PV2"),
]


@pytest.mark.parametrize("pv_bus_model, DSB_model, DSB_model_method", PV_DSB_METHODS)
@pytest.mark.parametrize("case_name", CASES)
def test_helm_matches_reference(case_name, pv_bus_model, DSB_model, DSB_model_method):
    bundle = get_case_bundle(case_name)
    scale = 1.02 if DSB_model else 1

    run, n_coefficients, diverged = helmpy.helm(
        bundle.case,
        mismatch=MISMATCH,
        scale=scale,
        pv_bus_model=pv_bus_model,
        DSB_model=DSB_model,
        DSB_model_method=DSB_model_method,
    )

    assert not diverged, f"HELM diverged on {case_name}"
    assert n_coefficients <= 100

    magnitude = np.abs(run.V_complex_profile)
    angle = np.angle(run.V_complex_profile, deg=True)
    if DSB_model:
        reference_magnitude = bundle.ds_magnitude
        reference_angle = bundle.ds_angle
    else:
        reference_magnitude = bundle.classic_magnitude
        reference_angle = bundle.classic_angle

    np.testing.assert_allclose(magnitude, reference_magnitude, rtol=0, atol=MAGNITUDE_ATOL)
    np.testing.assert_allclose(angle, reference_angle, rtol=0, atol=ANGLE_ATOL)


def test_non_consecutive_bus_numbers_load():
    """Guard for the Number_bus/phase_dict loader regressions: pegase cases have
    non-consecutive external bus numbers and phase-shifting transformers."""
    bundle = get_case_bundle("case1354pegase")
    assert bundle.case.N == 1354
    assert isinstance(bundle.case.Number_bus, dict)
    assert isinstance(bundle.case.phase_dict, dict)
    # phase shifters present -> phase_dict must carry list-valued entries
    assert any(bundle.case.phase_barras)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
