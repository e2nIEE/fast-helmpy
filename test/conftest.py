"""
Shared test infrastructure: import path setup and cached loading of grid cases
with their stored reference results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HELMPY_PATH = Path(__file__).parents[1]
sys.path.insert(0, str(HELMPY_PATH))

import helmpy  # noqa: E402  (needs the sys.path entry above)

DATA_PATH = HELMPY_PATH / "data"


class CaseBundle:
    """A grid case plus its stored reference voltage profiles.

    References were generated with HELM at mismatch=1e-8: classic slack at
    scale=1, distributed slack at scale=1.02 (see data/results/).
    """

    def __init__(self, name):
        self.name = name
        self.case = helmpy.create_case_data_object_from_xlsx(
            str(DATA_PATH / "cases" / f"{name}.xlsx")
        )
        classic = pd.read_excel(
            DATA_PATH / "results" / f"Results HELM {name} 1 1e-08.xlsx",
            sheet_name="Buses",
        )
        ds = pd.read_excel(
            DATA_PATH / "results" / f"Results HELM DS {name} 1.02 1e-08.xlsx",
            sheet_name="Buses",
        )
        self.classic_magnitude = np.asarray(classic["Voltages Magnitude"])
        self.classic_angle = np.asarray(classic["Voltages Phase Angle"])
        self.ds_magnitude = np.asarray(ds["Voltages Magnitude"])
        self.ds_angle = np.asarray(ds["Voltages Phase Angle"])


_bundles = {}


def get_case_bundle(name):
    """Load a case and its references once per session (xlsx parsing is slow)."""
    if name not in _bundles:
        _bundles[name] = CaseBundle(name)
    return _bundles[name]
