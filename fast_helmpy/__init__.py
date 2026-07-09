"""HELMpy — power flow solvers including the Holomorphic Embedding Load
Flow Method (HELM).

Importing the package needs only numpy/scipy. The xlsx frontend
(create_case_data_object_from_xlsx, save_results) and the Newton-Raphson
solvers (nr, nr_ds) additionally require pandas + openpyxl and are loaded
lazily on first use.
"""

from fast_helmpy.core.helm import helm
from fast_helmpy.core.classes import CaseData, create_case_data_object_from_xlsx
from fast_helmpy.api import solve_helm, HelmResults, create_case_from_arrays

__version__ = '0.2.0'

_LAZY_NR = {'nr': 'fast_helmpy.core.nr', 'nr_ds': 'fast_helmpy.core.nr_ds'}


def __getattr__(name):
    if name in _LAZY_NR:
        import importlib
        module = importlib.import_module(_LAZY_NR[name])
        return getattr(module, name)
    raise AttributeError(f"module 'fast_helmpy' has no attribute {name!r}")
