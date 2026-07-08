from helmpy.core.helm import helm
from helmpy.core.classes import CaseData, create_case_data_object_from_xlsx

_LAZY_NR = {'nr': 'helmpy.core.nr', 'nr_ds': 'helmpy.core.nr_ds'}


def __getattr__(name):
    # The Newton-Raphson modules import pandas at module level; load them
    # lazily so that importing helmpy.core needs only numpy/scipy.
    if name in _LAZY_NR:
        import importlib
        module = importlib.import_module(_LAZY_NR[name])
        return getattr(module, name)
    raise AttributeError(f"module 'helmpy.core' has no attribute {name!r}")
