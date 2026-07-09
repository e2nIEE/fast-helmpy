# fast-helmpy

fast-helmpy is an open source package of power flow solvers.

This package contains a **vectorized, ~16× faster** implementation of the
Holomorphic Embedding Load flow Method (HELM) and the Newton-Raphson (NR)
algorithm. It is a fork of [HELMpy](https://github.com/HELMpy/HELMpy)
(via [vogt31337/HELMpy](https://github.com/vogt31337/HELMpy)); the solver
core was rewritten for speed (sparse/`einsum` recurrence, batched Padé,
physical residual convergence check) and exposed as an array-based library
API. See `BENCHMARKS.md` for the measured speedups and `IMPROVEMENT_PLAN.md`
for the optimization history.

The import package is `fast_helmpy` (underscore); the distribution on PyPI is
`fast-helmpy` (hyphen).

## Installation

```
pip install git+https://github.com/e2nIEE/fast-helmpy.git
```

The core solver needs only `numpy` and `scipy`. Loading grids from `.xlsx`
files and the Newton-Raphson solvers additionally need `pandas`/`openpyxl`:
`pip install fast-helmpy[xlsx]`.

## Using fast-helmpy as a library

`solve_helm` solves a power flow directly from per-unit arrays — no files
involved. Bus types follow the ppc/pypower convention (1 = PQ, 2 = PV,
3 = slack):

```python
import numpy as np
from fast_helmpy import solve_helm

result = solve_helm(
    Ybus,          # (N, N) bus admittance matrix, dense or scipy.sparse
    Sbus,          # (N,) complex net power injection per bus (gen - load)
    bus_types,     # (N,) ints: 1=PQ, 2=PV, 3=slack (exactly one)
    V_specified,   # (N,) voltage magnitude setpoints (used at PV/slack)
    Qmin=Qmin, Qmax=Qmax,   # optional generator reactive limits per bus
    mismatch=1e-8,
)
if result.converged:
    V = result.V                    # complex bus voltages
    S = result.S_injection          # complex power injections
    print(result.n_coefficients, result.residual, result.switched_buses)
```

Calling it from **pandapower** takes a small adapter — the arrays already
exist after any `runpp` attempt (or via `pandapower.pd2ppc`):

```python
import numpy as np
import pandapower as pp
from fast_helmpy import solve_helm

net = ...                     # your pandapower net
try:
    pp.runpp(net)
except pp.LoadflowNotConverged:
    ppc = net._ppc
    internal = ppc["internal"]
    Ybus, Sbus = internal["Ybus"], internal["Sbus"]
    bus_types = np.ones(len(Sbus), dtype=int)
    bus_types[internal["pv"]] = 2
    bus_types[internal["ref"]] = 3
    result = solve_helm(Ybus, Sbus, bus_types, np.abs(internal["V0"]),
                        slack_angle_degrees=net.ext_grid.va_degree.iloc[0])
    # result.V is a robust start vector: pp.runpp(net, init_vm_pu=..., init_va_degree=...)
```

Solver progress is reported through the `fast_helmpy` `logging` logger (no
prints in library mode). Since fast-helmpy is used as a separate library
here, its AGPL license does not affect the license of the calling code base.

## Repository structure

- data: sample data of large-sized, complex practical grids for testing purposes. Already computed results can also be found
- helmm: matlab files for downloading and parsing to `.xlsx` matpower grids
- fast_helmpy: scripts with core functionality
- test: pytest regression suite (`pytest -m "not slow"` for the fast gate)
- benchmark: wall-clock benchmark harness (see `BENCHMARKS.md`)

## Compatibility

This package requires Python >= 3.9 and is tested on 3.10–3.13.

## History

This package was developed by Tulio Molina and Juan José Ortega
as a part of their thesis research
to obtain the degree of Electrical Engineer
at Universidad de los Andes (ULA) in Mérida, Venezuela.

## HELMpy Guide

Please refer to `HELMpy user's guide.pdf`.

## License - AGPLv3

    HELMpy, open source package of power flow solvers developed on Python 3
    Copyright (C) 2019 Tulio Molina tuliojose8@gmail.com and Juan José Ortega juanjoseop10@gmail.com

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
