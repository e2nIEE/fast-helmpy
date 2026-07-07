"""
Benchmark the HELM solver on the shipped cases.

Times only the helm() call (case loading from xlsx is excluded). Prints a
markdown table suitable for pasting into BENCHMARKS.md. Run after every
optimization phase and compare against the recorded baseline.

Usage:
    python benchmark/benchmark_helm.py
    python benchmark/benchmark_helm.py --repeat 5 --cases case2869pegase
    python benchmark/benchmark_helm.py --methods PV1 PV2 DS-M2-PV2
"""

import argparse
import contextlib
import io
import platform
import sys
import time
from pathlib import Path

HELMPY_PATH = Path(__file__).parents[1]
sys.path.insert(0, str(HELMPY_PATH))

import numpy as np
import scipy

import helmpy

ALL_CASES = ["case9", "case118", "case1354pegase", "case2869pegase"]

# name -> (pv_bus_model, DSB_model, DSB_model_method)
METHODS = {
    "PV1": (1, False, None),
    "PV2": (2, False, None),
    "DS-M1-PV1": (1, True, 1),
    "DS-M1-PV2": (2, True, 1),
    "DS-M2-PV1": (1, True, 2),
    "DS-M2-PV2": (2, True, 2),
}
DEFAULT_METHODS = ["PV2", "DS-M2-PV2"]


def benchmark(case_names, method_names, mismatch, repeat, enforce_q_limits):
    print(f"HELMpy benchmark - {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"python {platform.python_version()}, numpy {np.__version__}, "
          f"scipy {scipy.__version__}, {platform.platform()}")
    print(f"mismatch={mismatch:g}, enforce_Q_limits={enforce_q_limits}, "
          f"best of {repeat} runs\n")

    header = "| case | method | time [s] | coefficients | restarts |"
    print(header)
    print("|---|---|---|---|---|")

    for case_name in case_names:
        case = helmpy.create_case_data_object_from_xlsx(
            str(HELMPY_PATH / "data" / "cases" / f"{case_name}.xlsx")
        )
        for method_name in method_names:
            pv_bus_model, dsb_model, dsb_method = METHODS[method_name]
            scale = 1.02 if dsb_model else 1
            best = float("inf")
            for _ in range(repeat):
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    run, n_coefficients, diverged = helmpy.helm(
                        case, mismatch=mismatch, scale=scale,
                        pv_bus_model=pv_bus_model, DSB_model=dsb_model,
                        DSB_model_method=dsb_method,
                        enforce_Q_limits=enforce_q_limits,
                    )
                best = min(best, time.perf_counter() - t0)
            status = "" if not diverged else " DIVERGED"
            print(f"| {case_name} | {method_name} | {best:.3f} | "
                  f"{n_coefficients} | {len(run.list_coef)} |{status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--cases", nargs="+", default=ALL_CASES, choices=ALL_CASES)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=list(METHODS))
    parser.add_argument("--mismatch", type=float, default=1e-8)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--no-q-limits", action="store_true",
                        help="disable reactive-limit enforcement")
    args = parser.parse_args()

    benchmark(args.cases, args.methods, args.mismatch, args.repeat,
              not args.no_q_limits)
