#!/usr/bin/env python3
"""Independent check of the reference values: run Py-BOBYQA on the set.

Py-BOBYQA (https://github.com/numericalalgorithmsgroup/pybobyqa) is a general-objective
DFO solver, so it applies to all 90 problems and needs no residual structure. It is
used here only to sanity-check the dataset, not to benchmark anything: a solver that
gets close to the published f* from x0 within the budget corroborates that f* and the
SIF parameters belong to the same problem. Falling short does not: the stored budget
is not enough for every problem, which is the point of the benchmark.

  python validate_pybobyqa.py            # all 90 problems (slow: hours)
  python validate_pybobyqa.py LUKSAN     # only problems whose key matches

Writes validate_pybobyqa.csv.
"""
import csv
import json
import os
import sys

import cfmr

TOL = 1e-2  # relative gap to f* below which we call the reference corroborated


def run(key, meta):
    import numpy as np
    import pybobyqa

    p, params = cfmr._import(meta["name"], meta)
    is_nls = meta["is_nls"]
    obj = (lambda x: float(np.dot(p.cons(x), p.cons(x)))) if is_nls else (lambda x: float(p.obj(x)))
    x0 = p.x0
    bounds = None
    if meta["is_bound_constrained"]:
        bounds = (p.bl, p.bu)
        x0 = np.clip(x0, p.bl, p.bu)  # CUTEst x0 can sit on/outside a bound
    soln = pybobyqa.solve(obj, x0, bounds=bounds, maxfun=meta["budget"],
                          objfun_has_noise=False, print_progress=False)
    f_star = meta["f_star_expected"]
    gap = abs(soln.f - f_star) / max(abs(f_star), 1.0)
    return {"key": key, "n": p.n, "budget": meta["budget"], "sif_params": json.dumps(params),
            "f_x0_expected": meta["f_x0_expected"], "f_star_expected": f_star,
            "f_final": soln.f, "rel_gap": gap, "nf": soln.nf,
            "flag": soln.flag, "corroborated": gap <= TOL}


def main(argv):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    pattern = argv[1] if len(argv) > 1 else ""
    problems = {k: v for k, v in cfmr.load().items() if pattern in k}
    rows = []
    for key, meta in problems.items():
        try:
            row = run(key, meta)
        except Exception as e:  # noqa: BLE001 - a solver/import failure is a result too
            row = {"key": key, "n": meta["n"], "budget": meta["budget"], "sif_params": "",
                   "f_x0_expected": meta["f_x0_expected"], "f_star_expected": meta["f_star_expected"],
                   "f_final": None, "rel_gap": None, "nf": None,
                   "flag": "ERROR: %s" % str(e).replace("\n", " "), "corroborated": False}
        rows.append(row)
        print("%-28s f=%-14s f*=%-14s gap=%s" % (key, row["f_final"], row["f_star_expected"],
                                                 row["rel_gap"]))
    with open(os.path.join(cfmr.HERE, "validate_pybobyqa.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok = sum(r["corroborated"] for r in rows)
    print("\n%d/%d reached f* within %g relative gap; see validate_pybobyqa.csv"
          % (ok, len(rows), TOL))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
