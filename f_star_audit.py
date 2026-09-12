#!/usr/bin/env python3
"""Check every tabulated f* against a point actually reached on this CUTEst install.

The reference values in `data/cfmr_problems.json` are transcribed from the source
papers exactly as printed. They are *published reference values*, not certified
global minima, and for a multimodal problem a paper reports the minimum its own
solvers found. Where a solver in your comparison goes below the tabulated value,
the More-Wild test

    f(x0) - f(x_k) >= (1 - tau) * (f(x0) - f_L)      with f_L = f_star_expected

is satisfied at every tau at once, so the problem stops discriminating between
accuracy levels. This script measures how far that reaches on this installation.

What it does: run L-BFGS-B with CUTEst's exact gradients from the paper's own x0,
in the dataset's own objective convention, and record where it lands. Starting
from x0 is the point -- f_L must stay reachable from where the benchmark starts.
A value found by global search would make the target harder for every solver and
can leave a problem unsolvable for all of them, which is why Moré & Wild (2009)
define f_L as the best value *the compared solvers* attain, not the global minimum.

    python f_star_audit.py                        # run, write data/cfmr_observed.json
    python f_star_audit.py --report               # print the table from that file
    python f_star_audit.py --history run.json     # also fold in a solver run's f_final

`--history` accepts a `cfmr_profile.py` history file (repeatable). Its `f_final`
values were also produced from x0 and so are legitimate observations; folding them
in makes `f_ref_observed` the best value anything here has actually reached.

Writes `data/cfmr_observed.json`: measured values with the toolchain that produced
them. It is a companion to the dataset, deliberately a separate file --
`cfmr_problems.json` holds only what the papers print, so every row in it stays
checkable against the source by eye.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import datetime, json, subprocess, sys
import numpy as np
import cfmr

OUT = os.path.join(cfmr.HERE, "data", "cfmr_observed.json")
TAUS = [1e-1, 1e-3, 1e-5, 1e-7]

# L-BFGS-B is asked to converge as tightly as the arithmetic allows: a stationary
# point reported at a loose tolerance would be indistinguishable from a table value
# that is genuinely lower, which is the one thing this script exists to tell apart.
LBFGSB_OPTS = {"maxiter": 10000, "maxfun": 50000, "ftol": 1e-16, "gtol": 1e-12}


def _tool_versions():
    """Everything a reader needs to reproduce these numbers, or to distrust them."""
    def _first_line(path):
        try:
            with open(path) as f:
                return f.readline().strip().lstrip("* ")
        except OSError:
            return None

    def _mod(name):
        try:
            return __import__(name).__version__
        except Exception:                                   # noqa: BLE001
            return None

    root = os.path.dirname(os.environ.get("CUTEST", ""))
    mastsif = os.environ.get("MASTSIF", "")
    try:
        commit = subprocess.run(["git", "-C", mastsif, "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                       # noqa: BLE001
        commit = ""
    return {"date": datetime.date.today().isoformat(),
            "python": sys.version.split()[0],
            "numpy": _mod("numpy"), "scipy": _mod("scipy"),
            "pycutest": _mod("pycutest"),
            "cutest": _first_line(os.path.join(root, "cutest", "version")),
            "sifdecode": _first_line(os.path.join(root, "sifdecode", "version")),
            "mastsif_commit": commit or None}


def make_fg(p, is_nls):
    """(f, grad) in the dataset's convention: ||r||^2 for CR, f(x) as published for EXT."""
    if is_nls:
        def fg(x):
            c, J = p.cons(x, gradient=True)
            c = np.asarray(c, dtype=float)
            return float(c @ c), 2.0 * (np.asarray(J, dtype=float).T @ c)
    else:
        def fg(x):
            f, g = p.obj(x, gradient=True)
            return float(f), np.asarray(g, dtype=float)
    return fg


def audit_one(meta):
    from scipy.optimize import minimize
    p, how = cfmr.import_verified(meta)          # refuses anything but the paper's problem
    fg = make_fg(p, meta["is_nls"])
    x0 = np.asarray(p.x0, dtype=float)
    bounds = None
    if meta["is_bound_constrained"]:
        bounds = list(zip(p.bl, p.bu))
        x0 = np.clip(x0, p.bl, p.bu)
    f0 = fg(x0)[0]
    r = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds, options=LBFGSB_OPTS)
    f_end, g_end = fg(r.x)
    if bounds is None:
        gnorm = float(np.max(np.abs(g_end)))
    else:
        # at a bound the gradient need not vanish; the projected step is what does
        gnorm = float(np.max(np.abs(np.clip(r.x - g_end, p.bl, p.bu) - r.x)))
    return {"n": int(p.n), "f0_measured": f0, "f_lbfgsb": f_end,
            "grad_norm_inf": gnorm, "iterations": int(r.nit), "baked": how["baked"]}


def run(histories=()):
    extra = {}
    for path in histories:
        with open(path) as f:
            for rec in json.load(f).get("results", []):
                key, f_final = rec["key"], rec["f_final"]
                if key not in extra or f_final < extra[key][0]:
                    extra[key] = (f_final, os.path.basename(path))

    problems, out = cfmr.load(), {}
    for i, (key, meta) in enumerate(problems.items(), 1):
        try:
            rec = audit_one(meta)
        except Exception as e:                              # noqa: BLE001
            out[key] = {"error": str(e).replace("\n", " ")[:200]}
            print("%3d/%d %-18s ERROR %s" % (i, len(problems), key, out[key]["error"]),
                  flush=True)
            continue
        rec["f_star_expected"] = meta["f_star_expected"]
        best, src = rec["f_lbfgsb"], "lbfgsb_from_x0"
        if key in extra and extra[key][0] < best:
            best, src = extra[key][0], extra[key][1]
        rec["f_ref_observed"], rec["f_ref_source"] = best, src
        out[key] = rec
        print("%3d/%d %-18s n=%-4d f0=%-11.5g f_ref=%-13.7g f*=%-13.7g |g|=%.1e"
              % (i, len(problems), key, rec["n"], rec["f0_measured"],
                 rec["f_ref_observed"], rec["f_star_expected"], rec["grad_norm_inf"]),
              flush=True)

    doc = {"schema_version": 1,
           "description": "Measured companion to cfmr_problems.json: f(x0) and the "
                          "best objective value reached from x0 on this installation. "
                          "Not from the source papers.",
           "method": "scipy L-BFGS-B with CUTEst exact gradients from the paper's x0"
                     + (", plus f_final from: " + ", ".join(os.path.basename(h)
                                                            for h in histories)
                        if histories else ""),
           "toolchain": _tool_versions(),
           "problems": out}
    with open(OUT, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    print("\nwrote", OUT)
    return doc


def report(doc=None):
    """Which tabulated f* are beatable, and what that costs at each tau."""
    if doc is None:
        with open(OUT) as f:
            doc = json.load(f)
    problems = cfmr.load()
    rows = []
    for key, rec in doc["problems"].items():
        if "error" in rec:
            print("ERROR %-18s %s" % (key, rec["error"]))
            continue
        under = rec["f_star_expected"] - rec["f_ref_observed"]
        if under > 0:
            rows.append((key, rec, under))
    rows.sort(key=lambda r: -r[2] / max(abs(r[1]["f_star_expected"]), 1e-12))

    print("\n%d of %d problems reach below the tabulated f*.\n" % (len(rows), len(problems)))
    print("%-18s %-12s %-13s %-12s %-10s %s"
          % ("key", "f(x0)", "f*_table", "below by", "relative", "auto-solved at tau"))
    for key, rec, under in rows:
        span = rec["f0_measured"] - rec["f_star_expected"]
        hit = [t for t in TAUS if under > t * span]
        rel = under / max(abs(rec["f_star_expected"]), abs(rec["f_ref_observed"]), 1e-12)
        print("%-18s %-12.6g %-13.7g %-12.4g %-10.2e %s"
              % (key, rec["f0_measured"], rec["f_star_expected"], under, rel,
                 ", ".join("%g" % t for t in hit) or "-"))

    print("\nProblems the tabulated f* marks solved at every budget, by tau"
          " (denominator %d):" % len(problems))
    for t in TAUS:
        n = sum(1 for _, rec, u in rows
                if u > t * (rec["f0_measured"] - rec["f_star_expected"]))
        print("  tau=%-8g %d" % (t, n))

    bad = [k for k, r in doc["problems"].items()
           if "error" not in r
           and abs(r["f0_measured"] - problems[k]["f_x0_expected"])
           > 1e-3 * max(abs(problems[k]["f_x0_expected"]), 1.0)]
    print("\nf(x0) disagreeing with the table: %d %s" % (len(bad), bad or ""))


def main(argv):
    args = argv[1:]
    if "--report" in args:
        report()
        return 0
    histories = [args[i + 1] for i, a in enumerate(args) if a == "--history"]
    report(run(histories))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
