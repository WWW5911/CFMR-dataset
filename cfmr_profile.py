#!/usr/bin/env python3
"""Run Py-BOBYQA on the CFMR 90-problem set and draw its data profile.

Convergence test (More & Wild 2009, as used by Cartis et al.):
    f(x0) - f(x_k) >= (1 - tau) * (f(x0) - fL)
with fL from cfmr.f_L(): the tabulated f_star_expected, lowered to the best value
anything has actually reached from x0 where data/cfmr_observed.json records one.
The tabulated value is a published reference, not a certified global minimum, and
on a multimodal problem a solver can go below it -- which satisfies the test at
every tau at once and silently costs the problem its ability to separate accuracy
levels. See f_star_audit.py. x-axis is the budget in simplex gradients,
#evals / (n+1).

  python cfmr_profile.py            # all 90
  python cfmr_profile.py LUKSAN     # only keys matching
  python cfmr_profile.py --plot     # redraw from the saved history, no solving

Each result is written to the history file as soon as its problem finishes, so a
killed sweep resumes instead of starting over. Only --plot needs matplotlib, which
lets the solving run in a CUTEst env that has no plotting stack.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    # must precede numpy: these problems are n<=120, so a threaded BLAS only
    # spin-waits, and one worker per core is what we actually want
    os.environ.setdefault(_v, "1")

import json, sys
import numpy as np
import cfmr

TAUS = [1e-1, 1e-3, 1e-5, 1e-7]

# Py-BOBYQA settings. Written out in full -- including the two that merely restate
# Py-BOBYQA's own defaults -- so a result file states the configuration that produced
# it instead of leaving it to whatever the installed version happens to default to.
#
# BUDGET_SCALE 100, not the dataset's own 50: at 50 the tau=1e-5 and 1e-7 curves are
# still climbing when the budget runs out, so the run reports where the budget stopped
# rather than where the solver did.
# RHOEND 1e-12, not the default 1e-8: at 1e-8, 16 of 90 runs quit on "rho has reached
# rhoend" before their budget (FLOSP2HH at 13% of it), which truncates the high-accuracy
# taus for a reason that has nothing to do with the problem.
BUDGET_SCALE = 100                          # maxfun = BUDGET_SCALE * (n + 1)
RHOEND = 1e-12                              # Py-BOBYQA default is 1e-8
NPT = lambda n: 2 * n + 1                   # = Py-BOBYQA's default for a noiseless objfun
RHOBEG = lambda x0: 0.1 * max(np.max(np.abs(x0)), 1.0)   # = its default too

# Goes into every history file: a result that cannot say how it was produced is not
# a result anyone can compare against.
CONFIG = {"solver": "Py-BOBYQA", "budget_scale": BUDGET_SCALE, "rhoend": RHOEND,
          "npt": "2n+1", "rhobeg": "0.1*max(|x0|_inf, 1)", "objfun_has_noise": False,
          "seek_global_minimum": False, "f_star_source": "cfmr.f_L (CFMR_table lowered by cfmr_observed.json)"}

# read once, not per problem: it is the same file for the whole sweep
OBSERVED = cfmr.load_observed()

TAG = "_b%d_rhoend%g" % (BUDGET_SCALE, RHOEND)
OUT = os.path.join(cfmr.HERE, "cfmr_pybobyqa_history%s.json" % TAG)
PNG = os.path.join(cfmr.HERE, "cfmr_data_profile%s.png" % TAG)


def load_history(path=OUT):
    """Finished runs from an earlier invocation, keyed by problem. {} if none."""
    try:
        with open(path) as f:
            return {r["key"]: r for r in json.load(f).get("results", [])}
    except (OSError, ValueError):
        return {}


def save_history(done, failed, path=OUT):
    """Write via a temp file: a kill mid-write cannot leave a truncated history."""
    with open(path + ".tmp", "w") as f:
        json.dump({"config": CONFIG,
                   "results": sorted(done.values(), key=lambda r: r["key"]),
                   "failed": failed}, f)
    os.replace(path + ".tmp", path)


def draw(done, problems, failed=()):
    """Draw the profile for `problems`, counting anything unfinished as unsolved."""
    results = [done[k] for k in problems if k in done]
    missing = [k for k in problems if k not in done]
    print("\n%d runs, %d missing -> %s" % (len(results), len(missing), OUT))
    if missing:
        detail = dict(failed)
        print("\nWARNING: %d problem(s) did not resolve to the paper's problem and are\n"
              "counted as never solved. The denominator stays %d.\n%s"
              % (len(missing), len(problems),
                 "\n".join("  %-28s %s" % (k, detail.get(k, "not run")) for k in missing)))
    if not results:
        return 1 if missing else 0
    try:
        profile(results, PNG, n_total=len(problems), excluded=missing)
    except ImportError as e:
        # no matplotlib in the CUTEst env; the history is already on disk, so the
        # run is not lost -- redraw it from an env that has one
        print("plot skipped (%s).\nRedraw with:  python cfmr_profile.py --plot" % e)
    return 1 if missing else 0


def run(key, meta):
    import pybobyqa
    p, how = cfmr.import_verified(meta)     # refuses anything but the paper's problem
    params = how["sif_params"]
    is_nls = meta["is_nls"]
    hist = []

    def obj(x):
        f = float(np.dot(p.cons(x), p.cons(x))) if is_nls else float(p.obj(x))
        hist.append(f)
        return f

    x0, bounds = p.x0, None
    if meta["is_bound_constrained"]:
        bounds = (p.bl, p.bu)
        x0 = np.clip(x0, p.bl, p.bu)
    n, maxfun = len(x0), BUDGET_SCALE * (len(x0) + 1)
    soln = pybobyqa.solve(obj, x0, bounds=bounds, npt=NPT(n), maxfun=maxfun,
                          rhobeg=RHOBEG(x0), rhoend=RHOEND,
                          objfun_has_noise=False, seek_global_minimum=False,
                          print_progress=False)
    best = np.minimum.accumulate(np.array(hist)).tolist()
    return {"key": key, "n": int(p.n), "budget": maxfun, "how": how,
            "f_star": cfmr.f_L(key, meta, OBSERVED),
            "f_star_table": meta["f_star_expected"], "sif_params": params,
            # why the solver stopped: without it, a run that quit on rhoend is
            # indistinguishable from one the problem genuinely defeated
            "exit_flag": int(soln.flag), "exit_msg": str(soln.msg),
            "f0": best[0], "f_final": best[-1], "hist": best}


def profile(results, path, n_total=None, excluded=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_total = n_total or len(results)
    # A result below its own f_L means the floor is stale: this run found a better
    # point than anything on record, so every tau marked it solved. Say so rather
    # than draw a curve with a free point in it.
    # Compared against a band far tighter than the narrowest tau plotted, not
    # against zero: a solver that converges a little further into the same local
    # minimum undercuts a floor set by an earlier run of it by rounding, which
    # cannot move any point on any curve and is not what this warns about.
    def _undercut(r):
        floor = 0.01 * min(TAUS) * (r["f0"] - r["f_star"])
        return r["f_star"] - r["f_final"] > max(floor, 0.0)

    stale = [r["key"] for r in results if _undercut(r)]
    if stale:
        print("\nWARNING: %d problem(s) reached materially below the f_L they were "
              "scored against, so they count as solved at every tau:\n  %s\n"
              "Refresh the floor:  python f_star_audit.py --history %s"
              % (len(stale), ", ".join(stale), os.path.basename(OUT)))
    alphas = np.linspace(0, BUDGET_SCALE, 400)
    plt.figure(figsize=(7, 5))
    for tau in TAUS:
        frac = []
        for a in alphas:
            solved = 0
            for r in results:
                h = r["hist"]
                k = int(a * (r["n"] + 1))
                if k >= 1 and h[min(k, len(h)) - 1] <= r["f0"] - (1 - tau) * (r["f0"] - r["f_star"]):
                    solved += 1
            frac.append(solved / n_total)
        plt.step(alphas, frac, where="post",
                 label=r"$\tau=10^{%d}$" % round(np.log10(tau)))
    plt.xlabel("Budget in evals (gradients), #evals / (n+1)")
    plt.ylabel("Proportion of problems solved")
    plt.title("Py-BOBYQA on CFMR (%d problems)" % n_total)
    if excluded:
        # never let an unloadable problem shrink the denominator behind the reader's
        # back: it stays in it, counts as unsolved, and is named on the figure
        plt.gcf().text(0.5, 0.005, "%d of %d could not be resolved and count as "
                       "unsolved: %s" % (len(excluded), n_total, ", ".join(excluded)),
                       ha="center", fontsize=7, color="crimson", wrap=True)
    plt.ylim(0, 1.02); plt.xlim(0, BUDGET_SCALE); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150)
    print("wrote", path)


def main(argv):
    flags = [a for a in argv[1:] if a.startswith("--")]
    args = [a for a in argv[1:] if not a.startswith("--")]
    pattern = args[0] if args else ""
    problems = [k for k in cfmr.load() if pattern in k]
    if "--plot" in flags:
        return draw(load_history(), problems)
    metas = cfmr.load()
    done = {} if "--restart" in flags else load_history()
    todo = [k for k in problems if k not in done]
    if len(todo) < len(problems):
        print("resuming: %d of %d already done" % (len(problems) - len(todo),
                                                  len(problems)), flush=True)
    failed = []
    for key in todo:
        try:
            r = run(key, metas[key])
            done[key] = r
            print("%-28s n=%-4d f0=%-12.4g f=%-12.4g f*=%.4g" %
                  (key, r["n"], r["f0"], r["f_final"], r["f_star"]), flush=True)
        except Exception as e:  # noqa: BLE001 - an import/solve failure is a result too
            failed.append((key, str(e).replace("\n", " ")[:120]))
            print("%-28s FAILED: %s" % (key, failed[-1][1]), flush=True)
        save_history(done, failed)
    return draw(done, problems, failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
