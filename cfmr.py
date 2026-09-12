#!/usr/bin/env python3
"""CFMR 90-problem CUTEst benchmark set.

Unofficial third-party packaging: not published by, affiliated with, or endorsed by
the authors of the source papers. Contains no new problems or results.

CFMR pairs the 60 nonlinear least-squares problems of the CR set
[cartis2019derivative: Cartis & Roberts, MPC 11(4):631-674, 2019, Table 3] with an
extension set (EXT) of 30 general-objective problems, sourced from
[cartis2019improving: Cartis, Fiala, Marteau & Roberts, ACM TOMS 45(3):1-41, 2019],
which widens the set beyond least-squares structure. The CR+EXT pairing is specific
to CFMR; neither paper defines it.

Data lives in data/cfmr_problems.json (language-neutral, hand-curated from the papers).
This module only *verifies* it against a real CUTEst install and emits harness inputs.

  python cfmr.py verify   # import all 90 via pycutest, check n/m and f(x0)
  python cfmr.py build    # verify + write cfmr_registry.json + cfmr_problems.csv
  python cfmr.py list     # print the set (no pycutest needed)

pycutest requires CUTEst/SIFDecode/ARCHDefs and runs on Linux/macOS only.
"""
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "cfmr_problems.json")
OBSERVED = os.path.join(HERE, "data", "cfmr_observed.json")
FARM = os.path.join(HERE, "mastsif_cfmr")
TOL_REL = 1e-3

CSV_FIELDS = ["suite", "problem", "source", "n", "m", "budget", "budget_scale", "is_nls",
              "is_bound_constrained", "sif_params", "f_x0", "f_ref"]


SCHEMA_VERSION = 2

CONVENTIONS = ("sum_of_squares", "half_sum_squares", "as_published")


def load(path=DATA):
    """The 90 problem entries. Use load_meta() for the dataset-level block."""
    return _read(path)["problems"]


def load_meta(path=DATA):
    """Schema version, objective-convention definitions, citations, field docs."""
    return _read(path)["meta"]


def _read(path):
    with open(path) as f:
        d = json.load(f)
    got = d.get("meta", {}).get("schema_version")
    if got != SCHEMA_VERSION:
        raise RuntimeError("%s is schema version %r, this cfmr.py expects %d"
                           % (path, got, SCHEMA_VERSION))
    return d


def load_observed(path=OBSERVED):
    """Measured companion values from f_star_audit.py, or {} if it has not been run.

    Kept out of cfmr_problems.json on purpose: that file holds only what the papers
    print, so every row in it stays checkable against the source by eye. This one
    holds what a machine measured, and says which machine.
    """
    try:
        with open(path) as f:
            return json.load(f)["problems"]
    except (OSError, ValueError, KeyError):
        return {}


def f_L(key, meta, observed=None, extra=()):
    """The f_L to use in the More-Wild test for this problem.

    `f_star_expected` is a published reference value, not a certified global minimum:
    where a problem is multimodal a paper reports what its own solvers found, and a
    better solver can go below it. When that happens the test

        f(x0) - f(x_k) >= (1 - tau) * (f(x0) - f_L)

    is met at every tau at once and the problem stops separating accuracy levels --
    silently, since nothing about it raises an error. Taking the smallest value
    anyone has actually reached restores the ordering.

    `observed` is load_observed(); `extra` is any further values your own solvers
    reached, in this entry's objective convention. Every candidate must come from a
    run started at the problem's x0 -- a value from global search would tighten the
    target beyond what any solver starting there can reach, and Moré & Wild (2009)
    define f_L as the best value the compared solvers attain for exactly that reason.
    """
    best = meta["f_star_expected"]
    rec = (observed or {}).get(key, {})
    for cand in (rec.get("f_ref_observed"),) + tuple(extra):
        if cand is not None and cand < best:
            best = cand
    return best


def rescale(meta, to):
    """Factor to put this entry's f_x0_expected / f_star_expected into `to`'s units.

    The stored convention is `meta["objective_convention"]`; `to` is whatever the
    solver's objective actually returns. Multiply the stored values by this factor
    before comparing them with anything the solver reports:

        f_star = meta["f_star_expected"] * cfmr.rescale(meta, "half_sum_squares")

    The More-Wild test is invariant to a positive rescaling, so this matters only
    when f(x0), f(x_k) and f_L would otherwise come from different conventions --
    which is the one way to get a wrong data profile without any error being raised.
    """
    have = meta["objective_convention"]
    if to not in CONVENTIONS:
        raise ValueError("unknown convention %r, expected one of %s" % (to, CONVENTIONS))
    if have == to:
        return 1.0
    if {have, to} == {"sum_of_squares", "half_sum_squares"}:
        return 0.5 if to == "half_sum_squares" else 2.0
    raise ValueError("%s is stored as %r, which has no defined conversion to %r "
                     "(a general objective has no residual decomposition)"
                     % (meta["name"], have, to))


def _farm():
    """A writable MASTSIF: symlinks to every stock SIF, so single files can be
    overridden without touching the CUTEst installation. Returns the farm path."""
    stock = os.environ.get("MASTSIF")
    if not stock or not os.path.isdir(stock):
        raise RuntimeError("MASTSIF is not set to a readable directory")
    if os.path.realpath(stock) == os.path.realpath(FARM):
        return FARM
    os.makedirs(FARM, exist_ok=True)
    for f in os.listdir(stock):
        link = os.path.join(FARM, f)
        if not os.path.lexists(link):
            os.symlink(os.path.join(stock, f), link)
    os.environ["MASTSIF"] = FARM
    return FARM


def _bake(name, params):
    """Write a copy of <name>.SIF into the farm with `params` already selected.

    sifdecoder's -param is broken for some files (SEMICON2, SEMICN2U, ODC in
    CUTEst as of 2026: it drops the definition of the very parameter it is
    overriding, then reports that name as unrecognised). Editing the $-PARAMETER
    lines instead sidesteps the decoder entirely. Only ever called as a fallback,
    and each problem appears once in CFMR, so one baked file per name is enough.
    """
    farm = _farm()
    src = os.path.join(os.path.realpath(os.path.join(farm, name + ".SIF")))
    lines, hit = [], {k: 0 for k in params}
    for ln in open(src).read().splitlines(True):
        m = re.match(r'^([ *])(?:I|R)E (\S+)\s+(\S+)\s+\$-PARAMETER', ln)
        if m and m.group(2) in params:
            key, val = m.group(2), m.group(3)
            try:
                same = float(val.replace("D", "E")) == float(params[key])
            except ValueError:
                same = False
            take = same and not hit[key]
            lines.append((" " if take else "*") + ln[1:])
            hit[key] += bool(take)
            continue
        lines.append(ln)
    missing = [k for k, v in hit.items() if not v]
    if missing:
        raise RuntimeError("SIF %s lists no $-PARAMETER line for %s"
                           % (name, ", ".join("%s=%s" % (k, params[k]) for k in missing)))
    dst = os.path.join(farm, name + ".SIF")
    baked = "".join(lines)
    if not os.path.islink(dst) and open(dst).read() == baked:
        return                                  # already baked with these values
    if os.path.islink(dst):
        os.unlink(dst)
    with open(dst, "w") as f:
        f.write(baked)
    import pycutest
    try:
        pycutest.clear_cache(name)              # stale build would hide the new values
    except Exception:                           # noqa: BLE001 - nothing cached yet
        pass


def _candidates(meta):
    """(name, params) pairs to try, most faithful to the paper first."""
    names = [meta["name"]] + list(meta.get("name_alt") or [])
    param_sets = meta["sif_params_alt"] or [meta["sif_params"]]
    return [(n, p) for n in names for p in param_sets]


def import_verified(meta, drop_fixed=True):
    """Import a problem and refuse to return it unless it *is* the paper's problem.

    n, m and f(x0) together fingerprint a CUTEst problem: a rename, a wrong
    sifParams set or a different fixed-variable convention all move at least one
    of them. Checking the fingerprint before a benchmark spends its budget is what
    stops a mis-resolved problem from quietly becoming a wrong point on a plot.

    Each candidate is tried with sifParams first and, if the decoder chokes, with
    those values baked into a private copy of the SIF. Returns (problem, how),
    where `how` records what actually resolved. Raises with every attempt listed.
    """
    import pycutest
    tried = []
    for name, params in _candidates(meta):
        for baked in (False, True):
            try:
                if baked:
                    if not params:
                        continue
                    _bake(name, params)
                    p = pycutest.import_problem(name, drop_fixed_variables=drop_fixed)
                else:
                    _farm()
                    p = pycutest.import_problem(name, sifParams=params or None,
                                                drop_fixed_variables=drop_fixed)
            except Exception as e:  # noqa: BLE001 - report every failed candidate
                tried.append("%s %s%s -> %s" % (name, params, " (baked)" if baked else "",
                                                str(e).replace("\n", " ")[:100]))
                continue
            bad = fingerprint_mismatch(p, meta)
            if bad:
                tried.append("%s %s%s -> %s" % (name, params, " (baked)" if baked else "",
                                                "; ".join(bad)))
                continue
            return p, {"name": name, "sif_params": params, "baked": baked}
    raise RuntimeError("could not resolve %s to the problem the paper describes:\n    %s"
                       % (meta["name"], "\n    ".join(tried)))


def fingerprint_mismatch(p, meta):
    """Everything about `p` that disagrees with the paper. Empty list == it matches."""
    bad = []
    if p.n != meta["n"]:
        bad.append("n mismatch: got %d, paper %d" % (p.n, meta["n"]))
    if meta["is_nls"] and meta["m"] is not None and p.m != meta["m"]:
        bad.append("m mismatch: got %d, paper %d" % (p.m, meta["m"]))
    f0, exp = f_at_x0(p, meta["is_nls"]), meta["f_x0_expected"]
    if abs(f0 - exp) > TOL_REL * max(abs(exp), 1e-12) and abs(f0 - exp) > 1e-8:
        bad.append("f(x0) mismatch: got %.10g, paper %.10g" % (f0, exp))
    return bad


def f_at_x0(p, is_nls):
    """f(x0) in this dataset's convention.

    NLS: the sum of squares ||r(x0)||^2, which is what the CR paper tabulates as
    2f(x0) (it defines f = 0.5*||r||^2). Solvers minimising 0.5*||r||^2 report half
    of f_x0_expected / f_star_expected. Otherwise: the objective as published.
    """
    import numpy as np
    if is_nls:
        r0 = p.cons(p.x0)
        return float(np.dot(r0, r0))
    return float(p.obj(p.x0))


def verify(problems, verbose=True):
    """Resolve every problem and report whether it matches the paper.

    Returns (registry, rows) where rows are (key, status, detail) per problem.
    """
    registry, rows = {}, []
    for key, meta in problems.items():
        try:
            p, how = import_verified(meta)
            notes = []
            if how["name"] != meta["name"]:
                notes.append("resolved via alias %s" % how["name"])
            if how["baked"]:
                notes.append("sifParams baked into a private SIF copy (-param failed)")
            entry = dict(meta, sif_params=how["sif_params"], resolved_name=how["name"],
                         n=p.n, m=(p.m if meta["is_nls"] else None),
                         budget=meta["budget_scale"] * (p.n + 1),
                         f_x0_actual=f_at_x0(p, meta["is_nls"]))
            if notes:
                entry["param_note"] = "; ".join(notes)
            registry[key] = entry
            rows.append((key, "OK", "; ".join(notes)))
        except Exception as e:  # noqa: BLE001
            rows.append((key, "FAILED", str(e).replace("\n", " ")))
        if verbose:
            print("%-8s %s %s" % (rows[-1][1], rows[-1][0].ljust(28), rows[-1][2]))
    return registry, rows


def write_csv(registry, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, CSV_FIELDS)
        w.writeheader()
        for meta in registry.values():
            w.writerow({
                "suite": "cfmr_cr" if meta["suite"] == "CR" else "cfmr_ext",
                "problem": meta["name"],
                "source": meta["suite"],
                "n": meta["n"], "m": meta["m"], "budget": meta["budget"],
                "budget_scale": meta["budget_scale"],
                "is_nls": meta["is_nls"],
                "is_bound_constrained": meta["is_bound_constrained"],
                "sif_params": json.dumps(meta["sif_params"], sort_keys=True),
                "f_x0": meta["f_x0_expected"], "f_ref": meta["f_star_expected"],
            })


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "list"
    problems = load()
    if cmd == "list":
        for key, m in problems.items():
            print("%-28s n=%-4s m=%-5s nls=%-5s bound=%-5s budget=%s"
                  % (key, m["n"], m["m"], m["is_nls"], m["is_bound_constrained"], m["budget"]))
        print("%d problems (%d CR, %d EXT)" % (
            len(problems),
            sum(m["suite"] == "CR" for m in problems.values()),
            sum(m["suite"] == "EXT" for m in problems.values())))
        return 0

    os.environ.setdefault("OMP_NUM_THREADS", "1")  # BLAS oversubscription
    registry, rows = verify(problems)
    bad = [r for r in rows if r[1] != "OK"]
    print("\n%d/%d clean, %d need attention" % (len(rows) - len(bad), len(rows), len(bad)))
    if cmd == "build":
        with open(os.path.join(HERE, "cfmr_registry.json"), "w") as f:
            json.dump(registry, f, indent=2)
        write_csv(registry, os.path.join(HERE, "cfmr_problems.csv"))
        print("wrote cfmr_registry.json (%d entries) and cfmr_problems.csv" % len(registry))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
