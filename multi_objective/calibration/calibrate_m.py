"""Calibration sweep for M and the clamp multiple C (CMUMO_PORT_DESIGN.md section 2).

Readout 1 (plateau occupancy) is already settled and lives in README.md. This covers:

  readout 2  Phi-variance share   - how much of Var(Phi) each axis drives. A maintain axis
                                    should be nearly flat; if it drives a large share it is
                                    a second improve objective wearing a hinge.
  readout 3  front size           - |non-dominated front| over draws the size of one
                                    generation's candidate set (population + offspring).
                                    This is what predicts the budget burn, because the
                                    budget is charged per front admission.

No agent calls, no GA. Run it against a ZINC pool:

    cd AI4S/cmumo
    ../RSI/.venv-toolmol39/bin/python -I calibrate_m.py --n 1000

Improve/maintain assignment follows the rule the occupancy readout forced: **the two
lowest-occupancy properties in a combination are the improve axes, the rest are maintain.**
Occupancy is measured on the pool, so the rule is re-derived here rather than hardcoded.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # the arm root, so cmumo_oracles imports
from cmumo_oracles import paths  # noqa: E402
from cmumo_oracles.properties import SPECS, PropertyEvaluator  # noqa: E402

ZINC = paths.ZINC

# Bounds for the improve-axis rescale to [0,1], 1 = best. Probabilities are (0,1) or
# inverted. PlogP is unbounded, so it gets the ToolMol treatment it gave dG: a clamp wide
# enough that nothing observed reaches it (200 ZINC molecules spanned -6.66 to +3.38).
# Must match main/toolmol/objectives.py: span = D_p / 0.1, so one tolerance-sized
# improvement is worth 0.1 of Phi on every improve axis.
IMPROVE_BOUNDS = {
    "PlogP": (-5.0, 5.0),
}

# The C-MuMO property combinations, in their task codes.
COMBINATIONS = {
    "BPQ":   ["BBBP", "PlogP", "QED"],
    "ELQ":   ["hERG", "LIV", "QED"],
    "ACEP":  ["AMP", "CARC", "hERG", "PlogP"],
    "BDPQ":  ["BBBP", "DRD2", "PlogP", "QED"],
    "BCMQ":  ["BBBP", "CARC", "MUT", "QED"],
    "HLMPQ": ["HIA", "LIV", "MUT", "PlogP", "QED"],
}

ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0]      # M = alpha * D_p

# The configuration chosen from the sweep (CMUMO_PORT_DESIGN.md section 2). Per-property,
# because a uniform alpha makes CARC and LIV inert - their D_p is large relative to the
# [0,1] scale, so M = D_p already swallows their whole violation range.
CHOSEN_ALPHA = {"BBBP": 1.0, "AMP": 1.0, "MUT": 1.0, "PlogP": 1.0,
                "CARC": 0.25, "LIV": 0.25}
CHOSEN_C = 5.0
# HIA saturates near 1.0 on drug-like input (median 1.000, 10% active at best), so it buys
# an axis and carries almost no signal. Excluded from every combination.
EXCLUDED = {"HIA"}
CLAMPS = [3.0, 5.0, 10.0, float("inf")]  # f = 0 past C * M


def improve_value(prop, raw):
    """Rescale to [0,1], 1 = best, direction-aware."""
    sp = SPECS[prop]
    low, high = IMPROVE_BOUNDS.get(prop, (0.0, 1.0))
    v = (raw - low) / (high - low)
    if sp["direction"] == "-":
        v = 1.0 - v
    return min(1.0, max(0.0, v))


def maintain_value(prop, raw, M, C):
    """The hinge of design section 2: 1 on the good side of theta, min(1, M/d) below it,
    0 past C*M. Continuous at the threshold, because M/d exceeds 1 as d goes to 0."""
    sp = SPECS[prop]
    d = (sp["theta"] - raw) if sp["direction"] == "+" else (raw - sp["theta"])
    if d <= 0:
        return 1.0
    if d > C * M:
        return 0.0
    return min(1.0, M / d)


def occupancy(prop, raws):
    sp = SPECS[prop]
    if sp["direction"] == "+":
        return float(np.mean(raws >= sp["theta"]))
    return float(np.mean(raws <= sp["theta"]))


def split_axes(props, values):
    """The two lowest-occupancy properties become the improve axes."""
    occ = {p: occupancy(p, values[p]) for p in props}
    order = sorted(props, key=lambda p: occ[p])
    return order[:2], sorted(order[2:]), occ


def axis_matrix(props, improve, values, M_by_prop, C):
    """n_molecules x n_axes of per-axis values in [0,1], 1 = best."""
    cols = []
    for p in props:
        if p in improve:
            cols.append([improve_value(p, r) for r in values[p]])
        else:
            cols.append([maintain_value(p, r, M_by_prop[p], C) for r in values[p]])
    return np.array(cols, dtype=float).T


def maintain_activity(F, props, maintain):
    """Distinguish the two ways a maintain axis can have a small variance share:
    compliant-and-flat (wanted) from penalty-too-soft-to-bite (not wanted). Returns the
    mean axis value and the fraction of molecules the penalty actually touches."""
    out = {}
    for p in maintain:
        col = F[:, props.index(p)]
        out[p] = {"mean": float(col.mean()), "active": float(np.mean(col < 0.999))}
    return out


def variance_shares(F):
    """Cov(f_i, Phi) / Var(Phi) per axis. These sum to 1 by construction, so they are a
    genuine decomposition rather than a normalized list of marginal variances."""
    phi = F.sum(axis=1)
    var = float(np.var(phi))
    if var <= 0:
        return np.zeros(F.shape[1]), 0.0
    shares = np.array([float(np.cov(F[:, i], phi, bias=True)[0, 1]) / var
                       for i in range(F.shape[1])])
    return shares, var


def front_size(F):
    nds = NonDominatedSorting().do(1.0 - F, only_non_dominated_front=True)
    return len(nds)


def mean_front_over_draws(F, draw, n_draws, rng):
    sizes = []
    n = F.shape[0]
    for _ in range(n_draws):
        idx = rng.choice(n, size=min(draw, n), replace=False)
        sizes.append(front_size(F[idx]))
    return float(np.mean(sizes)), float(np.std(sizes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="ZINC molecules to score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draw", type=int, default=95,
                    help="candidate-set size per generation (population 60 + offspring 35)")
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(_HERE, "calibration.json"))
    ap.add_argument("--chosen", action="store_true",
                    help="report only the chosen per-property alpha / C, with HIA excluded")
    args = ap.parse_args()

    with open(ZINC) as f:
        lines = [l.strip().strip('"') for l in f][1:]
    random.seed(args.seed)
    smis = random.sample(lines, args.n)

    ev = PropertyEvaluator(cache_path=paths.PROPERTY_CACHE)
    ev.self_check(verbose=False)
    ev.prefetch(smis)
    rows = [ev.get(s) for s in smis]
    rows = [r for r in rows if r]
    ev.close()
    print(f"\nscored {len(rows)} molecules\n")

    values_all = {p: np.array([r[p] for r in rows]) for p in SPECS}
    rng = np.random.default_rng(args.seed)
    results = {}

    for combo, props in COMBINATIONS.items():
        if args.chosen:
            props = [p for p in props if p not in EXCLUDED]
        improve, maintain, occ = split_axes(props, values_all)
        values = {p: values_all[p] for p in props}
        print("=" * 78)
        print(f"{combo}:  improve = {improve}   maintain = {maintain}")
        print("  occupancy: " + "  ".join(f"{p} {occ[p]*100:.0f}%" for p in props))
        if not maintain:
            print("  no maintain axis - M does not apply")
            continue

        print(f"\n  {'alpha':>5} {'C':>5} | " + " ".join(f"{p:>7}" for p in props)
              + f" | {'maint%':>7} {'front':>12} {'gens':>5} | mean/active per maintain axis")
        combo_rows = []
        if args.chosen:
            grid = [(None, CHOSEN_C)]
        else:
            grid = [(a, C) for a in ALPHAS for C in CLAMPS]
        for alpha, C in grid:
            if alpha is None:
                M_by_prop = {p: CHOSEN_ALPHA[p] * SPECS[p]["delta"] for p in maintain}
            else:
                M_by_prop = {p: alpha * SPECS[p]["delta"] for p in maintain}
            if True:
                F = axis_matrix(props, improve, values, M_by_prop, C)
                shares, var = variance_shares(F)
                share_by = dict(zip(props, shares))
                maint_share = sum(share_by[p] for p in maintain)
                activity = maintain_activity(F, props, maintain)
                fsize, fstd = mean_front_over_draws(F, args.draw, args.draws, rng)
                gens = args.budget / fsize if fsize > 0 else float("inf")
                act = " ".join(f"{p}:{activity[p]['mean']:.2f}/{activity[p]['active']*100:.0f}%"
                               for p in maintain)
                label = "chosen" if alpha is None else f"{alpha:>5.2f}"
                print(f"  {label:>6} {C:>5.0f} | "
                      + " ".join(f"{share_by[p]:>7.3f}" for p in props)
                      + f" | {maint_share*100:>6.1f}% {fsize:>6.1f}+-{fstd:<4.1f} {gens:>5.1f}"
                      + f" | {act}")
                combo_rows.append({
                    "alpha": alpha if alpha is not None else CHOSEN_ALPHA,
                    "C": None if C == float("inf") else C,
                    "M": {p: M_by_prop[p] for p in maintain},
                    "shares": share_by, "maintain_share": maint_share,
                    "activity": activity,
                    "phi_var": var, "front_mean": fsize, "front_std": fstd,
                    "gens_to_budget": gens,
                })
        results[combo] = {"improve": improve, "maintain": maintain,
                          "occupancy": occ, "rows": combo_rows}
        print()

    with open(args.out, "w") as f:
        json.dump({"n": len(rows), "seed": args.seed, "draw": args.draw,
                   "budget": args.budget, "combinations": results}, f, indent=2)
    print(f"wrote {args.out}")
    print("\nReading it: 'maint%' is the share of Var(Phi) driven by the maintain axes "
          "(want it small, so\nsearch is driven by the improve axes). 'front' is the mean "
          f"non-dominated front over\n{args.draw}-molecule draws, and 'gens' is "
          f"{args.budget}/front - an UPPER bound on the budget burn, since a\nfront member "
          "already in storing_buffer is not charged again.")


if __name__ == "__main__":
    main()
