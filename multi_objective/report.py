"""End-of-run report for a C-MuMO arm run.

Reads an output directory's `discovered_*.jsonl` (every molecule the run considered) and
`results_*.yaml` (the final front), and prints the metrics design section 0 commits to:
front quality, hypervolume, and an FA-style filtered report.

    python report.py <output_dir>
    python report.py <output_dir> --top 10 --json report.json

Three things this deliberately does NOT report, because the pooled design puts them out of
reach (CMUMO_PORT_DESIGN.md section 0): **SR, SR_Th, RI and Sim.** They are defined
relative to an input molecule M_x, and dropping per-instance anchoring means there is no
M_x. Nothing here may be placed beside the GeLLMO-C table.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmumo_oracles.properties import SPECS  # noqa: E402


def _one(pattern, where):
    hits = sorted(glob.glob(os.path.join(where, pattern)))
    if not hits:
        raise SystemExit(f"no {pattern} in {where}")
    if len(hits) > 1:
        raise SystemExit(f"several {pattern} in {where}; point at one run's directory:\n  "
                         + "\n  ".join(hits))
    return hits[0]


def load(where):
    rows = []
    with open(_one("discovered_*.jsonl", where)) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("the discovery log is empty - did the run complete a generation?")
    return rows


def compliant(row, maintain):
    """On the good side of Th_p for every maintain axis. This is the filter the FA-style
    metric applies, and the only well-defined notion of 'acceptable' the pooled design
    has."""
    for p in maintain:
        spec = SPECS[p]
        v = row["raw"][p]
        if (v < spec["theta"]) if spec["direction"] == "+" else (v > spec["theta"]):
            return False
    return True


def hypervolume(points):
    """Volume dominated by the front, measured from the origin in rescaled space where
    1 = best on every axis. pymoo minimizes, so points go in as 1 - value and the
    reference point is (1, ..., 1)."""
    from pymoo.indicators.hv import HV
    if not len(points):
        return 0.0
    arr = 1.0 - np.asarray(points, dtype=float)
    ref = np.ones(arr.shape[1])
    return float(HV(ref_point=ref)(arr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--top", type=int, default=10, help="molecules in the filtered report")
    ap.add_argument("--json", default=None, help="also write the numbers here")
    args = ap.parse_args()

    rows = load(args.output_dir)
    axes = list(rows[0]["scaled"])
    meta_path = os.path.splitext(_one("discovered_*.jsonl", args.output_dir))[0] + ".meta.json"
    if not os.path.exists(meta_path):
        raise SystemExit(
            f"{meta_path} is missing. It names which axes were improve and which were "
            f"maintain; a run from before DiscoveryLog wrote it has to be reported by "
            f"passing those by hand.")
    with open(meta_path) as f:
        meta = json.load(f)
    improve, maintain = meta["improve"], meta["maintain"]

    front = [r for r in rows if r["on_front"]]
    rejected = [r for r in rows if r.get("rejected")]
    ok = [r for r in rows if compliant(r, maintain)]
    gens = sorted({r["gen"] for r in rows})

    print(f"run: {args.output_dir}")
    print(f"axes: improve={improve}  maintain={maintain}")
    print()
    print(f"  molecules considered        {len(rows)}")
    print(f"  reached a front             {len(front)}")
    print(f"  rejected by the pre-filter  {len(rejected)}")
    print(f"  compliant on all maintain   {len(ok)}  ({len(ok) / len(rows) * 100:.1f}%)")
    print(f"  generations                 {len(gens)}  (first {gens[0]}, last {gens[-1]})")
    print(f"  new molecules per generation"
          f"  mean {len(rows) / max(1, len(gens)):.1f}")
    print()

    phis = np.array([r["phi"] for r in rows])
    print(f"  Phi over everything   mean {phis.mean():.3f}   max {phis.max():.3f}")
    if front:
        fphi = np.array([r["phi"] for r in front])
        print(f"  Phi on the front      mean {fphi.mean():.3f}   max {fphi.max():.3f}")

    # Hypervolume over the final front, and over everything, for comparison.
    hv_all = hypervolume([[r["scaled"][a] for a in axes] for r in rows])
    hv_front = hypervolume([[r["scaled"][a] for a in front[0]["scaled"]] for r in front]) if front else 0.0
    print(f"  hypervolume (front)   {hv_front:.4f}")
    print(f"  hypervolume (all)     {hv_all:.4f}")
    print()

    # The FA-style report: filter to compliant molecules, then look at the improve axes.
    print(f"  FA-style report: compliant molecules only, top {args.top} by improve-axis sum")
    if not ok:
        print("    none compliant - nothing to report")
    else:
        key = lambda r: sum(r["scaled"][a] for a in improve)
        best = sorted(ok, key=key, reverse=True)[:args.top]
        for a in improve:
            vals = np.array([r["raw"][a] for r in best])
            print(f"    {a:<7} mean {vals.mean():8.3f}   best {vals.max():8.3f}")
        print(f"    (of {len(ok)} compliant molecules)")
        print()
        print(f"  best compliant molecule:")
        b = best[0]
        print(f"    {b['smi']}")
        print("    " + "  ".join(f"{a}={b['raw'][a]:.3f}" for a in axes)
              + f"   Phi={b['phi']:.3f}  gen {b['gen']}  origin {b['origin']}")

    if args.json:
        payload = {
            "output_dir": args.output_dir, "improve": improve, "maintain": maintain,
            "n_considered": len(rows), "n_front": len(front), "n_rejected": len(rejected),
            "n_compliant": len(ok), "generations": len(gens),
            "phi_mean": float(phis.mean()), "phi_max": float(phis.max()),
            "hv_front": hv_front, "hv_all": hv_all,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
