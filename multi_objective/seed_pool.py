"""Build a task-specific seed pool, by C-MuMO's own test-set selection rule.

Design section 1 seeds a run from the combination's own molecules rather than from random
ZINC, and C-MuMO defines those precisely (section 2.4): a molecule enters the test set for
a task improving Pi and maintaining Ps if it has **every p in Pi worse than Th_p and every
p in Ps better than Th_p**. That is the whole point of the seed - the improve axes start
sub-optimal, so there is something to optimize, and the maintain axes start compliant, so
there is something to preserve.

Seeding from random ZINC instead asks a vaguer question: on a random draw 71% of molecules
are already compliant on AMP and 76% on LIV, so a "maintain" axis is mostly satisfied by
accident and the improve axes start wherever they happen to start.

    python seed_pool.py --combination BPQ --n 500 --out ../../assets/seeds/BPQ.smi
    python seed_pool.py --improve QED MUT --maintain BBBP CARC --n 500 --out ../../assets/seeds/x.smi

Then:

    run.py ... --combination BPQ --smi_file ../../assets/seeds/BPQ.smi

**Write the pool to the assets directory, which is outside the repository** - two levels up
from here, not one. `../assets` from this directory is the repo root, and a pool written
there gets committed; that happened once. `cmumo_oracles.paths.ASSETS` is the reliable
answer, and the repo's root .gitignore now catches the mistake.

**The output is shuffled**, because the GA takes the FIRST `population_size` lines of a
--smi_file, not a random sample of it (main/toolmol/run.py). An unshuffled pool would make
every run start from the same handful of molecules.

Costs no oracle calls in the budget sense - it scores candidates through the property
evaluator directly, and the cache makes a second combination over the same ZINC sample
nearly free.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmumo_oracles import paths  # noqa: E402
from cmumo_oracles.properties import SPECS, PropertyEvaluator  # noqa: E402
from main.toolmol.combinations import resolve_combination  # noqa: E402


def satisfies(values, improve, maintain):
    """C-MuMO's rule: every improve property worse than Th_p, every maintain property
    better than Th_p."""
    for p in improve:
        spec = SPECS[p]
        better = values[p] >= spec["theta"] if spec["direction"] == "+" else values[p] <= spec["theta"]
        if better:
            return False          # already good: nothing to improve
    for p in maintain:
        spec = SPECS[p]
        better = values[p] >= spec["theta"] if spec["direction"] == "+" else values[p] <= spec["theta"]
        if not better:
            return False          # already bad: nothing to maintain
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combination", default=None, help="a C-MuMO task code (BPQ, ELQ, ...)")
    ap.add_argument("--improve", nargs="+", default=None)
    ap.add_argument("--maintain", nargs="+", default=[])
    ap.add_argument("--n", type=int, default=500, help="molecules to write")
    ap.add_argument("--scan", type=int, default=20000,
                    help="ZINC molecules to score while looking for them")
    ap.add_argument("--source", default=None, help="defaults to the linked data/zinc.tab")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.combination:
        improve, maintain = resolve_combination(args.combination)
    elif args.improve:
        improve, maintain = list(args.improve), list(args.maintain)
    else:
        raise SystemExit("give --combination or --improve/--maintain")
    print(f"improve (must start sub-optimal): {improve}")
    print(f"maintain (must start compliant):  {maintain}")
    for p in improve + maintain:
        spec = SPECS[p]
        print(f"   {p:<6} {spec['direction']}  Th_p = {spec['theta']:g}")

    source = args.source or paths.ZINC
    with open(source) as f:
        pool = [l.strip().strip('"') for l in f][1:]
    random.seed(args.seed)
    scan = random.sample(pool, min(args.scan, len(pool)))
    print(f"\nscanning {len(scan)} molecules from {source}")

    ev = PropertyEvaluator(improve + maintain, cache_path=paths.PROPERTY_CACHE)
    ev.self_check(verbose=False)

    kept, batch = [], 2000
    for i in range(0, len(scan), batch):
        chunk = scan[i:i + batch]
        ev.prefetch(chunk)
        for smi in chunk:
            values = ev.get(smi)
            if values and satisfies(values, improve, maintain):
                kept.append(smi)
        print(f"  {min(i + batch, len(scan)):>6}/{len(scan)} scanned, {len(kept)} match", flush=True)
        if len(kept) >= args.n * 3:
            break
    ev.close()

    if not kept:
        raise SystemExit(
            "no molecule satisfies the rule. Either the combination is too strict for this "
            "source, or an improve axis is one almost everything already passes - check the "
            "occupancy table in PROPERTIES.md.")
    if len(kept) < args.n:
        print(f"\nWARNING: only {len(kept)} of the requested {args.n} molecules match. "
              f"Raise --scan, or accept the smaller pool and say so with the results.")

    random.shuffle(kept)       # the GA takes the FIRST population_size lines
    kept = kept[:args.n]
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        for smi in kept:
            f.write(smi + "\n")
    print(f"\nwrote {len(kept)} molecule(s) to {args.out}")


if __name__ == "__main__":
    main()
