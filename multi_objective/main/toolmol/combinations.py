"""The C-MuMO task codes, and the rule that splits each into improve and maintain axes.

The rule is the one the calibration forced (CMUMO_PORT_DESIGN.md, "What the oracle install
settled"): **the two lowest-ZINC-occupancy properties in a combination become the improve
axes, the rest are maintain.** Occupancy is the fraction of a ZINC sample already on the
good side of Th_p, so a low-occupancy property is one almost nothing satisfies - which is
what makes it a target rather than something to hold.

Occupancies below are measured, from 1000 random ZINC-250K molecules (seed 0); the full
table is in cmumo/README.md. They are frozen here so a run is reproducible without
rescoring ZINC, and `calibrate_m.py` recomputes them from the pool when it runs.

HIA is excluded everywhere: it saturates near 1.0 on drug-like input (median 1.000) and is
10% active at best, so it buys a Pareto axis and carries almost no signal.
"""

ZINC_OCCUPANCY = {
    "QED": 0.07, "DRD2": 0.00, "hERG": 0.22, "PlogP": 0.24, "MUT": 0.41,
    "BBBP": 0.48, "AMP": 0.66, "CARC": 0.68, "LIV": 0.74, "HIA": 0.88,
}

EXCLUDED = {"HIA"}

COMBINATIONS = {
    "BPQ":   ["BBBP", "PlogP", "QED"],
    "ELQ":   ["hERG", "LIV", "QED"],
    "ACEP":  ["AMP", "CARC", "hERG", "PlogP"],
    "BDPQ":  ["BBBP", "DRD2", "PlogP", "QED"],
    "BCMQ":  ["BBBP", "CARC", "MUT", "QED"],
    "HLMPQ": ["HIA", "LIV", "MUT", "PlogP", "QED"],
}

N_IMPROVE = 2


def resolve_combination(code):
    """code -> (improve, maintain). Raises on an unknown code rather than guessing."""
    key = code.upper()
    if key not in COMBINATIONS:
        raise SystemExit(
            f"unknown combination {code!r}; known: {sorted(COMBINATIONS)}")
    props = [p for p in COMBINATIONS[key] if p not in EXCLUDED]
    order = sorted(props, key=lambda p: ZINC_OCCUPANCY[p])
    return order[:N_IMPROVE], sorted(order[N_IMPROVE:])
