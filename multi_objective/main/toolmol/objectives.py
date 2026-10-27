"""C-MuMO objective set: improve axes rescaled linearly, maintain axes as a hinge.

Drop-in replacement for the tdc-backed objectives.py. Same `Objective` surface (name,
bound_low, bound_high, raw(smi), rescaled(raw)), same `OBJECTIVE_BOUNDS` mapping, same
`build_objectives(names)` signature - so `ToolMolOracle.assign_evaluator`,
`select_pareto_front` and `online_fewshot._pool_init` all pick it up unchanged.

**Direction lives in the property table, not in the CLI.** `rescaled()` always returns
1 = best, for lower-is-better properties too, so every property goes in `--max_obj` and
`--min_obj` stays empty. Passing a property in `--min_obj` would double-invert it.

**Improve vs maintain comes from the environment**, `CMUMO_IMPROVE` / `CMUMO_MAINTAIN`,
set by run.py. Not module state: `_pool_init` rebuilds objectives in a worker process that
may be spawned rather than forked, and an env var survives that while a module global does
not.

Formulas, per CMUMO_PORT_DESIGN.md section 2:

    improve     f = (raw - low) / (high - low), clipped to [0,1], inverted if lower-is-better
    maintain    f = 1                     for d <= M          (plateau, extended M past Th_p)
                f = min(1, M / d)         for M < d <= C*M    (decay)
                f = 0                     for d > C*M         (clamp)
                where d = Th_p - raw  (higher-is-better)  or  raw - Th_p  (lower-is-better)

M = alpha * D_p with the calibrated per-property alpha; C = 5. Both were measured, not
chosen by taste - see the design doc's "The result".
"""
import os

from cmumo_oracles import paths
from cmumo_oracles.properties import SPECS, PropertyEvaluator

# Calibrated 2026-10-26 on 1000 ZINC molecules. alpha = 1 keeps the penalty active on
# 27-52% of the pool; CARC and LIV need 0.25 because their D_p is large enough relative to
# [0,1] that M = D_p makes them inert (2% and 0% active).
CALIBRATED_ALPHA = {
    "BBBP": 1.0, "AMP": 1.0, "MUT": 1.0, "PlogP": 1.0, "QED": 1.0,
    "hERG": 1.0, "DRD2": 1.0,
    "CARC": 0.25, "LIV": 0.25,
    # HIA is excluded from runs (saturated near 1.0), but keep an entry so a deliberate
    # experiment with it does not crash.
    "HIA": 0.25,
}
CLAMP_C = 5.0

# Bounds for the improve rescale. Probabilities use their natural (0,1). PlogP is unbounded
# and gets the treatment ToolMol gave dG: a clamp wide enough that nothing observed reaches
# it (1000 ZINC molecules spanned -6.66 to +3.38).
IMPROVE_BOUNDS = {"PlogP": (-10.0, 5.0)}

# Every property is a legal objective name; the per-name bounds here are what the improve
# path uses and what any code inspecting OBJECTIVE_BOUNDS expects to find.
OBJECTIVE_BOUNDS = {name: IMPROVE_BOUNDS.get(name, (0.0, 1.0)) for name in SPECS}

_EVALUATOR = None


def evaluator():
    """The shared batched/cached PropertyEvaluator. One per process; the ADMET-AI worker
    starts on first use, so a run that touches no ADMET property never loads torch."""
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = PropertyEvaluator(ALL_CONFIGURED or None,
                                       cache_path=paths.PROPERTY_CACHE)
    return _EVALUATOR


def prefetch(smiles_list):
    """Batch the whole list before any per-molecule scoring. The GA calls this once per
    generation; without it every raw() is a separate neural-net forward pass."""
    evaluator().prefetch(smiles_list)


def _configured(var):
    raw = os.environ.get(var, "")
    return [p for p in (x.strip() for x in raw.split(",")) if p]


IMPROVE = _configured("CMUMO_IMPROVE")
MAINTAIN = _configured("CMUMO_MAINTAIN")
ALL_CONFIGURED = IMPROVE + MAINTAIN


def configure(improve, maintain):
    """Set the split for this process and every child process. Call before
    build_objectives()."""
    global IMPROVE, MAINTAIN, ALL_CONFIGURED, _EVALUATOR
    IMPROVE, MAINTAIN = list(improve), list(maintain)
    ALL_CONFIGURED = IMPROVE + MAINTAIN
    os.environ["CMUMO_IMPROVE"] = ",".join(IMPROVE)
    os.environ["CMUMO_MAINTAIN"] = ",".join(MAINTAIN)
    _EVALUATOR = None  # rebuild with the new property list


class Objective:
    """Unchanged surface. `rescaled()` is overridden by the two subclasses."""

    def __init__(self, name, evaluator, bound_low, bound_high):
        self.name = name
        self.evaluator = evaluator
        self.bound_low = bound_low
        self.bound_high = bound_high

    def raw(self, smi):
        value = self.evaluator(smi)
        # An unparseable SMILES must score 0, not crash the GA's scoring loop and not
        # score *well*. Which raw value is "worst" depends on the direction and on whether
        # this is an improve or a maintain axis, so each subclass says.
        if value is None:
            return self.worst_raw()
        return float(value)

    def worst_raw(self):
        return self.bound_low

    def rescaled(self, raw_value):
        value = (raw_value - self.bound_low) / (self.bound_high - self.bound_low)
        return min(1.0, max(0.0, value))


class ImproveObjective(Objective):
    """Linear rescale to [0,1], 1 = best, direction from SPECS."""

    kind = "improve"

    def __init__(self, name, evaluator):
        low, high = OBJECTIVE_BOUNDS[name]
        super().__init__(name, evaluator, low, high)
        self.direction = SPECS[name]["direction"]

    def worst_raw(self):
        # rescaled() inverts for "-", so the worst raw value is the opposite bound.
        return self.bound_high if self.direction == "-" else self.bound_low

    def rescaled(self, raw_value):
        value = super().rescaled(raw_value)
        return (1.0 - value) if self.direction == "-" else value


class MaintainObjective(Objective):
    """The hinge. `violation(raw)` is how far past the threshold a molecule is, in units of
    M, which the pre-filter and the agent prompt both read."""

    kind = "maintain"

    def __init__(self, name, evaluator):
        super().__init__(name, evaluator, 0.0, 1.0)
        spec = SPECS[name]
        self.direction = spec["direction"]
        self.theta = spec["theta"]
        self.delta = spec["delta"]
        self.M = CALIBRATED_ALPHA[name] * self.delta
        self.C = CLAMP_C

    def worst_raw(self):
        # Past the clamp, so rescaled() returns 0.
        past = self.C * self.M + 1.0
        return (self.theta - past) if self.direction == "+" else (self.theta + past)

    def distance(self, raw_value):
        """Raw distance past the threshold; <= 0 means compliant."""
        if self.direction == "+":
            return self.theta - raw_value
        return raw_value - self.theta

    def violation(self, raw_value):
        """Distance past the threshold in units of M. 0 = on the plateau."""
        d = self.distance(raw_value)
        return 0.0 if d <= self.M else d / self.M

    def rescaled(self, raw_value):
        d = self.distance(raw_value)
        if d <= self.M:
            return 1.0
        if d > self.C * self.M:
            return 0.0
        return min(1.0, self.M / d)


def _make_evaluator(name):
    """One callable per property, reading through the shared cache."""
    def _eval(smi):
        values = evaluator().get(smi)
        return None if values is None else values[name]
    return _eval


def configured_objectives():
    """The objective set this process was configured with, improve axes first.

    The beam search used to hardcode `build_objectives(['jnk3', 'qed', 'sa'])`, which
    raises against this property table. It needs whatever set the run was configured with,
    and it runs in pool workers that may be spawned rather than forked - so the split has
    to come from the environment, which `configure()` writes and children inherit.
    """
    if not ALL_CONFIGURED:
        raise RuntimeError(
            "no objective set configured: CMUMO_IMPROVE / CMUMO_MAINTAIN are both empty. "
            "run.py sets them via objectives.configure(); set them yourself when calling "
            "the beam search outside a run.")
    return build_objectives(ALL_CONFIGURED)


def build_objectives(names):
    objectives = []
    for name in names:
        if name not in SPECS:
            raise ValueError(
                f"unknown property {name!r}; known: {sorted(SPECS)}")
        if name in MAINTAIN:
            objectives.append(MaintainObjective(name, _make_evaluator(name)))
        elif name in IMPROVE:
            objectives.append(ImproveObjective(name, _make_evaluator(name)))
        else:
            raise ValueError(
                f"property {name!r} is in neither CMUMO_IMPROVE nor CMUMO_MAINTAIN. "
                f"Set them with objectives.configure() or the env vars - the split is "
                f"not inferable from the property name.")
    return objectives
