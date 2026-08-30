import tdc


class Objective:
    """A single optimization objective, rescaled to [0,1] where 1 is always better.

    bound_low/bound_high are the raw values that map to rescaled 0 and 1 respectively.
    For a maximize objective already in [0,1] (e.g. QED), bound_low=0.0, bound_high=1.0.
    For a minimize objective (e.g. SA, naturally 1-10, lower better), bound_high < bound_low
    (e.g. bound_low=10.0, bound_high=1.0) so the flip is handled by the formula itself.
    """

    def __init__(self, name, evaluator, bound_low, bound_high):
        self.name = name
        self.evaluator = evaluator
        self.bound_low = bound_low
        self.bound_high = bound_high

    def raw(self, smi):
        return float(self.evaluator(smi))

    def rescaled(self, raw_value):
        value = (raw_value - self.bound_low) / (self.bound_high - self.bound_low)
        return min(1.0, max(0.0, value))


# Phase 1-3 stand-ins for ToolMol's real (affinity, QED, SA) objective set, using MOLLEO's
# existing cheap oracles so the toolbox/agent/GA pipeline can be validated before Boltz-2.
# (bound_low, bound_high): the raw value mapping to rescaled 0, and to rescaled 1.
OBJECTIVE_BOUNDS = {
    'qed': (0.0, 1.0),
    'sa': (10.0, 1.0),
    'jnk3': (0.0, 1.0),
    # Phase 4 (deferred): 'affinity': (0.0, -13.0), per the paper's footnote 2.
}


def build_objectives(names):
    objectives = []
    for name in names:
        if name not in OBJECTIVE_BOUNDS:
            raise ValueError(f"No OBJECTIVE_BOUNDS entry for objective '{name}' - add one to objectives.py")
        bound_low, bound_high = OBJECTIVE_BOUNDS[name]
        objectives.append(Objective(name, tdc.Oracle(name=name), bound_low, bound_high))
    return objectives
