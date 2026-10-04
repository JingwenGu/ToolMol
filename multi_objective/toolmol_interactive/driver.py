"""Launch one ToolMol run with an interactive operator standing in for the LLM.

This is a harness around the unmodified pipeline, not a reimplementation of it: it builds
the same argv ``multi_objective/run.py`` takes, points ToolMolAgent at the ``interactive``
backend (agent.py), and calls ``run.main()``. Sampling, the toolbox, the agent loop, Pareto
selection and the GA are all the repo's own code.

Two things are instrumented from out here rather than by editing the pipeline:

1. **Oracle-call ledger / hard budget.** ``main.toolmol.objectives.Objective.raw`` is wrapped
   with a per-SMILES cache that counts *distinct molecules scored* and appends every raw
   objective vector to ``oracle_ledger.jsonl``. One "oracle call" = one molecule scored on
   all objectives, the PMO convention. The wrapper is what enforces ``--budget``: once the
   budget is spent, a molecule never seen before is given the worst possible raw value for
   every objective instead of being evaluated (the same "you are out of budget, score 0"
   convention ``pareto_optimizer.Oracle.score_smi`` already uses), and
   ``Oracle.finish`` is redefined to read the ledger so the GA loop stops at the end of the
   generation that exhausts the budget - after that generation's front has been scored and
   saved, never mid-generation.

   The cache matters for correctness of the count as well as for speed: the repo's
   ``select_pareto_front`` re-scores the whole live population every generation, which would
   otherwise charge for the same molecule many times over.

2. **Episode boundaries in the log.** Only ``print`` statements the pipeline already emits.

Usage:
    python toolmol_interactive/driver.py --seed 1 --budget 50
"""

from __future__ import print_function

import argparse
import json
import os
import sys
import time

# Run from multi_objective/ so that `import main...`, the repo-local oracle/ pickles that
# PyTDC looks for next to the CWD, and data/zinc.tab all resolve the way run.py expects.
HERE = os.path.dirname(os.path.realpath(__file__))
MULTI_OBJ = os.path.dirname(HERE)
os.chdir(MULTI_OBJ)
sys.path.insert(0, MULTI_OBJ)


# Raw value that rescales to 0 for each objective, used for molecules requested after the
# budget is spent. Derived from objectives.OBJECTIVE_BOUNDS rather than hardcoded, so a
# minimize-type objective (whose bound_low is its *worst* raw value) is handled correctly.
def _worst_raw(objective):
    return objective.bound_low


class OracleLedger:
    """Per-SMILES cache + distinct-molecule counter + jsonl log, with a hard cap."""

    def __init__(self, budget, path):
        self.budget = budget
        self.path = path
        self.cache = {}          # smiles -> {objective_name: raw_value}
        self.order = []          # smiles, in the order each was first scored
        self.n_objectives = None  # set on first complete record
        self._logged = set()
        open(self.path, 'a').close()

    @property
    def n_scored(self):
        return len(self.order)

    @property
    def exhausted(self):
        return self.n_scored >= self.budget

    def lookup(self, smiles, name):
        rec = self.cache.get(smiles)
        if rec is None:
            return None
        return rec.get(name, None)

    def admits(self, smiles):
        """True if `smiles` may be scored: either already paid for, or budget remains."""
        return smiles in self.cache or not self.exhausted

    def record(self, smiles, name, value, objective_names):
        if smiles not in self.cache:
            self.cache[smiles] = {}
            self.order.append(smiles)
        self.cache[smiles][name] = value
        rec = self.cache[smiles]
        if smiles not in self._logged and all(n in rec for n in objective_names):
            self._logged.add(smiles)
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "call": self.order.index(smiles) + 1,
                    "smiles": smiles,
                    "raw": {n: rec[n] for n in objective_names},
                }) + "\n")


def install_ledger(budget, ledger_path):
    from main.toolmol import objectives as objectives_mod
    from main import pareto_optimizer

    ledger = OracleLedger(budget, ledger_path)
    # Objective names in build_objectives' order, filled in when the oracle is constructed.
    names = {"all": []}

    original_build = objectives_mod.build_objectives

    def build_objectives(obj_names):
        objs = original_build(obj_names)
        names["all"] = [o.name for o in objs]
        return objs

    objectives_mod.build_objectives = build_objectives
    # main.toolmol.oracle imported build_objectives by value at import time, so rebind
    # there too rather than relying on attribute lookup through the module.
    from main.toolmol import oracle as toolmol_oracle_mod
    toolmol_oracle_mod.build_objectives = build_objectives

    def raw(self, smi):
        hit = ledger.lookup(smi, self.name)
        if hit is not None:
            return hit
        if not ledger.admits(smi):
            # Out of budget and never scored: refuse the call, hand back the worst value.
            return _worst_raw(self)
        value = float(self.evaluator(smi))
        ledger.record(smi, self.name, value, names["all"] or [self.name])
        return value

    objectives_mod.Objective.raw = raw

    # The GA loop's stopping condition becomes "the budget is gone", checked where the repo
    # already checks it: once per generation, after the front is recomputed and saved.
    pareto_optimizer.Oracle.finish = property(lambda self: ledger.exhausted)

    return ledger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--budget', type=int, default=50,
                        help='distinct molecules that may be scored by the objective oracles')
    parser.add_argument('--run-dir', default=None,
                        help='defaults to toolmol_interactive/runs/seed<N>')
    parser.add_argument('--config', default=os.path.join(HERE, 'hparams_interactive.yaml'))
    parser.add_argument('--max-obj', nargs='+', default=['jnk3', 'qed'])
    parser.add_argument('--min-obj', nargs='+', default=['sa'])
    args = parser.parse_args()

    run_dir = args.run_dir or os.path.join(HERE, 'runs', f'seed{args.seed}')
    ipc_dir = os.path.join(run_dir, 'ipc')
    os.makedirs(ipc_dir, exist_ok=True)
    os.environ['TOOLMOL_INTERACTIVE_DIR'] = ipc_dir
    # agent.py only reads the key for the http backends, but keep the env clean anyway.
    os.environ.setdefault('OPENAI_API_KEY', 'unused-interactive-backend')

    ledger = install_ledger(args.budget, os.path.join(run_dir, 'oracle_ledger.jsonl'))

    with open(os.path.join(run_dir, 'run_config.json'), 'w') as f:
        json.dump({"seed": args.seed, "budget": args.budget, "config": args.config,
                   "max_obj": args.max_obj, "min_obj": args.min_obj,
                   "started": time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)

    sys.argv = [
        'run.py', 'toolmol',
        '--mol_lm', 'ToolMol',
        '--llm_backend', 'interactive',
        '--llm_model', 'claude-opus-5-interactive',
        '--config_default', args.config,
        '--output_dir', run_dir,
        '--max_oracle_calls', str(args.budget),
        '--freq_log', '10',
        '--patience', '100',       # budget, not patience, is what ends these short runs
        '--seed', str(args.seed),
        '--max_obj', *args.max_obj,
        '--min_obj', *args.min_obj,
    ]

    import run as repo_run
    try:
        repo_run.main()
    finally:
        print(f"\n=== ledger: {ledger.n_scored}/{ledger.budget} distinct molecules scored ===",
              flush=True)
        with open(os.path.join(ipc_dir, 'RUN_FINISHED'), 'w') as f:
            f.write(f"{ledger.n_scored}/{ledger.budget}\n")


if __name__ == '__main__':
    main()
