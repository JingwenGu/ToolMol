"""End-to-end smoke test of the generation loop with the LLM stubbed out.

Exercises main/toolmol/run.py::_optimize for real - mating pool, offspring, Pareto front,
snapshots, budget check - with edit_pair replaced by the repo's own Graph-GA operators, so it
runs in seconds and needs no model. What it checks is the property the scoring-order fix was
made for: the results file must contain every distinct molecule the run evaluated, not only the
ones that survived selection.

Run: python -m main.toolmol.test_generation_loop
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import yaml
from rdkit import Chem, rdBase

# main/toolmol/mutate.py does `from utils import ...`, a sibling import that only resolves with
# this directory on sys.path. The real entry point (multi_objective/run.py) appends it before
# importing the optimizer; do the same here rather than rewriting the import.
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import main.toolmol.crossover as co  # noqa: E402
import main.toolmol.mutate as mu  # noqa: E402
from main.toolmol.run import GB_GA_Optimizer  # noqa: E402

rdBase.DisableLog('rdApp.error')

MAX_CALLS = 34
POP = 12
OFFSPRING = 6


class _StubAgent:
    """Stands in for ToolMolAgent: one Graph-GA child per pair, deterministic enough to finish."""

    max_steps = 1
    goal_description = ''
    score_detail = None
    is_duplicate = None
    population_summary = None

    def edit_pair(self, m0, s0, m1, s1, mutation_rate):
        for _ in range(40):
            child = co.crossover(m0, m1)
            if child is not None:
                child = mu.mutate(child, 1.0) or child
            if child is not None and Chem.MolToSmiles(child):
                return child
        return None


def main():
    out = tempfile.mkdtemp()
    # BaseOptimizer.load_smiles_from_file is commented out in this repo, so smi_file must be
    # None - real runs draw the starting pool from the cached ZINC set, as here.
    np.random.seed(3)

    args = SimpleNamespace(
        max_obj=['jnk3', 'qed'], min_obj=['sa'], max_oracle_calls=MAX_CALLS,
        freq_log=10 ** 9, output_dir=out, smi_file=None, seed=3, patience=1000,
        n_jobs=1, log_results=False, mol_lm=None, resume_from=None, task='simple',
        llm_model=None, llm_base_url=None, llm_api_key_env=None, llm_extra_body=None,
        llm_backend=None, llm_device_map=None, llm_max_new_tokens=None,
        llm_io_dir=None, domain_brief=None, few_shot_file=None,
    )

    opt = GB_GA_Optimizer(args=args)
    opt.mol_lm = _StubAgent()
    opt.mol_lm.is_duplicate = opt._is_duplicate
    opt.mol_lm.score_detail = opt._score_detail
    opt.mol_lm.population_summary = opt._population_summary
    opt.oracle.task_label = 'smoke'

    opt._optimize({'population_size': POP, 'offspring_size': OFFSPRING, 'mutation_rate': 0.5,
                   'k': 10.0, 'max_steps': 1, 'share_radius': 0.4, 'max_parent_share': 0.25})

    recorded = opt.oracle.all_molecules()
    n = len(recorded)

    # The budget check runs once per generation, after a full batch, so the final generation may
    # overshoot - but it must terminate, which under the old ordering it could fail to do.
    assert n >= MAX_CALLS, f'run stopped at {n} molecules, below its own budget of {MAX_CALLS}'
    assert n <= MAX_CALLS + OFFSPRING, f'overshot by more than one generation: {n}'

    with open(os.path.join(out, 'results_smoke.yaml')) as f:
        saved = yaml.safe_load(f)
    assert len(saved) == n, f'results file holds {len(saved)} of {n} evaluated molecules'

    # Discovery indices must be a contiguous run-global order, or a trajectory cannot be
    # reconstructed from the file afterwards.
    idx = sorted(v[1] for v in saved.values())
    assert idx == list(range(1, len(idx) + 1)), f'discovery indices not contiguous: {idx[:12]}...'

    print(f'ok - {n} molecules evaluated, {len(saved)} recorded, indices 1..{len(idx)} contiguous')


if __name__ == '__main__':
    main()
