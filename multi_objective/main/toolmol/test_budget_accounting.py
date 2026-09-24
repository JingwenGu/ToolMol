"""Regression test for the scoring-order defect found in RSI iteration 3.

select_pareto_front() reads obj.raw(smi) directly instead of going through score_smi, so a
candidate it rejects is evaluated by the oracle and then leaves no trace: not in mol_buffer,
not in the results file, not counted against max_oracle_calls. The generation loop must
therefore score every candidate before computing the front.

The two assertions below are the properties that were violated:

  1. a dominated molecule is still recorded (the deliberate negatives of a controlled
     comparison are dominated by construction, so this is where a run's information goes);
  2. the budget counter advances by the number of distinct molecules evaluated, so a
     generation whose offspring all lose still moves the run toward its stopping condition.

Run: python -m main.toolmol.test_budget_accounting
"""
import tempfile
from types import SimpleNamespace

from rdkit import Chem, rdBase

from main.toolmol.oracle import ToolMolOracle

rdBase.DisableLog('rdApp.error')

# A clear winner, plus a molecule dominated on every objective at once. The dominated one is
# the molecule the defect used to discard.
WINNER = 'N#Cc1ccc(Nc2nccc(-c3ccccc3)n2)cc1'
DOMINATED = 'CC[C@]1(C)C[C@]2(CCO1)C[NH+]=C(N)N2C1CCCCCC1'


def _oracle(max_calls=50):
    # __call__ writes a results file on every invocation, so it needs a real directory.
    args = SimpleNamespace(max_obj=['jnk3', 'qed'], min_obj=['sa'], max_oracle_calls=max_calls,
                           freq_log=10 ** 9, output_dir=tempfile.mkdtemp(), task_label=None)
    oracle = ToolMolOracle(args=args)
    oracle.assign_evaluator(args)
    return oracle


def _generation(oracle, candidate_smi):
    """The ordering main/toolmol/run.py uses, reproduced exactly."""
    oracle(candidate_smi)
    oracle.clean_buffer()
    front = oracle.select_pareto_front(candidate_smi)
    oracle([Chem.MolToSmiles(m) for m in front])
    return front


def main():
    oracle = _oracle()
    candidates = [WINNER, DOMINATED]

    front = _generation(oracle, candidates)
    front_smi = {Chem.MolToSmiles(m) for m in front}

    assert WINNER in front_smi, 'winner should be on the front'
    assert DOMINATED not in front_smi, (
        'test is vacuous unless the second molecule is actually dominated - pick another')

    recorded = oracle.all_molecules()
    assert DOMINATED in recorded, (
        'REGRESSION: a dominated candidate was evaluated and then discarded unrecorded. '
        'Score every candidate before select_pareto_front().')
    assert len(recorded) == 2, f'expected both molecules recorded, got {sorted(recorded)}'

    # Budget must reflect molecules evaluated, not molecules that survived.
    assert oracle.finish is False
    before = len(oracle.all_molecules())
    _generation(oracle, candidates + ['c1ccc(Nc2nccc(-c3ccccc3)n2)cc1'])
    after = len(oracle.all_molecules())
    assert after == before + 1, f'budget should advance by the one new molecule, {before}->{after}'

    # And a generation of nothing but losers still moves the counter.
    losers = ['CCCCCCO', 'CCCCCCCCN']
    before = len(oracle.all_molecules())
    _generation(oracle, [WINNER] + losers)
    after = len(oracle.all_molecules())
    assert after == before + len(losers), (
        f'a generation whose offspring are all dominated must still consume budget, '
        f'{before}->{after}')

    print(f'ok - {after} molecules recorded, none discarded')


if __name__ == '__main__':
    main()
