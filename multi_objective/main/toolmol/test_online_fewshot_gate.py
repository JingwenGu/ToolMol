"""The online few-shot beam search honours --feasibility_gate. Run from multi_objective/ as a
script (the pool tests spawn worker processes, which on Windows must re-import this file):

    python main/toolmol/test_online_fewshot_gate.py

A candidate that builds an azete must be scored with the gate off and pruned with it on, at
every level (_eval_one is the one choke point every beam candidate passes through), and the
prune must be tallied in the gate_log. A benign candidate must be untouched either way.
"""
import os
import sys
from collections import Counter
from multiprocessing import Pool

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(REPO)
sys.path.insert(0, REPO)
os.environ.setdefault("OPENAI_API_KEY", "unused-by-this-test")

import main.toolmol.online_fewshot as of
from main.feasibility_gate import check

START = "c1ccncc1"                   # pyridine
AZETE, BENIGN = "C1=NC=C1", "C"      # add_substructure fragments
failures = []


def expect(cond, msg):
    print(f"  {'pass' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


def task(fragment):
    return (START, None, 0.0, 'add_substructure', {'idx': 0, 'substructure': fragment, 'bond': 'single'})


def main():
    print("direct _eval_one (in-process):")
    of._pool_init('off')
    r_off = of._eval_one(task(AZETE))
    expect(r_off is not None and 'gated' not in r_off and check(r_off['smi'])[0] == 'fatal',
           f"gate off: azete candidate is scored ({r_off and r_off.get('smi')})")
    of._pool_init('fatal')
    r_on = of._eval_one(task(AZETE))
    expect(r_on is not None and 'gated' in r_on and 'smi' not in r_on,
           f"gate fatal: same candidate is pruned ({r_on})")
    r_ok = of._eval_one(task(BENIGN))
    expect(r_ok is not None and 'gated' not in r_ok and check(r_ok['smi'])[0] == 'ok',
           "gate fatal: benign candidate still scored")
    of._pool_init('off')             # leave module state as found

    print("\nthrough a worker pool (_expand; the mode reaches workers via initargs):")
    for mode in ('off', 'fatal'):
        log = Counter()
        pool = Pool(processes=2, initializer=of._pool_init, initargs=(mode,))
        try:
            res = of._expand(pool, START, None, 0.0, [AZETE, BENIGN], gate_log=log)
        finally:
            pool.close()
            pool.join()
        n_bad = sum(check(r['smi'])[0] == 'fatal' for r in res)
        n_pruned = log.get('(candidates pruned)', 0)
        print(f"        mode={mode}: {len(res)} candidates returned, {n_bad} infeasible, {n_pruned} pruned")
        if mode == 'off':
            expect(n_bad > 0 and n_pruned == 0, "off: azete candidates survive, nothing tallied")
        else:
            expect(n_bad == 0 and n_pruned > 0, "fatal: no infeasible candidate returned; prunes tallied")
            expect(any('antiaromatic 4-ring' in k for k in log), "fatal: prune reason recorded in gate_log")

    print()
    if failures:
        for f in failures:
            print("FAILURE:", f)
        return 1
    print("beam-search gate behaves as expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
