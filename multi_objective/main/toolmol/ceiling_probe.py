"""Cheap undirected search, to be run BETWEEN iterations - not during one.

Why this exists
---------------
Four RSI iterations and roughly two hundred agent-designed molecules never left the first
scaffold they found. Not because the agent was searching badly, but because every single-variable
step out of that scaffold looks fatal, and the domain brief - built from exactly those
single-variable measurements - said so. A region reachable only by violating two rules at once is
invisible to a process that changes one thing at a time.

This script found that region in 4,000 evaluations and about four minutes, and it beat the best
agent result of four iterations (Phi 2.538 -> 2.662).

The division of labour that follows from this:

  cheap search   finds WHERE to look. Undirected, no chemistry knowledge, no respect for the
                 brief's rules, happily makes molecules that violate three of them at once.
  agent episodes find out WHAT IS THERE. Controlled comparison - hold everything constant, move
                 one thing, attribute the difference. Every reliable number in the brief came
                 from an episode; none of the regions did.

Run this after each iteration, before writing the brief. If it finds something well above the
agent's best, the next iteration's job is to understand that region rather than to search.

What NOT to do with it
----------------------
Do not feed its output molecules to the agent mid-run. That makes the run a transcription
exercise and destroys any claim that the harness transferred knowledge - the failure mode that
forced iteration 1 to be abandoned. Use it to decide what the brief should say about *method*,
and let the run rediscover the specifics.

Usage:
    python -m main.toolmol.ceiling_probe [n_evals] [pop] [out.csv] [--seed SMILES ...]
"""
import argparse
import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from tdc import Oracle
from tdc.generation import MolGen

import main.toolmol.crossover as co
import main.toolmol.mutate as mu

RDLogger.DisableLog('rdApp.*')


def make_scorer(max_obj=('jnk3', 'qed'), min_obj=('sa',)):
    """Phi, matching main/toolmol/oracle.py: objectives rescaled to [0,1] and summed.

    QED is computed directly rather than through TDC, which is much faster and identical.
    SA is rescaled (10 - sa) / 9, the same transform the run's oracle applies.
    """
    jnk3 = Oracle('jnk3') if 'jnk3' in max_obj else None
    sa = Oracle('SA') if 'sa' in min_obj else None
    cache = {}

    def phi(smi):
        if smi in cache:
            return cache[smi]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            cache[smi] = (-1.0, None)
            return cache[smi]
        try:
            parts = {}
            total = 0.0
            if 'qed' in max_obj:
                parts['qed'] = QED.qed(mol)
                total += parts['qed']
            if jnk3 is not None:
                parts['jnk3'] = jnk3(smi)
                total += parts['jnk3']
            if sa is not None:
                parts['sa'] = sa(smi)
                total += (10 - parts['sa']) / 9
        except Exception:
            cache[smi] = (-1.0, None)
            return cache[smi]
        cache[smi] = (total, parts)
        return cache[smi]

    return phi, cache


def run(n_evals=4000, pop=120, seeds=(), rng_seed=0):
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    phi, cache = make_scorer()

    library = MolGen(name='ZINC').get_data()['smiles'].tolist()
    random.shuffle(library)

    population = [Chem.MolFromSmiles(s) for s in seeds]
    population = [m for m in population if m is not None]
    population += [Chem.MolFromSmiles(s) for s in library[:max(0, pop - len(population))]]
    population = [m for m in population if m is not None]

    t0, gen = time.time(), 0
    while len(cache) < n_evals:
        gen += 1
        scored = []
        for m in population:
            smi = Chem.MolToSmiles(m)
            p, _ = phi(smi)
            if p > 0:
                scored.append((p, smi, m))
        scored.sort(key=lambda x: -x[0])
        population = [x[2] for x in scored[:pop]]

        if gen % 5 == 1:
            print(f'gen {gen:4d}  evals {len(cache):6d}  best {scored[0][0]:.3f}  '
                  f'{scored[0][1][:58]}  {time.time() - t0:.0f}s', flush=True)

        n = len(population)
        w = np.array([1.0 / (i + 2) for i in range(n)])
        w /= w.sum()
        children = []
        for _ in range(pop):
            i, k = np.random.choice(n, 2, p=w)
            child = co.crossover(population[i], population[k])
            if child is not None and random.random() < 0.6:
                child = mu.mutate(child, 1.0) or child
            if child is not None:
                children.append(child)
        population = population + children

    rows = rank(cache)
    print(f'\n{len(cache)} evaluations in {time.time() - t0:.0f}s')
    return rows


def rank(cache):
    """Best-first rows of (phi, parts, smiles) from a scorer cache.

    Sorts on Phi alone, and that is the whole point of the function existing. The rows carry a
    dict of per-objective parts, so a plain `sorted(..., reverse=True)` on the tuple falls through
    to comparing those dicts whenever two molecules tie on Phi, raising TypeError - after the
    entire search has run and with the results held only in memory.

    Ties are near-certain past a few thousand evaluations and essentially impossible in a small
    smoke test, so this failed on every real run and passed every quick check. It arrived when
    this was refactored out of a scratch script whose tuple was all scalars. Kept as a named
    function so a regression test can reach it without running a search; see
    test_ceiling_probe_rank.py.
    """
    return sorted(((v[0], v[1], k) for k, v in cache.items() if v[1]),
                  key=lambda r: r[0], reverse=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('n_evals', nargs='?', type=int, default=4000)
    ap.add_argument('pop', nargs='?', type=int, default=120)
    ap.add_argument('out', nargs='?', default='ceiling_probe.csv')
    ap.add_argument('--seed', action='append', default=[],
                    help='SMILES to seed the population with, e.g. the agent run\'s best. '
                         'Repeatable. Seeding biases the probe toward refining what you already '
                         'have; omit it entirely for an unbiased look at what else exists.')
    ap.add_argument('--rng', type=int, default=0,
                    help='RNG seed. Run several unseeded probes with different values to test '
                         'whether a ceiling is real: independent restarts converging on the same '
                         'region is evidence the region is the optimum, whereas one seeded run '
                         'converging on its own starting point is evidence of nothing.')
    args = ap.parse_args()

    rows = run(args.n_evals, args.pop, tuple(args.seed), rng_seed=args.rng)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['phi', 'jnk3', 'qed', 'sa', 'smiles'])
        for total, parts, smi in rows:
            w.writerow([f'{total:.4f}', parts.get('jnk3', ''), f"{parts.get('qed', 0):.4f}",
                        parts.get('sa', ''), smi])

    print('\ntop 15:')
    for total, parts, smi in rows[:15]:
        print(f'  {total:6.3f}  jnk3 {parts.get("jnk3", 0):4.2f}  qed {parts.get("qed", 0):5.3f}  '
              f'sa {parts.get("sa", 0):4.2f}  {smi}')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
