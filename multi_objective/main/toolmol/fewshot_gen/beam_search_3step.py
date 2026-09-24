"""Non-myopic beam search over 3 sequential tool calls, for the same 10 molecules used to
build the few-shot demonstration library (regret_gpt54_v1.json / PICKS below).

ASSUMPTION (load-bearing, not yet enforced by removing the mismatch it papers over):
crossover_molecules only ever wins the beam at level 1, never level 2/3. This matters because
at every beam level this script generates crossover_molecules candidates against *this node's*
current working molecule combined with the snapshot's parent2 - which differs from the live
agent's actual semantics (confirmed by reading agent.py's _agentic_edit): the live agent always
calls crossover_molecules against the frozen original parent1, never the evolved working
molecule, matching brute_force_gpt54.py's original convention. At level 1 the working molecule
and parent1 are identical by construction, so the mismatch is inert there - but a crossover
step winning at level 2/3 would be scored against the wrong base molecule and silently produce
a wrong path. run_one_molecule() below asserts this can't happen rather than trusting it
silently: a future run on different input molecules that DOES have crossover win beyond level 1
will raise instead of writing a quietly-wrong beam3_results.jsonl entry. If that ever fires,
this needs the real fix (generate crossover candidates against the frozen parent1 at every
level, not the evolving working molecule) before rerunning.

Beam search, not exhaustive: at each level, keep the global top BEAM candidates by delta
(not per-branch), expand all of them at the next level. Not a proof of 3-step optimality -
a branch outside the top BEAM at any level is pruned even if its descendants would be great -
but it explores BEAM independent lines simultaneously rather than one greedy path.
"""
import sys, os, json, time, argparse

# Must happen before numpy (imported transitively via rdkit/main.toolmol.*) ever loads its BLAS
# backend, which reads these once at init and otherwise defaults to one thread pool per process
# sized to the full core count. This script runs --workers separate processes (each importing
# numpy independently via _pool_init) on top of that per-process pool, so without this a
# --workers 16 run alone can try to spawn on the order of 16 x 64 = 1024 OS threads - fine on a
# dedicated SLURM allocation, but enough to exhaust process/thread limits machine-wide on a
# shared login node (observed directly: pthread_create failures for every other user on the
# node, not just this job, until the runaway processes were killed).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
MULTI_OBJECTIVE_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, MULTI_OBJECTIVE_DIR)

REGRET_DIR = HERE

PICKS = [357, 165, 84, 64, 132, 114, 383, 424, 318, 69]

ELEMENTS = ['C', 'N', 'O', 'F', 'Cl', 'Br', 'S']
BONDS = ['single', 'double', 'triple']

# ---- module-level globals, set once per worker process by _pool_init ----
_objectives = None
_group_smiles = None
_toolbox = None
_Chem = None


def _pool_init():
    global _objectives, _group_smiles, _toolbox, _Chem
    from rdkit import Chem as _C
    from main.toolmol import toolbox as _tb
    from main.toolmol.fg_lookup import FUNCTIONAL_GROUPS
    from main.toolmol.objectives import build_objectives
    _Chem = _C
    _toolbox = _tb
    _group_smiles = list(FUNCTIONAL_GROUPS.values())
    _objectives = build_objectives(['jnk3', 'qed', 'sa'])


def _phi_and_raw(smi):
    m = _Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    raw = {}
    total = 0.0
    for obj in _objectives:
        r = obj.raw(smi)
        raw[obj.name] = r
        total += obj.rescaled(r)
    return total, raw


def _ring_free_atom_indices(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if any(not b.IsInRing() for b in a.GetBonds())]


def _gen_candidates(mol, mol2_smi):
    n = mol.GetNumAtoms()
    for idx in range(n):
        for el in ELEMENTS:
            for bond in BONDS:
                yield 'add_atom', {'idx': idx, 'element': el, 'bond': bond}
            yield 'replace_atom', {'idx': idx, 'element': el}
        for frag in _group_smiles:
            for bond in BONDS:
                yield 'add_substructure', {'idx': idx, 'substructure': frag, 'bond': bond}

    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for anchor, branch in ((a1, a2), (a2, a1)):
            yield 'remove_substructure', {'anchor_idx': anchor, 'branch_idx': branch}
            for frag in _group_smiles:
                yield 'replace_substructure', {'anchor_idx': anchor, 'branch_idx': branch, 'new_substructure': frag}

    if mol2_smi:
        mol2 = _Chem.MolFromSmiles(mol2_smi)
        for i1 in _ring_free_atom_indices(mol):
            for i2 in _ring_free_atom_indices(mol2):
                yield 'crossover_molecules', {'idx1': i1, 'idx2': i2}


def _call_tool(tool_name, working_smi, mol2_smi, args):
    if tool_name == 'crossover_molecules':
        return _toolbox.crossover_molecules(working_smi, args['idx1'], mol2_smi, args['idx2'])
    return getattr(_toolbox, tool_name)(working_smi, **args)


def _eval_one(task):
    """task = (working_smi, mol2_smi, baseline_phi, tool_name, args) -> result dict or None"""
    working_smi, mol2_smi, baseline_phi, tool_name, args = task
    try:
        result = _call_tool(tool_name, working_smi, mol2_smi, args)
    except Exception:
        return None
    if not result.success:
        return None
    try:
        out_smi = _Chem.MolToSmiles(result.mol)
    except Exception:
        return None
    phi, raw = _phi_and_raw(out_smi)
    if phi is None:
        return None
    return {'tool': tool_name, 'args': args, 'smi': out_smi, 'phi': phi,
            'delta': phi - baseline_phi, 'raw': raw}


def expand(pool, smi, mol2_smi, baseline_phi, chunksize=200):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    tasks = [(smi, mol2_smi, baseline_phi, t, a) for t, a in _gen_candidates_main(mol, mol2_smi)]
    n_tried = len(tasks)
    results = pool.map(_eval_one, tasks, chunksize=chunksize)
    results = [r for r in results if r is not None]
    return results, n_tried


# main-process copies (for building task lists without needing worker globals)
def _gen_candidates_main(mol, mol2_smi):
    from rdkit import Chem
    n = mol.GetNumAtoms()
    for idx in range(n):
        for el in ELEMENTS:
            for bond in BONDS:
                yield 'add_atom', {'idx': idx, 'element': el, 'bond': bond}
            yield 'replace_atom', {'idx': idx, 'element': el}
        # functional groups enumerated lazily via FG_SMILES set at main-process import time
        for frag in FG_SMILES_MAIN:
            for bond in BONDS:
                yield 'add_substructure', {'idx': idx, 'substructure': frag, 'bond': bond}
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for anchor, branch in ((a1, a2), (a2, a1)):
            yield 'remove_substructure', {'anchor_idx': anchor, 'branch_idx': branch}
            for frag in FG_SMILES_MAIN:
                yield 'replace_substructure', {'anchor_idx': anchor, 'branch_idx': branch, 'new_substructure': frag}
    if mol2_smi:
        mol2 = Chem.MolFromSmiles(mol2_smi)
        for i1 in _ring_free_atom_indices_main(mol):
            for i2 in _ring_free_atom_indices_main(mol2):
                yield 'crossover_molecules', {'idx1': i1, 'idx2': i2}


def _ring_free_atom_indices_main(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if any(not b.IsInRing() for b in a.GetBonds())]


def _assert_crossover_only_at_step1(path, snap_id):
    """Enforces the module docstring's ASSUMPTION. A crossover_molecules step anywhere but
    path[0] was scored against the evolving working molecule (this script's convention) rather
    than the frozen parent1 (the live agent's actual convention) - see the docstring for why
    that's wrong. Fail loudly here rather than silently writing a wrong result."""
    for depth, step in enumerate(path):
        if step['tool'] == 'crossover_molecules' and depth != 0:
            raise AssertionError(
                f"[{snap_id}] crossover_molecules won the beam at depth {depth + 1} (path={path}) - "
                f"violates the ASSUMPTION documented at the top of this file. This candidate was "
                f"scored against the evolving working molecule, not the frozen parent1 the live "
                f"agent actually uses, so its delta/smi are wrong. Fix crossover candidate "
                f"generation to use the frozen parent1 at every beam level before proceeding.")


def run_one_molecule(pool, snap_id, snap, beam_width, log):
    working_smi = snap['working_smi_before']
    parent2 = snap['parent2']
    # compute baseline via a worker call to stay consistent with pool workers' objective defs
    base_phi, base_raw = pool.apply(_phi_and_raw, (working_smi,))

    t0 = time.time()
    level1, n1 = expand(pool, working_smi, parent2, base_phi)
    for r in level1:
        r['path'] = [{'tool': r['tool'], 'args': r['args'], 'smi': r['smi'], 'delta': r['delta']}]
    log(f"[{snap_id}] level 1: tried={n1} succeeded={len(level1)} ({time.time()-t0:.1f}s)")

    beam1 = sorted(level1, key=lambda r: -r['delta'])[:beam_width]

    t1 = time.time()
    level2 = []
    for node in beam1:
        res, n = expand(pool, node['smi'], parent2, base_phi)
        for r in res:
            r['path'] = node['path'] + [{'tool': r['tool'], 'args': r['args'], 'smi': r['smi'], 'delta': r['delta']}]
        level2.extend(res)
    log(f"[{snap_id}] level 2: expanded {len(beam1)} beam nodes -> {len(level2)} succeeded ({time.time()-t1:.1f}s)")

    beam2 = sorted(level2, key=lambda r: -r['delta'])[:beam_width]

    t2 = time.time()
    level3 = []
    for node in beam2:
        res, n = expand(pool, node['smi'], parent2, base_phi)
        for r in res:
            r['path'] = node['path'] + [{'tool': r['tool'], 'args': r['args'], 'smi': r['smi'], 'delta': r['delta']}]
        level3.extend(res)
    log(f"[{snap_id}] level 3: expanded {len(beam2)} beam nodes -> {len(level3)} succeeded ({time.time()-t2:.1f}s)")

    all_results = level1 + level2 + level3
    best = max(all_results, key=lambda r: r['delta']) if all_results else None
    best1 = max(level1, key=lambda r: r['delta']) if level1 else None
    best2 = max(level2, key=lambda r: r['delta']) if level2 else None
    best3 = max(level3, key=lambda r: r['delta']) if level3 else None

    for candidate in (best2, best3, best):
        if candidate is not None:
            _assert_crossover_only_at_step1(candidate['path'], snap_id)

    return {
        'snap_id': snap_id, 'gen': snap['gen'], 'pair': snap['pair'], 'step': snap['step'],
        'working_smi_before': working_smi, 'parent2': parent2, 'baseline_phi': base_phi,
        'chosen_tool': snap['name'], 'chosen_delta': None,  # filled by caller from regret_master
        'n1': n1, 'n_succ1': len(level1), 'n_succ2': len(level2), 'n_succ3': len(level3),
        'best_depth1': {'delta': best1['delta'], 'tool': best1['tool'], 'args': best1['args'], 'smi': best1['smi']} if best1 else None,
        'best_depth2': {'delta': best2['delta'], 'path': best2['path']} if best2 else None,
        'best_depth3': {'delta': best3['delta'], 'path': best3['path']} if best3 else None,
        'best_overall': {'delta': best['delta'], 'depth': len(best['path']), 'path': best['path']} if best else None,
        'elapsed_s': round(time.time() - t0, 1),
    }


FG_SMILES_MAIN = None


def main():
    global FG_SMILES_MAIN, PARENT2_BY_KEY
    ap = argparse.ArgumentParser()
    ap.add_argument('--beam', type=int, default=30)
    ap.add_argument('--workers', type=int, default=15)
    ap.add_argument('--picks', type=str, default=None, help='comma-separated snap_ids subset (for a quick test run)')
    ap.add_argument('--top-n-regret', type=int, default=None,
                     help='instead of --picks or the hardcoded PICKS list, auto-select the top N '
                          'snap_ids by regret from --master. Simple and reproducible, unlike the '
                          'original gpt-5.4 PICKS list, which was a hand curated subset (mixing '
                          'moderate and high regret across a spread of tools), not literally top-N.')
    ap.add_argument('--master', type=str, default=os.path.join(REGRET_DIR, 'regret_master_gpt54.json'),
                     help='regret-summary JSON to select snapshots from (schema: main/toolmol/oracle.py-'
                          'style list of dicts with snap_id/gen/pair/step/chosen_tool/working_smi_before/'
                          'chosen_delta/best_delta/regret - matches both this dir\'s regret_master_gpt54.json '
                          'and regret/*_regret_summary.json from the sibling brute_force.py tool)')
    ap.add_argument('--snapshots', type=str, default=os.path.join(REGRET_DIR, 'snapshots_gpt54.jsonl'),
                     help='snapshots.jsonl to pull parent2 SMILES from (same schema as regret/snapshots.jsonl)')
    ap.add_argument('--out', type=str, default=os.path.join(HERE, 'beam3_results.jsonl'))
    args = ap.parse_args()

    from main.toolmol.fg_lookup import FUNCTIONAL_GROUPS
    FG_SMILES_MAIN = list(FUNCTIONAL_GROUPS.values())

    master = json.load(open(args.master, encoding='utf-8'))
    by_id = {m['snap_id']: m for m in master}

    PARENT2_BY_KEY = {}
    for _line in open(args.snapshots, encoding='utf-8'):
        _d = json.loads(_line)
        PARENT2_BY_KEY[(_d['gen'], _d['pair'], _d['step'])] = _d['parent2']

    if args.picks:
        picks = [int(x) for x in args.picks.split(',')]
    elif args.top_n_regret:
        picks = [m['snap_id'] for m in sorted(master, key=lambda m: -m['regret'])[:args.top_n_regret]]
    else:
        picks = PICKS

    done_ids = set()
    if os.path.exists(args.out):
        with open(args.out, encoding='utf-8') as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)['snap_id'])
                except Exception:
                    pass

    def log(msg):
        print(msg, flush=True)

    log(f"beam width={args.beam} workers={args.workers} picks={picks} already_done={sorted(done_ids)}")

    with Pool(processes=args.workers, initializer=_pool_init) as pool:
        for snap_id in picks:
            if snap_id in done_ids:
                log(f"[{snap_id}] already done, skipping")
                continue
            snap_master = by_id[snap_id]
            snap = {
                'gen': snap_master['gen'], 'pair': snap_master['pair'], 'step': snap_master['step'],
                'working_smi_before': snap_master['working_smi_before'],
                'parent2': None,  # filled below from snapshots_gpt54.jsonl
                'name': snap_master['chosen_tool'],
            }
            # parent2 isn't stored in regret_master_gpt54.json directly; pull from snapshots_gpt54.jsonl
            snap['parent2'] = PARENT2_BY_KEY[(snap_master['gen'], snap_master['pair'], snap_master['step'])]

            t0 = time.time()
            result = run_one_molecule(pool, snap_id, snap, args.beam, log)
            result['chosen_delta'] = snap_master['chosen_delta']
            result['original_best_delta_1step'] = snap_master['best_delta']
            result['original_regret'] = snap_master['regret']
            with open(args.out, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + '\n')
            log(f"[{snap_id}] DONE in {time.time()-t0:.1f}s -- best_overall_delta={result['best_overall']['delta'] if result['best_overall'] else None} "
                f"(depth={result['best_overall']['depth'] if result['best_overall'] else None}) vs original 1-step best_delta={snap_master['best_delta']:.4f} "
                f"vs chosen_delta={snap_master['chosen_delta']:.4f}")

    log("ALL DONE")


# parent2 lookup, built from --snapshots inside main() (was module-level / import-time and
# hardcoded to snapshots_gpt54.jsonl; needed to depend on the CLI arg instead)
PARENT2_BY_KEY = {}


if __name__ == '__main__':
    main()
