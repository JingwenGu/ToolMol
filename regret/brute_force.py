"""Single-step regret: for each sampled (working_mol, chosen_action) snapshot from the new
run, brute-force every candidate action reachable from working_mol across all 7 tools, score
each resulting molecule with the same Phi() used by the GA, and compare the best achievable
delta to the model's actual delta.

add_functional_group is folded into add_substructure's enumeration: add_functional_group(idx,
group,bond) is implemented as add_substructure(idx, get_functional_group(group), bond), so it
can never reach an outcome add_substructure's group-SMILES sweep doesn't already cover - testing
both separately would just duplicate every candidate molecule under two tool-name labels.

add_substructure/replace_substructure's new_substructure is restricted to the 41-entry
FUNCTIONAL_GROUPS vocabulary (same one add_functional_group already draws from), not arbitrary
SMILES - true unrestricted search is unbounded/undecidable, so "optimal" here means "best
reachable with a rich but finite fragment library," not a true global optimum.
"""
import sys, os, json, random, time, csv

# Must happen before numpy (imported transitively via rdkit/main.toolmol.*) ever loads its BLAS
# backend, which reads these once at init and otherwise defaults to one thread pool per process
# sized to the full core count. --workers spawns that many separate processes, each importing
# numpy independently via _worker, so without this a large --workers run can try to spawn far
# more OS threads than there are cores - harmless-looking on a SLURM allocation sized to match
# (threads just contend within your own cores), but capable of exhausting process/thread limits
# machine-wide if ever run somewhere shared, like a login node (observed directly with the same
# pattern in fewshot_gen/beam_search_3step.py - see that file's history).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "multi_objective"))
from rdkit import Chem
from main.toolmol import toolbox
from main.toolmol.fg_lookup import FUNCTIONAL_GROUPS
from main.toolmol.objectives import build_objectives

random.seed(0)

HERE = os.path.dirname(__file__)
SNAPSHOTS = os.path.join(HERE, "snapshots.jsonl")
CANDIDATES_CSV = os.path.join(HERE, "candidates_log.csv")
SUMMARY_JSON = os.path.join(HERE, "regret_summary.json")

ELEMENTS = ['C', 'N', 'O', 'F', 'Cl', 'Br', 'S']
BONDS = ['single', 'double', 'triple']
GROUP_SMILES = list(FUNCTIONAL_GROUPS.values())

objectives = build_objectives(['jnk3', 'qed', 'sa'])


def phi_and_raw(smi):
    """Returns (combined_score, {name: raw_value}) for a SMILES, or (None, None) if invalid."""
    if Chem.MolFromSmiles(smi) is None:
        return None, None
    raw = {}
    total = 0.0
    for obj in objectives:
        r = obj.raw(smi)
        raw[obj.name] = r
        total += obj.rescaled(r)
    return total, raw


def ring_free_atom_indices(mol):
    """Atoms with at least one acyclic (non-ring) bond - valid crossover cut points."""
    out = []
    for atom in mol.GetAtoms():
        if any(not b.IsInRing() for b in atom.GetBonds()):
            out.append(atom.GetIdx())
    return out


def gen_candidates(working_smi, parent1_smi, parent2_smi):
    """Yields (tool_name, args_dict) for every candidate action reachable from working_smi
    (plus parent1/parent2 for crossover, which doesn't use working_smi at all)."""
    mol = Chem.MolFromSmiles(working_smi)
    n = mol.GetNumAtoms()

    for idx in range(n):
        for el in ELEMENTS:
            for bond in BONDS:
                yield 'add_atom', {'idx': idx, 'element': el, 'bond': bond}
            yield 'replace_atom', {'idx': idx, 'element': el}
        for frag in GROUP_SMILES:
            for bond in BONDS:
                yield 'add_substructure', {'idx': idx, 'substructure': frag, 'bond': bond}

    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for anchor, branch in ((a1, a2), (a2, a1)):
            yield 'remove_substructure', {'anchor_idx': anchor, 'branch_idx': branch}
            for frag in GROUP_SMILES:
                yield 'replace_substructure', {'anchor_idx': anchor, 'branch_idx': branch, 'new_substructure': frag}

    if parent1_smi and parent2_smi:
        mol1 = Chem.MolFromSmiles(parent1_smi)
        mol2 = Chem.MolFromSmiles(parent2_smi)
        idxs1 = ring_free_atom_indices(mol1)
        idxs2 = ring_free_atom_indices(mol2)
        for i1 in idxs1:
            for i2 in idxs2:
                yield 'crossover_molecules', {'idx1': i1, 'idx2': i2}


def call_tool(tool_name, working_smi, parent1_smi, parent2_smi, args):
    if tool_name == 'crossover_molecules':
        return toolbox.crossover_molecules(parent1_smi, args['idx1'], parent2_smi, args['idx2'])
    fn = getattr(toolbox, tool_name)
    return fn(working_smi, **args)


def brute_force_snapshot(snap_id, snap):
    # Returns (result_dict, candidate_rows) rather than writing straight to a CSV writer, so
    # this can run in a worker process (each snapshot is fully independent - no shared state
    # with any other snapshot - so this whole function parallelizes across a process pool one
    # snapshot per task; a shared file handle/writer can't cross that process boundary, so the
    # main process does all file writing after collecting each worker's return value instead).
    working_smi = snap["working_smi_before"]
    parent1_smi, parent2_smi = snap["parent1"], snap["parent2"]

    baseline_phi, baseline_raw = phi_and_raw(working_smi)

    n_tried = 0
    n_success = 0
    best = None  # (delta, phi, tool_name, args, smi, raw)
    rows = []
    for tool_name, args in gen_candidates(working_smi, parent1_smi, parent2_smi):
        n_tried += 1
        result = call_tool(tool_name, working_smi, parent1_smi, parent2_smi, args)
        if not result.success:
            continue
        smi = Chem.MolToSmiles(result.mol)
        phi, raw = phi_and_raw(smi)
        if phi is None:
            continue
        n_success += 1
        delta = phi - baseline_phi
        rows.append([snap_id, tool_name, json.dumps(args), smi,
                      f"{raw['qed']:.4f}", f"{raw['jnk3']:.4f}", f"{raw['sa']:.4f}",
                      f"{phi:.4f}", f"{delta:.4f}"])
        if best is None or delta > best[0]:
            best = (delta, phi, tool_name, args, smi, raw)

    chosen_success = snap["success"]
    chosen_smi = snap.get("result_smi")
    if chosen_success and chosen_smi:
        chosen_phi, chosen_raw = phi_and_raw(chosen_smi)
        chosen_delta = chosen_phi - baseline_phi
    else:
        chosen_phi, chosen_raw = None, None
        chosen_delta = 0.0

    regret = (best[0] - chosen_delta) if best is not None else None

    res = {
        "snap_id": snap_id, "gen": snap["gen"], "pair": snap["pair"], "step": snap["step"],
        "working_smi_before": working_smi, "baseline_phi": baseline_phi, "baseline_raw": baseline_raw,
        "chosen_tool": snap["name"], "chosen_args_str": snap["args_str"], "chosen_success": chosen_success,
        "chosen_result_smi": chosen_smi, "chosen_phi": chosen_phi, "chosen_delta": chosen_delta,
        "n_candidates_tried": n_tried, "n_candidates_succeeded": n_success,
        "best_delta": best[0] if best else None, "best_phi": best[1] if best else None,
        "best_tool": best[2] if best else None, "best_args": best[3] if best else None,
        "best_smi": best[4] if best else None, "best_raw": best[5] if best else None,
        "regret": regret,
    }
    return res, rows


def _worker(args):
    snap_id, snap = args
    t0 = time.time()
    res, rows = brute_force_snapshot(snap_id, snap)
    return res, rows, time.time() - t0


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--test", action="store_true", help="run on just 2 snapshots for timing")
    p.add_argument("--snapshots", default=SNAPSHOTS, help="snapshots.jsonl to sample from")
    p.add_argument("--out-prefix", default=None,
                    help="prefix for output files (default: unprefixed candidates_log.csv/regret_summary.json)")
    p.add_argument("--workers", type=int, default=1,
                    help="worker processes for parallel brute force across snapshots (each snapshot is "
                         "fully independent, so this scales ~linearly up to min(--n, cpu count); "
                         "default 1 = serial, matching the original behavior")
    args_cli = p.parse_args()

    SNAPSHOTS = args_cli.snapshots
    if args_cli.out_prefix:
        CANDIDATES_CSV = os.path.join(HERE, f"{args_cli.out_prefix}_candidates_log.csv")
        SUMMARY_JSON = os.path.join(HERE, f"{args_cli.out_prefix}_regret_summary.json")

    snapshots = []
    with open(SNAPSHOTS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["name"] != "undo_last_change":
                snapshots.append(d)

    by_tool = {}
    for s in snapshots:
        by_tool.setdefault(s["name"], []).append(s)

    if args_cli.test:
        sample = snapshots[:2]
    else:
        target_n = args_cli.n
        tool_names = sorted(by_tool)
        base_each = target_n // len(tool_names)
        remainder = target_n - base_each * len(tool_names)
        sample = []
        for i, name in enumerate(tool_names):
            k = base_each + (1 if i < remainder else 0)
            pool = by_tool[name]
            k = min(k, len(pool))
            sample.extend(random.sample(pool, k))
        random.shuffle(sample)
        print(f"stratified sample: {[(name, sum(1 for s in sample if s['name']==name)) for name in tool_names]}")

    print(f"running brute force over {len(sample)} snapshots ({args_cli.workers} worker(s))...")
    os.makedirs(HERE, exist_ok=True)
    results = []
    t0 = time.time()
    tasks = list(enumerate(sample))
    with open(CANDIDATES_CSV, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["snap_id", "tool", "args", "smiles", "qed", "jnk3", "sa", "combined_score", "delta_vs_baseline"])

        def handle(i, snap, res, rows, dt):
            for row in rows:
                w.writerow(row)
            print(f"[{len(results)+1}/{len(sample)}] gen={snap['gen']} pair={snap['pair']} tool={snap['name']} "
                  f"tried={res['n_candidates_tried']} success={res['n_candidates_succeeded']} "
                  f"regret={res['regret']} ({dt:.1f}s)", flush=True)
            results.append(res)

        if args_cli.workers <= 1:
            for i, snap in tasks:
                t1 = time.time()
                res, rows = brute_force_snapshot(i, snap)
                handle(i, snap, res, rows, time.time() - t1)
        else:
            # imap_unordered so results are handled (and progress printed) as each worker
            # finishes, not held back by whichever snapshot happens to be slowest/first in
            # input order. It doesn't preserve the tasks list's order, so recover which
            # snapshot a result belongs to via its own snap_id (== the enumerate index)
            # rather than zipping against tasks positionally.
            with mp.Pool(processes=args_cli.workers) as pool:
                for res, rows, dt in pool.imap_unordered(_worker, tasks, chunksize=1):
                    snap = sample[res["snap_id"]]
                    handle(res["snap_id"], snap, res, rows, dt)
    print(f"total time: {time.time()-t0:.1f}s")

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("wrote", CANDIDATES_CSV)
    print("wrote", SUMMARY_JSON)
