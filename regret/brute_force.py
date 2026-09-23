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

sys.path.append(r"D:\校外学习\Reading\Computer Science\ML papers\AI4S\MOLLEO\multi_objective")
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


def brute_force_snapshot(snap_id, snap, candidates_writer):
    working_smi = snap["working_smi_before"]
    parent1_smi, parent2_smi = snap["parent1"], snap["parent2"]

    baseline_phi, baseline_raw = phi_and_raw(working_smi)

    n_tried = 0
    n_success = 0
    best = None  # (delta, phi, tool_name, args, smi, raw)
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
        candidates_writer.writerow([snap_id, tool_name, json.dumps(args), smi,
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

    return {
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


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--test", action="store_true", help="run on just 2 snapshots for timing")
    args_cli = p.parse_args()

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

    print(f"running brute force over {len(sample)} snapshots...")
    os.makedirs(HERE, exist_ok=True)
    results = []
    t0 = time.time()
    with open(CANDIDATES_CSV, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["snap_id", "tool", "args", "smiles", "qed", "jnk3", "sa", "combined_score", "delta_vs_baseline"])
        for i, snap in enumerate(sample):
            t1 = time.time()
            res = brute_force_snapshot(i, snap, w)
            dt = time.time() - t1
            print(f"[{i+1}/{len(sample)}] gen={snap['gen']} pair={snap['pair']} tool={snap['name']} "
                  f"tried={res['n_candidates_tried']} success={res['n_candidates_succeeded']} "
                  f"regret={res['regret']} ({dt:.1f}s)", flush=True)
            results.append(res)
    print(f"total time: {time.time()-t0:.1f}s")

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("wrote", CANDIDATES_CSV)
    print("wrote", SUMMARY_JSON)
