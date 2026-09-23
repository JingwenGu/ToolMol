"""Reconstruct (working_mol_before, parent1, parent2, chosen_tool, args, success, result_smi)
snapshots by replaying the new (2-idx bond-cut) run's log as a state machine - no LLM calls,
pure text parsing plus the same state-transition logic agent.py itself follows."""
import re, json, os

LOG = os.path.join(os.path.dirname(__file__), "..", "multi_objective", "main", "toolmol",
                    "results", "task_cerebras_seed1_max200_idxredesign_2026-09-07", "run_log.txt")
OUT = os.path.join(os.path.dirname(__file__), "snapshots.jsonl")

start_re = re.compile(r"generation (\d+) pair (\d+)/(\d+): starting edit_pair")
done_re = re.compile(r"generation (\d+) pair (\d+)/(\d+): edit_pair done")
parent1_re = re.compile(r"\[state\] parent1: (\S+)")
parent2_re = re.compile(r"\[state\] parent2: (\S+)")
call_re = re.compile(r"agent step (\d+): tool call -> (\w+)\((.*)\)$")
before_re = re.compile(r"\[state\] before call: (\S+)")
result_re = re.compile(r"agent step \d+: tool result success=(True|False): (.*)")
undo_result_re = re.compile(r"agent step \d+: tool result: (success|failed): (.*)")
current_smiles_re = re.compile(r"Current SMILES: (\S+)")

all_calls = []
episode = None


def new_episode(gen, pair):
    # mol_history mirrors agent.py's own undo stack: pushed with the pre-call working_smi
    # right before any successful *non-undo* call overwrites it, popped by a successful undo.
    # Needed because undo's own console log line is truncated to its first line only (the
    # full "Current SMILES: ..." goes to the model, never to console) - so the only reliable
    # way to know what an undo restored is to replay the same stack discipline agent.py uses.
    return {"gen": gen, "pair": pair, "parent1": None, "parent2": None, "working_smi": None,
            "mol_history": [], "calls": []}


pending_call = None       # {'step','name','args_str'}
pending_finalized = None  # {'step','name','args_str','success'} - awaiting "Current SMILES" if success

with open(LOG, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = start_re.search(line)
        if m:
            episode = new_episode(int(m.group(1)), int(m.group(2)))
            pending_call = None
            pending_finalized = None
            continue
        if episode is None:
            continue

        m = parent1_re.search(line)
        if m:
            episode["parent1"] = m.group(1)
            episode["working_smi"] = m.group(1)
            continue
        m = parent2_re.search(line)
        if m:
            episode["parent2"] = m.group(1)
            continue

        m = call_re.search(line)
        if m:
            pending_call = {"step": int(m.group(1)), "name": m.group(2), "args_str": m.group(3),
                             "working_smi_before": episode["working_smi"],
                             "parent1": episode["parent1"], "parent2": episode["parent2"]}
            continue

        m = before_re.search(line)
        if m and pending_call:
            # sanity cross-check against our own state tracking
            if m.group(1) != pending_call["working_smi_before"]:
                print(f"WARNING mismatch gen={episode['gen']} pair={episode['pair']} step={pending_call['step']}: "
                      f"tracked={pending_call['working_smi_before']} logged={m.group(1)}")
            continue

        m = result_re.search(line)
        if m and pending_call:
            success = m.group(1) == "True"
            pending_finalized = dict(pending_call, success=success, result_smi=None)
            pending_call = None
            if not success:
                episode["calls"].append(pending_finalized)
                pending_finalized = None
            # else: wait for the "Current SMILES:" line (regular tools print full state)
            continue

        m = undo_result_re.search(line)
        if m and pending_call:
            success = m.group(1) == "success"
            finalized = dict(pending_call, success=success, result_smi=None)
            pending_call = None
            if success:
                # undo's own console line is truncated - replay the stack instead of
                # waiting for a "Current SMILES:" line that was never printed.
                if episode["mol_history"]:
                    restored_smi = episode["mol_history"].pop()
                    finalized["result_smi"] = restored_smi
                    episode["working_smi"] = restored_smi
                else:
                    print(f"WARNING undo succeeded but local history stack empty: "
                          f"gen={episode['gen']} pair={episode['pair']}")
            episode["calls"].append(finalized)
            continue

        m = current_smiles_re.search(line)
        if m and pending_finalized is not None:
            pending_finalized["result_smi"] = m.group(1)
            episode["mol_history"].append(pending_finalized["working_smi_before"])
            episode["working_smi"] = m.group(1)
            episode["calls"].append(pending_finalized)
            pending_finalized = None
            continue

        m = done_re.search(line)
        if m:
            for c in episode["calls"]:
                all_calls.append({"gen": episode["gen"], "pair": episode["pair"], **c})
            episode = None
            continue

print(f"total tool-call snapshots reconstructed: {len(all_calls)}")
by_tool = {}
for c in all_calls:
    by_tool.setdefault(c["name"], [0, 0])
    by_tool[c["name"]][0] += 1
    by_tool[c["name"]][1] += c["success"]
for name, (n, s) in sorted(by_tool.items(), key=lambda x: -x[1][0]):
    print(f"  {name}: {n} calls, {s} success")

with open(OUT, "w", encoding="utf-8") as f:
    for c in all_calls:
        f.write(json.dumps(c) + "\n")
print("wrote", OUT)
