"""Turn fewshot_examples_3step.json's post-rationalized (molecule, reasoning, action) sequences
into authentic OpenAI-protocol messages: for each step, re-invoke the REAL tool function (not
just trust the beam search's recorded smi) so tool-result content is byte-identical to what a
live multi-step episode would see. Chains steps together exactly as _agentic_edit does: each
step's tool call acts on the PREVIOUS step's real result (working_mol), and between steps a
"continue or FINAL ANSWER" reminder message is inserted, matching the live agent's actual
per-step message flow. crossover_molecules (only ever step 1 in these 10 examples) uses the
frozen original parent1/parent2, matching agent.py's real semantics.

Output: fewshot_messages_3step.json - flat message list, same shape as the original
fewshot_messages.json, ready to be prepended in agent.py's _agentic_edit via few_shot_file.
"""
import sys, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
MULTI_OBJECTIVE_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, MULTI_OBJECTIVE_DIR)
sys.path.insert(0, os.path.join(MULTI_OBJECTIVE_DIR, "main", "toolmol"))

from rdkit import Chem
from main.toolmol.agent import _format_context, _canonicalize
from main.toolmol.toolbox import TOOL_DISPATCH

CONTINUE_TEXT = (
    "You have made {n} modification(s) so far (up to 3 allowed). Output FINAL ANSWER if you "
    "have made sufficient modifications, or continue with another tool call to keep refining. "
    "Ensure that desired properties are maintained.\nCurrent SMILES: {smi}\n{ctx}")
CAPPED_TEXT = ("You have made 3 modifications already - please output FINAL ANSWER now.\n"
               "Current SMILES: {smi}\n{ctx}")


def invoke_step(tool_name, args, working_smi, mol1_smi, mol2_smi, target_smi):
    """Re-invoke the real tool. crossover_molecules always uses the frozen original parent1/
    parent2 (matching agent.py), every other tool acts on the current working_smi. Retries up
    to 25x for crossover_molecules' internal random.shuffle non-determinism, matching
    build_fewshot_library.py's original approach."""
    result, result_smi = None, None
    for attempt in range(25):
        if tool_name == 'crossover_molecules':
            result = TOOL_DISPATCH[tool_name](mol1_smi, args.get('idx1'), mol2_smi, args.get('idx2'))
        else:
            result = TOOL_DISPATCH[tool_name](working_smi, **args)
        if result.success:
            result_smi = Chem.MolToSmiles(_canonicalize(result.mol))
            if result_smi == target_smi:
                break
    if result is None or not result.success:
        raise RuntimeError(f"re-invoking {tool_name} failed on all 25 attempts: {result.message if result else '?'}")
    if result_smi != target_smi:
        print(f"  WARNING: never matched expected output in 25 attempts; "
              f"using last successful result {result_smi!r} instead of {target_smi!r}")
    return result, result_smi


def build(examples_path):
    examples = json.load(open(examples_path, encoding='utf-8'))
    messages = []
    for ex_i, ex in enumerate(examples):
        mol1_smi, mol2_smi = ex['mol1_smi'], ex['mol2_smi']
        n_steps = ex['n_steps']
        path_steps = ex['path_steps']
        reasonings = ex['per_step_reasoning']

        messages.append({"role": "user", "content": ex['public_user_msg']})

        working_smi = mol1_smi
        for i in range(n_steps):
            tool_name = path_steps[i]['tool']
            args = path_steps[i]['args']
            target_smi = path_steps[i]['smi']

            result, result_smi = invoke_step(tool_name, args, working_smi, mol1_smi, mol2_smi, target_smi)
            working_mol = _canonicalize(result.mol)
            result_smi = Chem.MolToSmiles(working_mol)

            call_id = f"fewshot3_{ex_i}_{i}_call"
            messages.append({
                "role": "assistant",
                "content": reasonings[i],
                "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(args, separators=(",", ":"))},
                }],
            })
            tool_content = (f"success: {result.message}\n"
                             f"Current SMILES: {result_smi}\n"
                             f"{_format_context(working_mol)}")
            messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_content})

            working_smi = result_smi
            is_last = (i == n_steps - 1)
            n_mods = i + 1
            ctx = _format_context(working_mol)
            if n_mods >= 3:
                reminder_content = CAPPED_TEXT.format(smi=working_smi, ctx=ctx)
            else:
                reminder_content = CONTINUE_TEXT.format(n=n_mods, smi=working_smi, ctx=ctx)
            messages.append({"role": "user", "content": reminder_content})
            if is_last:
                messages.append({"role": "assistant", "content": "FINAL ANSWER"})

        print(f"[{ex['snap_id']}] ok: {n_steps} steps, final smi = {working_smi}", flush=True)

    return messages, examples


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--examples', default=os.path.join(HERE, 'fewshot_examples_3step.json'),
                     help='post_rationalize_3step.py --out file to assemble')
    ap.add_argument('--out', default=os.path.join(HERE, 'fewshot_messages_3step.json'))
    args = ap.parse_args()

    messages, examples = build(args.examples)
    json.dump(messages, open(args.out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f"\nwrote {len(messages)} messages ({len(examples)} examples) to {args.out}")
