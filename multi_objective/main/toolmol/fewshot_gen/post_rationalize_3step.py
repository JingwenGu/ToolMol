"""Multi-step post-rationalization: for the 10 molecules searched by beam_search_3step.py,
show gpt-5.4 the same opening context a real episode would see, then privately reveal the
best discovered 2- or 3-step sequence and ask it to write one first-person reasoning block per
step (not knowing the answer), one combined LLM call per molecule. Then re-invoke the REAL
tool functions sequentially (matching build_fewshot_library.py's approach) to build an
authentic OpenAI-protocol message sequence with genuine tool-result content at every step.

Crossover note: the live agent (agent.py ~line 381) always crosses the FROZEN original
parent1 against parent2, never the evolved working molecule - confirmed by re-reading
_agentic_edit. All 10 winning paths only use crossover_molecules at step 1 (where the working
molecule and parent1 are identical anyway), so this doesn't affect any of these sequences.

Output: fewshot_examples_3step.json (post-rationalized text) and fewshot_messages_3step.json
(ready-to-splice OpenAI messages, same shape as the original fewshot_messages.json).
"""
import sys, os, json, re

HERE = os.path.dirname(os.path.abspath(__file__))
MULTI_OBJECTIVE_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, MULTI_OBJECTIVE_DIR)
sys.path.insert(0, os.path.join(MULTI_OBJECTIVE_DIR, "main", "toolmol"))

from rdkit import Chem
from main.toolmol.agent import SYSTEM_PROMPT, _format_context, _canonicalize
from main.toolmol.objectives import build_objectives
from main.toolmol.toolbox import TOOL_DISPATCH
from openai import OpenAI

objectives = build_objectives(['jnk3', 'qed', 'sa'])


def phi_and_raw(smi):
    if smi is None or Chem.MolFromSmiles(smi) is None:
        return None, None
    raw = {}
    total = 0.0
    for obj in objectives:
        r = obj.raw(smi)
        raw[obj.name] = r
        total += obj.rescaled(r)
    return total, raw


GOAL_DESCRIPTION = "maximize jnk3, qed scores; minimize sa score"


def fmt_args(tool, args):
    return f'{tool}({json.dumps(args, separators=(",", ":"))})'


def build_private_instruction(path_steps, base_raw, base_phi):
    """path_steps: list of dicts with tool, args, smi, raw, phi, delta (per-step delta vs
    the state right before that step)."""
    n = len(path_steps)
    lines = [
        "\n\n---",
        "[PRIVATE, DO NOT REFERENCE THIS BLOCK IN YOUR ANSWER]",
        f"A search over multi-step edit sequences to this pair found that the following "
        f"{n}-step sequence is a very strong plan, verified by direct evaluation:",
        "",
    ]
    prev_raw, prev_phi = base_raw, base_phi
    for i, s in enumerate(path_steps):
        applied_to = "molecule 1 and 2 (crossover)" if s['tool'] == 'crossover_molecules' else (
            "molecule 1" if i == 0 else "the molecule after the previous step")
        lines.append(f"Step {i+1}: {fmt_args(s['tool'], s['args'])}")
        lines.append(f"  Applied to {applied_to}, this produces:")
        lines.append(f"  {s['smi']}")
        lines.append(
            f"  jnk3={s['raw']['jnk3']:.4f}, qed={s['raw']['qed']:.4f}, sa={s['raw']['sa']:.4f} "
            f"(from jnk3={prev_raw['jnk3']:.4f}, qed={prev_raw['qed']:.4f}, sa={prev_raw['sa']:.4f}; "
            f"step score delta {s['phi']-prev_phi:+.4f})")
        lines.append("")
        prev_raw, prev_phi = s['raw'], s['phi']

    overall_delta = path_steps[-1]['phi'] - base_phi
    lines.append(f"Overall, this {n}-step sequence takes the total score from {base_phi:.4f} "
                 f"to {path_steps[-1]['phi']:.4f} (total delta {overall_delta:+.4f}).")
    lines.append("")
    lines.append(
        "Write the reasoning you would have given BEFORE knowing any of this, as separate short "
        "reasoning blocks - one written right before each step, in the same concise first-person "
        "style you'd normally use, analyzing the molecule state visible at that point and "
        "explaining why that specific action is the right move. Step 1's reasoning may briefly "
        "mention your general intention to keep refining, but must be based only on what's "
        "visible in molecules 1 and 2 - do not describe later steps as if already decided in "
        "detail. Do not mention that you were told the answer, a search, a multi-step sequence, "
        "or verification anywhere in your reasoning - write it as your own original analysis, "
        "one step at a time, exactly as you would if deciding each step live.")
    lines.append("")
    lines.append("Output exactly in this format and nothing else:")
    for i, s in enumerate(path_steps):
        lines.append(f"STEP{i+1}_REASONING: <1-3 sentences>")
        lines.append(f"STEP{i+1}_TOOL_CALL: {fmt_args(s['tool'], s['args'])}")
    return "\n".join(lines)


def parse_response(content, n_steps):
    out = []
    for i in range(1, n_steps + 1):
        r_pat = re.compile(rf"STEP{i}_REASONING:\s*(.*?)(?=STEP{i}_TOOL_CALL:)", re.DOTALL)
        t_pat = re.compile(rf"STEP{i}_TOOL_CALL:\s*(.*?)(?=STEP{i+1}_REASONING:|$)", re.DOTALL)
        rm = r_pat.search(content)
        tm = t_pat.search(content)
        reasoning = rm.group(1).strip() if rm else ""
        tool_call = tm.group(1).strip() if tm else ""
        out.append((reasoning, tool_call))
    return out


def build_examples(client, model, beam_results_path, snapshots_path):
    # Picks come from whichever snap_ids beam_search_3step.py actually wrote to
    # --out (in file order), rather than a separately-maintained PICKS list here -
    # the two lists drifting apart was a real footgun (they had to be kept in sync
    # by hand) and beam_results.jsonl is already the authoritative "what was searched".
    beam_results = {}
    with open(beam_results_path, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            beam_results[d['snap_id']] = d
    picks = list(beam_results.keys())

    raw_idx = {}
    with open(snapshots_path, encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            raw_idx[(d['gen'], d['pair'], d['step'])] = d

    results = []
    for sid in picks:
        br = beam_results[sid]
        raw = raw_idx[(br['gen'], br['pair'], br['step'])]
        mol1_smi = br['working_smi_before']
        mol2_smi = br['parent2']
        mol1, mol2 = Chem.MolFromSmiles(mol1_smi), Chem.MolFromSmiles(mol2_smi)
        base_phi, base_raw = phi_and_raw(mol1_smi)
        score1, _ = base_phi, base_raw
        score2, _ = phi_and_raw(mol2_smi)

        path = br['best_overall']['path']
        # recompute raw jnk3/qed/sa for every step (beam search only stored delta, not raw)
        path_steps = []
        for step in path:
            phi, raw_scores = phi_and_raw(step['smi'])
            path_steps.append({'tool': step['tool'], 'args': step['args'], 'smi': step['smi'],
                                'phi': phi, 'raw': raw_scores})

        public_user_msg = (
            f"Goal: I want to {GOAL_DESCRIPTION}. Please propose a new molecule better "
            f"than the current molecule. I have given you two candidate ligands. You are "
            f"encouraged to make a crossover between the candidate molecules on the first "
            f"step, then mutate the resulting molecule. Only make a few modifications (at "
            f"most 3), then respond with FINAL ANSWER. Do not let molecular weight exceed 700.\n\n"
            f"1. {mol1_smi}\nScore: {score1}\n{_format_context(mol1)}\n\n"
            f"2. {mol2_smi}\nScore: {score2}\n{_format_context(mol2)}"
        )
        private_instruction = build_private_instruction(path_steps, base_raw, base_phi)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": public_user_msg + private_instruction},
        ]
        resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
        content = resp.choices[0].message.content or ""
        parsed = parse_response(content, len(path_steps))

        results.append({
            'snap_id': sid, 'gen': br['gen'], 'pair': br['pair'], 'step': br['step'],
            'n_steps': len(path_steps),
            'mol1_smi': mol1_smi, 'mol1_score': score1, 'mol2_smi': mol2_smi, 'mol2_score': score2,
            'public_user_msg': public_user_msg,
            'baseline_raw': base_raw, 'baseline_phi': base_phi,
            'path_steps': path_steps,
            'per_step_reasoning': [p[0] for p in parsed],
            'per_step_tool_call_text': [p[1] for p in parsed],
            'overall_delta': path_steps[-1]['phi'] - base_phi,
            'original_1step_ceiling': br['original_best_delta_1step'],
            'self_chosen_delta': br['chosen_delta'],
            'raw_llm_response': content,
        })
        print(f"[{sid}] done: {len(path_steps)} steps, parsed {sum(1 for r,t in parsed if r and t)}/{len(path_steps)} cleanly", flush=True)

    return results


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt-5.4',
                     help='model that both wrote the episode being post-rationalized and will '
                          'later read these few-shot examples in-context - e.g. openai/gpt-oss-120b '
                          'for a local vLLM server (pair with --base-url/--api-key-env below). No '
                          'tools/tool_choice are passed on this call (free-text generation only), '
                          'so this works over plain /v1/chat/completions even for gpt-oss, whose '
                          'tool_calls field is otherwise unreliable there - see agent.py\'s '
                          '"responses" backend notes.')
    ap.add_argument('--base-url', default=None,
                     help='override API base URL, e.g. http://localhost:8000/v1 for a local vLLM '
                          'server. Default (None) talks to the real OpenAI API.')
    ap.add_argument('--api-key-env', default='OPENAI_API_KEY',
                     help='env var holding the API key. A local vLLM server ignores the value but '
                          'still needs the var set to something (e.g. OPENAI_API_KEY=not-needed).')
    ap.add_argument('--beam-results', default=os.path.join(HERE, 'beam3_results.jsonl'),
                     help='beam_search_3step.py --out file to post-rationalize')
    ap.add_argument('--snapshots', default=os.path.join(HERE, 'snapshots_gpt54.jsonl'),
                     help='snapshots.jsonl matching --beam-results, for the parent2 lookup')
    ap.add_argument('--out', default=os.path.join(HERE, 'fewshot_examples_3step.json'))
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ[args.api_key_env], base_url=args.base_url)
    out = build_examples(client, args.model, args.beam_results, args.snapshots)
    json.dump(out, open(args.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"\nwrote {len(out)} examples to {args.out}")
