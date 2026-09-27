"""Targeted fix for two reasoning steps in fewshot_examples_3step_gptoss_repick.json whose
free-text claims contradicted the actual measured jnk3/qed/sa deltas at that step
(snap_id 62 step 3 falsely claimed an SA improvement that didn't happen; snap_id 12 step 3
falsely claimed a QED/drug-likeness improvement that didn't happen). Re-prompts gpt-oss-120b
with the exact before/after raw values for just that one step and asks for a corrected
1-3 sentence reasoning block in the same first-person, hedged style as the rest of the library,
rather than regenerating all 30 steps for a single-word-level miscalibration.
"""
import sys, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
MULTI_OBJECTIVE_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, MULTI_OBJECTIVE_DIR)
sys.path.insert(0, os.path.join(MULTI_OBJECTIVE_DIR, "main", "toolmol"))

from openai import OpenAI

TARGETS = {
    12: 2,  # 0-indexed step 3
}


def build_fix_prompt(ex, step_idx):
    steps = ex['path_steps']
    reasoning = ex['per_step_reasoning']
    s = steps[step_idx]
    prev_raw = ex['baseline_raw'] if step_idx == 0 else steps[step_idx - 1]['raw']
    lines = [
        "You previously wrote this reasoning for one step of a molecule-editing sequence "
        "(maximize jnk3, qed; minimize sa):",
        "",
        f"Tool call: {s['tool']}({json.dumps(s['args'])})",
        f'Your reasoning: "{reasoning[step_idx]}"',
        "",
        "The actual measured property change for this exact step was:",
        f"  jnk3: {prev_raw['jnk3']:.4f} -> {s['raw']['jnk3']:.4f} ({s['raw']['jnk3']-prev_raw['jnk3']:+.4f})",
        f"  qed:  {prev_raw['qed']:.4f} -> {s['raw']['qed']:.4f} ({s['raw']['qed']-prev_raw['qed']:+.4f})",
        f"  sa:   {prev_raw['sa']:.4f} -> {s['raw']['sa']:.4f} ({s['raw']['sa']-prev_raw['sa']:+.4f}, lower sa is better)",
        "",
        "Your reasoning claimed a QED or SA improvement that did not actually happen at this "
        "step. Rewrite the reasoning (1-3 sentences, same first-person style, written as if "
        "before knowing the numeric outcome) so it does not assert a specific QED or SA "
        "direction contradicted by the numbers above, and does NOT use the words or concepts "
        "'drug-likeness', 'drug-like', 'QED', 'synthetic accessibility/tractability', or 'SA' "
        "at all, favorable or otherwise, since the actual measured direction for that property "
        "went the wrong way here. You may still justify the move on jnk3/binding affinity, "
        "scaffold/structural grounds (e.g. aromaticity, polarity, size, rigidity), or other "
        "properties not contradicted by the numbers. Do not mention that you were given these "
        "numbers or that this is a correction anywhere in the text.",
        "",
        "Output exactly one line: STEP_REASONING: <text>",
    ]
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='openai/gpt-oss-120b')
    ap.add_argument('--base-url', default=None)
    ap.add_argument('--api-key-env', default='OPENAI_API_KEY')
    ap.add_argument('--examples', default=os.path.join(HERE, 'fewshot_examples_3step_gptoss_repick.json'))
    args = ap.parse_args()

    with open(args.examples, encoding='utf-8') as f:
        examples = json.load(f)
    by_id = {ex['snap_id']: ex for ex in examples}

    client = OpenAI(api_key=os.environ[args.api_key_env], base_url=args.base_url)

    for sid, step_idx in TARGETS.items():
        ex = by_id[sid]
        prompt = build_fix_prompt(ex, step_idx)
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = resp.choices[0].message.content or ""
        new_reasoning = content.split("STEP_REASONING:", 1)[-1].strip()
        old_reasoning = ex['per_step_reasoning'][step_idx]
        print(f"=== snap_id {sid} step {step_idx+1} ===")
        print("OLD:", old_reasoning)
        print("NEW:", new_reasoning)
        print()
        ex['per_step_reasoning'][step_idx] = new_reasoning

    with open(args.examples, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"patched {len(TARGETS)} reasoning step(s) in {args.examples}")


if __name__ == '__main__':
    main()
