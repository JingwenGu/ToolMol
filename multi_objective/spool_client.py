"""The agent's interface to a `--llm_backend claude` run: read turns, answer them, search,
inject.

With that backend there is no model behind an endpoint. The run spools each agent turn to
`$TOOLMOL_CLAUDE_SPOOL/req_NNNNN.json` and blocks until `resp_NNNNN.json` appears, so the
**answering session is the agent** and the run's whole history accumulates in its context
for free rather than being resent every turn.

    export TOOLMOL_CLAUDE_SPOOL=<spool dir>
    python spool_client.py status
    python spool_client.py next
    python spool_client.py reply --content "..." --tool add_atom --args '{"idx":7,"element":"F","bond":"single"}'
    python spool_client.py beam "CCOc1ccccc1" --beam 5 --inject
    python spool_client.py rationalize 3 --endpoint "<the winning SMILES>" \
        --why "..." --why-optimal "..." --lesson "..."
    python spool_client.py skills
    python spool_client.py inject "CCOc1ccccc1"

`next` prints only the **delta** since the previous request, because turns 1..N-1 are
already in the answering session's context. It detects an episode boundary (the run rebuilds
`messages` from a fresh system prompt) and prints the whole list, labelled NEW EPISODE.

`beam` is the brute-force 3-step search, run on demand instead of on the `--online_fewshot`
timer: no second API model is involved, candidates are scored against the run's own
objective set, and nothing enters the oracle budget until a molecule is actually injected.
Searching is therefore free against the budget, which makes the search *volume* here a
behavioural difference from a timer-driven arm, not just the trigger. Record it with results.

Every `beam` writes a **pending** entry to the beam journal and `next`/`reply` refuse to
proceed until it is closed with `rationalize`, which demands the winning endpoint back (so
the result has actually been read) plus an explanation of why the sequence works and why it
is near-optimal. Closed entries become the skill library, and `next` prints a digest of it at
the top of every new episode. This is the paper's post-rationalise-then-few-shot loop with
the agent doing its own rationalising instead of a second model; it is also the manipulation
that distinguishes this arm, so its results are not directly comparable to one where the
search output merely sits in the log.

The journal is JSONL at $TOOLMOL_BEAM_JOURNAL, defaulting to beam_journal.jsonl in the run
directory (the parent of the spool). Point the variable at an earlier run's file to carry a
library across runs; leave it alone for a per-run library.
"""
import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# run.py appends main/<method> to sys.path, and main/toolmol/mutate.py relies on it with a
# bare `from utils import ...` that resolves to main/toolmol/utils.py. Importing anything
# from main.toolmol outside run.py needs the same entry.
sys.path.insert(0, os.path.join(_HERE, "main", "toolmol"))

STATE = "spool_client_state.json"


def spool_dir():
    d = os.environ.get("TOOLMOL_CLAUDE_SPOOL")
    if not d:
        raise SystemExit("TOOLMOL_CLAUDE_SPOOL is not set; point it at the run's spool dir.")
    if not os.path.isdir(d):
        raise SystemExit(f"{d} is not a directory.")
    return d


def _requests(d):
    ns = []
    for fn in os.listdir(d):
        if fn.startswith("req_") and fn.endswith(".json"):
            try:
                ns.append(int(fn[4:-5]))
            except ValueError:
                pass
    return sorted(ns)


def _answered(d, n):
    return os.path.exists(os.path.join(d, f"resp_{n:05d}.json"))


def _pending(d):
    return [n for n in _requests(d) if not _answered(d, n)]


def _load_request(d, n):
    with open(os.path.join(d, f"req_{n:05d}.json"), encoding="utf-8") as f:
        return json.load(f)


def _write_atomic(path, payload):
    """Same .tmp + os.replace discipline the run uses, so neither side can read a
    half-written file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def _state_path(d):
    return os.path.join(d, STATE)


def _load_state(d):
    p = _state_path(d)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except ValueError:
        return {}


def _save_state(d, state):
    _write_atomic(_state_path(d), state)


# -- beam journal / skill library -------------------------------------------------------
# Deliberately a file in the run directory rather than Claude's own memory store: that store
# is keyed per working directory and shared by every session in it, so findings written there
# would leak into other arms' sessions - the exact contamination channel this experiment
# closed. A run-local JSONL also survives context compaction, is inspectable alongside the
# results, and matches what --online_fewshot already does with online_fewshot_buffer.json.
MIN_WHY = 120
MIN_WHY_OPTIMAL = 80
DIGEST_ENTRIES = 12


def journal_path():
    override = os.environ.get("TOOLMOL_BEAM_JOURNAL")
    if override:
        return override
    return os.path.join(os.path.dirname(os.path.abspath(spool_dir())), "beam_journal.jsonl")


def _load_journal():
    p = journal_path()
    if not os.path.exists(p):
        return []
    entries = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    pass
    return entries


def _save_journal(entries):
    p = journal_path()
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def _open_entries(entries=None):
    """Entries whose rationale has not been filed yet."""
    return [e for e in (entries if entries is not None else _load_journal())
            if not e.get("rationale")]


def _gate(args, verb):
    """Refuse to go on while a beam search is un-rationalised. The whole point of this arm is
    that a search result is not allowed to just sit in the log, so the forcing function is
    mechanical rather than a line of prompt text. --force exists so a wedged run can always be
    driven to completion; it is recorded on the entry so the lapse shows up in the results."""
    open_entries = _open_entries()
    if not open_entries:
        return
    if getattr(args, "force", False):
        entries = _load_journal()
        for e in entries:
            if not e.get("rationale"):
                e["forced_past"] = e.get("forced_past", 0) + 1
        _save_journal(entries)
        print(f"  [forced past {len(open_entries)} un-rationalised beam search(es)]")
        return
    print(f"BLOCKED: {len(open_entries)} beam search(es) still need a rationale before you "
          f"{verb}.\n")
    for e in open_entries:
        print(f"  [{e['id']}] {e['start_smi']}")
        print(f"       -> {e['end_smi']}   delta {e['delta']:+.4f}")
        print(f"       steps: {' -> '.join(e['step_ids'])}")
    print("\nLook at the winning molecule and the sequence that reached it, then close each one:")
    e = open_entries[0]
    print(f'  python spool_client.py rationalize {e["id"]} --endpoint "{e["end_smi"]}" \\\n'
          f'      --why "why these edits move the objectives, step by step" \\\n'
          f'      --why-optimal "why this is near-optimal here - what the alternatives gave up" \\\n'
          f'      --lesson "the one-line generalisation to reuse in later episodes"')
    print("\n(--force overrides, and is recorded on the entry.)")
    raise SystemExit(2)


def _digest():
    entries = [e for e in _load_journal() if e.get("rationale")]
    if not entries:
        return ("=== BEAM SKILL LIBRARY: empty ===\n"
                "No rationalised beam searches yet. Run `spool_client.py beam \"<smiles>\"` on a\n"
                "molecule this run has actually reached to start building it.\n")
    shown = entries[-DIGEST_ENTRIES:]
    out = [f"=== BEAM SKILL LIBRARY: {len(entries)} entry(s)"
           + (f", showing the {len(shown)} most recent" if len(shown) < len(entries) else "")
           + " ==="]
    for e in shown:
        start = e["start_smi"] if len(e["start_smi"]) <= 34 else e["start_smi"][:31] + "..."
        end = e["end_smi"] if len(e["end_smi"]) <= 34 else e["end_smi"][:31] + "..."
        out.append(f"  [{e['id']}] {start}  ->  {end}   delta {e['delta']:+.4f}")
        out.append(f"       {' -> '.join(e['step_ids'])}")
        if e.get("lesson"):
            out.append(f"       lesson: {e['lesson']}")
    out.append("")
    out.append("Before choosing this episode's edits, check whether any of these applies to the")
    out.append("molecules below - the same move on a similar substructure, or the same reason for")
    out.append("preferring one edit over another. `spool_client.py skills` prints the full")
    out.append("reasoning for any entry.")
    out.append("")
    return "\n".join(out)


def _render(messages, start):
    out = []
    for i, m in enumerate(messages[start:], start):
        role = m.get("role", "?")
        content = m.get("content")
        out.append(f"--- [{i}] {role} ---")
        if content:
            out.append(content if isinstance(content, str) else json.dumps(content))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            out.append(f"  tool_call {fn.get('name')}({fn.get('arguments')})")
    return "\n".join(out)


def cmd_status(args):
    d = spool_dir()
    reqs = _requests(d)
    pending = _pending(d)
    print(f"spool: {d}")
    print(f"requests: {len(reqs)}   answered: {len(reqs) - len(pending)}   pending: {len(pending)}")
    if pending:
        print(f"next to answer: {pending[0]}")
    inj = os.path.join(d, "inject.txt")
    if os.path.exists(inj):
        with open(inj) as f:
            queued = [l for l in (x.strip() for x in f) if l]
        print(f"inject.txt: {len(queued)} molecule(s) queued, folded in at the next generation")


def cmd_next(args):
    _gate(args, "read the next turn")
    d = spool_dir()
    pending = _pending(d)
    if not pending:
        print("nothing pending.")
        return
    n = pending[0]
    payload = _load_request(d, n)
    messages = payload["messages"]
    state = _load_state(d)
    prev_len = state.get("last_rendered_len", 0)
    prev_req = state.get("last_rendered_request")
    prev_first_user = state.get("last_first_user")

    # Identify an episode by its opening two-parent prompt: that message is constant within
    # an episode and changes when the run starts the next one. Length alone is not enough -
    # `len(messages) < prev_len` misses the boundary whenever an episode is answered in a
    # single turn, because the next episode's first request is also 2 messages long, and the
    # agent then gets "0 new message(s)" with no molecules in it. The length test is kept as a
    # second signal for the case where consecutive episodes happen to draw the same parents.
    first_user = next((m.get("content") or "" for m in messages if m.get("role") == "user"), "")
    first_user_hash = hashlib.sha1(first_user.encode("utf-8")).hexdigest()

    if prev_req is not None and n == prev_req:
        mode = "reread"
    elif prev_req is None or first_user_hash != prev_first_user or len(messages) <= prev_len:
        mode = "new"
    else:
        mode = "delta"

    start = prev_len if mode == "delta" else 0
    if mode == "new":
        print(_digest())
        print(f"=== NEW EPISODE (request {n}, {len(messages)} messages) ===")
    elif mode == "reread":
        print(f"=== request {n} again ({len(messages)} messages, shown in full) ===")
    else:
        print(f"=== request {n}: {len(messages) - start} new message(s) ===")
    print(_render(messages, start))
    state["last_rendered_len"] = len(messages)
    state["last_rendered_request"] = n
    state["last_first_user"] = first_user_hash
    _save_state(d, state)


def cmd_reply(args):
    _gate(args, "answer this turn")
    d = spool_dir()
    pending = _pending(d)
    if not pending:
        raise SystemExit("nothing pending to reply to.")
    n = args.request or pending[0]
    if _answered(d, n):
        raise SystemExit(f"request {n} is already answered.")
    tool_calls = []
    if args.tool:
        try:
            parsed = json.loads(args.args) if args.args else {}
        except ValueError as e:
            raise SystemExit(f"--args is not valid JSON: {e}")
        tool_calls.append({"name": args.tool, "arguments": parsed})
    payload = {"content": args.content, "tool_calls": tool_calls}
    _write_atomic(os.path.join(d, f"resp_{n:05d}.json"), payload)
    print(f"answered request {n}"
          + (f" with {args.tool}({args.args})" if args.tool else " (no tool call)"))


def cmd_inject(args):
    """Queue molecules for the next generation. They are folded into the population before
    the front is selected, so they compete on the same terms as the agent's own offspring."""
    d = spool_dir()
    smis = [s for s in args.smiles if s.strip()]
    if not smis:
        raise SystemExit("no SMILES given.")
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    bad = [s for s in smis if Chem.MolFromSmiles(s) is None]
    if bad:
        raise SystemExit(f"unparseable SMILES, nothing queued: {bad}")
    path = os.path.join(d, "inject.txt")
    with open(path, "a") as f:
        for s in smis:
            f.write(s.strip() + "\n")
    print(f"queued {len(smis)} molecule(s) in {path}")


def cmd_beam(args):
    """Brute-force 3-step beam search from one molecule, scored against the run's own
    objective set (jnk3 + QED maximized, SA minimized)."""
    from multiprocessing import Pool
    from main.toolmol import objectives
    from main.toolmol.fg_lookup import FUNCTIONAL_GROUPS
    from main.toolmol.online_fewshot import _beam_search_3step, _pool_init

    # This arm's objective set is fixed rather than configured: online_fewshot._pool_init
    # builds ['jnk3', 'qed', 'sa'] in every worker, so the beam already scores against
    # exactly what the run's own ToolMolOracle sums. Rebuilt here only to name them.
    # (The cmumo fork reads a run-exported set via objectives.configured_objectives().)
    objs = objectives.build_objectives(['jnk3', 'qed', 'sa'])
    print("objectives: " + ", ".join(o.name for o in objs))

    group_smiles = list(FUNCTIONAL_GROUPS.values())
    with Pool(processes=args.workers, initializer=_pool_init) as pool:
        result = _beam_search_3step(pool, args.smiles, args.parent2, args.beam, group_smiles)
    if result is None:
        print("no path found.")
        return

    print(f"\nbase Phi {result['base_phi']:.4f}  ->  "
          f"{result['base_phi'] + result['delta']:.4f}  (delta {result['delta']:+.4f})")
    for i, step in enumerate(result["path"], 1):
        raws = " ".join(f"{k}={v:.3f}" for k, v in step["raw"].items())
        print(f"  {i}. {step['tool']}({json.dumps(step['args'])})")
        print(f"     -> {step['smi']}")
        print(f"        Phi {step['phi']:.4f}   {raws}")

    endpoint = result["path"][-1]["smi"]
    if args.inject:
        ns = argparse.Namespace(smiles=[endpoint])
        cmd_inject(ns)
    else:
        print(f"\nnot injected. To fold the endpoint into the next generation:\n"
              f"  python spool_client.py inject \"{endpoint}\"")

    # Step identity: the tool plus its *content* parameter, dropping atom indices - the same
    # notion online_fewshot._step_identity uses for its diversity gate. Indices are purely
    # positional and say nothing about what a move does, so two searches that both attach the
    # same fragment are the same skill even on different molecules and at different indices.
    # Precomputed here, while the heavy imports are already paid for, so the digest and the
    # gate stay import-free.
    step_ids = []
    for step in result["path"]:
        a = step["args"]
        content = (a.get("substructure") or a.get("new_substructure") or a.get("group")
                   or a.get("element"))
        step_ids.append(f"{step['tool']}({content})" if content else step["tool"])

    entries = _load_journal()
    entry = {
        "id": (max((e["id"] for e in entries), default=0) + 1),
        "start_smi": args.smiles,
        "parent2_smi": args.parent2,
        "beam_width": args.beam,
        "base_phi": result["base_phi"],
        "end_smi": endpoint,
        "end_phi": result["path"][-1]["phi"],
        "delta": result["delta"],
        "step_ids": step_ids,
        "path": [{"tool": st["tool"], "args": st["args"], "smi": st["smi"],
                  "phi": st["phi"], "raw": st["raw"]} for st in result["path"]],
        "injected": bool(args.inject),
        "rationale": None,
        "why_optimal": None,
        "lesson": None,
    }
    entries.append(entry)
    _save_journal(entries)

    print(f"\n--- beam journal entry [{entry['id']}] opened (pending a rationale) ---")
    print("next and reply are blocked until you close it. Read the winning molecule and the")
    print("sequence above, then explain why that sequence works and why it is near-optimal")
    print("for this molecule:")
    print(f'  python spool_client.py rationalize {entry["id"]} --endpoint "{endpoint}" \\\n'
          f'      --why "step by step, why each edit moves jnk3 / QED / SA the way it does" \\\n'
          f'      --why-optimal "why this beats the alternatives here - what a greedier or '
          f'different route gave up" \\\n'
          f'      --lesson "one line you want to see in later episodes"')


def cmd_rationalize(args):
    """Close an open beam-journal entry. Requires the winning endpoint back, so the entry
    cannot be closed without having actually read the search output, and refuses rationales
    too short to be an explanation. The length floors are a crude proxy for substance - they
    are there to stop a one-word placeholder, not to be hit exactly."""
    entries = _load_journal()
    match = [e for e in entries if e["id"] == args.id]
    if not match:
        raise SystemExit(f"no beam journal entry [{args.id}]. `skills --all` lists them.")
    entry = match[0]
    if entry.get("rationale"):
        raise SystemExit(f"entry [{args.id}] is already rationalised. `skills --id {args.id}` "
                         f"shows it.")

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    given, expected = Chem.MolFromSmiles(args.endpoint), Chem.MolFromSmiles(entry["end_smi"])
    if given is None:
        raise SystemExit(f"--endpoint is not parseable: {args.endpoint}")
    if Chem.MolToSmiles(given) != Chem.MolToSmiles(expected):
        raise SystemExit(f"--endpoint does not match entry [{args.id}]'s winning molecule.\n"
                         f"  you gave:  {Chem.MolToSmiles(given)}\n"
                         f"  entry has: {Chem.MolToSmiles(expected)}\n"
                         f"Re-read the search output for that entry.")
    if len(args.why.strip()) < MIN_WHY:
        raise SystemExit(f"--why is {len(args.why.strip())} chars; at least {MIN_WHY} are "
                         f"wanted. Explain what each step does to jnk3, QED and SA.")
    if len(args.why_optimal.strip()) < MIN_WHY_OPTIMAL:
        raise SystemExit(f"--why-optimal is {len(args.why_optimal.strip())} chars; at least "
                         f"{MIN_WHY_OPTIMAL} are wanted. Say what the alternatives gave up.")

    entry["rationale"] = args.why.strip()
    entry["why_optimal"] = args.why_optimal.strip()
    entry["lesson"] = (args.lesson or "").strip() or None
    _save_journal(entries)
    still_open = len(_open_entries(entries))
    print(f"entry [{args.id}] closed and added to the skill library.")
    if entry["lesson"]:
        print(f"  lesson: {entry['lesson']}")
    else:
        print("  no --lesson given; the digest will show this entry without a one-line takeaway.")
    print(f"  {still_open} entry(s) still open." if still_open
          else "  nothing open - next and reply are unblocked.")


def cmd_skills(args):
    """The library in full. The per-episode digest is one line per entry; this is where the
    reasoning lives, so an entry that looks relevant can be read properly."""
    entries = _load_journal()
    if not entries:
        print(f"beam journal is empty ({journal_path()}).")
        return
    if args.id is not None:
        entries = [e for e in entries if e["id"] == args.id]
        if not entries:
            raise SystemExit(f"no entry [{args.id}].")
    elif not args.all:
        entries = [e for e in entries if e.get("rationale")]
    print(f"beam journal: {journal_path()}\n")
    for e in entries:
        flag = "" if e.get("rationale") else "   [OPEN - needs a rationale]"
        print(f"=== [{e['id']}]{flag}")
        print(f"  start      {e['start_smi']}")
        if e.get("parent2_smi"):
            print(f"  parent2    {e['parent2_smi']}")
        print(f"  end        {e['end_smi']}")
        print(f"  Phi        {e['base_phi']:.4f} -> {e['end_phi']:.4f}  (delta {e['delta']:+.4f})")
        print(f"  injected   {e['injected']}   beam width {e['beam_width']}")
        for i, st in enumerate(e["path"], 1):
            raws = " ".join(f"{k}={v:.3f}" for k, v in st["raw"].items())
            print(f"  {i}. {st['tool']}({json.dumps(st['args'])})")
            print(f"     -> {st['smi']}")
            print(f"        Phi {st['phi']:.4f}   {raws}")
        if e.get("forced_past"):
            print(f"  NOTE: driven past un-rationalised {e['forced_past']} time(s) with --force")
        if e.get("rationale"):
            print(f"  why:         {e['rationale']}")
            print(f"  why optimal: {e['why_optimal']}")
            if e.get("lesson"):
                print(f"  lesson:      {e['lesson']}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="pending/answered counts and the injection queue"
                   ).set_defaults(fn=cmd_status)
    p = sub.add_parser("next", help="print the delta since the previous request")
    p.add_argument("--force", action="store_true",
                   help="proceed even with an un-rationalised beam search (recorded)")
    p.set_defaults(fn=cmd_next)

    p = sub.add_parser("reply", help="answer the pending request")
    p.add_argument("--content", default=None, help="assistant message text")
    p.add_argument("--tool", default=None, help="tool name to call")
    p.add_argument("--args", default=None, help="tool arguments as a JSON object")
    p.add_argument("--request", type=int, default=None, help="answer this request number")
    p.add_argument("--force", action="store_true",
                   help="proceed even with an un-rationalised beam search (recorded)")
    p.set_defaults(fn=cmd_reply)

    p = sub.add_parser("inject", help="queue SMILES for the next generation")
    p.add_argument("smiles", nargs="+")
    p.set_defaults(fn=cmd_inject)

    p = sub.add_parser("beam", help="brute-force 3-step beam search from one molecule")
    p.add_argument("smiles")
    p.add_argument("--parent2", default=None,
                   help="second parent, enabling crossover at level 1")
    p.add_argument("--beam", type=int, default=5, help="beam width (default 5)")
    p.add_argument("--workers", type=int, default=8,
                   help="pool workers. They only apply tools - scoring is batched in the "
                        "parent - so this costs no extra oracle processes.")
    p.add_argument("--inject", action="store_true",
                   help="queue the winning endpoint for injection")
    p.set_defaults(fn=cmd_beam)

    p = sub.add_parser("rationalize", help="close a beam-journal entry with your reasoning")
    p.add_argument("id", type=int)
    p.add_argument("--endpoint", required=True,
                   help="the winning SMILES from that search, read back to confirm you looked")
    p.add_argument("--why", required=True,
                   help="step by step, why the sequence moves jnk3 / QED / SA as it does")
    p.add_argument("--why-optimal", required=True, dest="why_optimal",
                   help="why it is near-optimal here - what the alternatives gave up")
    p.add_argument("--lesson", default=None,
                   help="one-line generalisation, shown in every later episode's digest")
    p.set_defaults(fn=cmd_rationalize)

    p = sub.add_parser("skills", help="the beam journal in full, with reasoning")
    p.add_argument("--id", type=int, default=None, help="just this entry")
    p.add_argument("--all", action="store_true", help="include entries still open")
    p.set_defaults(fn=cmd_skills)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
