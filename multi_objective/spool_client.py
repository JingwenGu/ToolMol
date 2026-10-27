"""The agent's interface to a `--llm_backend claude` run: read turns, answer them, search,
inject.

With that backend there is no model behind an endpoint. The run spools each agent turn to
`$TOOLMOL_CLAUDE_SPOOL/req_NNNNN.json` and blocks until `resp_NNNNN.json` appears, so the
**answering session is the agent** and the run's whole history accumulates in its context
for free rather than being resent every turn.

    export TOOLMOL_CLAUDE_SPOOL=<spool dir>
    python spool_client.py status
    python spool_client.py next
    python spool_client.py reply --content "..." --tool add_atom --args '{"idx":7,"element":"F","bond":"SINGLE"}'
    python spool_client.py beam "CCOc1ccccc1" --beam 5 --inject
    python spool_client.py inject "CCOc1ccccc1"

`next` prints only the **delta** since the previous request, because turns 1..N-1 are
already in the answering session's context. It detects an episode boundary (the run rebuilds
`messages` from a fresh system prompt) and prints the whole list, labelled NEW EPISODE.

`beam` is the brute-force 3-step search, run on demand instead of on the `--online_fewshot`
timer: no second API model is involved, candidates are scored against the run's own
objective set, and nothing enters the oracle budget until a molecule is actually injected.
Searching is therefore free against the budget, which makes the search *volume* here a
behavioural difference from a timer-driven arm, not just the trigger. Record it with results.
"""
import argparse
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

    # An episode boundary: the run rebuilds `messages` from a fresh system prompt, so the
    # list is shorter than the one we last rendered (or starts a new conversation).
    new_episode = prev_req is None or len(messages) < prev_len
    start = 0 if new_episode else prev_len
    if new_episode:
        print(f"=== NEW EPISODE (request {n}, {len(messages)} messages) ===")
    else:
        print(f"=== request {n}: {len(messages) - start} new message(s) ===")
    print(_render(messages, start))
    state["last_rendered_len"] = len(messages)
    state["last_rendered_request"] = n
    _save_state(d, state)


def cmd_reply(args):
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
    the front is selected, so they compete on the same terms as the agent's own offspring -
    and in this arm they must also clear the violator pre-filter and the front cap."""
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
    objective set (CMUMO_IMPROVE / CMUMO_MAINTAIN, which the run exports)."""
    from multiprocessing import Pool
    from main.toolmol import objectives as cmumo_objectives
    from main.toolmol.fg_lookup import FUNCTIONAL_GROUPS
    from main.toolmol.online_fewshot import _beam_search_3step, _pool_init

    objs = cmumo_objectives.configured_objectives()
    print("objectives: " + ", ".join(
        f"{o.name}({'maintain' if getattr(o, 'kind', None) == 'maintain' else 'improve'})"
        for o in objs))

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="pending/answered counts and the injection queue"
                   ).set_defaults(fn=cmd_status)
    sub.add_parser("next", help="print the delta since the previous request"
                   ).set_defaults(fn=cmd_next)

    p = sub.add_parser("reply", help="answer the pending request")
    p.add_argument("--content", default=None, help="assistant message text")
    p.add_argument("--tool", default=None, help="tool name to call")
    p.add_argument("--args", default=None, help="tool arguments as a JSON object")
    p.add_argument("--request", type=int, default=None, help="answer this request number")
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

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
