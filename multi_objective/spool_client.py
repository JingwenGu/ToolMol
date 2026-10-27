"""The agent's interface to a `--llm_backend claude` run: read turns and answer them.

With that backend there is no model behind an endpoint. The run spools each agent turn to
`$TOOLMOL_CLAUDE_SPOOL/req_NNNNN.json` and blocks until `resp_NNNNN.json` appears, so the
**answering session is the agent** and the run's whole history accumulates in its context
for free rather than being resent every turn.

    export TOOLMOL_CLAUDE_SPOOL=<spool dir>
    python spool_client.py status
    python spool_client.py next
    python spool_client.py reply --content "..." --tool add_atom --args '{"idx":7,"element":"F","bond":"SINGLE"}'

`next` prints only the **delta** since the previous request, because turns 1..N-1 are
already in the answering session's context. It detects an episode boundary (the run rebuilds
`messages` from a fresh system prompt) and prints the whole list, labelled NEW EPISODE.

This arm has **no beam search and no injection** - that is what separates it from
`cmumo_objectives_opfs`, where `spool_client.py` also carries `beam` and `inject`. Here the
population changes only through the agent's own tool edits, so a result is attributable to
the agent rather than to a brute-force search running beside it.
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






def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="pending and answered counts"
                   ).set_defaults(fn=cmd_status)
    sub.add_parser("next", help="print the delta since the previous request"
                   ).set_defaults(fn=cmd_next)

    p = sub.add_parser("reply", help="answer the pending request")
    p.add_argument("--content", default=None, help="assistant message text")
    p.add_argument("--tool", default=None, help="tool name to call")
    p.add_argument("--args", default=None, help="tool arguments as a JSON object")
    p.add_argument("--request", type=int, default=None, help="answer this request number")
    p.set_defaults(fn=cmd_reply)



    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
