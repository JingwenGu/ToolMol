"""Operator side of agent.py's `interactive` backend.

`driver.py` runs the pipeline in the background; whenever ToolMolAgent needs an assistant
turn it drops `request_<n>.json` in the run's ipc/ directory and blocks. This script shows
the pending request compactly and writes `response_<n>.json` back.

    python toolmol_interactive/respond.py wait  --seed 1
    python toolmol_interactive/respond.py reply --seed 1 '{"content": "...", "tool_calls":
        [{"name": "crossover_molecules", "arguments": {"idx1": 7, "idx2": 3}}]}'
    python toolmol_interactive/respond.py reply --seed 1 '[ {...turn k...}, {...turn k+1...} ]'
    python toolmol_interactive/respond.py status --seed 1
    python toolmol_interactive/respond.py ledger --seed 1

`reply` with a list applies each response to successive turns, waiting for each new request
in between, then prints whatever request is pending afterwards. That is purely a round-trip
saver for turns whose content is already decided (e.g. the closing FINAL ANSWER); each
response still lands on exactly the turn it would have.

Nothing here feeds the agent anything the prompt does not already contain. The one derived
number it prints, `jnk3~`, is arithmetic on the prompt's own text: Phi = jnk3 + QED +
(10 - SA)/9 for this task (main/toolmol/objectives.py OBJECTIVE_BOUNDS), and the prompt
states Phi ("Score:") and both QED and SA, so jnk3 = Phi - QED - (10 - SA)/9.
"""

from __future__ import print_function

import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))

POLL = 0.2


def run_dir(seed, explicit=None):
    return explicit or os.path.join(HERE, 'runs', f'seed{seed}')


def ipc_dir(seed, explicit=None):
    return os.path.join(run_dir(seed, explicit), 'ipc')


def pending(ipc):
    """(seq, path) of the lowest-numbered request with no response yet, or None."""
    for path in sorted(glob.glob(os.path.join(ipc, 'request_*.json'))):
        seq = int(re.search(r'request_(\d+)\.json$', path).group(1))
        if not os.path.exists(os.path.join(ipc, f'response_{seq:05d}.json')):
            return seq, path
    return None


def finished(ipc):
    return os.path.exists(os.path.join(ipc, 'RUN_FINISHED'))


def wait_for_pending(ipc, timeout, after_seq=None):
    """Block until a request is pending (optionally one newer than after_seq)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = pending(ipc)
        if found and (after_seq is None or found[0] > after_seq):
            return found
        if finished(ipc):
            return None
        time.sleep(POLL)
    return None


# --------------------------------------------------------------------------- rendering

PHI_NOTE = re.compile(r'^(\d+)\.\s+(\S+)\s*$')


def _derived_jnk3(block):
    """Given one parent's prompt block, return (phi, qed, sa, derived_jnk3) or None."""
    m_score = re.search(r'^Score:\s*([-\d.eE]+)\s*$', block, re.M)
    m_props = re.search(r'^Properties:\s*(.*)$', block, re.M)
    if not (m_score and m_props):
        return None
    props = dict(kv.split('=', 1) for kv in m_props.group(1).split() if '=' in kv)
    try:
        phi = float(m_score.group(1))
        qed = float(props['QED'])
        sa = float(props['SA'])
    except (KeyError, ValueError):
        return None
    return phi, qed, sa, phi - qed - (10.0 - sa) / 9.0


def render(seq, req, log_tail=None):
    messages = req['messages']
    out = []
    out.append(f"===== TURN {seq}  ({len(messages)} messages) =====")
    if log_tail:
        out.append(f"[driver] {log_tail}")

    user_idxs = [i for i, m in enumerate(messages) if m['role'] == 'user']
    first_turn = len(user_idxs) == 1

    if first_turn:
        body = messages[user_idxs[0]]['content']
        out.append(body)
        # Annotate each parent with the jnk3 implied by its Phi / QED / SA.
        parts = re.split(r'\n\n(?=\d+\.\s)', body)
        for part in parts[1:]:
            tag = part.split('\n', 1)[0].strip()
            d = _derived_jnk3(part)
            if d:
                phi, qed, sa, jnk3 = d
                out.append(f"[derived] ligand {tag}  Phi={phi:.4f} = jnk3~{jnk3:.4f} + "
                           f"QED {qed:.4f} + SA-term {(10 - sa) / 9:.4f}")
    else:
        # Everything since (and including) the previous assistant turn.
        last_assistant = max(i for i, m in enumerate(messages) if m['role'] == 'assistant')
        for m in messages[last_assistant:]:
            if m['role'] == 'assistant':
                for tc in m.get('tool_calls') or []:
                    out.append(f"[my last call] {tc['function']['name']}"
                               f"({tc['function']['arguments']})")
                if not (m.get('tool_calls')) and m.get('content'):
                    out.append(f"[my last message] {m['content'][:300]}")
            elif m['role'] == 'tool':
                head = m['content'].split('\n', 1)[0]
                out.append(f"[tool result] {head}")
            elif m['role'] == 'user':
                out.append(m['content'])
    out.append("===== respond with: content / tool_calls =====")
    return "\n".join(out)


def log_tail_line(seed, explicit):
    path = os.path.join(run_dir(seed, explicit), 'run_log.txt')
    if not os.path.exists(path):
        return None
    pat = re.compile(r'(generation \d+: population=.*|generation \d+ pair \d+/\d+: starting edit_pair)')
    last = None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = pat.search(line)
            if m:
                last = m.group(1)
    return last


# --------------------------------------------------------------------------- commands

def write_response(ipc, seq, payload):
    path = os.path.join(ipc, f'response_{seq:05d}.json')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def cmd_wait(a):
    ipc = ipc_dir(a.seed, a.run_dir)
    found = wait_for_pending(ipc, a.timeout)
    if found is None:
        print("RUN_FINISHED" if finished(ipc) else f"no pending request within {a.timeout}s")
        return 0 if finished(ipc) else 1
    seq, path = found
    with open(path, encoding='utf-8') as f:
        req = json.load(f)
    print(render(seq, req, log_tail_line(a.seed, a.run_dir)))
    return 0


def cmd_reply(a):
    ipc = ipc_dir(a.seed, a.run_dir)
    payload = json.loads(a.response)
    responses = payload if isinstance(payload, list) else [payload]

    last_seq = None
    for i, resp in enumerate(responses):
        found = wait_for_pending(ipc, a.timeout, after_seq=last_seq)
        if found is None:
            print(f"!! no request pending for response #{i + 1} "
                  f"({'run finished' if finished(ipc) else 'timed out'})")
            return 1
        seq, _ = found
        write_response(ipc, seq, resp)
        names = [tc.get('name') for tc in (resp.get('tool_calls') or [])] or ['(no tool call)']
        print(f"[ok] turn {seq} <- {', '.join(names)}")
        last_seq = seq

    if a.no_wait:
        return 0
    found = wait_for_pending(ipc, a.timeout, after_seq=last_seq)
    if found is None:
        print("RUN_FINISHED" if finished(ipc) else f"no next request within {a.timeout}s")
        return 0 if finished(ipc) else 1
    seq, path = found
    with open(path, encoding='utf-8') as f:
        req = json.load(f)
    print(render(seq, req, log_tail_line(a.seed, a.run_dir)))
    return 0


def read_ledger(seed, explicit):
    path = os.path.join(run_dir(seed, explicit), 'oracle_ledger.jsonl')
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _phi(raw):
    return raw.get('jnk3', 0.0) + raw.get('qed', 0.0) + (10.0 - raw.get('sa', 10.0)) / 9.0


def cmd_ledger(a):
    rows = read_ledger(a.seed, a.run_dir)
    for r in rows:
        r['phi'] = _phi(r['raw'])
    if a.sort == 'jnk3':
        rows.sort(key=lambda r: (-r['raw'].get('jnk3', 0.0), -r['phi']))
    elif a.sort == 'phi':
        rows.sort(key=lambda r: -r['phi'])
    rows = rows[:a.top] if a.top else rows
    print(f"{'call':>4} {'jnk3':>6} {'qed':>6} {'sa':>6} {'phi':>6}  smiles")
    for r in rows:
        raw = r['raw']
        print(f"{r['call']:>4} {raw.get('jnk3', 0):>6.3f} {raw.get('qed', 0):>6.3f} "
              f"{raw.get('sa', 0):>6.3f} {r['phi']:>6.3f}  {r['smiles']}")
    return 0


def cmd_status(a):
    ipc = ipc_dir(a.seed, a.run_dir)
    rows = read_ledger(a.seed, a.run_dir)
    print(f"run_dir      : {run_dir(a.seed, a.run_dir)}")
    print(f"oracle calls : {len(rows)}")
    print(f"finished     : {finished(ipc)}")
    found = pending(ipc)
    print(f"pending turn : {found[0] if found else None}")
    print(f"driver       : {log_tail_line(a.seed, a.run_dir)}")
    if rows:
        best = max(rows, key=lambda r: _phi(r['raw']))
        bj = max(rows, key=lambda r: r['raw'].get('jnk3', 0.0))
        print(f"best phi     : {_phi(best['raw']):.4f}  {best['smiles']}")
        print(f"best jnk3    : {bj['raw'].get('jnk3', 0):.4f}  {bj['smiles']}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--run-dir', default=None)
    p.add_argument('--timeout', type=float, default=900.0)
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('wait')
    r = sub.add_parser('reply')
    r.add_argument('response', help='a response object, or a JSON list of them')
    r.add_argument('--no-wait', action='store_true')
    sub.add_parser('status')
    l = sub.add_parser('ledger')
    l.add_argument('--sort', choices=['call', 'jnk3', 'phi'], default='call')
    l.add_argument('--top', type=int, default=0)

    a = p.parse_args()
    return {'wait': cmd_wait, 'reply': cmd_reply, 'status': cmd_status,
            'ledger': cmd_ledger}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
