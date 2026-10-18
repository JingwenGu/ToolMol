"""The online few-shot buffer's exact-duplicate check. Run from multi_objective/ as a script
(it starts a worker pool, which on Windows must re-import this file):

    python main/toolmol/test_online_fewshot_dedupe.py

Drives a real update cycle with the beam search and example builder stubbed out, so it needs
no API key, no network and takes a few seconds. Checks that
  - a (start, parent2) pair already accepted is skipped before any search runs, as is a pair
    repeated within one window, and neither costs an attempt;
  - a result whose final molecule was already in the buffer is rejected, including one that
    only matches an entry accepted earlier in the same cycle, and one matching an entry that
    has since been evicted from the FIFO buffer;
  - the same molecule written two ways counts as one;
  - the history survives a save/load round trip, so a restarted run does not re-accept.
"""
import json
import os
import sys
import tempfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(REPO)
sys.path.insert(0, REPO)
os.environ.setdefault("OPENAI_API_KEY", "unused-by-this-test")

import main.toolmol.online_fewshot as of

failures = []


def expect(cond, msg):
    print(f"  {'pass' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


def entry(start, parent2, final, delta=0.1):
    steps = [{'tool': 'add_atom', 'args': {'idx': 0, 'element': 'N', 'bond': 'single'}, 'smi': start, 'phi': 1.0}]
    return {'messages': [], 'path': steps, 'working_smi': start, 'parent2_smi': parent2,
            'delta': delta, 'final_smi': final, 'phi': 1.0}


# distinct, valid molecules
S0, P0, F0 = "CCO", "CCN", "CCCl"          # an entry already in the buffer
S1, P1 = "c1ccccc1", "CC(=O)O"
S2, P2, F2 = "c1ccncc1", "CCC", "COC"
S3, P3 = "CCS", "CCBr"
S4, P4, F4 = "CC(C)O", "CCCC", "CCCCO"
F0_ALT = "ClCCC"                            # 1-chloropropane: a different molecule from F0
F0_SAME_OTHER_SPELLING = "C(C)Cl"           # chloroethane, written differently from F0 "CCCl"

# fake beam search: the final molecule each start leads to, as a 3-step path with unique step identities
FINAL_FOR = {S1: F0, S2: F2, S3: F2, S4: F4}
beam_calls = []


def fake_beam(pool, working_smi, parent2_smi, beam_width, group_smiles, gate_log=None):
    beam_calls.append(working_smi)
    final = FINAL_FOR[working_smi]
    tag = len(beam_calls)
    path = [{'tool': 'add_atom', 'args': {'idx': 0, 'element': el, 'bond': 'single'}, 'smi': final, 'phi': 1.5}
            for el in ('C', 'N', 'O')]
    path = [dict(p, args=dict(p['args'], element=e)) for p, e in zip(path, ('C', 'N', 'O'))]
    # rotate the step identities per call so the step-identity gate never trips in this test
    rot = ('F', 'Cl', 'Br', 'S')[tag % 4]
    path[0]['args']['element'] = rot
    path[-1]['smi'] = final
    return {'base_phi': 1.0, 'base_raw': {}, 'path': path, 'delta': 0.5}


def fake_build(working_smi, parent2_smi, result):
    return entry(working_smi, parent2_smi, result['path'][-1]['smi'], delta=result['delta']), None


def make_buffer(state_file, **kw):
    return of.OnlineFewShotBuffer(buffer_size=3, update_every=10, sample_size=3, workers=1,
                                  state_file=state_file, log=lambda *a, **k: None, **kw)


def main():
    real_beam = of._beam_search_3step
    of._beam_search_3step = fake_beam
    tmp = tempfile.mkdtemp()
    state = os.path.join(tmp, "buf.json")
    buf = make_buffer(state)
    buf._build_example = fake_build
    try:
        print("canonicalisation:")
        expect(of._canon(F0) == of._canon(F0_SAME_OTHER_SPELLING), "same molecule, two spellings -> one key")
        expect(of._canon(F0) != of._canon(F0_ALT), "different molecules stay different")
        expect(of._exact_duplicate_reason(F0_SAME_OTHER_SPELLING, {of._canon(F0)}) is not None,
               "_exact_duplicate_reason matches across spellings")
        expect(of._exact_duplicate_reason(F2, {of._canon(F0)}) is None, "no false match on a new molecule")

        print("\none update cycle:")
        # pre-load the history as if (S0,P0)->F0 were already in the buffer
        with buf._lock:
            buf._buffer_entries = [entry(S0, P0, F0, delta=0.1)]
            buf._register_history(buf._buffer_entries)
        recent = [(S0, P0),          # pair already accepted -> skipped pre-search
                  (S1, P1),          # searches, lands on F0 -> exact-duplicate reject
                  (S2, P2),          # searches, new final F2 -> accepted
                  (S2, P2),          # repeated pair in the window -> skipped pre-search
                  (S3, P3),          # searches, lands on F2 (accepted this cycle) -> reject
                  (S4, P4)]          # searches, new final F4 -> accepted
        beam_calls.clear()
        buf._run_update_cycle(recent, list(recent))
        # candidates are shuffled, so compare as a set; sample_size=3 > the 2 accepts possible here
        # means no candidate is cut off by the per-cycle cap whatever the order
        expect(len(beam_calls) == 4 and set(beam_calls) == {S1, S2, S3, S4},
               f"exactly 4 searches ran, one per non-skipped pair ({sorted(beam_calls)})")
        finals = sorted(e['final_smi'] for e in buf._buffer_entries)
        expect(finals == sorted([F0, F2, F4]), f"buffer holds the original plus the two new finals ({finals})")
        expect(sorted(buf.drain_injections()) == sorted([F2, F4]), "only the two accepted finals queued for injection")
        expect(len([e for e in buf._buffer_entries if e['final_smi'] == F2]) == 1,
               "F2 reached by two different starts in one cycle is in the buffer once")

        # which of (S2,P2)/(S3,P3) won F2 depends on the shuffle; the loser is correctly NOT recorded
        accepted_pairs = [(e['working_smi'], e['parent2_smi']) for e in buf._buffer_entries
                          if e['final_smi'] in (F2, F4)]
        expect(len(accepted_pairs) == 2, f"two pairs were accepted this cycle ({accepted_pairs})")

        print("\nhistory outlives the FIFO buffer:")
        # buffer_size is 3 and now holds F0,F2,F4. Push F0 out with two more entries, then offer F0 again.
        with buf._lock:
            extra = [entry("CCCC", "CCCCC", "CCCCCC", 0.2), entry("CCCCCCC", "C", "CCCCCCCC", 0.2)]
            buf._buffer_entries = (buf._buffer_entries + extra)[-3:]
            buf._register_history(extra)
        expect(F0 not in [e['final_smi'] for e in buf._buffer_entries], "F0's entry has been evicted from the buffer")
        reason = of._exact_duplicate_reason(F0, buf._seen_finals)
        expect(reason is not None, "yet F0 is still rejected as an exact duplicate (run-long history)")

        print("\nsave / load round trip:")
        buf._save_state()
        saved = json.load(open(state))
        expect(of._canon(F0) in saved['seen_finals'] and of._canon(F4) in saved['seen_finals'],
               "state file records seen_finals, including the evicted F0")
        buf2 = make_buffer(None, load_buffer_file=state)
        buf2._build_example = fake_build
        expect(buf2._seen_finals == buf._seen_finals and buf2._seen_pairs == buf._seen_pairs,
               "a buffer loaded from the file has the same history")
        beam_calls.clear()
        buf2._run_update_cycle(accepted_pairs, list(accepted_pairs))
        expect(beam_calls == [], "restarted run skips pairs it accepted before the restart")
        buf2.shutdown()
    finally:
        of._beam_search_3step = real_beam
        buf.shutdown()

    print()
    if failures:
        for f in failures:
            print("FAILURE:", f)
        return 1
    print("exact-duplicate check behaves as expected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
