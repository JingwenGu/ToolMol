"""On-policy few-shot buffer: maintained live during a ToolMol run instead of built once
offline (see main/toolmol/fewshot_gen/ for the offline pipeline this reuses the method
from). Every `update_every` episodes, OnlineFewShotBuffer.record_episode() fires a
background thread that:

  1. samples a few starting states (working_smi, parent2) from the episodes just seen,
  2. runs a reduced-width non-myopic beam search on each (same method as
     fewshot_gen/beam_search_3step.py, just cheaper - default beam=5, not 30),
  3. rejects a candidate and draws a replacement from the same recent-episode pool if the
     search/message-building pipeline fails outright, the candidate (or its parent2, or its
     outcome molecule) is too structurally similar to what's already in the buffer, its
     action sequence overlaps too much with existing examples at the same step position
     (_passes_diversity_gate), or its final weighted score doesn't beat the weakest example
     currently in the buffer (the "quality bar" - skipped when the buffer is empty),
  4. post-rationalizes the surviving winners with one combined LLM call each (same method
     as fewshot_gen/post_rationalize_3step.py), and re-invokes the real tool functions to
     build authentic messages (same method as fewshot_gen/build_fewshot_library_3step.py),
  5. splices the result into a fixed-size FIFO buffer (oldest examples evicted first - the
     buffer is meant to track whatever the run has recently been encountering, not hold a
     permanent set).

Every `update_every`-episode window gets exactly one cycle - none are skipped. The main GA
loop runs freely alongside a cycle for one window, but if that cycle is still running when the
next window closes, record_episode() blocks the loop until it finishes before launching the
next one (backpressure: the live episodes can run at most one window ahead of the beam search).
Every read of the live few-shot text (`few_shot_messages` property) is a cheap lock-guarded
list copy, not a wait.

Deliberately reuses the exact candidate-generation/scoring code from beam_search_3step.py
rather than importing it, since that script is a fixed, git-tracked artifact of the earlier
10-molecule offline investigation - it should stay exactly as it was for that investigation
to remain reproducible, not become a shared dependency that changes out from under it.
"""
import os
import json
import random
import re
import threading
import time
from multiprocessing import Pool

from openai import OpenAI
from rdkit import Chem

from main.toolmol.agent import SYSTEM_PROMPT, _format_context, _canonicalize
from main.toolmol.toolbox import TOOL_DISPATCH
from main.toolmol.fg_lookup import FUNCTIONAL_GROUPS
from main.toolmol.objectives import build_objectives

ELEMENTS = ['C', 'N', 'O', 'F', 'Cl', 'Br', 'S']
BONDS = ['single', 'double', 'triple']

# ---------------------------------------------------------------------------
# Pool worker globals/functions - module-level so they're picklable across
# multiprocessing.Pool workers, exactly as in fewshot_gen/beam_search_3step.py.
# ---------------------------------------------------------------------------
_objectives = None
_group_smiles = None
_Chem = None
_toolbox = None


def _pool_init():
    global _objectives, _group_smiles, _Chem, _toolbox
    from rdkit import Chem as _C
    from main.toolmol import toolbox as _tb
    _Chem = _C
    _toolbox = _tb
    _group_smiles = list(FUNCTIONAL_GROUPS.values())
    _objectives = build_objectives(['jnk3', 'qed', 'sa'])


def _phi_and_raw(smi):
    m = _Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    raw = {}
    total = 0.0
    for obj in _objectives:
        r = obj.raw(smi)
        raw[obj.name] = r
        total += obj.rescaled(r)
    return total, raw


def _ring_free_atom_indices(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if any(not b.IsInRing() for b in a.GetBonds())]


def _call_tool(tool_name, working_smi, mol2_smi, args):
    if tool_name == 'crossover_molecules':
        # Crossover candidates only exist at beam level 1 (see _gen_candidates_main), where
        # working_smi is the frozen original parent1 - exactly what the live agent crosses over.
        return _toolbox.crossover_molecules(working_smi, args['idx1'], mol2_smi, args['idx2'])
    return getattr(_toolbox, tool_name)(working_smi, **args)


def _eval_one(task):
    working_smi, mol2_smi, baseline_phi, tool_name, args = task
    try:
        result = _call_tool(tool_name, working_smi, mol2_smi, args)
    except Exception:
        return None
    if not result.success:
        return None
    try:
        out_smi = _Chem.MolToSmiles(result.mol)
    except Exception:
        return None
    phi, raw = _phi_and_raw(out_smi)
    if phi is None:
        return None
    return {'tool': tool_name, 'args': args, 'smi': out_smi, 'phi': phi,
            'delta': phi - baseline_phi, 'raw': raw}


def _gen_candidates_main(mol, mol2_smi, group_smiles, include_crossover=False):
    """Every single-tool action available from `mol`. crossover_molecules is only included when
    asked: in the live agent it always acts on the frozen original parent1 and replaces the
    working molecule with the result, so a crossover after earlier edits silently discards them -
    and scoring it against an evolved molecule (as this search once did at levels 2-3) produces
    paths the live model cannot reproduce. Callers therefore enable it at level 1 only, where
    `mol` is the frozen parent1."""
    from rdkit import Chem
    n = mol.GetNumAtoms()
    for idx in range(n):
        for el in ELEMENTS:
            for bond in BONDS:
                yield 'add_atom', {'idx': idx, 'element': el, 'bond': bond}
            yield 'replace_atom', {'idx': idx, 'element': el}
        for frag in group_smiles:
            for bond in BONDS:
                yield 'add_substructure', {'idx': idx, 'substructure': frag, 'bond': bond}
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for anchor, branch in ((a1, a2), (a2, a1)):
            yield 'remove_substructure', {'anchor_idx': anchor, 'branch_idx': branch}
            for frag in group_smiles:
                yield 'replace_substructure', {'anchor_idx': anchor, 'branch_idx': branch, 'new_substructure': frag}
    if include_crossover and mol2_smi:
        mol2 = Chem.MolFromSmiles(mol2_smi)
        for i1 in _ring_free_atom_indices(mol):
            for i2 in _ring_free_atom_indices(mol2):
                yield 'crossover_molecules', {'idx1': i1, 'idx2': i2}


def _expand(pool, smi, mol2_smi, baseline_phi, group_smiles, chunksize=100, include_crossover=False):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    tasks = [(smi, mol2_smi, baseline_phi, t, a)
             for t, a in _gen_candidates_main(mol, mol2_smi, group_smiles, include_crossover)]
    if not tasks:
        return []
    results = pool.map(_eval_one, tasks, chunksize=chunksize)
    return [r for r in results if r is not None]


def _beam_search_3step(pool, working_smi, parent2_smi, beam_width, group_smiles):
    """Same algorithm as fewshot_gen/beam_search_3step.py's run_one_molecule, trimmed to
    just what the online buffer needs (no per-depth bookkeeping, no file I/O). Crossover is
    searched at level 1 only - see _gen_candidates_main."""
    from rdkit import Chem
    base_phi, base_raw = pool.apply(_phi_and_raw, (working_smi,))
    if base_phi is None:
        return None

    level1 = _expand(pool, working_smi, parent2_smi, base_phi, group_smiles, include_crossover=True)
    for r in level1:
        r['path'] = [{'tool': r['tool'], 'args': r['args'], 'smi': r['smi'], 'phi': r['phi'], 'raw': r['raw']}]
    beam1 = sorted(level1, key=lambda r: -r['delta'])[:beam_width]

    level2 = []
    for node in beam1:
        res = _expand(pool, node['smi'], parent2_smi, base_phi, group_smiles)
        for r in res:
            r['path'] = node['path'] + [{'tool': r['tool'], 'args': r['args'], 'smi': r['smi'], 'phi': r['phi'], 'raw': r['raw']}]
        level2.extend(res)
    beam2 = sorted(level2, key=lambda r: -r['delta'])[:beam_width]

    level3 = []
    for node in beam2:
        res = _expand(pool, node['smi'], parent2_smi, base_phi, group_smiles)
        for r in res:
            r['path'] = node['path'] + [{'tool': r['tool'], 'args': r['args'], 'smi': r['smi'], 'phi': r['phi'], 'raw': r['raw']}]
        level3.extend(res)

    all_results = level1 + level2 + level3
    if not all_results:
        return None
    best = max(all_results, key=lambda r: r['delta'])
    return {'base_phi': base_phi, 'base_raw': base_raw, 'path': best['path'], 'delta': best['delta']}


# ---------------------------------------------------------------------------
# Diversity gates
# ---------------------------------------------------------------------------

def _step_identity(tool, args):
    """What makes two steps "the same kind of move," for diversity purposes: same tool AND
    same content parameter - the inserted fragment for add_substructure/replace_substructure,
    the group name for add_functional_group, the target element for add_atom/replace_atom
    (add_atom(F) and add_atom(C) are different; three add_atom(F) calls at three different
    atom indices on three different molecules are the *same* move, since the index is purely
    positional and carries no information about what the step does). crossover_molecules and
    remove_substructure have no non-positional parameter at all, so their identity is the
    tool name alone - any two crossover steps count as the same move regardless of which
    atoms they cut, which is deliberate: crossover is exactly the kind of step we don't want
    several buffer examples leaning on identically."""
    if tool in ('add_substructure', 'replace_substructure'):
        return (tool, args.get('substructure') or args.get('new_substructure'))
    elif tool == 'add_functional_group':
        return (tool, args.get('group'))
    elif tool in ('add_atom', 'replace_atom'):
        return (tool, args.get('element'))
    else:  # crossover_molecules, remove_substructure
        return (tool,)


def _summarize_entry(entry):
    """One-line, human-scannable summary of a buffer entry for the mid-run visibility this
    was specifically added for: starting molecule (truncated - full SMILES can run past 100
    chars and would swamp the log), the step-by-step identity sequence, and the delta - so a
    log skim answers both "does this look diverse" and "does this look like a good example"
    without needing to reconstruct anything by hand."""
    smi = entry['working_smi']
    smi_short = smi if len(smi) <= 40 else smi[:37] + '...'
    steps = ' -> '.join(str(_step_identity(s['tool'], s['args'])) for s in entry['path'])
    delta = entry.get('delta')
    delta_str = f"delta={delta:+.4f}" if delta is not None else "delta=?"
    return f"{smi_short} | {steps} | {delta_str}"


def _passes_diversity_gate(candidate_path, existing_paths, max_repeats):
    """Position-aware: a candidate's step i is only compared against other examples' own
    step i - an opening crossover and a step-3 cleanup edit are different *roles* even when
    they happen to share a tool, so they never compete with each other. Up to `max_repeats`
    existing examples may already use a given (step index, identity) before a further match
    is rejected - not "any single shared step disqualifies," which was too strict for a
    small, finite action vocabulary (7 elements, a fixed functional-group list) where
    unrelated molecules landing on the same good single step is expected, not evidence of
    redundancy. Returns (True, None) on acceptance or (False, reason) on rejection."""
    for step_idx, step in enumerate(candidate_path):
        cand_id = _step_identity(step['tool'], step['args'])
        matches = [p for p in existing_paths
                   if step_idx < len(p) and _step_identity(p[step_idx]['tool'], p[step_idx]['args']) == cand_id]
        if len(matches) >= max_repeats:
            return False, (f"step {step_idx + 1} ({cand_id}) already used by {len(matches)} existing "
                            f"example(s) at that step - max {max_repeats} allowed to share it")
    return True, None


class OnlineFewShotBuffer:
    def __init__(self, buffer_size=10, update_every=100, sample_size=2, beam_width=5,
                 workers=8, llm_model="gpt-5.4", api_key_env="OPENAI_API_KEY",
                 goal_description="improve the configured objectives",
                 diversity_evaluator=None, diversity_floor_frac=0.6,
                 signature_max_repeats=2, seed_examples_file=None,
                 state_file=None, load_buffer_file=None, log=print):
        if seed_examples_file and load_buffer_file:
            raise ValueError("seed_examples_file and load_buffer_file are mutually exclusive - "
                             "both choose what the buffer starts with.")
        self.state_file = state_file
        self.buffer_size = buffer_size
        self.update_every = update_every
        self.sample_size = sample_size
        self.beam_width = beam_width
        self.diversity_evaluator = diversity_evaluator
        self.diversity_floor_frac = diversity_floor_frac
        self.signature_max_repeats = signature_max_repeats
        self.log = log

        self._group_smiles = list(FUNCTIONAL_GROUPS.values())
        self._objectives = build_objectives(['jnk3', 'qed', 'sa'])
        self._pool = Pool(processes=workers, initializer=_pool_init)
        self._client = OpenAI(api_key=os.environ[api_key_env])
        self._llm_model = llm_model
        self._goal_description = goal_description

        # Private RNG for every randomized draw this class makes (candidate shuffling, and
        # re-invoking crossover_molecules to rebuild authentic messages) - never the global
        # random module. This class's work runs on a background thread concurrently with the
        # live GA loop, which itself calls crossover_molecules (via the shared global random
        # module, deliberately seeded once by pareto_optimizer.py's optimize() for
        # reproducibility). Drawing from the same global stream here would interleave this
        # thread's draws with the live loop's in a wall-clock-dependent, non-reproducible
        # order - confirmed live: the first update cycle fires partway through generation 1,
        # before any buffer content has even changed, so runs already diverged before the
        # buffers differed at all.
        self._rng = random.Random()

        self._lock = threading.Lock()
        self._episode_log = []      # [(working_smi, parent2_smi), ...] - unbounded for this run
        self._episode_count = 0
        self._buffer_entries = []   # [{messages, path, base_phi, working_smi, parent2_smi}, ...]
        self._thread = None
        self._stop = threading.Event()  # set by shutdown() so an in-flight cycle ends after its current attempt
        self._population_diversity_ref = None  # set via update_population_diversity()
        self._pending_injections = []  # final_smi of each newly-accepted entry, drained once per
                                        # generation by the main loop (see drain_injections) - only
                                        # populated by genuinely on-policy accepts, never by seeding,
                                        # since seeded entries aren't a new discovery to fold in

        self.n_updates_started = 0
        self.n_updates_completed = 0

        if seed_examples_file:
            self._seed_from_file(seed_examples_file)
        elif load_buffer_file:
            self._load_state(load_buffer_file)
        self._save_state()

    def _save_state(self, entries=None):
        """Write the buffer's full entries (messages, exact tool args, full SMILES, delta, and
        the cycle/episode each was accepted at) to state_file, replacing it atomically. The run
        log only carries a lossy summary of each entry, so this file is the only way to recover
        the exact few-shot text a run was using - e.g. to inspect it, or restart with it via
        load_buffer_file. Failures are logged, never raised: losing a snapshot must not abort
        a cycle that took tens of minutes."""
        if not self.state_file:
            return
        try:
            if entries is None:
                with self._lock:
                    entries = list(self._buffer_entries)
            payload = {'saved_at_episode': self._episode_count, 'cycles_started': self.n_updates_started,
                       'entries': entries}
            tmp = self.state_file + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=1, ensure_ascii=False,
                          default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
            os.replace(tmp, self.state_file)
        except Exception as e:
            self.log(f"  [online-fewshot] could not save buffer state to {self.state_file}: "
                     f"{type(e).__name__}: {e}", flush=True)

    def _load_state(self, path):
        """Start from a buffer saved by _save_state (same entry shape the live code builds)."""
        with open(path, encoding='utf-8') as f:
            payload = json.load(f)
        entries = payload['entries'] if isinstance(payload, dict) else payload
        required = {'messages', 'path', 'working_smi', 'parent2_smi', 'delta', 'final_smi'}
        for i, e in enumerate(entries):
            missing = required - set(e)
            if missing:
                raise ValueError(f"{path}: entry {i} is missing {sorted(missing)}")
        self._buffer_entries = entries[-self.buffer_size:]
        self.log(f"  [online-fewshot] loaded {len(self._buffer_entries)} buffer example(s) from {path}:",
                 flush=True)
        for i, e in enumerate(self._buffer_entries):
            self.log(f"    [{i}] {_summarize_entry(e)}", flush=True)

    def _seed_from_file(self, path):
        """Warm-start the buffer from an offline fewshot_gen library (fewshot_examples_3step.json
        shape - post-rationalized path_steps + per_step_reasoning already generated by a full
        200-oracle-call run), instead of starting empty and slowly filling over the first
        several update cycles. Re-invokes the real tool functions to build authentic messages,
        exactly as build_fewshot_library_3step.py does for the static few-shot file, so seeded
        entries are indistinguishable in format from ones the buffer accepts on its own."""
        with open(path, encoding='utf-8') as f:
            examples = json.load(f)
        entries = []
        for ex in examples:
            mol1_smi, mol2_smi = ex['mol1_smi'], ex['mol2_smi']
            path_steps = ex['path_steps']
            reasonings = ex['per_step_reasoning']
            messages = _build_messages(ex['public_user_msg'], mol1_smi, mol2_smi, path_steps, reasonings, self._rng)
            entries.append({
                'messages': messages, 'path': path_steps,
                'working_smi': mol1_smi, 'parent2_smi': mol2_smi,
                'delta': path_steps[-1]['phi'] - ex['baseline_phi'],
                'final_smi': path_steps[-1]['smi'], 'phi': path_steps[-1]['phi'],
            })
        self._buffer_entries = entries[-self.buffer_size:]
        self.log(f"  [online-fewshot] seeded buffer with {len(self._buffer_entries)} static "
                 f"example(s) from {path}:", flush=True)
        for i, e in enumerate(self._buffer_entries):
            self.log(f"    [{i}] {_summarize_entry(e)}", flush=True)

    # -- called from the main GA loop --------------------------------------

    def update_population_diversity(self, smis):
        """Call once per generation with the current population's SMILES - the diversity
        gate below compares candidates against this live reference rather than a fixed
        magic number, so the gate tightens/loosens with however diverse the run actually
        is at the time."""
        if len(smis) >= 2:
            self._population_diversity_ref = self.diversity_evaluator(smis)

    def record_episode(self, working_smi, parent2_smi):
        """Call once per live episode, before that episode's own agent call. At each
        `update_every`-episode boundary, launches a cycle over the window just closed. If the
        cycle launched at the previous boundary is still running, this blocks the caller (the
        main GA loop) until it finishes first - so the loop gets one full window of overlap with
        the beam search, then waits, rather than skipping windows or letting the backlog grow.
        Callers should read few_shot_messages *after* this returns, so an episode that waited
        sees the buffer that wait produced."""
        with self._lock:
            self._episode_log.append((working_smi, parent2_smi))
            self._episode_count += 1
            at_boundary = (self._episode_count % self.update_every == 0)
            if at_boundary:
                recent_pool = self._episode_log[-self.update_every:]
                fallback_pool = list(self._episode_log)
        if not at_boundary:
            return

        prev = self._thread
        if prev is not None and prev.is_alive():
            t_wait = time.time()
            self.log(f"  [online-fewshot] backpressure: window {self._episode_count // self.update_every} "
                     f"closed while the previous cycle is still running - pausing the episode loop "
                     f"until it finishes", flush=True)
            prev.join()
            self.log(f"  [online-fewshot] backpressure: resumed after {time.time() - t_wait:.0f}s", flush=True)

        self.n_updates_started += 1
        self._thread = threading.Thread(target=self._run_update_cycle, args=(recent_pool, fallback_pool), daemon=True)
        self._thread.start()

    @property
    def few_shot_messages(self):
        with self._lock:
            msgs = []
            for entry in self._buffer_entries:
                msgs.extend(entry['messages'])
            return msgs if msgs else None

    def drain_injections(self):
        """Return and clear the queue of beam-search-discovered molecules (final_smi of each
        entry accepted since the last drain) waiting to be folded into the live GA population.
        Called once per generation from main/toolmol/run.py's _optimize(), right before
        Pareto-front selection - so a background cycle's winner gets a chance to compete for a
        population slot on equal footing with the LLM's own offspring, not just serve as a
        future few-shot demo. A no-op (returns []) unless --online_fewshot_inject is set."""
        with self._lock:
            smis, self._pending_injections = self._pending_injections, []
            return smis

    def shutdown(self):
        """Join any in-flight background update thread before closing the pool - closing
        first would race the pool's own shutdown against that thread's own pool.map()/
        pool.apply() calls, which fails with ValueError: Pool not running the moment the
        run's last generation finishes while an update cycle is still mid-search (caught
        live: the smoke test's max_oracle_calls=20 run ended with a cycle still running).
        The stop event tells the cycle to end after its current attempt (one beam search, a few
        minutes) instead of grinding through its remaining attempts - a cycle can now run for
        tens of minutes, and whatever it would still accept after the run's last generation
        can no longer influence anything. The 1500s cap covers a genuinely stuck API call: past
        it, give up waiting and close anyway; _run_update_cycle's own try/except means that
        thread just logs a failure internally rather than crashing anything here."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self.log("  [online-fewshot] shutdown: waiting for in-flight update cycle to finish...", flush=True)
            self._thread.join(timeout=1500)
            if self._thread.is_alive():
                self.log("  [online-fewshot] shutdown: in-flight cycle still running after 1500s, "
                         "closing the pool anyway", flush=True)
        self._pool.close()
        self._pool.join()

    # -- background worker ---------------------------------------------------

    def _run_update_cycle(self, recent_pool, fallback_pool):
        t0 = time.time()
        try:
            with self._lock:
                existing_paths = [e['path'] for e in self._buffer_entries]
                existing_mols = ([e['working_smi'] for e in self._buffer_entries]
                                  + [e['parent2_smi'] for e in self._buffer_entries])
                existing_finals = [e['final_smi'] for e in self._buffer_entries]
                existing_deltas = [e['delta'] for e in self._buffer_entries]
                existing_lens = [len(e['path']) for e in self._buffer_entries]
                pop_div_ref = self._population_diversity_ref

            candidates = list(recent_pool)
            self._rng.shuffle(candidates)
            extra = [c for c in fallback_pool if c not in candidates]
            self._rng.shuffle(extra)
            candidates.extend(extra)

            accepted = []
            accepted_paths = []
            accepted_mols = []
            accepted_finals = []
            accepted_deltas = []
            accepted_lens = []
            max_attempts = max(3 * self.sample_size, 6)
            attempts = 0
            for working_smi, parent2_smi in candidates:
                if len(accepted) >= self.sample_size or attempts >= max_attempts or self._stop.is_set():
                    break
                attempts += 1

                # cheap gate first: starting-molecule AND parent2 diversity vs what's already
                # in/going into the buffer, relative to the live population's own diversity -
                # skips the expensive beam search entirely for a redundant candidate. Checking
                # parent2 as well as working_smi matters because a dominant scaffold can keep
                # re-entering via crossover as the *other* parent even when working_smi varies.
                probe_set = existing_mols + accepted_mols + [working_smi, parent2_smi]
                if pop_div_ref is not None and len(probe_set) >= 2:
                    cand_div = self.diversity_evaluator(probe_set)
                    if cand_div < self.diversity_floor_frac * pop_div_ref:
                        self.log(f"  [online-fewshot] reject (low structural diversity: "
                                 f"{cand_div:.3f} vs {self.diversity_floor_frac}x pop ref "
                                 f"{pop_div_ref:.3f})", flush=True)
                        continue

                result = _beam_search_3step(self._pool, working_smi, parent2_smi,
                                             self.beam_width, self._group_smiles)
                if result is None:
                    self.log("  [online-fewshot] reject (beam search found nothing usable)", flush=True)
                    continue

                # step-length gate: a 1-step "sequence" doesn't demonstrate multi-step
                # planning at all, and beam=5's narrow search settled for these often enough
                # (see the manual buffer review) that they're worth excluding outright rather
                # than leaving it to the quality bar to catch. 2-step sequences are real but
                # weaker demonstrations than 3-step ones, so they're capped rather than banned.
                path_len = len(result['path'])
                if path_len == 1:
                    self.log("  [online-fewshot] reject (step-length gate: 1-step sequences "
                              "are not accepted)", flush=True)
                    continue
                if path_len == 2:
                    combined_2step = sum(1 for l in existing_lens + accepted_lens if l == 2)
                    if combined_2step >= 2:
                        self.log(f"  [online-fewshot] reject (step-length gate: buffer already "
                                 f"holds {combined_2step} two-step example(s), max 2 allowed)", flush=True)
                        continue

                # quality bar, on delta (improvement shown) rather than end phi - a candidate
                # that starts from an already-good molecule and barely improves it can still
                # have a high end phi despite being a weak demonstration of editing skill. The
                # static library's own edge traced back to being hand-picked for high regret
                # (i.e. high delta), not high end-quality, so this matches that selection
                # principle directly instead of a proxy for it.
                candidate_delta = result['delta']
                combined_deltas = existing_deltas + accepted_deltas
                if combined_deltas and candidate_delta <= min(combined_deltas):
                    self.log(f"  [online-fewshot] reject (quality bar: candidate delta "
                             f"{candidate_delta:+.4f} does not beat weakest buffer entry "
                             f"{min(combined_deltas):+.4f})", flush=True)
                    continue

                ok, reason = _passes_diversity_gate(result['path'], existing_paths + accepted_paths,
                                                     self.signature_max_repeats)
                if not ok:
                    self.log(f"  [online-fewshot] reject ({reason})", flush=True)
                    continue

                # outcome-molecule diversity: catches convergent scaffolds that reach the same
                # kind of result via different starting pairs, which the step-identity gate
                # (steps, not molecules) and the pre-search filter (starts/parent2s, not
                # outcomes) both miss.
                final_smi = result['path'][-1]['smi']
                final_probe_set = existing_finals + accepted_finals + [final_smi]
                if pop_div_ref is not None and len(final_probe_set) >= 2:
                    final_div = self.diversity_evaluator(final_probe_set)
                    if final_div < self.diversity_floor_frac * pop_div_ref:
                        self.log(f"  [online-fewshot] reject (low outcome-molecule diversity: "
                                 f"{final_div:.3f} vs {self.diversity_floor_frac}x pop ref "
                                 f"{pop_div_ref:.3f})", flush=True)
                        continue

                entry, reason = self._build_example(working_smi, parent2_smi, result)
                if entry is None:
                    self.log(f"  [online-fewshot] reject ({reason})", flush=True)
                    continue

                accepted.append(entry)
                accepted_paths.append(result['path'])
                accepted_mols.append(working_smi)
                accepted_mols.append(parent2_smi)
                accepted_finals.append(final_smi)
                accepted_deltas.append(candidate_delta)
                accepted_lens.append(path_len)
                self.log(f"  [online-fewshot] accept: {_summarize_entry(entry)}", flush=True)

            if accepted:
                with self._lock:
                    # n_updates_started still equals this cycle's index here: the next cycle is
                    # only launched after this one's thread has been joined (see record_episode).
                    for e in accepted:
                        e['cycle'] = self.n_updates_started
                        e['episode_at_accept'] = self._episode_count
                    self._buffer_entries.extend(accepted)
                    self._buffer_entries = self._buffer_entries[-self.buffer_size:]
                    buffer_snapshot = list(self._buffer_entries)
                    self._pending_injections.extend(e['final_smi'] for e in accepted)
                self.n_updates_completed += 1
                self._save_state(buffer_snapshot)
            else:
                with self._lock:
                    buffer_snapshot = list(self._buffer_entries)
            self.log(f"  [online-fewshot] update cycle: {len(accepted)}/{self.sample_size} accepted "
                     f"in {attempts} attempt(s), {time.time()-t0:.1f}s", flush=True)
            self.log(f"  [online-fewshot] buffer now holds {len(buffer_snapshot)}/{self.buffer_size} example(s):",
                     flush=True)
            for i, e in enumerate(buffer_snapshot):
                self.log(f"    [{i}] {_summarize_entry(e)}", flush=True)
        except Exception as e:
            self.log(f"  [online-fewshot] update cycle failed: {type(e).__name__}: {e}", flush=True)

    # -- post-rationalize + build authentic messages for one accepted path ---

    def _build_example(self, working_smi, parent2_smi, result):
        """Returns (entry, None) on success or (None, reason) on failure - always a 2-tuple,
        never a bare None, so a rejection always carries a human-readable cause into the log
        rather than forcing a step-through to find out why (cost real debugging time here -
        see the credit_balance_exhausted case this was written to fix)."""
        path = result['path']
        base_phi, base_raw = result['base_phi'], result['base_raw']
        mol1, mol2 = Chem.MolFromSmiles(working_smi), Chem.MolFromSmiles(parent2_smi)
        if mol1 is None or mol2 is None:
            return None, "starting molecule or parent2 SMILES did not parse"

        public_user_msg = (
            f"Goal: I want to {self._goal_description}. Please propose a new molecule better "
            f"than the current molecule. I have given you two candidate ligands. You are "
            f"encouraged to make a crossover between the candidate molecules on the first "
            f"step, then mutate the resulting molecule. Only make a few modifications (at "
            f"most 3), then respond with FINAL ANSWER. Do not let molecular weight exceed 700.\n\n"
            f"1. {working_smi}\nScore: {base_phi}\n{_format_context(mol1)}\n\n"
            f"2. {parent2_smi}\nScore: {self._phi(parent2_smi)}\n{_format_context(mol2)}"
        )
        private_instruction = _build_private_instruction(path, base_raw, base_phi)

        try:
            resp = self._client.chat.completions.create(
                model=self._llm_model, temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": public_user_msg + private_instruction},
                ],
            )
            content = resp.choices[0].message.content or ""
        except Exception as e:
            return None, f"post-rationalization API call failed: {type(e).__name__}: {e}"

        reasonings = _parse_step_reasoning(content, len(path))
        if reasonings is None:
            return None, f"could not parse STEP*_REASONING from response (first 200 chars): {content[:200]!r}"

        try:
            messages = _build_messages(public_user_msg, working_smi, parent2_smi, path, reasonings, self._rng)
        except Exception as e:
            return None, f"message-build (real tool re-invocation) failed: {type(e).__name__}: {e}"

        return {'messages': messages, 'path': path, 'working_smi': working_smi, 'parent2_smi': parent2_smi,
                'delta': result['delta'], 'final_smi': path[-1]['smi'], 'phi': path[-1]['phi']}, None

    def _phi(self, smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return sum(o.rescaled(o.raw(smi)) for o in self._objectives)


def _fmt_args(tool, args):
    return f'{tool}({json.dumps(args, separators=(",", ":"))})'


def _build_private_instruction(path_steps, base_raw, base_phi):
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
        lines.append(f"Step {i+1}: {_fmt_args(s['tool'], s['args'])}")
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
        lines.append(f"STEP{i+1}_TOOL_CALL: {_fmt_args(s['tool'], s['args'])}")
    return "\n".join(lines)


def _parse_step_reasoning(content, n_steps):
    out = []
    for i in range(1, n_steps + 1):
        r_pat = re.compile(rf"STEP{i}_REASONING:\s*(.*?)(?=STEP{i}_TOOL_CALL:)", re.DOTALL)
        rm = r_pat.search(content)
        if not rm or not rm.group(1).strip():
            return None
        out.append(rm.group(1).strip())
    return out


def _invoke_step(tool_name, args, working_smi, mol1_smi, mol2_smi, target_smi, rng):
    result, result_smi = None, None
    for attempt in range(25):
        if tool_name == 'crossover_molecules':
            result = TOOL_DISPATCH[tool_name](mol1_smi, args.get('idx1'), mol2_smi, args.get('idx2'), rng=rng)
        else:
            result = TOOL_DISPATCH[tool_name](working_smi, **args)
        if result.success:
            result_smi = Chem.MolToSmiles(_canonicalize(result.mol))
            if result_smi == target_smi:
                break
    if result is None or not result.success:
        raise RuntimeError(f"re-invoking {tool_name} failed on all 25 attempts")
    return result, result_smi


CONTINUE_TEXT = (
    "You have made {n} modification(s) so far (up to 3 allowed). Output FINAL ANSWER if you "
    "have made sufficient modifications, or continue with another tool call to keep refining. "
    "Ensure that desired properties are maintained.\nCurrent SMILES: {smi}\n{ctx}")
CAPPED_TEXT = ("You have made 3 modifications already - please output FINAL ANSWER now.\n"
               "Current SMILES: {smi}\n{ctx}")


def _build_messages(public_user_msg, mol1_smi, mol2_smi, path, reasonings, rng):
    """rng: a private random.Random() instance (OnlineFewShotBuffer._rng), never the global
    random module - this function re-invokes real tools (including crossover_molecules, which
    is randomized) from the background update thread, which runs concurrently with the live
    GA loop's own tool calls on the same shared global random-module state. Drawing from a
    private instance instead keeps this thread's randomness from interleaving with - and
    making non-reproducible - the live loop's own crossover_molecules calls."""
    messages = [{"role": "user", "content": public_user_msg}]
    working_smi = mol1_smi
    n_steps = len(path)
    for i in range(n_steps):
        tool_name, args, target_smi = path[i]['tool'], path[i]['args'], path[i]['smi']
        result, result_smi = _invoke_step(tool_name, args, working_smi, mol1_smi, mol2_smi, target_smi, rng)
        working_mol = _canonicalize(result.mol)
        result_smi = Chem.MolToSmiles(working_mol)

        call_id = f"online_{i}_call_{rng.randint(0, 1_000_000)}"
        messages.append({
            "role": "assistant", "content": reasonings[i],
            "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": tool_name, "arguments": json.dumps(args, separators=(",", ":"))}}],
        })
        tool_content = f"success: {result.message}\nCurrent SMILES: {result_smi}\n{_format_context(working_mol)}"
        messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_content})

        working_smi = result_smi
        is_last = (i == n_steps - 1)
        n_mods = i + 1
        ctx = _format_context(working_mol)
        reminder = CAPPED_TEXT.format(smi=working_smi, ctx=ctx) if n_mods >= 3 else \
            CONTINUE_TEXT.format(n=n_mods, smi=working_smi, ctx=ctx)
        messages.append({"role": "user", "content": reminder})
        if is_last:
            messages.append({"role": "assistant", "content": "FINAL ANSWER"})
    return messages
