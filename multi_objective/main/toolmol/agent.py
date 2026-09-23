"""AgentGen: the tool-calling LLM operator (Section 3.2 / Appendix B.2 of the paper).

Given two parent molecules, drives a multi-step tool-calling conversation (up to
max_steps) where the LLM calls the 7 toolbox functions to build a new molecule, never
touching SMILES syntax for the molecule being edited directly. Falls back to plain
Graph-GA crossover/mutation only if the LLM never successfully calls a tool within the
step budget - the paper's stated "extremely uncommon" failure path.
"""

import os
import json
import re
import threading
import time

from openai import OpenAI, RateLimitError
from rdkit import Chem

from main.toolmol.toolbox import TOOL_SCHEMAS, TOOL_DISPATCH
from main.toolmol import ligand_info
from main.toolmol import crossover as co
from main.toolmol import mutate as mu

MAX_RATE_LIMIT_RETRIES = 3
# Only auto-wait-and-retry a rate limit if the provider's suggested wait is short (a
# per-minute cap that will clear shortly). Longer suggested waits - e.g. a daily token
# cap, which some providers report as "try again in 7m..." or more - aren't worth
# blocking a single edit episode for; let those propagate to the normal fallback path.
RATE_LIMIT_RETRY_CAP_SECONDS = 20.0


def _parse_retry_after_seconds(message):
    match = re.search(r'try again in (?:(\d+)m)?([\d.]+)s', message)
    if not match:
        return None
    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2))
    return minutes * 60 + seconds

SYSTEM_PROMPT = (
    "You are a molecular design agent.\n"
    "You may ONLY modify molecules using tools.\n"
    "Only make one modification at a time.\n"
    "Read the parameter descriptions for the tools very carefully.\n"
    "Always ensure that your modifications don't break valence rules and do not result in a fragmented molecule.\n"
    "After each successful tool call, check its result message and the updated atom table against what "
    "you intended - if it modified the wrong part of the molecule, call undo_last_change before continuing.\n"
    "Every tool except crossover_molecules acts ONLY on your current working molecule (ligand 1) - none of "
    "them can see or touch the second candidate ligand. Treat ligand 2 as inspiration only: if you want to "
    "bring in one of its features, either call crossover_molecules to combine the two, or rebuild that "
    "feature yourself on your current working molecule (e.g. with add_substructure). An atom index you give "
    "to any tool other than crossover_molecules is always interpreted against your current working molecule, "
    "never against ligand 2 - do not reason about 'modifying ligand 2' and then call one of these tools, "
    "since the edit will land on the wrong molecule and likely do nothing you intended."
)


def _format_context(mol):
    # Compact table instead of a Python dict-repr dump - the raw dict form runs to
    # ~40 tokens/atom (repeated key names, quotes, braces); this table format runs to
    # roughly 4-5 tokens/atom, plus a bit more now that neighbor indices are included.
    # neighbor_indices was previously dropped (the tools took a single atom index, so
    # connectivity wasn't needed to place a call). It's restored now that
    # remove_substructure/replace_substructure take an (anchor_idx, branch_idx) bonded
    # pair - without it the model would have to guess which atoms are actually bonded.
    # num_available_valences is still dropped: in practice it almost always duplicates
    # num_substitutable_hydrogens for neutral atoms, so it stays redundant.
    atoms = ligand_info.get_ligand_structure(mol)
    rows = [
        f"{a['atom_index']},{a['element']},{a['num_substitutable_hydrogens']},"
        f"{a['num_neighboring_atoms']},{'Y' if a['is_in_ring'] else 'N'},{a['centrality']:.2f},"
        f"{';'.join(str(n) for n in a['neighbor_indices'])}"
        for a in atoms
    ]
    structure_block = "idx,elem,subH,deg,ring,cent,nbrs\n" + "\n".join(rows)

    props = ligand_info.calculate_properties(mol)
    props_line = " ".join(f"{k}={v}" for k, v in props.items())

    return f"Atoms:\n{structure_block}\nProperties: {props_line}"


def _canonicalize(mol):
    """Re-parse mol through its own canonical SMILES so its atom ordering matches what
    Chem.MolFromSmiles(Chem.MolToSmiles(mol)) will always reproduce for it from here on - a
    fixed point, since RDKit's canonical-SMILES algorithm is a deterministic function of the
    molecule alone, not of the input atom order (parsing the same canonical string always
    gives the same order back). This matters because every tool call receives the molecule
    as Chem.MolToSmiles(working_mol) and re-parses it internally (toolbox._load) - if
    working_mol's own atom order doesn't already match that round trip's result (true of any
    mol built via CombineMols/RWMol edits, whose order reflects the edit history, not
    canonical traversal), the atom indices _format_context shows the model can silently
    diverge from the indices a tool call actually acts on, with no error raised - a real,
    previously undetected root cause of "the model picked the wrong atom" failures, not
    specific to any one tool. Falls back to the original mol (leaving it however it was) in
    the rare case the round trip itself fails, rather than crashing the episode."""
    canon = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    return canon if canon is not None else mol


def _strip_context(entry):
    # Drop the atom table + properties from a message once a newer turn's context has
    # been sent - the older molecule state is stale, so there's no reason to keep paying
    # for it in every subsequent request. Keeps the short summary line(s) before it intact.
    idx = entry["content"].find("\nAtoms:\n")
    if idx != -1:
        entry["content"] = entry["content"][:idx] + "\n[context omitted - superseded by a later turn]"


# Some providers occasionally stall a request indefinitely instead of erroring (observed
# hanging 8+ hours against DigitalOcean, and up to 3.6 hours on isolated calls against
# Cerebras, with no response). httpx's own read-timeout - the client's `timeout=` below -
# only fires after a gap with *zero* bytes received; a connection that trickles occasional
# keep-alive bytes without ever completing the response defeats it. So this bound is
# enforced twice: once via the client's own timeout (catches most stalls, e.g. a dead
# connect phase), and again as a hard wall-clock deadline around the call in a daemon
# thread (catches the trickling-connection case the client-level timeout misses). A stall
# that blows the deadline raises TimeoutError - caught by edit_pair's broad except, which
# falls back to Graph-GA for that pair - instead of hanging the whole run. The abandoned
# daemon thread is left to finish or die on its own; it doesn't block later calls since a
# fresh thread is started per call rather than routed through a shared worker pool.
REQUEST_TIMEOUT_SECONDS = 180.0


def load_few_shot_messages(path):
    """Load a few-shot conversation history from a JSON file: a flat list of message dicts
    (role: user/assistant/tool) in the exact same shape _agentic_edit builds internally -
    matching pairs/triples of {user molecule prompt} -> {assistant reasoning + tool_calls} ->
    {tool result}, closed out with a final {user reminder} -> {assistant FINAL ANSWER} so the
    pattern the model sees includes recognizing when to stop, not just how to open. Generate
    one with main/toolmol scratch tooling that reuses _format_context/TOOL_DISPATCH directly,
    rather than hand-assembling messages, so tool-result text is byte-identical to what a live
    episode would actually produce."""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


class ToolMolAgent:
    def __init__(self, model="gpt-4", base_url=None, max_steps=10, api_key_env="OPENAI_API_KEY",
                 extra_body=None, system_prompt_suffix=None, few_shot_file=None):
        self.client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url,
                              timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
        self.model = model
        self.max_steps = max_steps
        # Provider-specific request body extensions, e.g. OpenRouter's
        # {"provider": {"only": ["groq"]}} to pin routing to a specific underlying provider.
        self.extra_body = extra_body
        # Appended to SYSTEM_PROMPT verbatim when set - not baked into the shared constant,
        # since it's only needed for providers/models (e.g. OpenAI's gpt-5-family via plain
        # Chat Completions) that don't expose reasoning_content/reasoning the way gpt-oss
        # does; those models return content=null alongside the tool call with no visible
        # rationale at all unless explicitly asked to write one into content. Leaving this
        # None preserves the exact original prompt for every other provider/model.
        self.system_prompt_suffix = system_prompt_suffix
        # Few-shot conversation history prepended before the live query on every call - a
        # toggle, not a permanent prompt change: None (the default) reproduces the exact
        # original behavior for every existing run; pass a path to turn it on for a specific
        # experiment. See load_few_shot_messages() for the expected file format.
        self.few_shot_messages = load_few_shot_messages(few_shot_file) if few_shot_file else None
        # Objective description used in the initial prompt; set by run.py before optimizing
        # (adapted wording for the Phase 1-3 cheap-oracle stand-ins - see plan ambiguity #9 -
        # the paper's literal "[PROTEIN TARGET]" phrasing applies once Boltz-2 is wired in).
        self.goal_description = "improve the configured objectives"

    def edit_pair(self, mol1, score1, mol2, score2, mutation_rate=0.0):
        try:
            result_mol, n_modifications = self._agentic_edit(mol1, score1, mol2, score2)
        except Exception as e:
            print(f"{type(e).__name__} {e}", flush=True)
            result_mol, n_modifications = None, 0

        if result_mol is not None and n_modifications > 0:
            return result_mol

        # Paper's stated "extremely uncommon" fallback: the LLM never successfully called
        # a tool within the step budget. crossover.crossover() Kekulizes its inputs in
        # place (clearing aromaticity flags), which would otherwise corrupt the caller's
        # population Mol objects - pass copies so mol1/mol2 (and whatever they alias in
        # the population) are never mutated.
        print("ToolMolAgent: no successful tool call in budget, falling back to Graph-GA", flush=True)
        new_child = co.crossover(Chem.Mol(mol1), Chem.Mol(mol2))
        if new_child is not None:
            new_child = mu.mutate(new_child, mutation_rate)
        return new_child

    def _call_with_deadline(self, messages):
        """Run the completions call in a daemon thread and enforce a hard wall-clock
        deadline via Thread.join(timeout=...), independent of whatever the client's own
        (bytes-since-last-read) timeout does or doesn't catch. Returns the response, or
        raises TimeoutError if the deadline is hit, or re-raises whatever exception the
        call itself raised."""
        box = {}

        def _call():
            try:
                box['response'] = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=0,
                    extra_body=self.extra_body,
                )
            except Exception as e:
                box['error'] = e

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()
        thread.join(timeout=REQUEST_TIMEOUT_SECONDS)

        if thread.is_alive():
            raise TimeoutError(f"LLM call exceeded hard {REQUEST_TIMEOUT_SECONDS:.0f}s deadline")
        if 'error' in box:
            raise box['error']
        return box['response']

    def _create_with_retry(self, messages):
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            t0 = time.time()
            print(f"    -> LLM call starting ({len(messages)} messages)...", flush=True)
            try:
                response = self._call_with_deadline(messages)
                print(f"    <- LLM call returned in {time.time() - t0:.1f}s", flush=True)
                return response
            except RateLimitError as e:
                print(f"    <- LLM call rate-limited after {time.time() - t0:.1f}s", flush=True)
                wait = _parse_retry_after_seconds(str(e))
                if wait is None or wait > RATE_LIMIT_RETRY_CAP_SECONDS or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                print(f"rate limited, waiting {wait:.1f}s before retry ({attempt + 1}/{MAX_RATE_LIMIT_RETRIES})", flush=True)
                time.sleep(wait)
            except Exception as e:
                print(f"    <- LLM call failed after {time.time() - t0:.1f}s: {type(e).__name__}: {e}", flush=True)
                raise

    def _agentic_edit(self, mol1, score1, mol2, score2):
        # Canonicalize both parents up front - see _canonicalize's docstring for why this is
        # required for atom indices to stay consistent between what's shown to the model and
        # what any tool call actually acts on, starting from the very first tool call.
        mol1, mol2 = _canonicalize(mol1), _canonicalize(mol2)
        smi1, smi2 = Chem.MolToSmiles(mol1), Chem.MolToSmiles(mol2)

        system_content = SYSTEM_PROMPT
        if self.system_prompt_suffix:
            system_content = system_content + "\n" + self.system_prompt_suffix

        messages = [{"role": "system", "content": system_content}]
        # Prepended verbatim, before the live query, whenever the agent was constructed with
        # few_shot_file set - a toggle, not a permanent prompt change (see __init__/
        # load_few_shot_messages). Absent by default, so every existing call site is unaffected.
        if self.few_shot_messages:
            messages.extend(self.few_shot_messages)
        messages.append(
            {"role": "user", "content": (
                f"Goal: I want to {self.goal_description}. Please propose a new molecule better "
                f"than the current molecule. I have given you two candidate ligands. You are "
                f"encouraged to make a crossover between the candidate molecules on the first "
                f"step, then mutate the resulting molecule. Only make a few modifications (at "
                f"most 3), then respond with FINAL ANSWER. Do not let molecular weight exceed 700.\n\n"
                f"1. {smi1}\nScore: {score1}\n{_format_context(mol1)}\n\n"
                f"2. {smi2}\nScore: {score2}\n{_format_context(mol2)}"
            )}
        )

        # Log both parents' full state up front - previously this (and every subsequent
        # molecule state) only ever went into the API message payload, never to the console,
        # making it impossible to reconstruct after the fact what the model was actually
        # looking at when it made a given tool call.
        print(f"  [state] parent1: {smi1}\n{_format_context(mol1)}", flush=True)
        print(f"  [state] parent2: {smi2}\n{_format_context(mol2)}", flush=True)

        working_mol = mol1
        parent1_smi, parent2_smi = smi1, smi2
        n_modifications = 0
        # Stack of (previous_working_mol, description_of_the_change_that_replaced_it), pushed
        # right before working_mol is overwritten by any successful tool call. undo_last_change
        # pops this to recover from a tool call that "succeeded" (no RDKit error) but modified
        # the wrong part of the molecule - e.g. the model misidentified which atom index it was
        # targeting. A stack (not just one slot) costs nothing extra and lets the model step
        # back through more than one recent change if needed.
        mol_history = []

        # Only the most recent per-turn context dump is kept in full; every earlier one
        # (across all turns) gets its atom table/properties stripped as soon as a newer
        # one is added, since the older molecule state is stale by then. The one-time
        # initial prompt (both parents) is intentionally not tracked here - it's paid once,
        # not once per turn, so it doesn't need pruning.
        context_msgs = []

        def track_context(entry):
            for m in context_msgs:
                _strip_context(m)
            context_msgs.append(entry)

        for step in range(self.max_steps):
            print(f"  agent step {step + 1}/{self.max_steps}", flush=True)
            response = self._create_with_retry(messages)
            msg = response.choices[0].message

            # gpt-oss models return chain-of-thought in a separate field, not included in
            # content - log it so failure analysis can see *why* a step was chosen, not just
            # what tool call resulted. The field name is provider-specific: DigitalOcean uses
            # reasoning_content, Cerebras uses plain reasoning. Not all providers populate this.
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
            if reasoning:
                print(f"  agent step {step + 1}: reasoning: {reasoning}", flush=True)
            # Separate from the above: msg.content itself was never logged to console before -
            # only carried forward in the in-memory messages list. For models with no exposed
            # reasoning_content/reasoning field (e.g. gpt-5-family via plain Chat Completions,
            # which return content=null alongside a tool call unless explicitly prompted
            # otherwise - see system_prompt_suffix above), this content field is the only place
            # any rationale shows up at all, so it must be logged too.
            if msg.content and not reasoning:
                print(f"  agent step {step + 1}: content: {msg.content}", flush=True)

            assistant_msg = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                if msg.content and "FINAL ANSWER" in msg.content.upper():
                    print(f"  agent step {step + 1}: FINAL ANSWER, no tool call", flush=True)
                    break
                print(f"  agent step {step + 1}: no tool call, no FINAL ANSWER - nudging", flush=True)
                messages.append({"role": "user", "content": (
                    "Please call a tool to make a modification, or respond with FINAL ANSWER "
                    "if you are done.")})
                continue

            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                print(f"  agent step {step + 1}: tool call -> {name}({tool_call.function.arguments})", flush=True)
                if name != "crossover_molecules":
                    print(f"  [state] before call: {Chem.MolToSmiles(working_mol)}\n{_format_context(working_mol)}", flush=True)

                if name == "undo_last_change":
                    if not mol_history:
                        tool_content = "failed: no previous change to undo"
                    else:
                        working_mol, undone_desc = mol_history.pop()
                        n_modifications = max(0, n_modifications - 1)
                        tool_content = (f"success: reverted - undone change was: {undone_desc}\n"
                                         f"Current SMILES: {Chem.MolToSmiles(working_mol)}\n"
                                         f"{_format_context(working_mol)}")
                    print(f"  agent step {step + 1}: tool result: {tool_content.splitlines()[0]}", flush=True)
                    if tool_content.startswith("success"):
                        # The line above only logs the summary; print the restored state too,
                        # the same way every other successful tool call's state gets printed -
                        # otherwise the console log has no record of what an undo actually
                        # produced, only that it succeeded.
                        print(f"  [state] after call: {tool_content}", flush=True)
                    tool_msg = {"role": "tool", "tool_call_id": tool_call.id, "content": tool_content}
                    messages.append(tool_msg)
                    if tool_content.startswith("success"):
                        track_context(tool_msg)
                    continue

                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                      "content": "error: could not parse tool call arguments as JSON"})
                    continue

                fn = TOOL_DISPATCH.get(name)
                if fn is None:
                    messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                      "content": f"error: unknown tool '{name}'"})
                    continue

                try:
                    if name == "crossover_molecules":
                        result = fn(parent1_smi, args.get("idx1"), parent2_smi, args.get("idx2"))
                    else:
                        result = fn(Chem.MolToSmiles(working_mol), **args)
                except TypeError as e:
                    # The model can hand a tool arguments belonging to a different tool's
                    # schema (e.g. a stray 'bond' kwarg on replace_substructure) - report it
                    # back like any other tool failure instead of aborting the whole episode.
                    print(f"  agent step {step + 1}: tool call raised TypeError: {e}", flush=True)
                    messages.append({"role": "tool", "tool_call_id": tool_call.id,
                                      "content": f"error: invalid arguments for '{name}': {e}"})
                    continue

                print(f"  agent step {step + 1}: tool result success={result.success}: {result.message}", flush=True)
                if result.success:
                    mol_history.append((working_mol, result.message))
                    working_mol = _canonicalize(result.mol)
                    n_modifications += 1
                    tool_content = (f"success: {result.message}\n"
                                     f"Current SMILES: {Chem.MolToSmiles(working_mol)}\n"
                                     f"{_format_context(working_mol)}")
                    print(f"  [state] after call: {tool_content}", flush=True)
                else:
                    tool_content = f"failed: {result.message}"
                tool_msg = {"role": "tool", "tool_call_id": tool_call.id, "content": tool_content}
                messages.append(tool_msg)
                if result.success:
                    track_context(tool_msg)

            if n_modifications >= 3:
                reminder = "You have made 3 modifications already - please output FINAL ANSWER now."
            else:
                reminder = (f"You have made {n_modifications} modification(s) so far (up to 3 "
                            f"allowed). Output FINAL ANSWER if you have made sufficient "
                            f"modifications, or continue with another tool call to keep refining. "
                            f"Ensure that desired properties are maintained.")
            reminder_msg = {"role": "user", "content": (
                f"{reminder}\nCurrent SMILES: {Chem.MolToSmiles(working_mol)}\n"
                f"{_format_context(working_mol)}")}
            messages.append(reminder_msg)
            track_context(reminder_msg)

        return working_mol, n_modifications
