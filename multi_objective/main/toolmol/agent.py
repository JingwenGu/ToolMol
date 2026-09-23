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
import time

from openai import OpenAI, RateLimitError
from rdkit import Chem

from main.toolmol.toolbox import TOOL_SCHEMAS, TOOL_DISPATCH

# The Responses API (needed for gpt-oss - see _create_responses) uses a flat function-tool
# shape ({"type": "function", "name": ..., ...}) instead of chat completions' nested one
# ({"type": "function", "function": {"name": ..., ...}}).
RESPONSES_TOOL_SCHEMAS = [
    {"type": "function", **schema["function"]} for schema in TOOL_SCHEMAS
]
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
    "Always ensure that your modifications don't break valence rules and do not result in a fragmented molecule."
)


def _format_context(mol):
    # Compact table instead of a Python dict-repr dump - the raw dict form runs to
    # ~40 tokens/atom (repeated key names, quotes, braces); this table format runs to
    # roughly 4-5 tokens/atom. Two fields from get_ligand_structure are dropped here:
    # neighbor_indices (the tools take an atom index directly, they don't need the LLM
    # to pre-verify connectivity - the tool call itself fails with a clear message if a
    # plan doesn't make sense) and num_available_valences (in practice this almost always
    # duplicates num_substitutable_hydrogens for neutral atoms, so it's redundant).
    atoms = ligand_info.get_ligand_structure(mol)
    rows = [
        f"{a['atom_index']},{a['element']},{a['num_substitutable_hydrogens']},"
        f"{a['num_neighboring_atoms']},{'Y' if a['is_in_ring'] else 'N'},{a['centrality']:.2f}"
        for a in atoms
    ]
    structure_block = "idx,elem,subH,deg,ring,cent\n" + "\n".join(rows)

    props = ligand_info.calculate_properties(mol)
    props_line = " ".join(f"{k}={v}" for k, v in props.items())

    return f"Atoms:\n{structure_block}\nProperties: {props_line}"


def _strip_context(entry):
    # Drop the atom table + properties from a message once a newer turn's context has
    # been sent - the older molecule state is stale, so there's no reason to keep paying
    # for it in every subsequent request. Keeps the short summary line(s) before it intact.
    idx = entry["content"].find("\nAtoms:\n")
    if idx != -1:
        entry["content"] = entry["content"][:idx] + "\n[context omitted - superseded by a later turn]"


# Some providers occasionally stall a request indefinitely instead of erroring (observed
# hanging 8+ hours against DigitalOcean with no response). Bound every call so a stall
# raises openai.APITimeoutError - caught by edit_pair's broad except, which falls back
# to Graph-GA for that pair - instead of hanging the whole run.
REQUEST_TIMEOUT_SECONDS = 180.0

# Qwen's chat template (Hermes-style, shared by Qwen2.5/Qwen3) renders each tool call the
# model makes as its own <tool_call>{"name": ..., "arguments": ...}</tool_call> block, and
# accepts one or more of these per assistant turn.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class _ToolCallFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = _ToolCallFunction(name, arguments)


class _Message:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls or None


class _Response:
    def __init__(self, message):
        self.choices = [type("_Choice", (), {"message": message})()]


def _messages_to_responses_input(messages):
    # Converts our internal chat-completions-shaped message list into Responses API
    # input items. gpt-oss's tool_calls never come back populated on /v1/chat/completions
    # (see agent.py history/PR notes) - only /v1/responses, OpenAI's native Harmony-format
    # endpoint, surfaces them - so this is the on-the-fly translation layer that lets
    # _agentic_edit's message bookkeeping stay backend-agnostic, same as _create_local's
    # chat-template rendering does for the transformers backend.
    #
    # Per vllm's response_input_to_harmony (entrypoints/openai/responses/harmony.py): a
    # function_call_output item is matched back to its call by call_id, and that lookup
    # requires the corresponding function_call item to appear earlier in the *same*
    # request's input list - so every tool_calls entry is echoed back as its own
    # function_call item, immediately followed by the tool result as function_call_output.
    # Reasoning items are optional to replay and are deliberately skipped here - Codex/
    # OpenCode-style clients that echo them back have hit vLLM validation bugs replaying
    # them (github.com/vllm-project/vllm/issues/33089), and _agentic_edit never needs its
    # own past reasoning restated to make its next decision.
    input_items = []
    for m in messages:
        role = m["role"]
        if role in ("system", "user"):
            input_items.append({
                "type": "message",
                "role": role,
                "content": [{"type": "input_text", "text": m["content"]}],
            })
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                input_items.append({
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                })
        elif role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": m["tool_call_id"],
                "output": m["content"],
            })
        else:
            raise ValueError(f"Unknown message role for Responses API conversion: {role!r}")
    return input_items


def _parse_responses_output(response):
    # Mirrors the shape _agentic_edit expects (.content / .tool_calls[].id /
    # .function.{name,arguments}) - same contract _parse_local_response fills for the
    # transformers backend. tool_call.id is set to the Responses API's call_id (not the
    # separate, unrelated `id` field on the item) since that's what _messages_to_responses_input
    # needs back to build the next turn's function_call/function_call_output pair.
    tool_calls = []
    content = None
    for item in response.output:
        if item.type == "function_call":
            tool_calls.append(_ToolCall(id=item.call_id, name=item.name, arguments=item.arguments))
        elif item.type == "message":
            content = "".join(part.text for part in item.content if getattr(part, "text", None))
    return _Response(_Message(content=content, tool_calls=tool_calls))


def _parse_local_response(text):
    # Mirrors the shape _agentic_edit expects back from an OpenAI ChatCompletion message
    # (.content / .tool_calls[].id / .function.{name,arguments}) so the rest of the agent
    # loop doesn't need to know which backend produced the response.
    tool_calls = []
    for i, match in enumerate(_TOOL_CALL_RE.finditer(text)):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name", "")
        arguments = payload.get("arguments", {})
        arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)
        tool_calls.append(_ToolCall(id=f"call_{i}", name=name, arguments=arguments))

    content = _TOOL_CALL_RE.sub("", text).strip()
    return _Response(_Message(content=content or None, tool_calls=tool_calls))


class ToolMolAgent:
    def __init__(self, model="gpt-4", base_url=None, max_steps=10, api_key_env="OPENAI_API_KEY",
                 extra_body=None, backend="api", device_map="auto", max_new_tokens=1024):
        # "api" talks to any OpenAI-compatible chat-completions endpoint - the official
        # OpenAI API, or a self-hosted server (e.g. vLLM/Ollama) pointed to via base_url.
        # "responses" talks to the same kind of server's /v1/responses endpoint instead -
        # needed for gpt-oss models, whose tool_calls never come back populated over chat
        # completions (confirmed against vLLM 0.28.0: the model's tool-call intent only
        # shows up in an unstructured `reasoning` string, never as msg.tool_calls) - see
        # _create_responses. "transformers" loads the model in-process with Hugging Face
        # transformers instead, for local inference with no server to run - see _create_local.
        self.backend = backend
        self.model = model
        self.max_steps = max_steps
        # Provider-specific request body extensions, e.g. OpenRouter's
        # {"provider": {"only": ["groq"]}} to pin routing to a specific underlying provider.
        self.extra_body = extra_body
        # Objective description used in the initial prompt; set by run.py before optimizing
        # (adapted wording for the Phase 1-3 cheap-oracle stand-ins - see plan ambiguity #9 -
        # the paper's literal "[PROTEIN TARGET]" phrasing applies once Boltz-2 is wired in).
        self.goal_description = "improve the configured objectives"

        if backend in ("api", "responses"):
            self.client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url,
                                  timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
        elif backend == "transformers":
            # Imported lazily so the "api" backend (the default) never requires torch/
            # transformers to be importable, let alone a GPU.
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.max_new_tokens = max_new_tokens
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype="auto", device_map=device_map)
        else:
            raise ValueError(f"Unknown backend: {backend!r} (expected 'api', 'responses', or 'transformers')")

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

    def _create_local(self, messages):
        # No server round-trip, so none of the OpenAI-specific retry/rate-limit handling
        # in _create_with_retry applies here - just render the prompt with Qwen's chat
        # template (which knows how to lay out our tool schema and prior tool_calls/tool
        # results - see _parse_local_response) and run one greedy generation.
        t0 = time.time()
        print(f"    -> local LLM call starting ({len(messages)} messages)...", flush=True)
        # Qwen's chat template renders message.content with a string filter, which chokes
        # on the None content assistant messages carry when they contain only tool_calls.
        template_messages = [{**m, "content": m.get("content") or ""} for m in messages]
        prompt = self.tokenizer.apply_chat_template(
            template_messages, tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.hf_model.device)
        output_ids = self.hf_model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id)
        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"    <- local LLM call returned in {time.time() - t0:.1f}s", flush=True)
        return _parse_local_response(generated)

    def _create_with_retry(self, messages):
        if self.backend == "transformers":
            return self._create_local(messages)
        if self.backend == "responses":
            return self._call_with_rate_limit_retry(
                messages, lambda: self.client.responses.create(
                    model=self.model,
                    input=_messages_to_responses_input(messages),
                    tools=RESPONSES_TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=0,
                    extra_body=self.extra_body,
                ),
                parse=_parse_responses_output,
            )
        return self._call_with_rate_limit_retry(
            messages, lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0,
                extra_body=self.extra_body,
            ),
            parse=lambda response: response,
        )

    def _call_with_rate_limit_retry(self, messages, request_fn, parse):
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            t0 = time.time()
            print(f"    -> LLM call starting ({len(messages)} messages)...", flush=True)
            try:
                response = request_fn()
                print(f"    <- LLM call returned in {time.time() - t0:.1f}s", flush=True)
                return parse(response)
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
        smi1, smi2 = Chem.MolToSmiles(mol1), Chem.MolToSmiles(mol2)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Goal: I want to {self.goal_description}. Please propose a new molecule better "
                f"than the current molecule. I have given you two candidate ligands. You are "
                f"encouraged to make a crossover between the candidate molecules on the first "
                f"step, then mutate the resulting molecule. Only make a few modifications (at "
                f"most 3), then respond with FINAL ANSWER. Do not let molecular weight exceed 700.\n\n"
                f"1. {smi1}\nScore: {score1}\n{_format_context(mol1)}\n\n"
                f"2. {smi2}\nScore: {score2}\n{_format_context(mol2)}"
            )},
        ]

        working_mol = mol1
        parent1_smi, parent2_smi = smi1, smi2
        n_modifications = 0

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
                    working_mol = result.mol
                    n_modifications += 1
                    tool_content = (f"success: {result.message}\n"
                                     f"Current SMILES: {Chem.MolToSmiles(working_mol)}\n"
                                     f"{_format_context(working_mol)}")
                else:
                    tool_content = f"failed: {result.message}"
                tool_msg = {"role": "tool", "tool_call_id": tool_call.id, "content": tool_content}
                messages.append(tool_msg)
                if result.success:
                    track_context(tool_msg)

            if n_modifications >= 3:
                reminder = "You have made 3 modifications already - please output FINAL ANSWER now."
            else:
                reminder = ("Output FINAL ANSWER if you have made sufficient modifications "
                            "(make at most 3). Ensure that desired properties are maintained.")
            reminder_msg = {"role": "user", "content": (
                f"{reminder}\nCurrent SMILES: {Chem.MolToSmiles(working_mol)}\n"
                f"{_format_context(working_mol)}")}
            messages.append(reminder_msg)
            track_context(reminder_msg)

        return working_mol, n_modifications
