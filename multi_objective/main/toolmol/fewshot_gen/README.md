# 3-step few-shot generation pipeline

How `../fewshot_examples/regret_gpt54_3step_v1.json` was built, in order:

1. **`regret_master_gpt54.json`** / **`snapshots_gpt54.jsonl`** (checked in as-is) - the
   500-molecule single-step regret dataset for the gpt-5.4 baseline run
   (`../results/task_gpt54_seed1_max200_2026-09-13`), and the raw tool-call-by-tool-call
   snapshot log it was sampled from. Produced by a 1-step brute-force search
   (`_gen_candidates`: every single possible tool call from each snapshot's starting
   molecule) - not included here, this is its output.

2. **`beam_search_3step.py`** - takes the 10 hand-picked highest-regret snapshots from
   `regret_master_gpt54.json` (`PICKS` at the top of the file) and runs a non-myopic beam
   search (default width 30) over *sequences* of up to 3 tool calls from each one, instead
   of the single-step ceiling above. True exhaustive 3-step search is intractable (~10^10
   candidates/molecule); beam search keeps the global top-B states after each step and
   expands all of them, rather than committing to one greedy path. Writes
   `beam3_results.jsonl` (checked in - the best 1/2/3-step sequence found per molecule).

   ```
   python beam_search_3step.py --beam 30 --workers 15 --out beam3_results.jsonl
   ```

3. **`post_rationalize_3step.py`** - for each of the 10 molecules, shows gpt-5.4 the same
   opening context a real episode would see, privately reveals the beam-search-verified
   best sequence, and asks it to write one first-person reasoning block per step (as if it
   didn't know the answer) in a single combined API call. Writes
   `fewshot_examples_3step.json` (checked in - post-rationalized text, not yet in
   OpenAI-message form). Requires `OPENAI_API_KEY`.

   ```
   python post_rationalize_3step.py
   ```

4. **`build_fewshot_library_3step.py`** - re-invokes the real toolbox functions for each
   step (rather than trusting the beam search's recorded SMILES) so tool-result content is
   byte-identical to what a live episode would produce, and assembles the final flat
   OpenAI-message list - user/assistant/tool turns, with the same "continue or FINAL
   ANSWER" reminder flow `agent.py`'s `_agentic_edit` uses. Writes
   `fewshot_messages_3step.json`, which was then copied to
   `../fewshot_examples/regret_gpt54_3step_v1.json` (the file `--few_shot_file` actually
   points at).

   ```
   python build_fewshot_library_3step.py
   ```

## A known imprecision, not yet fixed

`beam_search_3step.py` generates `crossover_molecules` candidates against each beam node's
*current* working molecule. The live agent (`agent.py`'s `_agentic_edit`) actually always
calls `crossover_molecules` against the episode's *frozen original* parent1, never the
evolved working molecule - confirmed by reading the code, not by assumption. This didn't
affect the 10 results actually produced here, since `crossover_molecules` only ever won the
beam at step 1 in all 10 cases (where the working molecule and parent1 are identical by
construction) - but it would matter for a future run of this script where crossover wins at
step 2 or 3. Worth fixing in `_gen_candidates_main`/`_gen_candidates` before reusing this
beyond the original 10-molecule investigation.
