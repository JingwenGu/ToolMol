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

## The crossover-only-at-step-1 issue, and how it was actually fixed

`beam_search_3step.py` used to generate `crossover_molecules` candidates against each beam
node's *current* working molecule. The live agent (`agent.py`'s `_agentic_edit`) actually
always calls `crossover_molecules` against the episode's *frozen original* parent1, never the
evolved working molecule - confirmed by reading the code, not by assumption. This didn't
affect the original 10 gpt-5.4 results, since `crossover_molecules` only ever won the beam at
step 1 there (where the working molecule and parent1 are identical by construction) - but it
did matter once this pipeline was reused for gpt-oss-120b: `_assert_crossover_only_at_step1()`
(added as a regression guard for exactly this scenario) fired for real on that data, proving
crossover was winning beyond depth 1 and silently overscoring candidates. `_gen_candidates`/
`_gen_candidates_main`/`_call_tool`/`_eval_one`/`expand` were then all updated to thread a
separate `mol1_smi` (captured once, frozen) through the whole call chain so
`crossover_molecules` always scores against the true frozen parent1 at every beam level,
matching `agent.py`. The assertion is left in place as a permanent regression guard, not
removed now that the underlying bug is fixed.

## v2: re-picking for strategy diversity (gpt-oss-120b only)

The first gpt-oss-120b library (`regret_gptoss120b_3step_v1.json`, 10 examples) turned out to
have a real diversity problem discovered by manually reading the reasoning text: 9 of the 10
winning paths used `add_substructure`/`replace_substructure` with morpholine (`N1CCOCC1`)
somewhere in the sequence. Investigating further (by beam-searching an additional 24, then 30
more snapshots deliberately chosen to avoid molecules whose *1-step* best move was morpholine)
showed this isn't a sampling artifact of which starting molecules got picked - morpholine
addition is close to a universally-rewarding move under this objective (jnk3+qed, sa
minimized) once a suitable ring/linker anchor exists. Even in the deliberately-morpholine-avoiding
expansion pool, roughly 65-85% of winning paths still used it. Out of 54 candidates
beam-searched in total across all rounds, only 16 had genuinely morpholine-free winning paths.

`regret_gptoss120b_3step_v2.json` was built from a curated 10 (snap_ids 52, 12, 62, 28, 25, 30,
93, 99, 20, 61) that caps morpholine at 3/10 (vs 9/10 in v1) while maximizing diversity in the
other two steps of each path (tool choice, ring/fragment choice, crossover vs. non-crossover).
Beam-search outputs for the expansion pools live in `beam3_results_gptoss_pool2.jsonl` (15
molecules) and `beam3_results_gptoss_pool3.jsonl` (30 molecules); the final 10 selected
records are in `beam3_results_gptoss_repick.jsonl`.

A second problem found by manually cross-checking each step's post-rationalized reasoning
against the actual measured jnk3/qed/sa deltas (not just eyeballing plausibility): 2 of the 30
reasoning steps in the first post-rationalization pass asserted a QED or SA improvement that
had not actually happened at that step (snap_id 62 step 3 claimed "balancing the SA score"
when SA got substantially worse; snap_id 12 step 3 claimed improved "drug-likeness" when QED
dropped notably). `fix_reasoning_repick.py` re-prompts the model for just the flawed step(s),
showing it the real before/after numbers and explicitly forbidding QED/SA/drug-likeness
framing when the measured direction contradicts it, then patches the examples JSON in place
rather than regenerating the whole library. The first attempt at the drug-likeness fix still
implied the same contradicted claim in softer language ("should be favorable for overall
drug-likeness") - the prompt had to be tightened to explicitly ban those specific words/
concepts, not just ask for a "correction," before the rewrite actually stopped implying a QED
improvement.
