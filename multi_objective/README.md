# cmumo/cmumo_objectives_base — the GA integration

Written 2026-10-26, reorganized 2026-10-27 to be committable on its own. Sourced from
`RSI/OPFS_context1/multi_objective` (which carries the `claude` backend and the injection
spool). Implements sections 2-5 and 7 of `CMUMO_PORT_DESIGN.md` (in the AI4S workspace, outside this repo). **Not a sandbox** —
the predictor *is* the ground truth for this task, so there is no surrogate/truth split to
protect.

**Nothing heavy is in this tree** (~700 KB, no file over 1 MB). Model weights, the ZINC
pool, the property cache and the ADMET-AI virtualenv live in `../assets` and `../`;
`setup_assets.py` links in the two that tdc insists on finding relative to the cwd. A
fresh clone needs one command:

```bash
python setup_assets.py          # links assets in, names anything missing
```

No run output is committed either: `results/` holds only a `.gitkeep`.

## Run it

```bash
cd AI4S/cmumo/cmumo_objectives_base
../../RSI/.venv-toolmol39/bin/python -u run.py toolmol \
    --mol_lm ToolMol --llm_backend stub --combination BPQ \
    --max_oracle_calls 40 --patience 40 --seed 0 \
    --config_default main/toolmol/hparams_smoketest.yaml --output_dir $PWD/results
```

`--llm_backend stub` runs **no model**: every turn returns "no tool call", which drops each
episode into the documented Graph-GA fallback. That exercises objectives, prefetch,
pre-filter, cap, Pareto selection, budget accounting and logging without an API key or a
GPU. It is a smoke test, not an arm — no LLM means no agentic operator. Swap in `api`,
`responses`, `transformers` or `claude` for a real run.

Two foot-guns in the invocation, both inherited:

- **`--config_default` is resolved against the cwd first.** There is a
  `hparams_smoketest.yaml` at the arm root *and* one in `main/toolmol/`, and only the
  latter has `max_steps`. Pass the path explicitly as above.
- **`-I` breaks the run.** `run.py` imports `main.*` relative to the cwd, which isolated
  mode removes from `sys.path`.

## What changed from the OPFS arm

| file | change |
|---|---|
| `cmumo_oracles/` | moved in from `cmumo/oracles/`: the evaluator, the ADMET-AI worker, and `paths.py` |
| `calibration/` | moved in from `cmumo/`: `calibrate_m.py` and the sweep's two JSON results |
| `PROPERTIES.md` | moved in from `cmumo/README.md`: the property table, the oracle traps, the calibration |
| `setup_assets.py`, `.gitignore` | new: link the assets in, keep them and run output out of git |
| `spool_client.py` | new: agent interface to a `claude` run, plus `beam` and `inject` |
| `main/toolmol/online_fewshot.py`, `fewshot_gen/*` | objective set from the env; scoring moved from the pool workers to a batched parent-side call per beam level |
| `main/toolmol/objectives.py` | replaced: `ImproveObjective` (linear rescale) + `MaintainObjective` (the hinge), reading the batched `PropertyEvaluator` |
| `main/toolmol/cmumo_feedback.py` | new: the run's instruction, and the per-episode per-objective breakdown |
| `main/toolmol/combinations.py` | new: the C-MuMO task codes and the occupancy rule that splits each into improve / maintain |
| `main/toolmol/run.py` | `_prefilter_violators`, `_cap_front`, `prefetch` hooks, breakdown into `edit_pair`, goal text from the objectives |
| `main/toolmol/agent.py` | `edit_pair`/`_agentic_edit` take optional breakdowns; `_score_block` renders them; `stub` backend |
| `run.py` | `--improve`, `--maintain`, `--combination`, `--reject_at`, `--front_cap`; patience check; the dead `task_mode` branch replaced |

## Layout

```
run.py                        CLI: --combination, --improve/--maintain, --reject_at, --front_cap
main/toolmol/                 the GA: objectives, run loop, agent, feedback, combinations
cmumo_oracles/                the property evaluator and the ADMET-AI worker
calibration/                  calibrate_m.py + calibration.json + chosen_config.json
setup_assets.py               links ../assets in; --check reports without creating
PROPERTIES.md                 the property table, the four oracle traps, the calibration
results/                      run output (gitignored; holds only .gitkeep)
oracle/ data/                 symlinks into ../assets (gitignored)
```

## Before and after a run

Three pieces added 2026-10-27, because without them a run is hard to set up honestly and
impossible to analyse afterwards.

**Seeding — `seed_pool.py`.** Design section 1 seeds from the combination's own molecules,
and C-MuMO defines those precisely: every improve property worse than Th_p, every maintain
property better than Th_p. Random ZINC asks a vaguer question (71% of a random draw is
already compliant on AMP, 76% on LIV), so a maintain axis would be satisfied by accident.

```bash
python seed_pool.py --combination BPQ --n 500 --scan 6000 --out ../../assets/seeds/BPQ.smi
run.py ... --combination BPQ --smi_file ../../assets/seeds/BPQ.smi
```

For BPQ, 1,952 of 6,000 ZINC molecules match. The output is **shuffled**, because the GA
takes the first `population_size` lines of a `--smi_file`, not a random sample. Write it to
the assets directory, **two** levels up - `../assets` from here is the repo root, and a
pool written there gets committed (which happened once; the root `.gitignore` now catches
it).

This also needed `BaseOptimizer.load_smiles_from_file`, which was **commented out
upstream** - so `--smi_file` raised `AttributeError` and the design's seeding path had
never actually worked in any arm cut from this code.

**Recording — `main/toolmol/discovery_log.py`.** `save_result` writes only molecules that
reached a front, as `[Phi, index]`. Everything else is scored during front selection and
dropped, so "what did the agent propose" had no answer and the FA-style metric, defined
over the whole generated pool, could not be computed. Every run now writes
`discovered_<label>.jsonl`, one line per distinct molecule at first sighting, with Phi,
per-objective raw and scaled values, generation, origin, whether it reached the front, and
whether the pre-filter rejected it. Rejects are logged **before** the pre-filter runs,
deliberately: a proposal killed by a constraint is exactly what you want to count. A
sidecar `.meta.json` names which axes were improve and which maintain, with their
theta/delta/M/C, so nothing downstream has to infer it.

**Reporting — `report.py`.** Front quality, hypervolume, and the FA-style filtered report
(compliant molecules only, then the improve axes).

```bash
python report.py <output_dir> --top 10
```

It will not print SR, SR_Th, RI or Sim: those need an input molecule M_x, and the pooled
design has none. Nothing it prints may be placed beside the GeLLMO-C table.

## Beam search and injection

Wired for C-MuMO on 2026-10-27. The brute-force 3-step search and the injection path came
with the OPFS arm but could not run here: three call sites hardcoded
`build_objectives(['jnk3', 'qed', 'sa'])`, which raises against this property table.

```bash
export TOOLMOL_CLAUDE_SPOOL=<spool dir>          # only needed for injection
python spool_client.py beam "CCOc1ccccc1" --beam 5 --inject
python spool_client.py inject "CCOc1ccccc1"
python spool_client.py status
```

`spool_client.py` is also the agent's interface to a `--llm_backend claude` run (`next`,
`reply`), written fresh for this arm rather than ported from the surrogate-oracle sandbox,
whose client carried that experiment's adapter/budget/forecast machinery.

Three changes made it work:

- **The objective set comes from the environment.** `objectives.configured_objectives()`
  reads `CMUMO_IMPROVE` / `CMUMO_MAINTAIN`, which `configure()` exports so spawned pool
  workers inherit them. Applied to `online_fewshot.py` (two sites),
  `fewshot_gen/beam_search_3step.py` and `fewshot_gen/post_rationalize_3step.py`, whose
  module-level build made the module unimportable.
- **Scoring moved out of the pool workers.** They now only apply tools (pure rdkit) and
  return SMILES; the parent prefetches the whole beam level in one batched oracle call and
  scores it. Measured: a width-3 search scored **3,321 molecules in 20 ADMET-AI batch
  calls with zero per-molecule fallbacks**. Before, each candidate was its own round trip
  and each of N workers built its own evaluator, so N copies of torch.
- **Searching stays free against the oracle budget.** The beam search scores through
  `build_objectives` directly, never `ToolMolOracle.score_smi`, so nothing enters
  `mol_buffer` until a molecule is injected and admitted to the front. Search *volume* is
  therefore a behavioural difference from a timer-driven arm, not just the trigger. Record
  it with results.

### A front of one, and why `--population_floor` exists

Injected molecules are strong by construction, and in this arm compliant molecules tie at
1 on every maintain axis - so dominance is decided by the improve axes alone. An injected
winner can therefore dominate the whole population, the front collapses to a single
molecule, and the GA can only cross it with itself.

**How likely that is, measured.** Same code, same molecules, 40 random draws of 95 (one
generation's candidate set at the real config):

| formulation | mean front | min | max | ever 1? |
|---|---|---|---|---|
| 3 continuously-varying axes (jnk3/qed/sa-like) | 11.7 | 6 | 22 | never |
| C-MuMO: 2 improve + 1 plateaued maintain | 5.7 | 3 | 11 | never |

The plateau halves the front, but **neither formulation reached 1 at the real config**.
The collapse was observed with `hparams_smoketest.yaml` (population 8, offspring 4): twelve
candidates, one dropped by the violator pre-filter, and the injected winner dominated the
remaining seven. Same combination and seed without injection held at 4. So this is a rare
event needing both a small population and an injected winner - not something to expect at
60/35.

It is worth understanding anyway, because it says what injection does to the front.
Ordinary offspring are single-tool edits, better on one axis and worse on another, so they
join the front rather than displacing anyone - which is why fronts normally grow and the
other arms never shrank. A beam-search winner is selected to maximize the *sum* of axes, so
it is high on all of them at once. It is the only thing in the loop that can evict many
members at a stroke.

Rare or not, it exposed an inherited bug. The improvement/patience check sat inside
`if len(self.oracle) > 1`, so at a front of one it never ran, while the budget had also
stopped growing (only front members are charged, and the front was no longer changing).
**The run span forever** - observed at 4,878 generations with `population=1`. The guard is
now `>= 1`, so patience fires correctly at any front size.

Correct is not the same as useful, though: a run that stops right after a good injection
is barely better than one that spins. `--population_floor N` tops a collapsed front back
up to N with the next-best molecules by Phi. **Off by default**, because it changes the
inherited survival semantics, and a loud warning fires whenever it would have helped.
Pass `--population_floor 4` whenever you inject.

## What the pilot changed  [2026-10-27]

One episode driven by hand through the spool backend, two turns, and it earned its keep.

- **Three of the per-objective-feedback edits had never been saved.** A multi-edit script
  hit a failed assertion on its last change, so Python raised before the write and the
  earlier three were discarded; only the failed one got re-applied. The prompt had been
  falling back to `Goal: I want to maximize QED, PlogP, BBBP scores` with no breakdown at
  all. Recovered, and each edit now verified by re-reading the file rather than trusting
  the assert.
- **`goal_description` was built too early.** It was set in `__init__`, which runs before
  `assign_evaluator`, so `self.oracle.objectives` did not exist and `_goal_description`
  silently listed `args.max_obj`. It now runs right after `assign_evaluator`, and the run
  prints the goal so this cannot hide again.
- **Intermediate turns had no objective feedback.** After a tool call the agent saw rdkit
  descriptors but no BBBP, no Phi, no maintain status - it could break the constraint it
  had just been told to preserve and not know. `agent.breakdown_fn` now renders the working
  molecule's own breakdown into every reminder message.
- **PlogP's axis was weighted at two thirds of a probability axis.** See `span = D_p / 0.1`
  in `objectives.py`.
- **Cosmetics:** a double period where the goal block met the template, and a redundant
  `Score: 2.5325904221752205` line above `Phi 2.533 of 3.0`.

Both tool paths were exercised: `crossover_molecules` and `add_atom` each executed and
returned a correct result message and atom table.

## Two inherited loose ends

Both came with the OPFS copy and neither affects the C-MuMO path; recorded so they are not
mistaken for reorganization damage.

- `main/toolmol/fewshot_gen/README.md` points at
  `../fewshot_examples/regret_gpt54_3step_v1.json`, which does not exist in the source arm
  either. That whole path is the `--online_fewshot` machinery, which this arm does not use
  (design section 6).
- The `molleo_*` methods under `main/` are kept as source but their stale `oracle/`
  downloads were deleted. Nothing in the code refers to those nested paths — tdc resolves
  `./oracle` against the cwd, which is always the arm root — so they were leftovers from
  someone running inside those directories. A `--method molleo_*` run would re-download
  into the arm root's linked `oracle/`.

## Four things to know before reading a result

**Direction lives in the property table, not the CLI.** `rescaled()` returns 1 = best for
lower-is-better properties too, so `run.py` puts every property in `--max_obj` and leaves
`--min_obj` empty. Passing a property in `--min_obj` would invert it twice.

**The improve/maintain split travels in the environment.** `CMUMO_IMPROVE` /
`CMUMO_MAINTAIN`, set by `objectives.configure()`. Not module state: `_pool_init` rebuilds
objectives in a worker that may be spawned rather than forked, and an env var survives that
while a global does not.

**`prefetch()` is not optional for performance.** The GA reads properties one molecule at a
time; without the two prefetch hooks (seed population, and each generation's candidate set)
every `raw()` is a separate ADMET-AI forward pass. `PropertyEvaluator.get()` still works
without them and prints a cache-miss line per molecule, so a missing hook is loud but not
fatal.

**The pre-filter and cap run before the oracle call, deliberately.** `self.oracle(...)` is
what charges front admissions against the budget, so truncating the already-scored list
would save nothing. `oracle.evaluate()` is free, which is what makes a pre-scoring cap
possible at all.

## What the smoke test showed

`--combination BPQ` resolves to improve = QED, PlogP; maintain = BBBP. The run completed
(exit 0), the pre-filter dropped 1 violator in generation 1, the front settled at 4
molecules, and both output files were written.

It also reproduced the budget pathology the design describes: **`n_oracle` stopped at 8 of
40** and the run ended on patience after 40 generations, because `storing_buffer`
accumulates only front members and the Graph-GA fallback never put a new molecule on the
front. With a real operator admissions do accumulate — but it is a clean demonstration that
the budget counts admissions, not evaluations.

One unit mismatch worth remembering: **`--patience` counts generations, `--max_oracle_calls`
counts admissions.** Setting them equal, as section 7 requires, does not make them the same
quantity; it makes patience effectively non-binding. Fine as a convention, misleading if
read as "the two stop conditions agree".
