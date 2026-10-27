# cmumo/cmumo_objectives_base — the GA integration

Written 2026-10-26, reorganized 2026-10-27 to be committable on its own. Sourced from
`RSI/OPFS_context1/multi_objective` (which carries the `claude` backend and the injection
spool). Implements sections 2-5 and 7 of `../../CMUMO_PORT_DESIGN.md`. **Not a sandbox** —
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
