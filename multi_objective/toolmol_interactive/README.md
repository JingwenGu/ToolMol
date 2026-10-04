# ToolMol with an interactive operator as the agent

Runs the repo's ToolMol pipeline with a human/LLM operator producing the assistant turns
instead of an HTTP model provider. Nothing about the algorithm changes.

## What was changed in the codebase (the only two edits)

- `main/toolmol/agent.py` — new `backend="interactive"` for `ToolMolAgent`
  (`_create_interactive` + `_parse_interactive_response`). It writes the exact message list
  the `api` backend would have POSTed to `$TOOLMOL_INTERACTIVE_DIR/request_<n>.json`, blocks
  until `response_<n>.json` appears, and parses it into the same `_Response` shape every
  other backend returns. The agent loop, toolbox, context injection, oracle, Pareto
  selection and GA are untouched.
- `run.py` — `interactive` added to `--llm_backend`'s choices.

## Harness (this directory; nothing here is on the pipeline's import path)

- `driver.py` — builds run.py's argv, points the agent at the interactive backend, and
  instruments the oracle budget from outside the pipeline (see its docstring). One
  "oracle call" = one distinct molecule scored on all three objectives.
- `respond.py` — operator CLI: `wait` / `reply` / `status` / `ledger`.
- `hparams_interactive.yaml` — 20 starting molecules, 10 offspring/generation, 3 turns
  per episode.
- `memory/` — persistent artifacts carried across runs: `STRATEGY.md` (the playbook),
  `FRAGMENTS.md` (what the oracle actually rewards), `RUNLOG.md`, `EPISODES.md`.
- `runs/seed<N>/` — per-run output: `ipc/`, `oracle_ledger.jsonl`, `results_*.yaml`,
  `pareto_snapshots_*.yaml`, `run_log.txt`.

## Usage

The pipeline must run with `multi_objective/` as CWD (PyTDC resolves `oracle/*.pkl` and
`data/zinc.tab` relative to it); `driver.py` chdirs there itself.

```bash
# terminal 1 - start a run (background)
python toolmol_interactive/driver.py --seed 1 --budget 50 \
    > toolmol_interactive/runs/seed1/run_log.txt 2>&1 &

# terminal 2 - serve each turn
python toolmol_interactive/respond.py --seed 1 wait
python toolmol_interactive/respond.py --seed 1 reply '{"content": "...", "tool_calls":
    [{"name": "replace_substructure", "arguments":
      {"anchor_idx": 12, "branch_idx": 13, "new_substructure": "[1*]c1ccnc(N)n1"}}]}'
python toolmol_interactive/respond.py --seed 1 status
python toolmol_interactive/respond.py --seed 1 ledger --sort jnk3 --top 15
```

`reply` also takes a JSON *list* of responses, applied to successive turns — a round-trip
saver for turns already decided (e.g. the closing `FINAL ANSWER`).

## Environment

Python 3.9 + rdkit 2023.09.6, PyTDC 0.4.1, **scikit-learn 1.2.2** (the pinned version is
required: `oracle/jnk3_current.pkl` is an old pickled RandomForest that newer sklearn
refuses to load), pymoo 0.6.1.5, selfies, torch.
