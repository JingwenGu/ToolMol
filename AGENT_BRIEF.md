# Agent brief: running this arm with Claude Code as the agent

Hand this file to a **fresh** session. It is the whole context that session should need.

This repo is an implementation of ToolMol (arXiv 2605.12784), an LLM-agent framework for
molecule discovery. Upstream it calls an external model over an API. It has been modified so
that **the answering session is the agent**: the run spools each agent turn to disk and blocks
until that turn is answered.

Run it once: **pool of 60 initial molecules drawn from seed 1, total budget 200 oracle calls,
objectives jnk3 + QED (maximize) and SA (minimize)**. In each episode, reason chemically about
the best move on the molecules in front of you, and build on what you learn across episodes.

Use **seed 1**. Run-to-run variance at a fixed seed is large enough to matter, so repeat runs
hold the seed constant rather than changing it; a different seed is a separate experiment and
`--seed` is the only thing that needs to change for one.

## Hard rules

An earlier attempt at this arm was aborted for knowledge contamination. The contaminating
artifacts were deleted. Do not go looking for them or for replacements.

1. **Do not read any prior-run output.** No `results_*.yaml`, `pareto_snapshots_*.yaml`,
   `run_log*`, few-shot library, beam-search dump, or episode log from any earlier run, in this
   repo or anywhere else. If you find one, do not open it — say so and move on.
2. **Do not use git** in this repo. Not `log`, `show`, `stash`, `checkout`, `diff` of deleted
   paths — nothing. Earlier results may still exist in history, and reading them would
   invalidate the run.
3. **Do not leave this directory.** In particular nothing else under `RSI/`, and not
   `MOLLEO/`, `regret/`, `cmumo/`, `surrogate_oracles/`, `scratch.ipynb`, or `MOLLEO.pdf`.
   `ToolMol.pdf` is the method paper and is fine to read.
4. **No free oracle calls.** `multi_objective/oracle/jnk3_current.pkl` lets you score any
   molecule you like for nothing, outside the 200-call budget. Don't. The only legitimate ways
   to learn an objective value are (a) what the run itself reports, and (b) the beam search,
   run on a molecule the run actually reached. Never score a molecule you merely thought of.
5. **Every edit comes from chemistry.** Choose each tool call by reasoning about the molecule in
   front of you plus the feedback you have seen inside the run. No offline enumeration or
   brute-force sweeps standing in for that reasoning.
6. **Do not change the codebase.** The modifications that put you in the agent's seat are
   already in place (below), as is the operator client. Anything further is out of scope — and
   if you think you have found a bug, say so before changing it.
7. **No persistent artifacts.** Keep your strategy, your read of the landscape and your episode
   history in your own context, not in files. `--output_dir` points outside the repo so the tree
   stays clean. Notes-to-self are not wanted.

## Do this first, before anything else

This project may auto-inject memory files into your context at session start. Check whether you
received any memory mentioning jnk3 values, Phi values, molecule SMILES, or findings from earlier
runs of this experiment. **If you did, stop and tell the user — do not start the run.** The arm
is only valid if you begin without them. Memory about the Python environment is harmless.

## Environment

Interpreter (do not rebuild, do not use another):

    /Users/gujw/Desktop/Research/AI4S/RSI/.venv-toolmol39/bin/python

**CWD rule.** PyTDC cannot download on this network. The arm ships
`multi_objective/oracle/fpscores.pkl`, `multi_objective/oracle/jnk3_current.pkl` and
`multi_objective/data/zinc.tab`, but PyTDC only resolves them when the process CWD is this
repo's `multi_objective/`. Always run from there.

**Check the oracles before you start.** A failed model load answers `0.0` for everything
silently, which would waste the whole run:

    cd multi_objective
    /Users/gujw/Desktop/Research/AI4S/RSI/.venv-toolmol39/bin/python -c "
    import tdc
    for n in ['jnk3','qed','sa']: print(n, tdc.Oracle(name=n)('c1ccccc1C(=O)NC'))"

Expect three **non-zero** numbers. All zeros, or an sklearn unpickling traceback, means a bad
load or the wrong interpreter — stop and report it.

## How you are wired in

Four files differ from upstream. Read them before you start so you trust the loop:

- `multi_objective/main/toolmol/agent.py` — the `claude` backend. `_create_claude()` writes each
  agent turn to `$TOOLMOL_CLAUDE_SPOOL/req_NNNNN.json` and blocks, polling for
  `resp_NNNNN.json`; `_parse_claude_response()` turns the reply into the same object shape an
  OpenAI chat-completions message would have. No timeout, no retry — it waits as long as needed.
- `multi_objective/run.py` — `claude` added to `--llm_backend` choices.
- `multi_objective/main/toolmol/run.py` — `_drain_inject_spool()`, the `claude`-backend
  counterpart of `OnlineFewShotBuffer.drain_injections()`: reads and clears
  `$TOOLMOL_CLAUDE_SPOOL/inject.txt` (one SMILES per line) once per generation, right before
  Pareto-front selection, so injected molecules compete for a population slot alongside your own
  offspring.
- `multi_objective/spool_client.py` — your interface to the spool (not part of the method).

## Launch

    ./multi_objective/run_claude_arm.sh                 # seed 1, 200 calls, paper hyperparameters
    ./multi_objective/run_claude_arm.sh my_run_name     # same, under a named run directory

It prints the spool directory, the output directory and the log path, and backgrounds the run.
Startup takes about a minute (oracle loading) before the first request appears.

**Use a fresh spool directory for every run.** Relaunching or resuming restarts the request
counter at 1, so a stale `resp_00001.json` from a previous run would be read as the answer to the
new run's first turn. The launch script gives each run its own directory; don't reuse one.

Hyperparameters come from `main/toolmol/hparams_default.yaml` — the paper setting: population 60,
offspring 35, mutation_rate 0.0, k 10.0, max_steps 10. Leave it alone.

Note a trap if you hand-roll the command: `--config_default hparams_smoketest.yaml` run from
`multi_objective/` resolves to the copy in that directory, which has no `max_steps`/`k` and dies
with `KeyError: 'max_steps'`. Pass an absolute path to `main/toolmol/hparams_smoketest.yaml`
instead. `hparams_default.yaml` is fine, since no copy shadows it there.

Stop a run with `pkill -f "run.py toolmol"`.

## Driving the run

    export TOOLMOL_CLAUDE_SPOOL=<the spool dir the launch script printed>
    cd multi_objective
    python spool_client.py status     # pending/answered counts, injection queue
    python spool_client.py next       # the delta since the previous request
    python spool_client.py reply --content "..." --tool add_atom \
        --args '{"idx":7,"element":"F","bond":"single"}'
    python spool_client.py beam "<smiles>" --beam 5 --inject
    python spool_client.py inject "<smiles>"

`next` prints only what is new since the last request, because earlier turns are already in your
context; it detects an episode boundary and prints the full list, labelled NEW EPISODE.

Bond symbols are lowercase: `single`, `double`, `triple`.

## Episode-loop mechanics

All of this is readable in `agent.py`; it is stated here only to save you the rediscovery. If an
experiment is meant to measure cold-start discovery of the harness too, delete this section
before handing the brief over.

- One assistant turn may carry **several tool calls**; they apply sequentially, each to the
  result of the last. Atom indices are re-canonicalized after every successful call, so an index
  in a later call of the same turn refers to the molecule as it stands after the earlier ones —
  batch only when you can predict that, otherwise act one turn at a time and read the state back.
- `FINAL ANSWER` ends an episode **only in a turn with no tool calls**. A turn with tool calls is
  always followed by a reminder, so the minimum is two turns per episode.
- The budget is 3 modifications per episode; `undo_last_change` does not count against it.
- Every tool result echoes the new SMILES plus the full atom table and properties, so you always
  see what your edit actually did. Use it — a call can "succeed" and still hit the wrong atom.
- `crossover_molecules` is the only tool that can touch ligand 2, and it acts on the *original*
  parent 1, so calling it after other edits discards them.
- If you never land a successful tool call in an episode, the harness silently falls back to
  Graph-GA for that pair and the episode contributes nothing you chose.

## The beam search

`online_fewshot._beam_search_3step`, exposed as `spool_client.py beam`. In the original pipeline
it runs on a timer during the run, its near-optimal sequences few-shot the following episodes,
and its molecules are injected into the Pareto front. Use it the same way: run it on molecules
you have actually encountered, generalize the moves it finds, and `--inject` its winners so they
compete for a front slot and can serve as parents later. Roughly 15-20s at beam width 5 on 8
workers. It is **not** a scorer for molecules you invented — see rule 4.

Searching is free against the oracle budget, so search *volume* is itself a behavioural
difference between runs, not just a detail. Record how much you used it with the results.

## Reporting

When the run ends (budget exhausted or patience convergence), report: the best molecule and its
Phi, how the Pareto front evolved, how many oracle calls went where, which tool-call strategies
worked and which wasted episodes, and what the beam search contributed versus your own edits.
Say plainly what did not work.
