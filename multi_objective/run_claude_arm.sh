#!/bin/bash
# Launch this arm with Claude Code as the agent (--llm_backend claude).
#
# Each run gets its own directory outside the repo, holding the request/response spool, the
# oracle output and the log - so the tree stays clean and no run can read another's spool.
# A fresh spool per run is load-bearing: relaunching restarts the request counter at 1, so a
# stale resp_00001.json would be read as the answer to the new run's first turn.
#
#   ./run_claude_arm.sh                 # seed 1, 200 oracle calls, paper hyperparameters
#   ./run_claude_arm.sh my_run_name     # the same, under a named run directory
#
# Seed is pinned to 1 deliberately. Run-to-run variance at a fixed seed is large enough to
# matter, so repeat runs hold the seed constant; a different seed is a separate experiment -
# override with SEED=2 ./run_claude_arm.sh if that is what you actually want.
set -euo pipefail

PYTHON="${PYTHON:-/Users/gujw/Desktop/Research/AI4S/RSI/.venv-toolmol39/bin/python}"
SEED="${SEED:-1}"
MAX_ORACLE_CALLS="${MAX_ORACLE_CALLS:-200}"
PATIENCE="${PATIENCE:-5}"
CONFIG="${CONFIG:-hparams_default.yaml}"
# Not under /tmp: the system temp cleaner will gut a long run's spool out from under it.
RUNS_DIR="${RUNS_DIR:-$HOME/toolmol_runs}"

RUN_NAME="${1:-$(date +%Y%m%d_%H%M%S)_seed${SEED}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$RUNS_DIR/$RUN_NAME"

if [ -e "$RUN_DIR" ]; then
    echo "error: $RUN_DIR already exists - pick another run name so the spool starts empty." >&2
    exit 1
fi
mkdir -p "$RUN_DIR/spool" "$RUN_DIR/out"

# PyTDC only resolves the shipped oracle/data files when the process CWD is multi_objective/.
cd "$HERE"

"$PYTHON" -c "
import tdc
vals = [float(tdc.Oracle(name=n)('c1ccccc1C(=O)NC')) for n in ('jnk3','qed','sa')]
assert any(v != 0.0 for v in vals), 'all oracles returned 0.0 - model load failed silently'
print('  oracle check ok: jnk3=%.3f qed=%.3f sa=%.3f' % tuple(vals))" \
    || { echo "error: oracle check failed - not launching." >&2; exit 1; }

TOOLMOL_CLAUDE_SPOOL="$RUN_DIR/spool" nohup "$PYTHON" run.py toolmol \
    --mol_lm ToolMol \
    --llm_backend claude \
    --llm_model claude-opus-5 \
    --max_obj jnk3 qed \
    --min_obj sa \
    --seed "$SEED" \
    --max_oracle_calls "$MAX_ORACLE_CALLS" \
    --patience "$PATIENCE" \
    --config_default "$CONFIG" \
    --output_dir "$RUN_DIR/out" \
    > "$RUN_DIR/run.log" 2>&1 &
PID=$!

echo "  pid     $PID"
echo "  seed    $SEED   budget $MAX_ORACLE_CALLS   patience $PATIENCE   config $CONFIG"
echo "  spool   $RUN_DIR/spool"
echo "  out     $RUN_DIR/out"
echo "  log     $RUN_DIR/run.log"
echo "  journal $RUN_DIR/beam_journal.jsonl   (beam skill library; set TOOLMOL_BEAM_JOURNAL"
echo "                                         to an earlier run's file to carry it over)"
echo
echo "Drive it with:"
echo "  export TOOLMOL_CLAUDE_SPOOL=$RUN_DIR/spool"
echo "  cd $HERE && $PYTHON spool_client.py status"
echo
echo "The first request takes about a minute to appear (oracle loading)."
echo "Stop this run with: kill $PID"
echo "  (or: pkill -f \"$RUN_DIR\" - do NOT use pkill -f 'run.py toolmol', which would also"
echo "   kill any other arm's run active on this machine)"
