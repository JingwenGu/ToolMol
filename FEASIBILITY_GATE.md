# Feasibility gate — handoff notes

Status: implemented and tested in this tree (`MOLLEO/`) on 2026-10-12. **Default off**; no
existing invocation changes behaviour unless `--feasibility_gate` is passed. Not yet ported to
the `RSI/*` arm trees — see [Porting to the RSI arms](#7-porting-to-the-rsi-arms).

Audience: an agent picking this up later. Everything below marked *measured* was verified by
running it; everything marked *heuristic* is a threshold choice you may need to revisit.

---

## 1. Why this exists

The SA oracle cannot see ring strain, and nothing else in the objective can either. So an
optimizer can bank score on molecules that cannot exist, at essentially zero cost.

`tdc.Oracle(name='SA')` → `tdc/chem_utils/oracle/oracle.py:SA()` → `calculateScore()`
(Ertl & Schuffenhauer 2009; identical code to rdkit `Contrib/SA_Score/sascorer.py`):

```
sascore = score1 + score2 + score3
sascore = 11.0 - (sascore + 5) / 6.5 * 9.0      # 1 raw unit = 1.385 SA units
```

Four mechanisms, all *measured*:

1. **`score1` is an occurrence-weighted mean, not a sum** (`score1 /= nf`). One hard motif in
   an otherwise ordinary molecule is diluted by every common fragment. For the azete test case
   below, 2 of 42 bits were absent from the fragment table; their −4 each became −0.118 raw,
   i.e. 0.23 SA units.
2. **The unknown-fragment default (−4) is not an extreme.** `fpscores.pkl` holds 705,292 bits
   with range **[−3.13, +2.84]**, median −3.13, and 54.6% of entries at or below −3. "Never seen
   in PubChem" scores only ~0.87 below "rarest thing in the table". **No SA value means
   unsynthesizable.** The azete's own environments are mostly *in* the table with mild or
   positive scores (`C1=CN=C1` → −1.04, `C=NC` → **+0.19**, `C=CC` → **+0.28**).
3. **A third of the signal is bare atom types.** Every atom emits a radius-0 bit scoring ~+2.7.
   For the azete test case: radius 0 = 33.8% of fragment occurrences supplying **+0.916 of
   score1 (≈60%)**, while radius 2 — the only place a strained ring could register —
   contributed **+0.046 out of +1.519**. `score1` is mostly element composition.
4. **`score2` has no strain term.** Only `nAtoms**1.005 - nAtoms`, chiral centres, spiro,
   bridgeheads, macrocycles (>8). Since the size penalty is a function of atom count, swapping
   a pyridine for an azete *lowers* `score2` by 0.043 raw — the molecule is rewarded for having
   two fewer atoms. `score3` is a *bonus* for symmetry.

Measured consequences:

| molecule | SMILES | SA |
|---|---|---|
| azete on a kinase scaffold | `C1=NC(c2ccnc(Nc3ccc(N4CCOCC4)cc3)c2)=C1` | **2.48** |
| same, pyridine instead of azete | `c1ccncc1-c2ccnc(Nc3ccc(N4CCOCC4)cc3)c2` | 2.08 |
| bare azete | `C1=NC=C1` | **3.04** |
| cyclobutadiene | `C1=CC=C1` | **1.99** |
| aspirin | `CC(=O)Oc1ccccc1C(=O)O` | 1.58 |
| imatinib | — | 2.33 |
| morphine | — | 5.27 |

The azete costs **0.39 SA units**, i.e. **0.044 of Phi** via `(10-SA)/9` (0.836 vs 0.880) —
far below the noise in jnk3. Note cyclobutadiene and bare azete need no dilution at all: the
function simply has no channel for strain.

## 2. What the gate does

It answers a narrower question than synthesizability: **can this molecule exist at room
temperature?** It is a *stability* gate. It says nothing about whether a synthetic route exists
to a stable but exotic scaffold — for that, score the final Pareto front with AiZynthFinder.

Two mechanisms:

- **Programmatic ring analysis** over the SSSR (not SMARTS — the relevant facts are endocyclic
  bond orders per ring plus ring size, which is relational and awkward as one pattern):
  antiaromatic 4-ring (≥2 endocyclic double bonds), 3-ring with 2 double bonds, 5-ring with 3,
  endocyclic triple bond in a ring ≤7, cyclic allene ≤8, anti-Bredt alkene, cyclopropene/azirine,
  cyclobutene, trans alkene in a small ring.
- **SMARTS rules** for labile groups: peroxide, N–N–N chain (with a real-azide exemption),
  gem-diol, hemiketal, aminal, enol (exempting vinylogous acids / 1,3-dicarbonyl enols),
  orthoester, acyl halide, thiol-ketone gem, diazo, C-nitroso.

Two tiers, deliberately:

| verdict | meaning |
|---|---|
| `ok` | nothing flagged |
| `hard` | labile but real. **Artemisinin is `hard`** (endoperoxide), so this tier is advisory |
| `fatal` | geometric / electronic impossibility |
| `invalid` | unparseable SMILES |

Three modes: `off` rejects nothing; `fatal` rejects `fatal` + `invalid`; `strict` also rejects
`hard`.

### Validated

`main/test_feasibility_gate.py` — 30 molecules, exit-code driven. 18 real drugs and natural
products (aspirin, imatinib, sunitinib, penicillin G, morphine, camphor, cubane, caffeine,
vitamin C, norbornene, trans-cyclooctene, β-lactam, azetidine, oxetane, pyrrole, furan, phenyl
azide, artemisinin) must never be `fatal`; 12 impossible or non-isolable structures (azete,
cyclobutadiene, oxete, cyclopropyne, benzyne, anti-Bredt, bicyclo[2.2.2]oct-1-ene, cyclic
allene, 1,2-dioxetane, gem-diol, triazane) must never be `ok`. All 30 pass.

**The gate is only useful if its false-positive rate on real chemistry stays at zero. Re-run
this panel after touching any threshold.**

```bash
cd MOLLEO/multi_objective && python main/test_feasibility_gate.py
```

### Known limitations

- **trans-cycloalkenes in rings ≤7 are NOT caught.** RDKit does not retain E/Z stereo on bonds
  in small rings, so `_trans_small_ring_alkene` cannot fire there. `C1CC/C=C/C1` returns `ok`.
- The anti-Bredt size threshold (two smallest rings summing to ≤13) is *heuristic*. It
  correctly rejects bicyclo[2.2.2]oct-1-ene and correctly passes norbornene; raising it will
  start rejecting isolable large-bridge alkenes.
- `hard` contains real drugs. Do not reject on it unless you mean to.
- Aromatic systems are skipped — rdkit sanitization already validated them.

## 3. Files and API

| path | role |
|---|---|
| `multi_objective/main/feasibility_gate.py` | the gate + standalone CLI |
| `multi_objective/main/test_feasibility_gate.py` | validation panel |
| `single_objective/main/feasibility_gate.py` | identical copy |
| `single_objective/main/test_feasibility_gate.py` | identical copy |

Duplicated across the two trees because the repo already duplicates `utils/` and `oracle/` that
way, and because `run.py` is invoked with the tree as cwd so `main.*` resolves per-tree.
**If you change one copy, change both.**

```python
from main.feasibility_gate import check, rejects, is_feasible, penalty, audit_results_yaml

check(smiles)            # -> (verdict, [reasons]);  verdict in ok|hard|fatal|invalid
rejects(smiles, mode)    # -> (bool, [reasons]);     mode in off|fatal|strict
is_feasible(smiles, mode="fatal")
penalty(smiles, fatal=1.0, hard=0.25)   # objective-term form; see the warning in §6
audit_results_yaml(path, top=None)      # -> [(smiles, score, verdict, reasons)]
```

## 4. How it is wired

`BaseOptimizer.sanitize()` is the seam. It is called on parents+offspring in every GA loop,
always *before* `select_pareto_front`, so rejected molecules never enter the front, never reach
`mol_buffer`, and never consume oracle budget.

Three edited files; `toolmol` only overrides `reset()`, so patching the base classes covers all
four methods:

| file | covers |
|---|---|
| `multi_objective/main/optimizer.py` | `molleo_multi` |
| `multi_objective/main/pareto_optimizer.py` | `molleo_multi_pareto` **and** `toolmol` |
| `single_objective/main/optimizer.py` | `molleo` |

Plus `--feasibility_gate {off,fatal,strict}` added to `multi_objective/run.py` and
`single_objective/run.py`.

Call sites, as of 2026-10-12 (re-find with
`grep -rn "self.sanitize" */main/*/run.py` — line numbers drift):
`molleo_multi/run.py:156`, `molleo_multi_pareto/run.py:162`, `toolmol/run.py:209`,
`single_objective/main/molleo/run.py:158`.

Three design details that matter:

1. **`self.gate_mode = getattr(args, 'feasibility_gate', 'off')`** — via `getattr`, because the
   `fewshot_gen` and `online_fewshot` paths construct their own args namespaces. Do not change
   this to `args.feasibility_gate`.
2. **Empty-population guard.** If the gate would reject *every* molecule in a generation,
   `sanitize` keeps them and prints why. `select_pareto_front` and `make_mating_pool` both
   assume at least one molecule; the alternative is a crash.
3. **Rejections are logged** to `feasibility_rejections_<task_label>.json` in `--output_dir`,
   written by `save_gate_log()` from `optimize()` just before `reset()`. Gated molecules leave
   no trace in `mol_buffer`, so this file is the only place the count survives — and how often
   a backend proposes impossible chemistry is itself a result.

## 5. Using it

```bash
cd MOLLEO/multi_objective

# audit a finished run (reads only the yaml you name)
python main/feasibility_gate.py --audit main/toolmol/results/results_ToolMol_x_y1.yaml --top 100

# one-off SMILES
python main/feasibility_gate.py "C1=NC=C1" "CC(=O)Oc1ccccc1C(=O)O"

# new run with the gate on
python run.py toolmol --mol_lm ToolMol --min_obj sa --max_obj jnk3 qed --feasibility_gate fatal
```

`--audit` prints an `ok/hard/fatal/invalid` breakdown and lists every flagged molecule with its
score. It is the non-invasive way to find out whether an already-finished run was banking score
on strain.

## 6. Methodology warnings

- **A gated arm and an ungated arm are different experiments.** Enabling the gate changes what
  the run measures. Do not compare top-1/top-10/AUC across the boundary.
- **Do not fold `penalty()` into `Oracle.evaluate()` casually.** It redefines Phi, so every
  completed run becomes incomparable. If strain should be priced rather than forbidden, add it
  as a fourth *reported* objective instead.
- **Interaction with the budget.** `max_oracle_calls` in `pareto_optimizer.py` counts
  `len(storing_buffer)` — Pareto-front *admissions*, not oracle evaluations — while
  `select_pareto_front` evaluates every candidate for free. Gating at `sanitize` removes
  molecules *before* both, so it reduces true evaluations and (if any would have been
  non-dominated) slows nominal budget consumption. Account for this before comparing run
  lengths.
- Prefer the `sanitize`-time gate over a penalty term; prefer `fatal` over `strict` unless you
  specifically want labile groups gone.

## 7. Porting to the RSI arms

Not done. *Measured* 2026-10-12: all **21** optimizer files and **14** `run.py` files across the
seven arm trees (`OPFS_context`, `OPFS_context1`, `RSI_baseline`, `RSI_baseline_no_context`,
`RSI_context_and_memory`, `RSI_context_memory_code`, `RSI_context_only`) match the patch anchors
**exactly**, and none is already patched. The port is mechanical.

Per arm tree: copy `feasibility_gate.py` and `test_feasibility_gate.py` into both
`multi_objective/main/` and `single_objective/main/`, then apply the same five edits to
`multi_objective/main/optimizer.py`, `multi_objective/main/pareto_optimizer.py` and
`single_objective/main/optimizer.py`:

1. after `import os` → `import json` + `from collections import Counter`
2. after `from main.utils.chem import *` → `from main.feasibility_gate import rejects`
3. after the `self.smi_file = args.smi_file` / `self.oracle = Oracle(args=self.args)` pair
   (unique to `__init__`, **not** the identical line in `reset()`) → set `self.gate_mode` and
   call `self._reset_gate_log()`
4. replace the `sanitize()` body and append `_reset_gate_log`, `_record_gate_rejections`,
   `save_gate_log`
5. in `optimize()`: `self._reset_gate_log()` before `self._optimize(...)`, and
   `self.save_gate_log(self.oracle.task_label)` before `self.reset()`

and add the `--feasibility_gate` argument after the `--patience` line in each `run.py`.

The simplest correct port is to diff this tree and replay it, since the anchors are identical:

```bash
cd /Users/gujw/Desktop/Research/AI4S
# the four new files
for arm in RSI/*/; do
  for tree in multi_objective single_objective; do
    [ -d "$arm$tree/main" ] || continue
    cp MOLLEO/$tree/main/feasibility_gate.py       "$arm$tree/main/"
    cp MOLLEO/$tree/main/test_feasibility_gate.py  "$arm$tree/main/"
  done
done
# the edits — MOLLEO is a git repo, so its own diff is the patch
cd MOLLEO && git diff -- multi_objective/main multi_objective/run.py \
                         single_objective/main single_objective/run.py > /tmp/gate.patch
for arm in ../RSI/*/; do (cd "$arm" && patch -p1 --dry-run < /tmp/gate.patch); done
```

*Measured* 2026-10-12: this exact `patch -p1 --dry-run` was run against all seven arm trees and
succeeded in every one, so the context lines match too — not just the anchor strings. Still run
the dry-run before applying, in case an arm has moved since. Afterwards, re-run
`python main/test_feasibility_gate.py` inside each patched tree and confirm
`--feasibility_gate` appears in `python run.py --help`.

**Respect the arm-isolation rule**: do not read one arm's result artifacts while operating
another. Copying code files is fine; `--audit` reads only the yaml you name.

## 8. Unrelated gotchas found while doing this

- `/Users/gujw/Desktop/Research/AI4S/oracle/fpscores.pkl` is a **199-byte HTML error page**, not
  a pickle (a failed Dataverse download). `readFragmentScores()` opens `oracle/fpscores.pkl`
  *relative to the current working directory* and only catches `EOFError`, so anything calling
  the SA oracle from the AI4S root dies with an uncaught `UnpicklingError`. The copies under
  `MOLLEO/*/oracle/` and `RSI/*/*/oracle/` are intact at 9,048,931 bytes. Delete or replace the
  root one.
- Upstream `zoom-wang112358/MOLLEO` ships all three `GPT4.py` files with placeholder blanks
  (`openai.api_base = `, `"engine": ,`) that are **syntax errors**, and `run.py` imports `GPT4`
  unconditionally, so the BioT5 path breaks too. This fork has already ported them to the modern
  OpenAI SDK.
- Upstream `multi_objective/run.py` tests `args.max_obj == ['gsk3', 'qed']` for task 2 while the
  README and TDC use `gsk3b`, so the documented task-2 command prints `Undefined Task!`, leaves
  `task_mode` at `'1'`, and hands the LLM the **JNK3** prompt while scoring gsk3b. Check whether
  this fork still has it before running task 2.
- `BaseOptimizer` builds `tdc MolFilter(['PAINS','SureChEMBL','Glaxo'])` in `__init__` and
  **never calls it**. Same seam as the gate (`sanitize`) if you want nuisance filtering too —
  different axis, not strain.
- `Oracle.__init__(self, args=None, mol_buffer={})` has a **mutable default argument**, and
  `reset()` constructs `Oracle(args=self.args)` with no buffer, so every seed in
  `--seed 1 2 3` shares one dict. `storing_buffer` is per-instance so the Pareto budget escapes
  it, but scores, patience and the saved yaml do not.
