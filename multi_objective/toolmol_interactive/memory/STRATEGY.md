# ToolMol operator playbook — JNK3 / QED / SA, 50-oracle-call budget

Persistent artifact. Read at the start of every ToolMol run and at the start of every
episode where the next move is not already obvious. Updated at the end of every run with
whatever the oracle actually said (see `FRAGMENTS.md`, `RUNLOG.md`).

Status: **v6, after run 4 (seed 4): 50/50 calls, best Phi 2.3906.** Runs reached 2.2134,
2.3031, 2.3799, 2.3906 — still improving, but the gains are shrinking fast (+0.090, +0.077,
+0.011).

**Read `FRAGMENTS.md` first.** Both rings, the linker and the C4 position are now fully
mapped (24 rules). Section 4 of this file is obsolete literature prior, kept only as a list of
attachable fragments. `RUNLOG.md` has per-run history, `EPISODES.md` the process notes.

**The one-line plan for a new run:** in generation 1 build
`O=C(Nc1cccc(Nc2cc(C(F)(F)F)ccn2)c1)C1CC1` (jnk3 0.62, Phi 2.3906), then attack the conflict
in rule 24 — activity wants a flat aromatic ring at C4, QED charges for a third aromatic ring,
and the two cancel exactly. Look for **non-aromatic bulk** at C4: bicyclo[1.1.1]pentyl, cubyl,
cyclohexenyl, spiro and fused saturated systems, isopropenyl, butadienyl. Nothing else on this
chemotype is open.

**The habit that has cost budget in every run so far.** Four times now I have written a rule
from a comparison whose range was too narrow, and all four were wrong: "meta ≈ para" (measured
at jnk3 0.10), "keep the central ring bare" (one fluorine), "C4 is substituent-insensitive"
(measured in a 0.41-0.43 band), and "C4 has a size ceiling at ~0.61" (ten *compact* groups; a
phenyl gave 0.70). **An apparent ceiling is usually the edge of what you sampled.** Budget one
deliberately out-of-range probe per generation — in run 4 that single probe was the entire
result.

**Cheapest information available: deletion controls.** Removing a group entirely, or
methylating a donor, costs one call and bounds that feature's whole contribution. They never
produce good molecules; take them anyway, early.

**Know which objective is binding before designing.** Phi is QED-limited whenever jnk3 is below
about 0.6 and activity-limited above it. In run 4 the aryl series raised jnk3 by 0.13 and moved
Phi by nothing, because QED fell by the same amount. Check the arithmetic before spending a
generation on one objective.

## 1. What the score actually is

`Phi(m) = jnk3 + QED + (10 - SA)/9`, each term in [0,1] (`main/toolmol/objectives.py`,
`OBJECTIVE_BOUNDS`: qed (0,1), sa (10,1), jnk3 (0,1)). Max Phi = 3.

Typical random ZINC molecule: QED ≈ 0.70-0.90, SA ≈ 2.2-3.5 → SA-term ≈ 0.72-0.87,
jnk3 ≈ **0.00-0.05**.

**All the headroom is in jnk3.** QED and SA are already near their practical ceiling in
ZINC; squeezing +0.05 out of QED is a rounding error next to moving jnk3 from 0.02 to 0.40.
So the policy is: *push jnk3 hard, and only spend effort on QED/SA to stop them collapsing.*

Guard rails that keep QED/SA from collapsing while chasing jnk3:
- MW ≤ ~450 (hard ceiling 700 from the prompt, but QED falls off a cliff past ~500).
- LogP roughly 1-4. TPSA < 140. HBD ≤ 3, HBA ≤ 8. Rotatable bonds ≤ 8.
- Aromatic rings ≤ 4. Don't add fused polycycles unless they're common drug cores —
  SA punishes unusual ring systems far more than it punishes extra atoms.
- Never leave a charged group (`[NH3+]`, carboxylate) if it can be avoided — QED dislikes it.

### Reading jnk3 off a parent's prompt
The prompt gives `Score:` (= Phi) and the `Properties:` line gives QED and SA. So

    jnk3(parent) = Phi - QED - (10 - SA)/9

`respond.py` prints this as `jnk3~` for both ligands on the first turn of every episode.
**Use it.** It is the only per-molecule jnk3 feedback the agent gets inside an episode, and
it tells you which parent is actually carrying activity vs. which one is just a
high-QED/low-SA ZINC molecule. A parent with jnk3~0.00 contributes nothing but drug-likeness;
a parent with jnk3~0.10+ has something the oracle recognizes and is worth *preserving*.

---

## 2. Mechanics (confirmed from source — don't re-learn these)

- **Atom indices** in the prompt's table always refer to the *current working molecule*
  (ligand 1 / the edit product). Only `crossover_molecules` sees ligand 2, and its `idx2`
  is the only index ever interpreted against ligand 2.
- **`add_substructure` / `replace_substructure` take arbitrary SMILES.** `[1*]X...` attaches
  through `X` (the dummy's neighbour; the dummy is deleted). With no `[1*]`, the *first atom
  written* is the attachment point. This is the single most powerful tool here: it can graft
  a whole pharmacophore in one call. There is no size check on it.
- **`replace_substructure(anchor_idx, branch_idx, smiles)`** cuts the anchor-branch bond,
  throws away the branch side, and attaches `smiles` at anchor. Prefer it over
  `add_substructure`: it installs the motif *and* removes dead weight, so MW stays flat.
  The bond must be acyclic — pick `branch_idx` from anchor's `nbrs` such that the bond
  leaves a ring rather than being part of one.
- **`crossover_molecules` is stochastic and weakly controllable.** It cuts each parent at
  one acyclic bond of the named atom, then picks *at random* among the up-to-4 fragment
  recombinations that pass `mol_ok`/`ring_OK` (`main/toolmol/toolbox.py`). `mol_ok` also
  rejects anything with ≥ ~39 heavy atoms (`crossover.py: average_size=39.15, stdev=3.50`).
  So you cannot predict which half of each parent survives.
  → **Do not open with crossover when the episode has a specific design intent.** The
  prompt "encourages" it; it is not required, and a directed `replace_substructure` is worth
  far more than a coin flip. Use crossover only when both parents carry real jnk3 signal
  and the goal is genuinely to merge two unknown actives.
- Ring bonds cannot be cut by `remove_substructure` / `replace_substructure`.
- `undo_last_change` is free (doesn't count against the 3-modification budget). Use it the
  moment a tool result says it removed something other than what was intended.

### Anchors that `replace_substructure` refuses (measured, costs a whole offspring)
`cut_at_bond` calls `FragmentOnBonds(..., addDummies=False)`. For some atoms RDKit then
pins the stub's hydrogens as **explicit** with `noImplicit=True`, so the subsequent
`AddBond` overflows the valence and the whole call fails with
*"Explicit valence for atom # N … is greater than permitted"*. Verified cases:

| anchor atom | stub after cut | result |
|---|---|---|
| aromatic ring **nitrogen** (e.g. pyrazole N1 bearing an N-aryl) | becomes `[nH]`, explicit | **fails** |
| sp3 carbon that is a **stereocentre** (`[C@H]`/`[C@@H]`) | explicitHs=2, noImplicit=True | **fails** |
| sp3 carbon, no stereo tag (e.g. azetidine C3) | implicitHs=2 | works |
| aromatic carbon | implicitHs=1 | works |
| sulfonyl **S** | works |
| aliphatic **amide N** (its H stays implicit) | works |

**Workaround for a stereocentre anchor** (verified): `remove_substructure(anchor, branch)`
succeeds and drops the chirality tag, then `add_substructure` on that atom in the *next*
tool call works. Costs two modifications instead of one. Simply pick a different anchor if
one is available.

### Atom-index stability after an edit
Working-molecule indices are re-derived from the canonical SMILES after every successful
tool call, so they can renumber. In practice a pendant chain written early in the canonical
SMILES keeps its indices across a graft elsewhere in the molecule (the
`CC(C)OCCS(=O)(=O)N1CC(X)C1` scaffold kept `S`=6 / `CH2`=5 across three different grafts
at X). That makes it *sometimes* safe to batch two edits blind - but a wrong guess can
"succeed" while cutting the wrong bond, which is worse than a failure, so only batch the
second edit when the scaffold has already been seen to hold its numbering.

### The one-call pharmacophore graft (run 2's biggest efficiency win)
`replace_substructure(anchor, branch, SMILES)` keeps the **anchor** side and discards the
branch side, and the fragment SMILES is unrestricted. So anchoring on a peripheral atom whose
own side is tiny discards the whole host and installs a complete designed molecule in a single
tool call. Two idioms that cover most ZINC hosts:

- **terminal methyl**: `replace_substructure(0, 1, "[1*]c1ccnc(N<rest>)c1")` - atom 0 is the
  leading `C` of a canonical SMILES starting `C...`, and the surviving CH3 becomes the
  pyridine 4-methyl.
- **cycloalkyl CH**: `replace_substructure(ring_CH, carbonyl, "[1*]C(=O)N<rest>")` - the
  surviving ring becomes the acyl cap.

The trick is choosing *where the surviving atom lands in the product*. Run 1 spent 2-3 calls
per molecule doing the same work.

**Choosing the anchor chooses what you can build.** A terminal-methyl anchor forces a methyl
somewhere in every product, which in run 3 repeatedly produced 4-methyl azines when a halogen
was wanted — and methyl is the *worst* C4 group. When the pending host carries a cycloalkyl,
anchor on that ring instead and the azine is free to be anything. Check the host for a
cycloalkyl before reaching for the methyl out of habit. A host with neither (no terminal
methyl, no pendant ring) needs two cuts, so budget two calls for a generation-1 rebuild.

### Reusable atom indices
For the reference molecule `Cc1ccnc(Nc2ccc(NC(=O)C3CC3)cc2)c1`: atom 0 = pyridine 4-methyl,
atom 1 = pyridine C4, atom 12 = amide carbonyl, atom 14 = cap cycloalkyl CH. These held
across every analogue sharing that canonical prefix, including when the cap ring grew.
`add_atom` inserts the new atom immediately after the atom it bonded to and shifts only
later indices by one. Predicting these is safe *on a scaffold already seen*; never on a new
one, because a wrong guess silently cuts the wrong bond.

### Turn budget
`max_steps: 3`, so three assistant turns per episode. Standard shape:

1. the one high-value directed edit (usually `replace_substructure`),
2. a second edit informed by the *actual* resulting atom table (trim/tune, or install a
   second motif),
3. `FINAL ANSWER`.

Turn 3 is predictable, so batch it with turn 2 in a single `respond.py reply '[...]'` call.
If the episode's plan is a single edit, batch turns 2 and 3 as `[FINAL ANSWER]` right away.

---

## 3. Budget policy across a run

### Measured from run 1 — a generation is a blind parallel batch
All `offspring_size` episodes of a generation are generated *before* any of them is scored
(`run.py` builds every offspring, then calls `select_pareto_front`, then `oracle(...)`).
So there is **no feedback inside a generation**. Design each generation as a batch of
simultaneous experiments, not as a sequence. With 20 + 3x10 = 50 calls that is three blind
batches, which is exactly why generation 1 is worth spending on controlled probes: ten
matched-pair comparisons resolved more than ten greedy edits would have.

Also measured: offspring that are dominated still get evaluated by `select_pareto_front`,
so **every distinct offspring costs a call whether or not it survives**. Duplicates are
free. A failed tool call that leaves 0 modifications costs the episode (Graph-GA fallback)
but still costs a call.

### Where Phi actually comes from (run 1)
Phi rose 1.721 → 2.213. QED stayed in 0.85-0.90 and the SA term in 0.90-0.93 the whole way;
**every bit of the gain after generation 1 was jnk3** (0.06 → 0.41). But that only became
true once a real pharmacophore was found. Before that, QED discipline dominated: a 413 Da
molecule with jnk3 0.10 scored Phi 1.404, while a 241 Da molecule with jnk3 0.07 scored
1.829. **Rule: hold MW ≤ ~290 and QED ≥ 0.85 until jnk3 is above ~0.2; only then consider
trading drug-likeness for activity - and in run 1 that trade never became worth making,
because the two decorations that mattered most were both QED-neutral or QED-positive.**

50 distinct molecules, of which 20 are spent on the random initial pool. ~30 offspring,
10 per generation, ~3 generations. That means:

- **Generation 1 = probe.** 10 episodes, each installing a *different* candidate jnk3
  pharmacophore onto a good-QED/low-SA host. Maximum information per oracle call. Do not
  spend all ten episodes on variations of one idea.
- **Generation 2 = exploit + branch.** The generation-1 parents that survived onto the
  Pareto front show up as ligands with a readable `jnk3~`. Elaborate the winners (decorate,
  add a second recognized motif) and probe 2-3 fresh chemotypes with the remainder.
- **Generation 3 = exploit only.** Push the best chemotype; tune QED/SA on it.

Pareto selection keeps anything non-dominated across (jnk3, QED, SA), so a very high-QED
low-jnk3 ZINC molecule will *not* be removed from the population. Expect the pool of parents
to stay polluted with jnk3=0 molecules; that is normal and it is why a run can waste
episodes crossing two inactive parents. Prefer the parent with the higher `jnk3~` as the
host (= ligand 1 is the working molecule, so when ligand 2 is the active one, either
crossover or rebuild its motif onto ligand 1 with `add_substructure`).

---

## 4. Chemotype priors for JNK3 — SUPERSEDED, see `FRAGMENTS.md`

Measured outcome of testing the table below: **a lone hinge heterocycle does essentially
nothing** (indazole 0.02, carbazole 0.01, bare aminopyrimidine 0.03, benzimidazole 0.03).
What the oracle rewards is an **(hetero)aryl-NH-(hetero)aryl linkage** - an arylamine or an
anilide bridging two rings - and it rewards two such linkages more than one (0.07 for one,
0.08-0.10 for two). Keep the table as a source of attachable fragments, not as a ranking.

### Original priors (literature)

The oracle is a random forest on ECFP-style fingerprints trained on ExCAPE-DB JNK3
actives/inactives, so it rewards *presence of fragments characteristic of known actives*,
not binding physics. Grafting a canonical JNK3/kinase hinge-binder motif is the move.

Ready-to-paste fragments (all written `[1*]`-first so the attachment point is explicit):

| # | motif | SMILES fragment | why |
|---|-------|-----------------|-----|
| A | 2-aminopyrimidin-4-yl | `[1*]c1ccnc(N)n1` | canonical ATP-site hinge donor/acceptor pair |
| B | 2-anilinopyrimidine (N-linked) | `[1*]Nc1ncccn1` | the classic anilinopyrimidine kinase chemotype |
| C | 1H-indazol-5-yl | `[1*]c1ccc2[nH]ncc2c1` | indazole hinge binder, very common in JNK series |
| D | 7-azaindol-3-yl | `[1*]c1c[nH]c2ncccc12` | pyrrolopyridine hinge binder |
| E | 1H-pyrazol-4-yl | `[1*]c1cn[nH]c1` | small, cheap, SA-friendly aminopyrazole surrogate |
| F | 2-aminopyridin-5-yl | `[1*]c1ccc(N)nc1` | aminopyridine hinge binder |
| G | anilide / N-phenyl amide | `[1*]C(=O)Nc1ccccc1` | the amide spine of most JNK3 series |
| H | 3-cyanoanilide | `[1*]C(=O)Nc1cccc(C#N)c1` | meta-CN aniline recurs in JNK3 actives |
| I | benzimidazol-2-yl | `[1*]c1nc2ccccc2[nH]1` | fused hinge binder |
| J | imidazo[1,2-a]pyridin-3-yl | `[1*]c1cnc2ccccn12` | kinase-active fused core |
| K | carbazol-3-yl | `[1*]c1ccc2c(c1)[nH]c1ccccc12` | carbazole-carboxamide JNK3 series (heavy: watch MW) |
| L | 4-pyridyl | `[1*]c1ccncc1` (= `add_functional_group` "pyridyl") | cheapest hinge acceptor |
| M | aminopyrimidinyl-aniline composite | `[1*]Nc1ccc(-c2ccnc(N)n2)cc1` | B+A in one graft, bigger bet |
| N | 2-(methylamino)pyrimidin-4-yl | `[1*]c1ccnc(NC)n1` | A with the NH methylated |
| O | thiophene-2-carboxamide | `[1*]C(=O)c1cccs1` | acyl cap seen in JNK amide series |

Expected failure mode: a single small heteroaryl (E, L) is usually **not** enough for an
ECFP random forest to call a molecule active — one fragment match rarely moves the vote.
Larger, more distinctive grafts (M, K, H, C, D) are the better bets per oracle call. Probe
both sizes in generation 1 so this prior is actually tested rather than assumed.

---

## 5. Episode checklist

1. Read both ligands' `jnk3~`. Identify which (if either) carries signal.
2. Pick the host: the working molecule is **ligand 1**, always. If ligand 2 is the active
   one and ligand 1 is not, either crossover, or graft ligand 2's motif onto ligand 1.
3. Find a disposable branch on ligand 1: a terminal alkyl/alkoxy/halogen chain, a
   solubilizing amine, a non-ring substituent with low `cent`. Its neighbour toward the core
   is `anchor_idx`; the branch atom itself is `branch_idx`. Check `nbrs` to confirm they're
   bonded and that the bond isn't in a ring (`ring` column = N on the branch side).
4. Graft the chosen motif with `replace_substructure`.
5. Read the result's SMILES and properties. If MW > 450 or QED dropped hard, trim with
   `remove_substructure`; otherwise add one complementary small group (F, Cl, CN, methyl,
   methoxy) or a second motif.
6. `FINAL ANSWER`.
7. Record the (host, motif, product SMILES) triple in the run's notes so the ledger can be
   joined against it afterwards.

## 6. Hygiene

- The oracle is only ever called by the pipeline. Never score a candidate out-of-band —
  that would silently spend budget the experiment does not have.
- `oracle_ledger.jsonl` is legitimate feedback (it is the result of calls already paid for),
  and carrying it across runs is the point of this setup. `FRAGMENTS.md` is its distillate.
