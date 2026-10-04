# Episode notes

Per-episode record of (generation, pair, parents, intent, motif grafted, product SMILES).
Joined against `oracle_ledger.jsonl` after each generation to attribute jnk3 to motifs.

_(no episodes yet)_

## Run 1 (seed 1) — 30 designed episodes across 3 generations

Full per-turn transcript is in `runs/seed1/run_log.txt`; per-molecule oracle output is in
`runs/seed1/oracle_ledger.jsonl`. The distilled chemistry is in `FRAGMENTS.md` - that is
what to read. Recorded here are only the episode-level process notes worth carrying.

- **Generation 1 (ep 1-10), blind probe sweep.** One chemotype per episode. Three episodes
  landed on the *same* host (a pyrazolyl-acetamide) and three on the same minimal
  azetidine-sulfonamide, which is what made the graft-size and chemotype comparisons
  controlled. Worth repeating: when the sampler hands you a host you have already edited,
  reuse the identical anchor so the comparison is clean.
- **ep 1** failed on an aromatic-ring-nitrogen anchor; **ep 9** failed on a stereocentre
  anchor and produced no molecule at all. Both failure modes are now in STRATEGY.md §2.
- **ep 19 (N-methyl control)** was the highest-value episode of the run despite producing a
  dead molecule (jnk3 0.00): it proved the N-H donor is essential and it cost nothing in
  Phi terms because removing a donor *raised* QED to 0.896. Deliberately building a control
  that is designed to fail is cheap here and worth doing early.
- **ep 25 (pyridin-3-yl)** and **ep 26 (monocyclic thiazole)** were the other two pure
  probes, and both came back 0.00-0.01. Between them they pinned down the regiochemistry
  requirement and corrected my wrong explanation for the benzothiazole failure.
- **Batching turns.** After a graft, the canonical atom ordering of this scaffold family was
  stable enough to predict the next edit's indices and send both turns in one go. Verified
  rules: a `CC(=O)N...` or `CS(=O)(=O)N1CC(...)` prefix keeps its low indices across a graft
  elsewhere in the molecule, and `add_atom` inserts the new atom immediately after the atom
  it bonded to, shifting only later indices by one. Predicted indices were correct in all
  four cases where I relied on them. Still do not guess when the scaffold is new: a wrong
  guess *succeeds* while cutting the wrong bond, which is worse than an error.

## Run 2 (seed 2) — 30 episodes

- **The one-call pharmacophore graft.** The single biggest efficiency win of run 2. Because
  `replace_substructure` takes arbitrary fragment SMILES and *keeps the anchor side*,
  anchoring on a peripheral atom whose own side is tiny - a terminal methyl, or a cycloalkyl
  CH bonded to the rest by one acyclic bond - discards the entire host and installs a
  complete designed molecule in **one** tool call. Run 1 took 2-3 calls per molecule.
  Two useful anchor idioms:
  - terminal CH3, `replace_substructure(0, 1, "[1*]c1ccnc(N<rest>)c1")` → the surviving
    methyl becomes the pyridine 4-methyl;
  - cycloalkyl CH, `replace_substructure(ring_CH, carbonyl, "[1*]C(=O)N<rest>")` → the
    surviving ring becomes the acyl cap.
  Choosing *where the surviving atom lands in the product* is what makes this general.
- **Rebuilding a known winner costs one call and is worth it.** A generation gives no
  feedback, so without a known-good parent in the population the whole generation is a gamble.
  Episode 1 did this and generations 2-3 refined from Phi 2.21 instead of from random ZINC.
- **Reusable atom tables.** The reference molecule `Cc1ccnc(Nc2ccc(NC(=O)C3CC3)cc2)c1` has
  atom 0 = pyridine 4-methyl and atom 14 = cap cyclopropane CH (carbonyl at 12). Those
  indices held across every analogue with the same canonical prefix, including when the cap
  ring grew to cyclobutyl. Predicting them saved a fetch per episode; they were right every
  time I relied on them.
- **Episodes 11/12/18/20 (pyridine C3 and C6 substitution) all came back 0.05-0.11.** Four
  calls to establish one rule, but it was worth it: it closed off an entire direction and
  redirected the remaining budget to the central ring, which is where the meta result came
  from.

## Run 3 (seed 3) — 30 episodes

- **Deletion controls are the cheapest information in this project.** Three episodes removed
  the cap, methylated the anilide N–H, and deleted the azine ring nitrogen. Together with
  run 1's arylamine N-methyl control they reduced three runs of scattered structure-activity
  to one sentence. None of them was a good molecule; all four were worth their call.
- **An amide-nitrogen anchor works.** Added to the anchor table: aromatic ring N and
  stereocentres fail, aliphatic amide N is fine (its hydrogen stays implicit through the cut).
  That matters because it is often the only way into a molecule with no terminal methyl.
- **A host with no terminal methyl needs two cuts.** Episode 1 had to trim to the
  cyclobutanecarboxamide first and then replace the whole aryl-azine half. Worth knowing the
  one-call graft is not always available — budget two calls for the generation-1 rebuild.
- **Choosing the anchor chooses what you can build.** With a terminal-methyl anchor every
  product must contain a methyl somewhere, which repeatedly forced 4-methyl molecules when I
  wanted a halogen. When the pending host has a cycloalkyl, anchor on *that* instead and the
  azine is free. Several mid-run episodes were weaker than they needed to be because I took
  the methyl anchor by habit.
- **Four consecutive episodes landing within 0.01 is the signal to stop tweaking.** It
  happened in run 2 and again in run 3 generation 2; both times the plateau broke only when a
  previously "settled" variable was re-opened at the new activity level.

## Run 4 (seed 4) — 30 episodes

- **The best result of the run came from a probe I expected to fail.** The phenyl-at-C4
  episode was framed as "find the size ceiling"; it found 0.70 instead. Three runs in a row
  have now had an apparent ceiling turn out to be the edge of what I had sampled rather than a
  real limit. Budget one deliberately out-of-range probe per generation.
- **Anchor choice decided what each episode could test.** A terminal-methyl anchor forces a
  methyl into the product, which is why generation 1 ran on substituents that *contain* a
  methyl - methylthio, acetyl, methylsulfonyl, propynyl, methylamino, methyl ester. That was
  a happy accident here, since it produced a clean electronic spread, but it was not a choice
  I made deliberately until partway through.
- **Ordering two batched edits by atom count.** When both a cap swap and an azine swap are
  wanted, do the cap first if its fragment changes the atom count: the cap sits late in the
  canonical ordering, so the azine indices survive, while the reverse shifts everything. Used
  successfully in four episodes.
- **A failed index guess is usually safe.** Predicting the cap CH index risks a mis-cut, but
  the realistic failure is "atoms are not bonded", which the tool rejects cleanly. The
  dangerous case needs the guessed atom to be bonded to the anchor, which on this scaffold
  only the cap carbon is. That reasoning made several batched episodes worth the risk.
