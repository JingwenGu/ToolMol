# Run log — one block per ToolMol run

Append-only. Read `FRAGMENTS.md` for the chemistry and `STRATEGY.md` for the method.

---

## Run 1 — seed 1, budget 50, task jnk3+qed / sa

**Result: best Phi 2.2134** (`Cc1ccnc(Nc2ccc(NC(=O)C3CC3)cc2)c1`, jnk3 0.410, QED 0.891, SA 1.791)
on the last of exactly 50 oracle calls. Starting pool's best was Phi 1.721 with best jnk3 0.06.

| stage | oracle calls | best Phi | best jnk3 |
|-------|--------------|----------|-----------|
| initial pool (20 random ZINC) | 20 | 1.721 | 0.06 |
| generation 1 (10 probes)      | 29 | 1.829 | 0.10 |
| generation 2 (10 designs)     | 39 | 1.921 | 0.14 |
| generation 3 (10 designs)     | 49 | 2.066 | 0.28 |
| final call                    | 50 | **2.213** | **0.41** |

**What I did.** Generation 1 was spent deliberately as a *blind probe sweep* - ten different
candidate chemotypes, one per episode, each grafted onto a decent-QED host, several on
matched hosts so the comparison was controlled. Generations 2 and 3 exploited and refined,
each still blind within itself (a generation's offspring are all produced before any of them
is scored - see STRATEGY.md §3). The final call combined the three separately-measured
positive effects into one molecule that had never been built.

**The single most transferable thing learned.** jnk3 here is read almost entirely off one
feature: **an aryl-N(H)-azine link whose azine is monocyclic and whose ring nitrogen sits
ortho to a free N-H**. Every part of that sentence was established by a matched pair that
broke when it was violated - N-methylation → 0.00, pyridin-3-yl instead of -2-yl → 0.00,
fused azine → 0.01-0.05. Then two decorations on top (a 4-methyl on the pyridine, a
cyclopropanecarboxamide cap) each roughly doubled it. Chemotype intuition about "kinase
hinge binders" was actively misleading: indazole, carbazole, benzimidazole, benzothiazole
and the textbook anilinoquinazoline all scored ≈ 0.

**Cost of mistakes.** Two offspring were wasted: one on an aromatic-ring-nitrogen anchor and
one on a stereocentre anchor (both now documented in STRATEGY.md §2 with a workaround), and
one episode produced no molecule at all. Early episodes also over-weighted jnk3 against QED -
the 0.10-jnk3 molecule at 413 Da scored Phi 1.404, worse than a 241 Da molecule with jnk3
0.07. Keeping MW ≤ ~290 was worth more than any single activity gain until jnk3 passed ~0.2.

**For the next run:** build `O=C(C1CC1)Nc1ccc(Nc2cc(C)ccn2)cc1` or a close analogue within
the first generation instead of probing chemotypes again, then spend the remaining budget
pushing past jnk3 0.41 (untested: larger 4-alkyl on the pyridine, a second ring methyl,
cyclobutane/cyclopentane caps, and whether a bis-arylamine version of the methylpyridine
link beats the mono - bis-pyridylamine reached 0.23 against 0.15 for mono, so the stacking
effect is real but cost 0.09 of QED when both links were unmethylated pyridines).

---

## Run 2 — seed 2, budget 50, same task

**Result: best Phi 2.3031** (`O=C(Nc1cccc(Nc2cc(Cl)ccn2)c1)C1CCC1`, jnk3 0.520, QED 0.889,
SA 1.951). Run 1 finished at 2.2134, so carrying the memory forward was worth **+0.09 of
Phi** — and it got there far earlier: Phi 2.2134 was reached on **call 21**, where run 1
needed all 50.

| stage | oracle calls | best Phi | best jnk3 |
|-------|--------------|----------|-----------|
| initial pool (20 random ZINC) | 20 | ~1.76 | ~0.03 |
| generation 1 | 30 | 2.2134 | 0.43 |
| generation 2 | 40 | 2.2962 | 0.50 |
| generation 3 | 50 | **2.3031** | **0.52** |

**What I did differently.** No chemotype probe sweep — `FRAGMENTS.md` already answered that.
Episode 1 rebuilt the run-1 winner from a fresh ZINC host (one call, scored 0.410/0.891/1.791,
*identical to run 1*), which both confirmed the oracle is deterministic and put a Phi-2.21
parent into the population from generation 1. The other nine generation-1 episodes went
straight to the untested extensions listed at the end of run 1's notes. Generations 2 and 3
then ran systematic position scans of first the pyridine and then the central ring, which is
what found the meta geometry.

**Biggest wins:** the anilide belongs *meta* to the arylamine, not para (+0.09) — and a
4-chloro plus cyclobutyl cap added another +0.02 while improving QED.

**Two of my own run-1 rules were wrong, both for the same reason.** I had written "meta ≈
para, placement is free" and "keep the central ring bare". The first came from a comparison
made at jnk3 ≈ 0.10, where a 0.09 effect is invisible; the second was generalized from a
single fluorine. *A null result is only as strong as the signal it was measured against,
and one substituent is not a position.*

**And additivity failed.** The central-ring methyl is +0.03 on the para core but −0.06 on the
meta core, and the full methyl scan of the meta central ring is below the bare ring at every
one of its four positions. The best molecule of the run is the *simplest*: meta, bare centre,
one halogen on the pyridine, cycloalkyl cap.

**For run 3:** build `O=C(Nc1cccc(Nc2cc(Cl)ccn2)c1)C1CCC1` in generation 1 — one tool call
from most hosts — then spend the budget on the untested list at the end of `FRAGMENTS.md`.
Both runs have now plateaued at jnk3 0.50-0.52 on this chemotype, so the honest open question
is whether that is the ceiling for 2-anilinopyridines against this oracle, or whether a
different chemotype entirely (never found in two runs of probing) scores higher. Consider
spending generation 1 of run 3 on a fresh chemotype sweep *seeded with the knowledge that
the aryl-N(H)-azine motif is what works* — e.g. azines other than pyridine with an ortho N.

---

## Run 3 — seed 3, budget 50, same task

**Result: best Phi 2.3799** (`O=C(Nc1cccc(Nc2cc(Br)ccn2)c1)C1CC1`, jnk3 0.600, QED 0.892,
SA 2.007). Every metric improved again, and the margin over run 2 was larger than run 2's
over run 1.

| run | top-1 | top-10 | top-100 | best jnk3 |
|-----|-------|--------|---------|-----------|
| 1 | 2.2134 | 1.9827 | 1.6781 | 0.41 |
| 2 | 2.3031 | 2.2428 | 1.8705 | 0.52 |
| 3 | **2.3799** | **2.3287** | **1.9393** | **0.60** |

| stage | calls | best Phi | best jnk3 |
|-------|-------|----------|-----------|
| initial pool | 20 | 1.6706 | 0.07 |
| generation 1 | 30 | 2.3031 | 0.52 |
| generation 2 | 40 | 2.3252 | 0.53 |
| generation 3 | 50 | **2.3799** | **0.60** |

**What I did.** Episode 1 rebuilt run 2's best molecule from a fresh ZINC host (two cuts,
since that host had no terminal methyl) and it scored identically again, so the run opened
at Phi 2.3031. The rest of generation 1 went to the question run 2 left open: is the
2-anilinopyridine chemotype at its ceiling? Nine episodes swept alternative azines and
alternative linkers. Generation 2 decomposed the pharmacophore with three deletion controls.
Generation 3 scanned the C4 substituent properly and found the chemotype was *not* at its
ceiling after all.

**The decomposition is the most valuable thing here.** Deleting the azine ring nitrogen
takes jnk3 from 0.53 to 0.04; methylating the arylamine N–H took it to 0.00 back in run 1.
Those two atoms are the whole pharmacophore. Removing the acyl cap costs only 0.11 and
methylating the anilide N–H costs 0.17 — real contributions, but modifiers. Three runs of
work reduce to: *an aryl N–H with a ring nitrogen ortho to it, on a benzene, meta to an
anilide.*

**And a rule of mine was wrong for the third time, the same way.** "C4 is
substituent-insensitive" came from run 2, where Me/Cl/OMe/cyclopropyl/cyclobutyl all landed
within 0.41-0.43 on the para core. On the meta core at double the activity the same position
spreads 0.49 (methyl) to 0.60 (bromine). **A structure-activity rule is only valid in the
activity band it was measured in — re-test the important ones whenever the series moves up.**
This has now cost me something in all three runs; it is the single most expensive habit in
this project.

**For run 4:** build `O=C(Nc1cccc(Nc2cc(Br)ccn2)c1)C1CC1` in generation 1 and then push C4
hard — iodine, SMe, SCN, ethynyl, nitro — because the series was still climbing when the
budget ended (Me 0.49 → F/Cl/OMe 0.50-0.54 → CN 0.58 → Br 0.60). Pair the winners with the
oxetane cap, which carries the best QED of any cap (0.911). Both rings and the linker are
fully mapped, so spend nothing re-probing them.

---

## Run 4 — seed 4, budget 50, same task

**Result: best Phi 2.3906** (`O=C(Nc1cccc(Nc2cc(C(F)(F)F)ccn2)c1)C1CC1`, jnk3 0.620, QED 0.887,
SA 2.052). Best jnk3 **0.74**, a large jump — but it did not convert into Phi.

| run | top-1 | top-10 | top-100 | best jnk3 |
|-----|-------|--------|---------|-----------|
| 1 | 2.2134 | 1.9827 | 1.6781 | 0.41 |
| 2 | 2.3031 | 2.2428 | 1.8705 | 0.52 |
| 3 | 2.3799 | 2.3287 | 1.9393 | 0.60 |
| 4 | **2.3906** | **2.3431** | **1.9777** | **0.74** |

| stage | calls | best Phi | best jnk3 |
|-------|-------|----------|-----------|
| initial pool | 20 | ~1.78 | 0.03 |
| generation 1 (C4 scan) | 30 | 2.3799 | 0.61 |
| generation 2 | 40 | 2.3906 | 0.70 |
| generation 3 (aryl series) | 50 | 2.3906 | 0.74 |

**What I did.** Episode 1 rebuilt run 3's best (two cuts — the host had no terminal methyl).
Generation 1 then spent nine episodes on the C4 lead run 3 handed over, scanning ten
substituents across every electronic class. Generation 2 found the real result almost by
accident: a *phenyl* at C4 gives 0.70, far outside the 0.51-0.61 band everything compact had
occupied. Generation 3 spent all ten episodes trying to make that affordable, replacing the
benzene with eight different polar heteroaryls.

**It did not work, and the reason is the useful part.** Every aryl-at-C4 molecule lands at QED
0.71-0.75 regardless of the ring's polarity — phenyl and methyltriazolyl differ by two and a
half log units and score 0.623 and 0.749. The penalty is the *third aromatic ring itself*,
which QED charges for directly, so no amount of polarising fixes it. Activity +0.13, QED
−0.14, SA worse: a wash. Generation 3 moved top-1 by nothing.

**Diminishing returns, honestly.** Phi gained +0.090, +0.077, +0.011 across the three
memory-carrying runs. Run 4's real product is knowledge, not a molecule: C4 is now fully
characterised, and the two objectives are shown to be in direct conflict there.

**For run 5:** build `O=C(Nc1cccc(Nc2cc(C(F)(F)F)ccn2)c1)C1CC1` in generation 1, then attack
the conflict rule 24 names — find bulk at C4 that is **not aromatic**. Bicyclo[1.1.1]pentyl,
cubyl, cyclohexenyl, spiro and fused saturated systems, isopropenyl, butadienyl. The saturated
series is barely sampled: cyclopropyl managed 0.57 and cyclobutyl was never tried. A saturated
group reaching 0.70 would clear Phi 2.45. If nothing does, the chemotype is finished and the
honest answer is that ~2.39 is its ceiling under this scoring function.
