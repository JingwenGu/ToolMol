# Domain brief

Appended to the agent's system prompt via `--domain_brief`. Everything here was measured by
running this task and watching the oracle respond, over five runs and ~250 designed molecules.
It transfers *method and costs*, not answers: no winning molecule or fragment is named, because
naming one lets a later run reproduce a result without searching and makes runs incomparable.

## 0. How to read every rule below

Each rule came from the same procedure: take one good molecule, change one thing, record the
cost. That is what makes the numbers attributable — and it makes each one **a gradient at the
point it was measured, not a constraint everywhere**. At least one inverts.

Dropping the hinge's distal ring nitrogen, by what the donor ring carries:

| donor-ring group | two-N hinge | one-N hinge |
|---|---|---|
| hydrogen | 0.70 | **0.12** |
| a primary carboxamide | 0.85 | **0.28** |
| a saturated N-linked heterocycle | 0.82 | **0.86** |

The same deletion costs 0.58 twice and *gains* 0.04 once. Nothing in the first two rows predicts
the third.

**So: when two rules each forbid a change, that is not evidence the two changes are forbidden
together.** Each was measured with the other feature held at its original value; the joint case
has never been observed. Doubly-forbidden regions are untested, not closed — and on this task
that is exactly where a better optimum sits. Spend some budget deliberately violating two rules
at once, especially when each cost was measured on a molecule unlike the one you are building.

**Corollary on interpreting a failed edit.** If your edit also changes a feature with its own
known penalty, subtract that penalty before concluding anything. Testing the pocket change alone
with an *open chain* gave 0.32 and looked like proof that three changes were inseparable; chains
carry a 0.22 penalty of their own, and the correct value with a ring is 0.58.

**Corollary on search.** Episodes are good at controlled comparison and poor at search. Four runs
never left their first scaffold, because every single step out of it looks fatal. What found the
better region was `main/toolmol/ceiling_probe.py` — cheap undirected search, run *between*
iterations, 4,000 evaluations. Use that to find where to look; use episodes to find out what is
there.

## 1. Which objective has room

Φ is a sum of objectives each rescaled to [0,1]. Read the per-objective breakdown before editing.
Drug-likeness terms start high and saturate; the activity term starts near zero and holds nearly
all the headroom. A molecule at QED 0.9 has a few hundredths left there. Polishing what is
already good is the most common way to waste an episode.

## 2. The pharmacophore

Activity is scored by a fingerprint model over known actives, so it rewards recognisable
arrangements, not general drug-likeness.

**(a) A donor N–H carrying an *aryl*** — not an alkyl, not an acyl. Confirmed on five scaffolds.
Adjacency to a ring nitrogen is not enough: an alkyl or amide neighbour gives 0.01–0.05, an aryl
gives 0.70. Methylating the N–H costs 0.72 → 0.06; acylating it, 0.70 → 0.02. The hydrogen
itself is recognised.

**(b) A conjugated ring at the hinge position beside that nitrogen**, worth +0.4 to +0.6.
Two separate requirements, easy to conflate:

*Conjugation, not aromaticity.* A non-aromatic ring bearing a C=C conjugated to the hinge scores
like a benzene (~0.87 vs ~0.86); the fully saturated version of the same ring collapses to ~0.59.
This matters for score, not just taxonomy: a non-aromatic pocket drops the molecule from three
aromatic rings to two, worth **>0.1 of QED** — the largest single lever on this scaffold, and
bigger than the entire spread across every donor substituent ever measured.

*But it must be a ring.* Conjugation alone is insufficient:

| pocket | activity |
|---|---|
| 5-ring with C=N / with C=C | 0.88 / 0.87 |
| nitrile / vinyl / isopropenyl / butenyl | 0.66 / 0.64 / 0.62 / 0.57 |
| saturated ring | 0.59 |

Rings 0.85–0.88 across five compositions and two sizes; every chain loses ~0.22. **The chains
have the best drug-likeness in the series**, so QED and SA point the wrong way here — do not
follow them. Inside the ring, composition only moves LogP and therefore QED by a few hundredths;
pick one for polarity and stop.

*Ring identity, if you keep it aromatic:* phenyl 0.70, naphthyl 0.65, thiophen-3-yl 0.60,
**pyridin-4-yl 0.04** — a basic ring nitrogen here is worse than no ring at all. Do not reach for
a pyridyl to fix lipophilicity.

**(c) Coplanar and directly bonded.** One sp3 methylene between hinge and pocket costs
0.74 → 0.25. Fluorine on the pocket ring: para 0.69, meta 0.63, ortho 0.50.

**(d) The 1,3 relationship between pocket and donor is required.** Moving the pocket to 1,4 —
same atoms, QED identical to four decimals — costs **0.88 → 0.29**. This cannot be tested on a
two-nitrogen hinge at all, so it is easy to inherit unexamined for many runs.

## 3. What is closed

**The hinge ring tolerates nothing.** Not substitution: methyl adjacent to the ring nitrogen
0.70 → 0.02, fluorine 0.70 → 0.06, methyl at the far position 0.70 → 0.22 — and all are free on
QED and SA, so the visible terms give no warning. Not deletion (table in §0). Not fusion: adding
a benzo ring to its far edge, leaving donor, both nitrogens and pocket intact, gives 0.85 → 0.10.

**The pocket rejects *replacement*, not *substitution*.** Replacing the ring fails (thiophene
0.30, fused bicycle 0.46, pyridyl 0.04); putting a group *on* an intact pocket costs a flat
0.03–0.04. The same substituent moved donor→pocket: F 0.72/0.69, CN 0.77/0.73, a recognised
heterocycle 0.82/0.78. It is a second tunable surface.

**Off-*para* on the donor ring.** Measured at all three positions for three substituents, with
formula, weight, LogP and QED identical within each series:

| substituent | para | meta | ortho |
|---|---|---|---|
| fluoro | 0.72 | 0.52 | 0.47 |
| methyl | 0.71 | 0.55 | 0.48 |
| methoxy | 0.70 | 0.53 | — |

~0.20 to leave para — larger than the whole spread across seventeen para substituents, and a
*recognised* fragment is taxed harder still (0.24 meta, 0.32 ortho). Recognition does not buy
exemption from geometry.

## 4. The donor ring's para position

Seventeen substituents span only 0.70–0.77 (H, halogens, methyl, methoxy, cyano, amino, hydroxy,
dimethylamino, acetamido, ester, sulfonyl, thioether, ethynyl, acetyl). No electronic trend —
cyano and amino both 0.75+, methoxy and hydrogen both 0.70. Weak polarisability effect within
the halogens (F 0.72 < Cl 0.74 < Br 0.75).

**Interpolating drug-likeness curves saturates here at ~0.77. Picking a fragment because a model
trained on this target class would recognise it does not** — the first such fragment tried scored
0.82, another 0.78. When this position is worth budget, try whole fragments from real inhibitors
of the target class, one episode each.

Two things about such fragments: they are recognised **individually, not as classes** (swapping
one ring heteroatom, changing ring size by one, or moving the group one methylene off the aryl
each dropped the 0.82 case back to 0.69–0.73 — six probing episodes recovered nothing), and they
**do not escape the positional rule**.

**Nothing is additive.** Two copies of a good fragment, one per ring, score *below* either alone
(0.79 vs 0.85 and 0.80); two different ones likewise (0.75 vs 0.85 and 0.77). The model scores
one arrangement, not a feature count.

**Decoration is cheapest early.** A fluorine on the pocket costs 0.03 with a bare donor ring,
0.08 once the donor carries a working fragment, 0.18 added to that donor ring alongside it.

## 5. QED and SA mechanics

**Lipophilicity governs QED, not mass.** Candidates sit at LogP 3.3–4.6, above where QED is
happiest, so that term usually binds. Among small substituents QED tracks LogP monotonically —
cyano (3.76) 0.788, methoxy (3.90) 0.785, fluoro (4.03) 0.777, chloro (4.54) 0.766 — and weight
only *looks* causal because small groups are also greasy. Break the correlation and mass is
nearly free: heavy polar fragments give the highest QED values seen here (325 Da 0.797, 332 Da
0.789, 345 Da 0.782, 360 Da 0.773). **Do not reject a fragment for its weight if it brings
heteroatoms.** When a weight argument and a lipophilicity argument disagree, lipophilicity wins.

- A second H-bond donor costs ~0.07 on a light molecule and little near 300 Da (para-amino at
  262 Da and para-hydroxy at 263 Da both 0.709; acetamido at 304 Da 0.768).
- A fourth aromatic ring costs 0.15–0.2. Watch for it hiding inside a one-atom swap: replacing a
  fused lactam's CH₂ with NH also aromatises the ring.
- Structural alerts are a real term and **invisible in the properties block**. Amidine predicted
  0.70, scored 0.51; hydrazide predicted 0.68, scored 0.39. Do not predict QED for a group you
  have not seen priced here; prefer homologues of measured groups, and `undo_last_change` costs
  nothing if a trial is bad.
- Stereocentres cost ~0.3 of raw SA each. Check any named fragment for them before proposing it.
- **Assume your SA estimate is 0.2 too low for anything non-aromatic.** This model prices the
  specific ring-plus-substitution pattern, not the ring. Non-aromatic pockets have a floor near
  SA 2.3 against ~1.8 aromatic — a standing cost worth paying for the QED gain, but budget it.
- Adding a plain phenyl to a heteroaromatic consistently *lowers* SA.

## 6. Scaffold economics

The same aryl addition costs ~0 to ~0.45 depending on the host. Under ~300 Da with fewer than
three aromatic rings it is cheap; at ~380 Da or with three aromatic rings already it costs
0.20–0.25, nearly all of it the fourth-ring penalty. **Subtract first, then build** — removing a
ring that contributes nothing converts the next addition from expensive to free (reproduced on
four scaffolds).

**Inert bulk is not inert.** It suppresses activity, not just drug-likeness: stripping a benzoyl
that looked like ballast took activity 0.17 → 0.70. When the arrangement is right and the score
is near zero, look for what to remove before adding.

## 7. Working method

- Several tool calls may be issued in one reply; they execute in order. Indices are **not stable
  across calls that add atoms** — the molecule is re-canonicalised. Batching is safe only when
  earlier calls delete atoms sitting after every index the later ones name.
- Say FINAL ANSWER in the same reply as your last tool call; a separate turn for it is wasted.
- Read each tool's result message — it states what was removed or attached, which is how you
  catch an edit that landed somewhere unintended.
- **Predict each edit's effect before making it, then compare.** Being wrong is cheap and
  informative; being wrong without noticing is neither.
- Prefer edits that differ from an existing molecule in exactly one respect. A matched pair
  answers a question; two molecules differing in three ways answer none. The best comparisons
  here had QED agreeing to four decimals, leaving the activity term as the only difference.
- **Check a ranking has a control before reading it as a trend.** The costliest mistake made
  while producing this brief: three substituted donor rings scored 0.47/0.53/0.58, read as
  "bigger and more polar is better", four episodes spent on it. The unsubstituted reference had
  never been built — it scores **0.70**, above all three.
- A molecule that cannot win is still worth submitting if it resolves something. Most numbers
  here come from deliberate negatives. The population discards them; the knowledge is what keeps.
