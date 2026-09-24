# Domain brief

Appended to the agent's system prompt via `--domain_brief`. Everything here was learned by
running this task and watching the oracle respond. It is written as transferable guidance
rather than a list of answers: it names what to look for and what things cost, so a model that
has never seen this pool can use it. It deliberately contains no winning molecules or
scaffolds, which would only overfit to one starting population.

Revised after iteration 3 (seed 11). Several claims in earlier versions were wrong and are
corrected below; where a rule rests on a controlled comparison, the numbers are given so a later
run can check them rather than trust them. The three corrections that changed how budget should
be spent, in case you read nothing else:

1. On the donor's aryl ring, **position matters far more than which substituent you pick** —
   ~0.20 for moving off *para*, against a 0.12 spread across seventeen different para groups.
2. The pocket ring rejects being *replaced*, not being *substituted*. Earlier versions treated
   it as frozen; it is a second tunable surface at a flat cost of about 0.03.
3. Picking substituents by interpolating drug-likeness curves saturates around 0.77. Picking a
   fragment because a model trained on this target class would recognise it does not.

---

## Which objective actually has room

The score you are shown is a sum of objectives each rescaled to [0,1]. Read the per-objective
breakdown before planning an edit. In practice the drug-likeness terms (QED, synthetic
accessibility) start reasonably high and saturate quickly, while the binding/activity term
starts near zero and holds nearly all the available headroom. A molecule whose QED is already
above ~0.9 has at most a few hundredths left there; the activity term may have an order of
magnitude more. Spending every edit polishing what is already good is the most common way to
waste an episode.

## The pharmacophore, in order of how much it matters

For ATP-competitive kinase targets, activity is scored by a fingerprint model trained on known
actives, so it responds to recognisable arrangements rather than to general drug-likeness.
Three elements, in descending order of importance:

**1. The donor must be an NH that carries an aryl — not an alkyl or an acyl.**
This is the single most reliable finding, confirmed on five unrelated scaffolds. An NH sitting
adjacent to an aromatic ring nitrogen looks correct and scores ~0.01–0.05 when its other
substituent is an sp3 carbon, an amide or a urea. Replacing that substituent with a phenyl,
holding everything else fixed, is what moves the molecule. Adjacency alone is not enough; an
anilino NH is conjugated and roughly planar, an alkylamino NH is neither.

**2. An aryl at the ring position beside the hinge nitrogen — and it must be a carbocycle.**
Occupying the adjacent hydrophobic pocket is worth roughly +0.4 to +0.6 of activity. What goes
there matters more than anything else in this brief:

| pocket group | activity |
|---|---|
| phenyl | 0.70 |
| naphthyl | 0.65 |
| thiophen-3-yl | 0.60 |
| cyclohexyl / cyclopentyl (saturated) | ~0.31 |
| **pyridin-4-yl** | **0.04** |

A basic ring nitrogen pointing into this pocket is catastrophic — worse than removing the ring
entirely. Sulfur is fine. Saturated rings lose about half the benefit, so the site wants an
aromatic carbocycle specifically. **Do not reach for a pyridyl here to control lipophilicity;
that instinct is correct for QED and destroys the binding.**

**3. Keep the two rings coplanar and directly bonded.** Substituents that twist the biaryl bond
cost activity: fluorine on the pocket ring scores 0.69 para, 0.63 meta, 0.50 ortho. Inserting a
single sp3 methylene between the hinge and the pocket ring — leaving the benzene itself
untouched and only breaking conjugation — costs far more: **0.74 → 0.25**. The same principle
explains why the donor must be an aryl (conjugation) and why the hinge ring must stay bare.

**The pocket rejects *replacement*, not *substitution*.** These are easy to conflate and the
distinction is worth about a third of the molecule's tunable surface. Replacing the ring fails
badly — thiophene 0.30, a fused isoquinoline 0.46, saturated rings ~0.31, pyridyl 0.04. But
putting a substituent *on* an intact phenyl pocket is nearly free. The same group moved from the
donor ring to the pocket ring, with formula and QED identical, costs a flat −0.03 to −0.04:

| same substituent, para | on donor ring | on pocket ring |
|---|---|---|
| fluoro | 0.72 | 0.69 |
| cyano | 0.77 | 0.73 |
| a recognised saturated heterocycle | 0.82 | 0.78 |

So the pocket is a *second* tunable surface, not a frozen one. Earlier versions of this brief
inferred it was untouchable from three failed ring replacements; that inference was wrong.

## Do not substitute the hinge ring

The ring carrying the hinge nitrogen and the pocket aryl should be left unsubstituted. This is
not a preference — it is the largest single loss observed:

- methyl at the carbon adjacent to the ring nitrogen: **0.70 → 0.02**
- fluorine at that same position: 0.70 → 0.06
- methyl at the far ring position (not adjacent to a nitrogen): 0.70 → 0.22

Even the smallest substituent at the adjacent position is fatal, so this is positional and
electronic rather than steric. Both of those edits are essentially free on QED and SA — **cheap
to add is not the same as safe to add**, and the visible terms give no warning at all.

*The previous version of this brief recommended a methyl beside the ring nitrogen as the
best-value edit available. That advice was exactly backwards.*

### Nor remove any of its atoms

Every atom of the hinge is load-bearing, including one that looks decorative:

| single-atom change | activity | reference |
|---|---|---|
| methylate the donor N–H (nothing else changes) | 0.06 | 0.72 |
| acylate the donor N — amide instead of arylamine | 0.02 | 0.70 |
| pyrimidine → pyridine, dropping the **distal** ring N | 0.12 | 0.70 |

The dropped nitrogen in the third row touches neither the donor nor the pocket and might
reasonably be read as spare mass; removing it still costs 0.58. The N–H is not merely a
convenient linker either — methylating it is nearly as destructive as acylating it, so the
hydrogen itself is what is recognised, not the two-ring arrangement around it.

## The donor's own ring: tolerant at *para*, closed everywhere else

Substituents **para** to the donor NH range only 0.70–0.77 across seventeen groups — hydrogen,
fluoro, chloro, bromo, iodo, methyl, methoxy, cyano, amino, hydroxy, dimethylamino, acetamido,
methyl ester, methylsulfonyl, methylthio, ethynyl and acetyl. There is no usable electronic
trend: cyano (strongly withdrawing) and amino (strongly donating) both score 0.75+, while
methoxy and plain hydrogen both sit at 0.70. The one weak regularity is polarisability within a
homologous series — F 0.72 < Cl 0.74 < Br 0.75.

**Position, however, costs more than any substituent choice.** Measured for three substituents
at all three positions, with formula, weight, LogP and QED identical inside each series:

| substituent | para | meta | ortho |
|---|---|---|---|
| fluoro | 0.72 | 0.52 | 0.47 |
| methyl | 0.71 | 0.55 | 0.48 |
| methoxy | 0.70 | 0.53 | — |
| best fragment found (see next section) | **0.82** | 0.58 | — |

Moving off para costs ~0.20 — larger than the entire 0.12 spread across all seventeen para
substituents, and it costs the same whether the group is a fluorine or the best fragment
available. So: treat the para position as a free surface for tuning solubility, mass and
lipophilicity; treat ortho and meta as closed. A 2,5-dimethylphenyl donor ring scores 0.47, and
its ortho methyl alone accounts for that.

## How to search this position: provenance beats property interpolation

The seventeen para substituents above were each chosen by interpolating QED against molecular
weight, LogP and donor count. That approach plateaus: every one landed in 0.70–0.77, and three
successive property-based explanations of the spread (structural alerts, then electronics, then
hydrogen-bond donation) each fitted the molecules already measured and were falsified by the
next one.

The activity model is a fingerprint model over known actives, so it rewards **recognisable
substructure**, not interpolatable properties. Choosing candidates because a model trained on
this target class would have seen them constantly — rather than because their properties fit a
curve — is what broke the ceiling. In the run that established this, the first such fragment
tried scored **0.82**, five points clear of the best of seventeen property-chosen substituents,
and a second scored 0.78.

*(The specific fragment is deliberately not named here. This brief is meant to transfer method,
not answers; naming it would let a later run reproduce the result without searching and would
make the comparison between runs meaningless.)*

Two things about such fragments are worth knowing in advance, because they change how you
follow one up:

- **They are recognised as specific fragments, not as classes, so nothing generalises from
  them.** For the 0.82 case, swapping one ring heteroatom for S, SO₂, CH₂ or NMe, changing the
  ring size by one atom in either direction, or moving the whole group one methylene off the
  aryl each returned it to the 0.69–0.73 pack. Only decoration of its periphery was partly
  tolerated. Once you find one, install it and stop — six episodes spent probing around it
  recovered nothing and the probes are individually uninformative about any other fragment.
- **They do not escape the positional rule.** Moving the 0.82 fragment from *para* to *meta*
  cost 0.24, the same penalty an ordinary fluorine pays. Recognition does not override geometry.

So when this position is worth budget, spend it trying whole fragments drawn from real
inhibitors of the target class — and spend only one episode on each.

## Cost is set by the host scaffold, not by the group you add

The same aryl addition can cost anywhere from ~0 to ~0.45 of summed score depending entirely on
where it lands. Before adding an aromatic ring, check the host:

- **Room to spare** — MW well under ~300, fewer than three aromatic rings: the addition is
  cheap and may raise QED.
- **No room** — MW near or above ~380, or already carrying three aromatic rings: the same
  addition costs 0.20–0.25, almost all of it the fourth-aromatic-ring penalty.

**Subtract first, then build.** Removing a substituent that contributes nothing — and in
particular removing an aromatic ring — converts the next addition from expensive to free. The
identical pocket-aryl addition cost 0.25 made directly and ~0 when made after trimming, on four
different scaffolds. The constraint is the fourth ring, not mass in general.

## Inert bulk is not inert

A substituent that forms no part of the pharmacophore does not merely cost drug-likeness — it
can suppress activity outright. Stripping a benzoyl that looked like neutral ballast took
activity from 0.17 to 0.70 on one scaffold. When a molecule has the right arrangement and still
scores near zero, look for what to remove before adding anything.

## How QED responds

**Lipophilicity governs it, not mass.** This is the correction most worth carrying: on a typical
three-ring molecule of this class every candidate sits at LogP 3.3–4.6, well above where QED's
lipophilicity term is happiest, so that term is almost always the binding constraint. Among small
lipophilic substituents QED tracks LogP monotonically and weight looks like it matters only
because small groups are also greasy — cyano (LogP 3.76) 0.788, methoxy (3.90) 0.785, fluoro
(4.03) 0.777, chloro (4.54) 0.766.

Break that correlation and mass turns out to be nearly free. The three highest QED values ever
measured on this scaffold are all *heavy* polar fragments: methylsulfonyl at 325 Da 0.797,
a saturated N-heterocycle at 332 Da 0.789, a larger one at 345 Da 0.782, and an aryl amide
carrying one still holds 0.773 at 360 Da. **Do not reject a fragment for its weight if it brings heteroatoms.** A rule of
"stay between 265 and 290 Da", derived from small substituents, is a LogP rule in disguise and
will cost you the best fragments available.

- A second hydrogen-bond donor costs about 0.07 — but only on a light molecule. Para-amino at
  262 Da and para-hydroxy at 263 Da both give exactly 0.709, while acetamido at 304 Da reaches
  0.768 and a tertiary alcohol at 305 Da reaches 0.761 with the same donor count.
- A fourth aromatic ring is penalised sharply — about 0.15–0.2 of QED, regardless of which ring.
  Watch for this when swapping an atom *inside* a ring: replacing a fused lactam's CH₂ with NH also
  aromatises the ring, so what looks like a one-atom change is two.
- Structural alerts are a real term. An aromatic nitro group is penalised both as an alert and
  through TPSA; an aryl methyl ketone costs ~0.05 on its own (acetyl at 289 Da gives 0.735 where
  the LogP line predicts 0.78).
- Stereocentres are the most reliable lever on synthetic accessibility — about 0.3 of raw SA
  each. a 2,6-dimethylated saturated ring (two stereocentres) scores SA 2.82; its 2,2-dimethyl isomer, same
  formula and same QED, scores 2.23. Check any fragment you name for them before proposing it.
- Adding a plain phenyl to a heteroaromatic *lowers* SA, consistently. A biaryl between two
  commodity rings reads as easy to make regardless of what it is attached to.

## Working method

- You may issue several tool calls in one reply; they execute in order against the updated
  molecule. `undo_last_change` followed by a corrected call works in a single turn, and a known
  two-step sequence (trim, then build) can be issued as one turn once you know the intermediate.
- Say FINAL ANSWER in the same reply as your last tool call when you are done — a separate turn
  for it is wasted.
- Read each tool's result message. It states exactly what was removed or attached, which is how
  you catch an edit that landed somewhere other than intended.
- Predict the effect of an edit before making it and compare against what comes back. Being
  wrong is informative and cheap; being wrong without noticing is neither. Two systematic biases
  worth correcting for: swapping a small aliphatic group for a nitrogen heteroaryl costs far
  less QED than it appears it should, and a fourth aromatic ring costs far more.
- When you have budget, prefer edits that differ from an existing molecule in exactly one
  respect. A matched pair answers a question; two molecules differing in three ways answer none.
- **Before reading a ranking as a trend, check that it has a control.** This is the single
  costliest mistake made while producing this brief. Three substituted donor rings scored 0.47,
  0.53 and 0.58; that ordering was read as "larger and more polar is better" and four episodes
  were spent building fused bicycles on the strength of it. The unsubstituted reference had never
  been made. It scores **0.70** — above all three. The "trend" was three substituted rings all
  doing worse than nothing, ranked among themselves. Building the missing baseline cost one
  molecule and inverted the conclusion.
- A molecule that cannot win is still worth submitting if it resolves something. Most of the
  numbers in this brief come from deliberate negatives: an ortho isomer, a methylated N–H, a
  ring with one atom removed. The population will discard them; the knowledge is what you keep.
