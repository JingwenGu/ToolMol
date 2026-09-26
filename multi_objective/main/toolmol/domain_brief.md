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
entirely. Sulfur is fine. **Do not reach for a pyridyl here to control lipophilicity; that
instinct is correct for QED and destroys the binding.**

*Correction, and an important one.* Earlier versions concluded from the saturated-ring row that
the site "wants an aromatic carbocycle specifically". That is wrong, and the error was reading a
two-point comparison — benzene against cyclohexyl — as being about aromaticity when it was about
conjugation. A five- or six-membered carbocycle carrying a C=C conjugated to the hinge scores
like a benzene (~0.87 against ~0.86 in the context where both were measured), while the fully
saturated version of that same ring collapses to ~0.59. What the pocket requires is a flat,
conjugated ring attached directly — not an aromatic one.

**It must be a ring, though — conjugation alone is not enough.** This is worth stating sharply
because the obvious next inference from the paragraph above is wrong, and a whole generation was
spent finding that out. If conjugation is the requirement, an open chain bearing the same double
bond ought to do, and it does not:

| pocket, everything else held constant | activity |
|---|---|
| five-membered ring with a C=N | 0.88 |
| five-membered ring with a C=C | 0.87 |
| nitrile | 0.66 |
| vinyl | 0.64 |
| isopropenyl | 0.62 |
| but-1-en-2-yl | 0.57 |

Every ring lands at 0.85–0.88; every chain loses about 0.22. Conjugation is necessary and
ring-shaped bulk is the other half. Note that the chains have the *best* drug-likeness in the
whole series — the isopropenyl gives the highest QED-plus-SA of any molecule ever built here —
so the readable terms actively point the wrong way at this position. Do not follow them.

**Inside the ring, nothing else matters.** Five compositions (all-carbon, imine, ether away from
the double bond, enamine on the double bond, imine-plus-ether) and two ring sizes all score
0.85–0.88. Heteroatoms move LogP and therefore QED, which is worth a few hundredths, but they do
not move binding. Pick one for its polarity and stop; this position will not repay further
sampling.

**The 1,3 relationship between pocket and donor is required.** Moving the pocket to the 1,4
position on the hinge — same atoms, QED identical to four decimal places — costs **0.88 → 0.29**.
On a two-nitrogen hinge this cannot even be tested, because the alternative positions sit next
to a ring nitrogen; it only becomes visible with a one-nitrogen hinge, and it is easy to carry
the 1,3 arrangement forward for many runs without ever noticing it was never chosen.

That distinction is worth real score rather than being a technicality. A non-aromatic pocket
drops the molecule from three aromatic rings to two, and QED's aromatic-ring term is steep
enough that this is worth more than 0.1 of QED — larger than the entire spread across every
donor-ring substituent ever measured here. It is the single biggest QED gain available on this
scaffold, and it was invisible for four runs because this section told the agent the pocket had
to be aromatic.

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

## Read this before trusting any rule below: they are local, not universal

Every structural rule in this brief was measured the same way — take one good molecule, change
one thing, record what it costs. That is the right way to get an attributable number, and it is
why the numbers here are trustworthy *as gradients at the point they were measured*. It is not
evidence that the rule holds everywhere, and on this task at least one of them inverts.

The clearest case. "Both hinge ring nitrogens are load-bearing" rests on dropping the distal one
and watching activity collapse. It does, twice:

| group on the donor ring | two-nitrogen hinge | one-nitrogen hinge |
|---|---|---|
| hydrogen | 0.70 | **0.12** |
| a primary carboxamide | 0.85 | **0.28** |
| a saturated N-linked heterocycle | 0.82 | **0.86** |

In the third context the rule reverses. The same deletion that costs 0.58 twice *gains* 0.04
once, and nothing about the first two measurements predicts it. The rule is real, conditional,
and was written down as absolute.

The practical consequence is specific and worth more than any single number here. **When two
rules each forbid a change, that is not evidence the two changes are forbidden together.** Each
was measured with the other feature held at its original value, so the joint case has never been
observed. Regions that look doubly closed are exactly the regions nobody has tested, and on this
scaffold that is where a better optimum sits: the pocket's requirement turns out to be
conjugation rather than aromaticity (a non-aromatic ring bearing a C=C scores like a benzene,
while the fully saturated version collapses), and dropping to two aromatic rings is worth more
than 0.1 of QED — but only in combination with a hinge and donor that the single-variable rules
say should not work.

So: spend some budget deliberately violating two rules at once, especially when each rule's cost
was measured on a molecule unlike the one you are building. Treat everything below as "this is
what it cost there", not "this is what it costs".

**How much of the gain needs the combination.** Once the better region was found, the obvious
follow-up was to ask whether its three changes — conjugated non-aromatic pocket, one-nitrogen
hinge, saturated N-linked donor — are separable. They are partly. Making only the pocket change
on the older molecule, keeping its original hinge and donor, gives **0.58** against 0.85 for the
original and 0.87 in the new region. So that one change recovers roughly half its value alone
and needs the other two for the rest.

That number took two attempts to get right, and the first attempt is the more instructive. I
made the pocket change alone using an *open chain*, got 0.32, and concluded the three changes
were an inseparable package. The chain penalty above is 0.22 on its own, so that experiment was
measuring two things and I attributed all of it to one. **When an edit changes a feature that
has its own known penalty, subtract that penalty before interpreting the result** — or better,
make the edit in the form that carries no second penalty.

**On finding regions at all.** Four runs and roughly two hundred designed molecules never left
the first scaffold, because every single-variable step out of it looks fatal and the brief's
rules said so. What found the better region was not an episode: it was a few minutes of cheap
undirected search (`main/toolmol/ceiling_probe.py`) run between iterations, which located it in
4,000 evaluations. Episodes are expensive and are good at *controlled comparison* — holding
everything constant and moving one thing, which is what produced every reliable number in this
brief. They are poor at search. Use the cheap search to find where to look, then use episodes to
understand what you found.

## Four rules that save whole episodes

These come from dissecting one strong fragment with twelve single-variable edits. Each one rules
out a class of edit in advance, which is worth more than any individual substituent result.

**Nothing is additive.** Two copies of a recognised fragment, one on each ring, score *below*
either copy alone (0.79 against 0.85 and 0.80). Two *different* recognised fragments, one per
ring, also score below both (0.75 against 0.85 and 0.77). The activity model scores one
arrangement; it does not count features. So do not build combinations hoping to stack gains —
find the single best substituent and stop.

**Adding anything to a molecule that already works costs more than the same group added to a
bare one.** A fluorine on the pocket ring costs 0.03 when the donor ring is empty, but 0.08 once
the donor ring carries a working fragment, and 0.18 when added to that donor ring alongside it.
Budget for decoration is cheapest early and most expensive exactly when you most want it.

**Recognition does not buy exemption from geometry — it is taxed harder.** Inert substituents
lose about 0.20 moving off *para*; a recognised fragment lost 0.24 at meta and 0.32 at ortho.
Never place a good fragment anywhere but para because the properties would prefer it elsewhere.

**The hinge tolerates nothing — not substitution, not deletion, not fusion.** Beyond the
substitution and deletion results above, fusing a benzo ring onto the far edge of the hinge
while leaving the donor N–H, both ring nitrogens and the pocket untouched collapsed activity
from 0.85 to **0.10**. Fused hinge cores are standard in real kinase chemistry and this model
does not recognise them. Leave the hinge exactly as it is.

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
- **When a weight argument and a lipophilicity argument disagree, lipophilicity wins.** This
  cost me four predictions in one run. Adding a methylene to reach QED's ~300 Da optimum looks
  free and is not: on a molecule at LogP 3.7 it added 0.4 of LogP and cost 0.03 of QED, and the
  five daltons returned nothing. Enlarging a donor ring by one atom did the same, 0.9 of LogP
  for 0.03 of QED. Before adding any carbon for weight, price it as lipophilicity first.
- **Assume your SA estimate is 0.2 too low for anything non-aromatic.** Six pocket rings in one
  run, every prediction optimistic by 0.2 to 0.3. This model prices the specific ring-plus-
  substitution pattern, not the ring: a trisubstituted dihydrofuran is not "a common fragment"
  to it even though dihydrofuran is. Non-aromatic pockets appear to have a floor around SA 2.3
  regardless of what you do, against ~1.8 for an aromatic one — that gap is the standing cost of
  the aromatic-ring correction and it is worth paying, but budget for it rather than hoping.
- **Do not predict QED for a functional group you have not already seen priced on this
  scaffold.** The structural-alert term is invisible in the properties block, and it dominates
  when it fires. An amidine was predicted at 0.70 and scored 0.51; a hydrazide at 0.68 and
  scored 0.39 — both on standard alert lists, both at weights and donor counts where comparable
  groups score 0.77. Prefer homologues of groups already measured when the point of the episode
  is a binding comparison, and if you must try something exotic, check the result before
  committing the episode: `undo_last_change` costs nothing and returns the modification.
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
