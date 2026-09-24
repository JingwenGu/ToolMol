# Domain brief

Appended to the agent's system prompt via `--domain_brief`. Everything here was learned by
running this task and watching the oracle respond. It is written as transferable guidance
rather than a list of answers: it names what to look for and what things cost, so a model that
has never seen this pool can use it. It deliberately contains no winning molecules or
scaffolds, which would only overfit to one starting population.

Revised after iteration 2 (seed 7, 50 oracle calls). Several claims in the first version were
wrong and are corrected below; where a rule rests on a controlled comparison, the numbers are
given so a later run can check them rather than trust them.

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

**3. Keep the two rings coplanar.** Substituents that twist the biaryl bond cost activity:
fluorine on the pocket ring scores 0.69 para, 0.63 meta, 0.50 ortho. The same principle
explains why the donor must be an aryl (conjugation) and why the hinge ring must stay bare.

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

## The donor's own ring is tolerant — spend property budget there

Substituents on the aryl attached to the donor NH range only 0.70–0.77 across cyano, iodo,
amino, bromo, chloro, fluoro, methyl, hydrogen, methoxy and trifluoromethyl. There is no usable
electronic trend: cyano (strongly withdrawing) and amino (strongly donating) both score 0.75+,
while trifluoromethyl and methoxy both sit at 0.70. Treat this ring as a free surface for tuning
solubility, mass and lipophilicity without disturbing binding — which is exactly what the hinge
ring is not.

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

- Its molecular-weight term peaks near 300. Trimming mass from a molecule already near 300 gains
  nothing; trimming from 380 helps; trimming below ~260 hurts.
- A fourth aromatic ring is penalised sharply — about 0.2 of QED, regardless of which ring it is.
  A more polar ring does not soften this; the decision is whether to add one at all.
- Structural alerts are a real term: an aromatic nitro group is penalised both as an alert and
  through TPSA, so removing one is often the single largest available gain.
- Stereocentres are the most reliable lever on synthetic accessibility. Removing the group
  carrying one usually improves SA by a large margin and costs nothing, because such groups
  rarely do binding work.

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
