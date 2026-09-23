# Domain brief

Appended to the agent's system prompt via `--domain_brief`. Everything here was learned by
running this task and watching the oracle respond, not taken from the paper. It is written as
transferable guidance rather than a list of answers: it names what to look for and what things
cost, so a model that has never seen this pool can use it. It deliberately contains no winning
molecules or scaffolds, which would only overfit to one starting population.

Revise this file between iterations as runs teach more.

---

## Which objective actually has room

The score you are shown is a sum of objectives each rescaled to [0,1]. Read the per-objective
breakdown before planning an edit. In practice the drug-likeness terms (QED, synthetic
accessibility) start reasonably high and saturate quickly, while the binding/activity term
starts near zero and holds nearly all the available headroom. A molecule whose QED is already
above ~0.9 has at most a few hundredths left there; the activity term may have an order of
magnitude more. Spending every edit polishing what is already good is the most common way to
waste an episode.

## Reaching the activity term (ATP-competitive kinase targets)

Activity is scored by a fingerprint model trained on known actives, so it responds to
recognisable pharmacophores rather than to general "drug-likeness".

- **Hinge motif first.** These inhibitors bind through an aromatic ring nitrogen paired with an
  adjacent NH — a 2-aminopyrimidine is the canonical form. The donor and acceptor must be
  *adjacent*: an NH para to a ring nitrogen does not work, an NH flanked by ring nitrogens does.
  Installing one on a scaffold that has none typically lifts the activity term off zero.
- **Then occupy the pocket.** An aryl substituent on the hinge heterocycle, positioned next to
  a ring nitrogen, is usually worth substantially more than the hinge motif alone. Hinge motif
  and pocket group are largely independent and compound.
- **Small hydrophobic contacts are cheap.** A methyl ortho to a ring nitrogen can pay well for
  almost no drug-likeness cost. Try these before expensive additions.
- **A molecule with zero H-bond donors cannot engage a hinge at all.** Treat that as a
  structural dead end and fix it before anything else.

## Cost is set by the host scaffold, not by the group you add

The same aryl addition can cost anywhere from a few hundredths to ~0.45 of summed score
depending entirely on where it lands. Before adding an aromatic ring, check the host:

- **Room to spare** — MW well under ~300, LogP under ~2, high TPSA: the addition is cheap and
  may even raise QED, because mass and lipophilicity move toward their optima.
- **No room** — MW already near or above ~380, or a scaffold that already carries an
  unsubstituted phenyl: the same addition drives LogP past 4 and collapses QED.

When several parents are available, choosing *which* scaffold to modify matters more than
choosing the modification.

## How QED responds

- Its molecular-weight term peaks near 300. Trimming mass from a molecule already near 300
  gains nothing; trimming from 380 helps; trimming below ~260 hurts.
- A fourth aromatic ring is penalised sharply.
- Structural alerts are a real term: an aromatic nitro group is penalised both as an alert and
  through TPSA, so removing one is often the single largest available gain.
- Subtraction is a strategy, not just a cleanup. Once the pharmacophore is established, removing
  a redundant methoxy, a spare halogen or a floppy linker often improves QED and synthetic
  accessibility together while leaving binding untouched. It only works when the molecule is
  genuinely overweight.

## Working method

- You may issue several tool calls in one reply; they execute in order against the updated
  molecule. `undo_last_change` followed by a corrected call works in a single turn.
- Say FINAL ANSWER in the same reply as your last tool call when you are done — a separate turn
  for it is wasted.
- Read each tool's result message. It states exactly what was removed or attached, which is how
  you catch an edit that landed somewhere other than intended.
- Predict the effect of an edit before making it and compare against what comes back. Being
  wrong is informative and cheap; being wrong without noticing is neither.
