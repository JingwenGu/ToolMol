import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# Morgan radius 2 / 2048 bits, the usual default for Tanimoto similarity between small
# molecules. Built once: constructing a generator per call dominates the cost of the
# similarity matrix itself for populations this size.
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _niche_counts(population_smi, share_radius):
    """Fitness-sharing niche count for each molecule.

    Standard triangular sharing: a molecule's count is the sum over the population of
    sh(d) = 1 - d/share_radius for d < share_radius and 0 beyond, where d is Tanimoto distance.
    A molecule sitting alone scores 1 (itself); one in a cluster of near-identical neighbours
    scores close to the cluster size, and dividing its selection weight by that count is what
    stops the cluster monopolising the mating pool.

    Molecules whose SMILES will not parse get a count of 1, which leaves their weight untouched
    rather than silently dropping them from selection.
    """
    fps = []
    for smi in population_smi:
        mol = Chem.MolFromSmiles(smi)
        fps.append(None if mol is None else _MORGAN.GetFingerprint(mol))

    n = len(fps)
    counts = np.ones(n, dtype=float)
    for i in range(n):
        if fps[i] is None:
            continue
        total = 0.0
        for j in range(n):
            if fps[j] is None:
                continue
            d = 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
            if d < share_radius:
                total += 1.0 - d / share_radius
        counts[i] = max(total, 1e-9)
    return counts


def make_mating_pool_phi(population_mol, population_smi, oracle, k=10.0, offspring_size=35,
                         share_radius=0.0, max_parent_share=1.0):
    """Sample (m0, m1) parent pairs with probability proportional to k^Phi(m), where Phi(m) is
    the sum of each objective rescaled to [0,1] (ToolMol paper, Section 3.1). Self-pairing
    (m0 is m1) is allowed, matching this codebase's existing GPT4.py/reproduce() convention of
    drawing two parents via independent random choices.

    Two optional diversity controls, both off by default so the paper's behaviour is preserved.
    Their measured effects differ a lot, and the numbers below are from replaying iteration 2's
    two observed failure shapes rather than from theory - the first version of this docstring
    claimed more for fitness sharing than the measurements support.

    max_parent_share caps the fraction of the 2 * offspring_size parent slots any one molecule
    may take in a generation. This is the control that works. Iteration 2's worst generation gave
    a single molecule 6 of 16 slots, because it led the front by ~0.7 of Phi and k^Phi turns that
    into a ~5x per-draw advantage. Replaying that shape: worst-case usage falls from a mean of
    6.3 slots (95th percentile 10) to a hard 4.0 at max_parent_share=0.25.

    share_radius enables fitness sharing over Tanimoto distance - each molecule's weight divided
    by its niche count, the standard remedy for a population collapsing onto one neighbourhood.
    It helps much less here than expected, and it is worth being precise about why:

      * Against a single dominant molecule it does nothing at all (6.3 slots before and after),
        because a structurally unique leader has no neighbours to share with. That was iteration
        2's actual worst case, so sharing does not address the failure that motivated it.
      * Against a cluster of near-identical leaders it is weak: at radius 0.4 the cluster's share
        of parent slots falls only from 92% to 86%, and pushing the radius to 0.8 reaches 77%
        against a count-proportional baseline of 75% - i.e. it only stops the cluster dominating
        by erasing the score preference entirely, which defeats the selection.

    Kept at a mild setting because a modest nudge is still worth having, but the cap is doing the
    work. The deeper cause of late-run stagnation is not selection at all: the agent exhausts the
    distinct edits available around a converged architecture. That is addressed on the prompt
    side, by showing it what the population already contains (see run.py's _population_summary).
    """
    phis = np.array([oracle.evaluate(smi) for smi in population_smi])
    weights = np.power(float(k), phis)

    if share_radius and share_radius > 0:
        counts = _niche_counts(population_smi, share_radius)
        weights = weights / counts

    n = len(population_mol)
    slots = 2 * offspring_size
    cap = int(np.ceil(max_parent_share * slots)) if max_parent_share < 1.0 else slots
    used = np.zeros(n, dtype=int)

    def draw():
        live = weights * (used < cap)
        total = live.sum()
        if total <= 0:
            # Every molecule is at its cap (possible when the population is smaller than
            # slots/cap). Fall back to unconstrained sampling rather than failing.
            live, total = weights, weights.sum()
        idx = int(np.random.choice(n, p=live / total))
        used[idx] += 1
        return idx

    pairs = []
    for _ in range(offspring_size):
        i0, i1 = draw(), draw()
        pairs.append((population_mol[i0], population_mol[i1]))
    return pairs
