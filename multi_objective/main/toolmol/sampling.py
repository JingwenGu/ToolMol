import numpy as np


def make_mating_pool_phi(population_mol, population_smi, oracle, k=10.0, offspring_size=35):
    """Sample (m0, m1) parent pairs with probability proportional to k^Phi(m), where Phi(m) is
    the sum of each objective rescaled to [0,1] (ToolMol paper, Section 3.1). Self-pairing
    (m0 is m1) is allowed, matching this codebase's existing GPT4.py/reproduce() convention of
    drawing two parents via independent random choices.
    """
    phis = np.array([oracle.evaluate(smi) for smi in population_smi])
    weights = np.power(float(k), phis)
    probs = weights / weights.sum()

    pairs = []
    for _ in range(offspring_size):
        i0, i1 = np.random.choice(len(population_mol), size=2, replace=True, p=probs)
        pairs.append((population_mol[i0], population_mol[i1]))
    return pairs
