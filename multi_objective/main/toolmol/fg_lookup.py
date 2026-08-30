"""Plain-English functional group / ring name -> SMILES-with-attachment-point.

The paper (ToolMol) says it "builds a table mapping common functional groups and rings
to their SMILES encodings" for the add_functional_group tool, but never publishes the
table's actual contents - this is our own reasonable approximation, not a reproduction
of anything in the paper. Each entry uses the same "[1*]" dummy-atom attachment-point
convention already used by crossover.py's fragment recombination.
"""

FUNCTIONAL_GROUPS = {
    'methyl': '[1*]C',
    'ethyl': '[1*]CC',
    'propyl': '[1*]CCC',
    'isopropyl': '[1*]C(C)C',
    'butyl': '[1*]CCCC',
    'tert-butyl': '[1*]C(C)(C)C',
    'phenyl': '[1*]c1ccccc1',
    'benzyl': '[1*]Cc1ccccc1',
    'hydroxyl': '[1*]O',
    'amino': '[1*]N',
    'methylamino': '[1*]NC',
    'dimethylamino': '[1*]N(C)C',
    'carboxyl': '[1*]C(=O)O',
    'carboxamide': '[1*]C(=O)N',
    'amide': '[1*]C(=O)N',
    'methoxy': '[1*]OC',
    'ethoxy': '[1*]OCC',
    'cyano': '[1*]C#N',
    'nitrile': '[1*]C#N',
    'nitro': '[1*][N+](=O)[O-]',
    'fluoro': '[1*]F',
    'chloro': '[1*]Cl',
    'bromo': '[1*]Br',
    'iodo': '[1*]I',
    'trifluoromethyl': '[1*]C(F)(F)F',
    'acetyl': '[1*]C(C)=O',
    'sulfonyl': '[1*]S(C)(=O)=O',
    'methylsulfonyl': '[1*]S(C)(=O)=O',
    'thiol': '[1*]S',
    'cyclopropyl': '[1*]C1CC1',
    'cyclobutyl': '[1*]C1CCC1',
    'cyclopentyl': '[1*]C1CCCC1',
    'cyclohexyl': '[1*]C1CCCCC1',
    'pyridyl': '[1*]c1ccncc1',
    'furyl': '[1*]c1ccco1',
    'thienyl': '[1*]c1ccsc1',
    'imidazolyl': '[1*]c1cnc[nH]1',
    'morpholino': '[1*]N1CCOCC1',
    'piperidinyl': '[1*]N1CCCCC1',
    'piperazinyl': '[1*]N1CCNCC1',
    'pyrrolidinyl': '[1*]N1CCCC1',
}


def get_functional_group(name):
    """Look up a plain-English functional group / ring name (case/whitespace-insensitive).
    Returns the SMILES-with-[1*]-attachment-point, or None if unrecognized."""
    return FUNCTIONAL_GROUPS.get(name.strip().lower())
