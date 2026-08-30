from rdkit import Chem

BOND_TYPE_TO_SYMBOL = {
    Chem.BondType.SINGLE: 'single',
    Chem.BondType.DOUBLE: 'double',
    Chem.BondType.TRIPLE: 'triple',
}
BOND_SYMBOL_TO_TYPE = {v: k for k, v in BOND_TYPE_TO_SYMBOL.items()}


def bond_symbol_to_type(symbol):
    bt = BOND_SYMBOL_TO_TYPE.get(symbol)
    if bt is None:
        raise ValueError(f"unknown bond type '{symbol}', expected one of {list(BOND_SYMBOL_TO_TYPE)}")
    return bt


def bond_type_to_symbol(bond_type):
    return BOND_TYPE_TO_SYMBOL.get(bond_type, 'single')


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_mol.AddAtom(copy_atom(atom))
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        new_mol.AddBond(a1, a2, bond.GetBondType())
    return new_mol


def attach_fragment(mol, idx, frag_smiles, bond_symbol):
    """Attach frag_smiles (must contain exactly one [*]/[1*]-style dummy atom marking its
    attachment point) to atom idx of mol via a bond of the given symbol ('single'/'double'/
    'triple'). Returns (new_mol, None) on success or (None, error_message) on failure."""
    try:
        bond_type = bond_symbol_to_type(bond_symbol)
    except ValueError as e:
        return None, str(e)

    if not (0 <= idx < mol.GetNumAtoms()):
        return None, f"atom index {idx} out of range (molecule has {mol.GetNumAtoms()} atoms)"

    frag = Chem.MolFromSmiles(frag_smiles, sanitize=False)
    if frag is None:
        return None, f"invalid substructure SMILES '{frag_smiles}'"

    dummy_indices = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy_indices) != 1:
        return None, (f"substructure must specify exactly one attachment point "
                       f"(e.g. [*] or [1*]), found {len(dummy_indices)}")
    dummy_idx = dummy_indices[0]
    dummy_atom = frag.GetAtomWithIdx(dummy_idx)
    if dummy_atom.GetDegree() != 1:
        return None, "substructure's attachment-point atom must have exactly one bond"
    frag_neighbor_idx = dummy_atom.GetNeighbors()[0].GetIdx()

    offset = mol.GetNumAtoms()
    combo = Chem.RWMol(Chem.CombineMols(mol, frag))
    combo.AddBond(idx, offset + frag_neighbor_idx, bond_type)
    combo.RemoveAtom(offset + dummy_idx)

    try:
        new_mol = combo.GetMol()
        Chem.SanitizeMol(new_mol)
        return new_mol, None
    except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException, ValueError) as e:
        return None, f"invalid modification: {e}"


def match_and_remove_substructure(mol, idx, smarts):
    """Remove the substructure matched by smarts that contains atom idx (used as an anchor
    to disambiguate multiple matches). Only supports terminal substructures - i.e. removal
    must leave exactly one bond connecting the removed atoms to the rest of the molecule.
    Returns (hole_mol, stub_atom_idx, bond_type, None) on success, marking the surviving atom
    that used to be bonded to the removed substructure so a replacement can be attached there,
    or (None, None, None, error_message) on failure."""
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return None, None, None, f"invalid SMARTS pattern '{smarts}'"

    matches = [m for m in mol.GetSubstructMatches(patt) if idx in m]
    if len(matches) == 0:
        return None, None, None, f"no substructure match for '{smarts}' contains atom index {idx}"
    if len(matches) > 1:
        return None, None, None, (f"ambiguous: {len(matches)} substructure matches for '{smarts}' "
                                   f"contain atom index {idx}")
    match_set = set(matches[0])

    external_bonds = []
    for atom_idx in match_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in match_set:
                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                external_bonds.append((atom_idx, neighbor.GetIdx(), bond.GetBondType()))
    if len(external_bonds) != 1:
        return None, None, None, (f"substructure removal must leave exactly one attachment point "
                                   f"(found {len(external_bonds)}); only terminal substructures are supported")
    _, stub_neighbor_idx, bond_type = external_bonds[0]

    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(stub_neighbor_idx).SetAtomMapNum(99)
    for atom_idx in sorted(match_set, reverse=True):
        rw.RemoveAtom(atom_idx)

    stub_idx = None
    for atom in rw.GetAtoms():
        if atom.GetAtomMapNum() == 99:
            stub_idx = atom.GetIdx()
            atom.SetAtomMapNum(0)
            break
    if stub_idx is None:
        return None, None, None, "internal error: could not track attachment point through removal"

    return rw, stub_idx, bond_type, None
