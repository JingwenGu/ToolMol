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
    """Attach frag_smiles to atom idx of mol via a bond of the given symbol ('single'/'double'/
    'triple'). The fragment may mark its attachment point with exactly one [*]/[1*]-style dummy
    atom; if no dummy atom is present, the fragment's first atom (SMILES atom order) is used as
    the attachment point instead - this only changes behavior for calls that previously errored
    with "found 0" (a fragment with a dummy atom resolves exactly as before). Returns
    (new_mol, used_fallback, None) on success - used_fallback is True iff the first-atom rule
    was applied - or (None, False, error_message) on failure."""
    try:
        bond_type = bond_symbol_to_type(bond_symbol)
    except ValueError as e:
        return None, False, str(e)

    if not (0 <= idx < mol.GetNumAtoms()):
        return None, False, f"atom index {idx} out of range (molecule has {mol.GetNumAtoms()} atoms)"

    frag = Chem.MolFromSmiles(frag_smiles, sanitize=False)
    if frag is None:
        return None, False, f"invalid substructure SMILES '{frag_smiles}'"
    if frag.GetNumAtoms() == 0:
        return None, False, "substructure SMILES has no atoms"

    dummy_indices = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummy_indices) > 1:
        return None, False, (f"substructure must specify at most one attachment point "
                              f"(e.g. [*] or [1*]), found {len(dummy_indices)}")

    offset = mol.GetNumAtoms()
    combo = Chem.RWMol(Chem.CombineMols(mol, frag))
    used_fallback = False

    if dummy_indices:
        dummy_idx = dummy_indices[0]
        dummy_atom = frag.GetAtomWithIdx(dummy_idx)
        if dummy_atom.GetDegree() != 1:
            return None, False, "substructure's attachment-point atom must have exactly one bond"
        frag_attach_idx = dummy_atom.GetNeighbors()[0].GetIdx()
        combo.AddBond(idx, offset + frag_attach_idx, bond_type)
        combo.RemoveAtom(offset + dummy_idx)
    else:
        # No explicit attachment point marked - fall back to the fragment's first atom
        # (SMILES atom order), matching how substituent fragments are conventionally written
        # attachment-atom-first (e.g. "C(=O)O" for a carboxyl attaching via the carbonyl
        # carbon). No dummy atom to remove in this case.
        used_fallback = True
        combo.AddBond(idx, offset + 0, bond_type)

    try:
        new_mol = combo.GetMol()
        Chem.SanitizeMol(new_mol)
        return new_mol, used_fallback, None
    except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException, ValueError) as e:
        return None, False, f"invalid modification: {e}"


def _describe_atoms(mol, atom_indices):
    """Human-readable summary of a set of atoms for tool-result messages, e.g.
    "'C(F)(F)F' (4 atoms: C, F, F, F)". The SMILES is generated from the intact original
    mol restricted to just these atoms, so it reflects each atom's actual bonding within
    the fragment - the one bond leaving the fragment (to the rest of the molecule) isn't
    part of it, so this is a display string only, not necessarily something that would
    itself parse back into a valid standalone molecule. Falls back to just the element
    list if fragment-SMILES generation fails for any reason."""
    elements = sorted(mol.GetAtomWithIdx(i).GetSymbol() for i in atom_indices)
    formula = ", ".join(elements)
    try:
        frag_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atom_indices), canonical=True)
        return f"'{frag_smiles}' ({len(atom_indices)} atoms: {formula})"
    except (ValueError, RuntimeError, KeyError):
        return f"{len(atom_indices)} atoms: {formula}"


def cut_at_bond(mol, anchor_idx, branch_idx):
    """Cut the single bond between anchor_idx and branch_idx, keeping the fragment containing
    anchor_idx and discarding the fragment containing branch_idx (and everything only reachable
    through it). Replaces the old SMARTS-based match_and_remove_substructure: rather than asking
    the model to write a pattern describing the substructure to remove (a real source of
    confusion - SMARTS vs SMILES syntax, regex-style escaping, misplaced markers), the model
    just names two atoms it already sees in the atom table, and the exact bond between them is
    cut - unambiguous by construction, no matching/uniqueness step involved.

    Only acyclic (non-ring) bonds can be cut: a non-ring bond is always a graph bridge, so
    cutting it is guaranteed to split the molecule into exactly two pieces; a ring bond's
    removal wouldn't disconnect anything (the ring's other path still connects both sides).

    Returns (keep_mol, stub_idx, bond_type, removed_desc, None) on success - keep_mol is an
    already-sanitized Mol, stub_idx is anchor_idx's own index within keep_mol's (possibly
    renumbered) atoms, and removed_desc describes (see _describe_atoms) exactly what was cut
    away, so the caller can report it back to the model. This is the model's main defense
    against its own atom-misidentification mistakes: an anchor/branch pair that's validly
    bonded still "succeeds" even if the model misjudged which atoms they were, so the
    resulting description is what lets it notice and self-correct (e.g. via undo_last_change).
    Returns (None, None, None, None, error_message) on failure."""
    for label, i in (("anchor_idx", anchor_idx), ("branch_idx", branch_idx)):
        if not (0 <= i < mol.GetNumAtoms()):
            return None, None, None, None, f"{label} {i} out of range (molecule has {mol.GetNumAtoms()} atoms)"
    if anchor_idx == branch_idx:
        return None, None, None, None, "anchor_idx and branch_idx must be different atoms"

    bond = mol.GetBondBetweenAtoms(anchor_idx, branch_idx)
    if bond is None:
        neighbors = ", ".join(f"{n.GetIdx()} ({n.GetSymbol()})"
                               for n in mol.GetAtomWithIdx(anchor_idx).GetNeighbors())
        return None, None, None, None, (f"atoms {anchor_idx} and {branch_idx} are not bonded; "
                                         f"atom {anchor_idx}'s actual neighbors are: {neighbors}")
    if bond.IsInRing():
        return None, None, None, None, (f"the bond between atom {anchor_idx} and atom {branch_idx} is part of "
                                         f"a ring; only acyclic (non-ring) bonds can be cut")

    bond_type = bond.GetBondType()
    fragmented = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=False)
    frag_mapping = []
    try:
        frags = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=frag_mapping)
    except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException, ValueError) as e:
        return None, None, None, None, f"invalid modification: {e}"

    if len(frags) != 2:
        # A non-ring bond is a bridge in any single connected component, so this only fires
        # if mol itself was already disconnected (e.g. a multi-fragment/salt SMILES) before the cut.
        return None, None, None, None, (f"internal error: cutting that bond produced {len(frags)} fragments, "
                                         f"expected 2 (is the molecule already disconnected?)")

    keep_i = next((i for i, idxs in enumerate(frag_mapping) if anchor_idx in idxs), None)
    removed_i = next((i for i, idxs in enumerate(frag_mapping) if branch_idx in idxs), None)
    if keep_i is None or removed_i is None or keep_i == removed_i:
        return None, None, None, None, "internal error: could not track fragments through the cut"

    stub_idx = frag_mapping[keep_i].index(anchor_idx)
    removed_desc = _describe_atoms(mol, frag_mapping[removed_i])

    return frags[keep_i], stub_idx, bond_type, removed_desc, None
