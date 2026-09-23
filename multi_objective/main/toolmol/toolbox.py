"""The 7 LLM-callable RDKit-backed molecule-editing tools (Appendix B.1 of the paper).

Every tool function's first argument(s) are molecule state supplied by the agent (the
current working molecule, or - for crossover_molecules - both original parent molecules);
these are NOT part of the tool's exposed JSON schema, since the LLM never sees or specifies
raw SMILES for the molecule being edited. Only the remaining arguments (idx, element,
bond, group, substructure, ...) are LLM-facing.

Every tool returns a ToolResult(success, mol, message): mol is a valid RDKit Mol on
success, or None with an explanatory message on failure (e.g. bad valence, ambiguous
substructure match, non-2-fragment cut index).
"""

import random
from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

from main.toolmol import mol_ops
from main.toolmol import crossover as co
from main.toolmol.fg_lookup import get_functional_group


@dataclass
class ToolResult:
    success: bool
    mol: Optional[Chem.Mol]
    message: str


def _load(mol_smiles):
    mol = Chem.MolFromSmiles(mol_smiles)
    if mol is None:
        return None, "invalid current molecule SMILES"
    return mol, None


def _check_idx(mol, idx):
    if not (0 <= idx < mol.GetNumAtoms()):
        return f"atom index {idx} out of range (molecule has {mol.GetNumAtoms()} atoms)"
    return None


def add_atom(mol_smiles, idx, element, bond):
    mol, err = _load(mol_smiles)
    if err:
        return ToolResult(False, None, err)
    err = _check_idx(mol, idx)
    if err:
        return ToolResult(False, None, err)
    try:
        bond_type = mol_ops.bond_symbol_to_type(bond)
    except ValueError as e:
        return ToolResult(False, None, str(e))

    rw = mol_ops.copy_edit_mol(mol)
    try:
        new_idx = rw.AddAtom(Chem.Atom(element))
    except RuntimeError as e:
        return ToolResult(False, None, f"invalid element '{element}': {e}")
    rw.AddBond(idx, new_idx, bond_type)
    try:
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
        return ToolResult(True, new_mol, f"added {element} atom (bond={bond}) to atom {idx}")
    except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException, ValueError) as e:
        return ToolResult(False, None, f"invalid modification: {e}")


def replace_atom(mol_smiles, idx, element):
    mol, err = _load(mol_smiles)
    if err:
        return ToolResult(False, None, err)
    err = _check_idx(mol, idx)
    if err:
        return ToolResult(False, None, err)

    rw = mol_ops.copy_edit_mol(mol)
    try:
        rw.ReplaceAtom(idx, Chem.Atom(element))
    except RuntimeError as e:
        return ToolResult(False, None, f"invalid element '{element}': {e}")
    try:
        new_mol = rw.GetMol()
        Chem.SanitizeMol(new_mol)
        return ToolResult(True, new_mol, f"replaced atom {idx} with {element}")
    except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException, ValueError) as e:
        return ToolResult(False, None, f"invalid modification: {e}")


def add_functional_group(mol_smiles, idx, group, bond):
    frag_smiles = get_functional_group(group)
    if frag_smiles is None:
        return ToolResult(False, None, f"unknown functional group '{group}'")
    return add_substructure(mol_smiles, idx, frag_smiles, bond)


def add_substructure(mol_smiles, idx, substructure, bond):
    mol, err = _load(mol_smiles)
    if err:
        return ToolResult(False, None, err)
    err = _check_idx(mol, idx)
    if err:
        return ToolResult(False, None, err)

    new_mol, used_fallback, err = mol_ops.attach_fragment(mol, idx, substructure, bond)
    if err:
        return ToolResult(False, None, err)
    note = " (no [*] found - attached via the first atom written in the fragment)" if used_fallback else ""
    return ToolResult(True, new_mol, f"attached '{substructure}' to atom {idx} (bond={bond}){note}")


def replace_substructure(mol_smiles, anchor_idx, branch_idx, new_substructure):
    mol, err = _load(mol_smiles)
    if err:
        return ToolResult(False, None, err)

    hole_mol, stub_idx, bond_type, removed_desc, err = mol_ops.cut_at_bond(mol, anchor_idx, branch_idx)
    if err:
        return ToolResult(False, None, err)
    bond_symbol = mol_ops.bond_type_to_symbol(bond_type)
    new_mol, used_fallback, err2 = mol_ops.attach_fragment(hole_mol, stub_idx, new_substructure, bond_symbol)
    if err2:
        return ToolResult(False, None, err2)
    note = " (no [*] found in new_substructure - attached via its first atom)" if used_fallback else ""
    return ToolResult(True, new_mol, (f"removed branch at atom {branch_idx} (kept atom {anchor_idx}, "
                                       f"removed: {removed_desc}) and attached '{new_substructure}'{note}"))


def remove_substructure(mol_smiles, anchor_idx, branch_idx):
    mol, err = _load(mol_smiles)
    if err:
        return ToolResult(False, None, err)

    new_mol, stub_idx, bond_type, removed_desc, err = mol_ops.cut_at_bond(mol, anchor_idx, branch_idx)
    if err:
        return ToolResult(False, None, err)
    return ToolResult(True, new_mol, f"removed branch at atom {branch_idx} (kept atom {anchor_idx}): {removed_desc}")


def _cut_at_index(mol, idx):
    if not (0 <= idx < mol.GetNumAtoms()):
        return None
    atom = mol.GetAtomWithIdx(idx)
    acyclic_bonds = [b for b in atom.GetBonds() if not b.IsInRing()]
    random.shuffle(acyclic_bonds)
    for bond in acyclic_bonds:
        fragments_mol = Chem.FragmentOnBonds(mol, [bond.GetIdx()], addDummies=True, dummyLabels=[(1, 1)])
        try:
            frags = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
        except ValueError:
            continue
        if len(frags) == 2:
            return frags
    return None


def crossover_molecules(mol1_smiles, idx1, mol2_smiles, idx2):
    mol1, err = _load(mol1_smiles)
    if err:
        return ToolResult(False, None, err)
    mol2, err = _load(mol2_smiles)
    if err:
        return ToolResult(False, None, err)

    frags1 = _cut_at_index(mol1, idx1)
    if frags1 is None:
        return ToolResult(False, None, (f"atom {idx1} in molecule 1 does not yield exactly 2 fragments "
                                         f"(likely in a ring or has no acyclic bond)"))
    frags2 = _cut_at_index(mol2, idx2)
    if frags2 is None:
        return ToolResult(False, None, (f"atom {idx2} in molecule 2 does not yield exactly 2 fragments "
                                         f"(likely in a ring or has no acyclic bond)"))

    pairings = [(a, b) for a in frags1 for b in frags2]
    random.shuffle(pairings)
    rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
    for fa, fb in pairings:
        try:
            products = rxn.RunReactants((fa, fb))
        except Exception:
            continue
        for prod in products:
            candidate = prod[0]
            if co.mol_ok(candidate) and co.ring_OK(candidate):
                try:
                    Chem.SanitizeMol(candidate)
                    return ToolResult(True, candidate, f"crossover between atom {idx1} (mol1) and atom {idx2} (mol2)")
                except ValueError:
                    continue
    return ToolResult(False, None, "crossover failed to produce a valid molecule from any of the 4 fragment combinations")


TOOL_DISPATCH = {
    'add_atom': add_atom,
    'replace_atom': replace_atom,
    'add_functional_group': add_functional_group,
    'add_substructure': add_substructure,
    'replace_substructure': replace_substructure,
    'remove_substructure': remove_substructure,
    'crossover_molecules': crossover_molecules,
}

# Tools whose first argument is the single "current working molecule" (agent-injected).
# crossover_molecules is the only one operating on two agent-injected parent molecules.
SINGLE_MOL_TOOLS = {'add_atom', 'replace_atom', 'add_functional_group', 'add_substructure',
                    'replace_substructure', 'remove_substructure'}

# undo_last_change is deliberately absent from TOOL_DISPATCH: unlike every other tool here,
# it isn't a stateless mol-in/mol-out function - it needs the agent's own history of prior
# working-molecule states, which only agent.py's _agentic_edit loop owns. It's still declared
# in TOOL_SCHEMAS (so the LLM can call it) and handled by a special case in that loop, the
# same way crossover_molecules gets a special-cased call (there, only the call signature
# differs; here, there's no toolbox function at all to call).

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "add_atom",
            "description": "Add a single new atom to the current molecule, bonded to an existing atom.",
            "parameters": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer", "description": "Atom index in the current molecule to bond the new atom to."},
                    "element": {"type": "string", "description": "Element symbol of the new atom, e.g. 'C', 'N', 'O', 'F', 'Cl'."},
                    "bond": {"type": "string", "enum": ["single", "double", "triple"], "description": "Bond order connecting the new atom to atom idx."},
                },
                "required": ["idx", "element", "bond"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_atom",
            "description": "Replace the atom at the given index with a new element, preserving its existing bonds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer", "description": "Atom index in the current molecule to replace."},
                    "element": {"type": "string", "description": "New element symbol, e.g. 'N', 'O', 'S'."},
                },
                "required": ["idx", "element"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_functional_group",
            "description": ("Add a predefined common functional group or ring (e.g. 'methyl', 'phenyl', "
                             "'hydroxyl', 'carboxyl', 'cyclopropyl', 'pyridyl') to the current molecule "
                             "at the given atom index."),
            "parameters": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer", "description": "Atom index in the current molecule to attach the group to."},
                    "group": {"type": "string", "description": "Plain-English name of the functional group/ring, e.g. 'methyl', 'phenyl', 'hydroxyl'."},
                    "bond": {"type": "string", "enum": ["single", "double", "triple"], "description": "Bond order connecting the group to atom idx."},
                },
                "required": ["idx", "group", "bond"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_substructure",
            "description": ("Add a custom substructure (given as SMILES) to the current molecule at the "
                             "given atom index. By default it attaches via the first atom written in the "
                             "SMILES - e.g. 'C(=O)O' attaches via the carbonyl carbon. To attach via a "
                             "different atom instead, mark exactly one atom in the SMILES with [*] or [1*]; "
                             "that marked atom's own neighbor becomes the attachment point instead of the "
                             "first-written atom, and the marker itself is removed. Use this tool when no "
                             "predefined functional group matches what you want to add."),
            "parameters": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer", "description": "Atom index in the current molecule to attach the substructure to."},
                    "substructure": {"type": "string", "description": "SMILES of the substructure to add. Attaches via the first atom written by default (e.g. 'C(=O)O' attaches via the carbonyl carbon, 'NC3CCNCC3' via the nitrogen). To attach via a different atom, mark it with [*] or [1*] instead, e.g. '[1*]CCO' attaches via the CH2 next to the marker, not via the terminal O."},
                    "bond": {"type": "string", "enum": ["single", "double", "triple"], "description": "Bond order connecting the substructure to atom idx."},
                },
                "required": ["idx", "substructure", "bond"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_substructure",
            "description": ("Cut the bond between two directly-bonded atoms, discard the branch on "
                             "branch_idx's side (everything only reachable through it), and attach a new "
                             "substructure to anchor_idx in its place. Only acyclic (non-ring) bonds can be "
                             "cut - use a bond from the 'nbrs' column of the atom table where neither atom "
                             "has ring='Y' on that side of the connection, i.e. pick a branch that isn't "
                             "part of a ring."),
            "parameters": {
                "type": "object",
                "properties": {
                    "anchor_idx": {"type": "integer", "description": "Atom index to keep - the new substructure attaches here."},
                    "branch_idx": {"type": "integer", "description": "Atom index on the branch to discard. Must be one of anchor_idx's neighbors (see the 'nbrs' column in the atom table) - the whole branch reachable from this atom, without going back through anchor_idx, is removed."},
                    "new_substructure": {"type": "string", "description": "SMILES of the replacement substructure. Attaches via the first atom written by default (e.g. 'C(=O)O' attaches via the carbonyl carbon). To attach via a different atom, mark it with [*] or [1*] instead, e.g. '[1*]CCO' attaches via the CH2 next to the marker, not via the terminal O."},
                },
                "required": ["anchor_idx", "branch_idx", "new_substructure"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_substructure",
            "description": ("Cut the bond between two directly-bonded atoms and discard the branch on "
                             "branch_idx's side (everything only reachable through it), keeping anchor_idx "
                             "and the rest of the molecule. Only acyclic (non-ring) bonds can be cut - pick "
                             "a branch_idx that leads away from any ring, not into one."),
            "parameters": {
                "type": "object",
                "properties": {
                    "anchor_idx": {"type": "integer", "description": "Atom index to keep."},
                    "branch_idx": {"type": "integer", "description": "Atom index on the branch to discard. Must be one of anchor_idx's neighbors (see the 'nbrs' column in the atom table) - the whole branch reachable from this atom, without going back through anchor_idx, is removed."},
                },
                "required": ["anchor_idx", "branch_idx"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "crossover_molecules",
            "description": ("Split both of the two original parent molecules at the given atom indices "
                             "and recombine one fragment from each into a new molecule. Typically the "
                             "first modification you make, combining the two parents before further edits. "
                             "This is the ONLY tool that can access ligand 2 at all - every other tool acts "
                             "solely on your current working molecule (ligand 1) and cannot see ligand 2."),
            "parameters": {
                "type": "object",
                "properties": {
                    "idx1": {"type": "integer", "description": "Atom index in parent molecule 1 to cut at (must not be in a ring)."},
                    "idx2": {"type": "integer", "description": "Atom index in parent molecule 2 to cut at (must not be in a ring)."},
                },
                "required": ["idx1", "idx2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo_last_change",
            "description": ("Revert the most recent successful modification, restoring the molecule to "
                             "the state before that change. Use this if a tool's result message or the "
                             "updated atom table shows it did not do what you intended - e.g. it removed "
                             "or attached something different from what you expected. Can be called "
                             "repeatedly to step back through several recent changes. Does not count "
                             "against your modification budget. Fails if there is no change to undo."),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
