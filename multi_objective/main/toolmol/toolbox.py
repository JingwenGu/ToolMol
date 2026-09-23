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

    # Try the plain substitution first, then - only if that fails to kekulize - retry with an
    # explicit hydrogen on the new atom. copy_edit_mol rebuilds every atom as a bare
    # Chem.Atom(symbol), which drops aromaticity and hydrogen state, so swapping an aromatic
    # ring heteroatom for nitrogen produced a bare two-connected aromatic N with no hydrogen.
    # That is not a valid aromatic system, and benzoxazole -> benzimidazole, furan -> pyrrole
    # and every other aromatic O/S -> NH swap failed with "Can't kekulize mol". A pyrrole-type
    # nitrogen needs NumExplicitHs=1; carbons and pyridine-type nitrogens do not, hence the
    # retry rather than setting it unconditionally.
    for explicit_hs in (0, 1):
        rw = mol_ops.copy_edit_mol(mol)
        try:
            new_atom = Chem.Atom(element)
            if explicit_hs:
                new_atom.SetNumExplicitHs(explicit_hs)
            rw.ReplaceAtom(idx, new_atom)
        except RuntimeError as e:
            return ToolResult(False, None, f"invalid element '{element}': {e}")
        try:
            new_mol = rw.GetMol()
            Chem.SanitizeMol(new_mol)
            return ToolResult(True, new_mol, f"replaced atom {idx} with {element}")
        except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException, ValueError) as e:
            err = e
    return ToolResult(False, None, f"invalid modification: {err}")


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


def crossover_molecules(mol1_smiles, anchor1, branch1, mol2_smiles, anchor2, branch2):
    """Cut one named bond in each parent, keep the anchor side of each, and join the two kept
    pieces with a single bond between the two anchor atoms.

    This replaces an earlier signature that took one atom index per parent. That version was
    doubly random: it shuffled the chosen atom's acyclic bonds and cut whichever happened to
    work, then shuffled all four fragment pairings and returned the first valid product. The
    same call on the same inputs produced five different molecules across eight invocations,
    so the model's stated intent barely constrained the result - asking to keep parent 1's
    heteroaryl and add parent 2's aryl could just as easily return the two fragments it had
    asked to discard, joined at an unintended position.

    That mattered beyond the wasted edit: in the regret taxonomy those episodes score as
    "executed something unrelated to the stated plan" and the blame lands on the model's
    reasoning rather than on the tool. Naming a bond on each side - the same anchor/branch
    convention remove_substructure and replace_substructure already use - makes the product a
    deterministic function of the arguments.
    """
    mol1, err = _load(mol1_smiles)
    if err:
        return ToolResult(False, None, err)
    mol2, err = _load(mol2_smiles)
    if err:
        return ToolResult(False, None, err)

    keep1, stub1, _bt1, removed1, err = mol_ops.cut_at_bond(mol1, anchor1, branch1)
    if err:
        return ToolResult(False, None, f"molecule 1: {err}")
    keep2, stub2, _bt2, removed2, err = mol_ops.cut_at_bond(mol2, anchor2, branch2)
    if err:
        return ToolResult(False, None, f"molecule 2: {err}")

    offset = keep1.GetNumAtoms()
    combo = Chem.RWMol(Chem.CombineMols(keep1, keep2))
    combo.AddBond(stub1, offset + stub2, Chem.BondType.SINGLE)
    mol_ops._consume_explicit_hs(combo.GetAtomWithIdx(stub1), Chem.BondType.SINGLE)
    mol_ops._consume_explicit_hs(combo.GetAtomWithIdx(offset + stub2), Chem.BondType.SINGLE)

    try:
        new_mol = combo.GetMol()
        Chem.SanitizeMol(new_mol)
    except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException, ValueError) as e:
        return ToolResult(False, None, f"invalid modification: {e}")

    return ToolResult(True, new_mol, (f"joined molecule 1 at atom {anchor1} (discarded {removed1}) "
                                       f"to molecule 2 at atom {anchor2} (discarded {removed2})"))


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
            "description": ("Combine the two original parent molecules: cut one bond in each, keep the "
                             "anchor side of each, and join those two kept pieces together with a single "
                             "bond between the two anchor atoms. You choose exactly which part of each "
                             "parent survives, so the product is fully determined by your arguments. "
                             "Typically the first modification you make, combining the two parents before "
                             "further edits. This is the ONLY tool that can access ligand 2 at all - every "
                             "other tool acts solely on your current working molecule (ligand 1) and cannot "
                             "see ligand 2."),
            "parameters": {
                "type": "object",
                "properties": {
                    "anchor1": {"type": "integer", "description": "Atom index in parent molecule 1 to KEEP - the new bond to molecule 2 is formed here."},
                    "branch1": {"type": "integer", "description": "Atom index in parent molecule 1 on the side to DISCARD. Must be one of anchor1's neighbours (see the 'nbrs' column of molecule 1's atom table), and the bond between them must not be part of a ring."},
                    "anchor2": {"type": "integer", "description": "Atom index in parent molecule 2 to KEEP - the new bond to molecule 1 is formed here."},
                    "branch2": {"type": "integer", "description": "Atom index in parent molecule 2 on the side to DISCARD. Must be one of anchor2's neighbours (see the 'nbrs' column of molecule 2's atom table), and the bond between them must not be part of a ring."},
                },
                "required": ["anchor1", "branch1", "anchor2", "branch2"],
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
