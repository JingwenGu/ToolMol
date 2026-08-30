"""Non-callable context-injection functions - computed deterministically before every LLM
turn and included in the prompt as text, never invoked by the LLM itself (Section 3.2 /
Appendix B.2 of the paper)."""

import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
import tdc

_sa_scorer = tdc.Oracle(name='SA')

# Common max valence for typical organic elements; used only for the descriptive
# "num_available_valences" field shown to the LLM, not for validating edits (RDKit's own
# SanitizeMol is the actual source of truth for valence correctness).
_COMMON_MAX_VALENCE = {'C': 4, 'N': 3, 'O': 2, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1, 'S': 2, 'P': 3, 'B': 3}


def _available_valence(atom):
    max_valence = _COMMON_MAX_VALENCE.get(atom.GetSymbol())
    if max_valence is None:
        max_valence = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
        if max_valence < 0:
            max_valence = atom.GetTotalValence()
    return max(0, max_valence - atom.GetTotalValence())


def get_ligand_structure(mol):
    """Per-atom structural info for every atom: index, element, substitutable hydrogens,
    available valence, neighbor count/indices, ring membership, and betweenness centrality.
    Centrality is computed on a heavy-atom-only graph (atoms as nodes, bonds as edges,
    hydrogens not represented as explicit nodes) - the paper doesn't specify this choice,
    but it's the standard convention."""
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx())
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    if graph.number_of_nodes() > 2:
        centrality = nx.betweenness_centrality(graph)
    else:
        centrality = {n: 0.0 for n in graph.nodes}

    info = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        info.append({
            'atom_index': idx,
            'element': atom.GetSymbol(),
            'num_substitutable_hydrogens': atom.GetTotalNumHs(),
            'num_available_valences': _available_valence(atom),
            'num_neighboring_atoms': atom.GetDegree(),
            'neighbor_indices': [n.GetIdx() for n in atom.GetNeighbors()],
            'is_in_ring': atom.IsInRing(),
            'centrality': round(centrality.get(idx, 0.0), 4),
        })
    return info


def calculate_properties(mol):
    """Standard molecular descriptors: QED, SA, molecular weight, LogP, TPSA, H-bond
    donors/acceptors, rotatable bonds, aromatic rings."""
    smi = Chem.MolToSmiles(mol)
    return {
        'QED': round(QED.qed(mol), 4),
        'SA': round(float(_sa_scorer(smi)), 4),
        'molecular_weight': round(Descriptors.MolWt(mol), 2),
        'LogP': round(Descriptors.MolLogP(mol), 4),
        'TPSA': round(Descriptors.TPSA(mol), 4),
        'num_HBond_donors': Descriptors.NumHDonors(mol),
        'num_HBond_acceptors': Descriptors.NumHAcceptors(mol),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
    }
