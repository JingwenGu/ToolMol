"""Structural feasibility gate - the failure class the SA oracle cannot see.

SA score (tdc.Oracle('SA'), i.e. Ertl & Schuffenhauer 2009) is an
occurrence-weighted MEAN of ECFP4 fragment frequencies plus penalties for size,
stereo, spiro, bridgeheads and macrocycles. It contains no term for ring strain,
antiaromaticity, or chemical stability, and its unknown-fragment default (-4) sits
only ~0.9 below the true minimum of the fragment table (-3.13, median -3.13), so no
SA value can mean "unsynthesizable". Measured consequences: bare azete C1=NC=C1
scores 3.04 and cyclobutadiene C1=CC=C1 scores 1.99 (easier than aspirin, 1.58);
hanging an azete off a kinase-inhibitor scaffold costs 0.39 SA units, i.e. 0.044 of
Phi via (10-SA)/9.

This module answers a narrower question than synthesizability: can the molecule
exist at room temperature? Two tiers:

  'fatal'  geometric / electronic impossibility - antiaromatic 4-rings, endocyclic
           triple bonds in small rings, cyclic allenes, anti-Bredt alkenes.
  'hard'   labile but real - peroxides, gem-diols, aminals, unstabilised enols,
           N-N-N chains. Artemisinin is 'hard' (endoperoxide), so this tier is
           advisory: do not reject on it unless you mean to.

Validated by main/test_feasibility_gate.py: a panel of real drugs, natural
products and fused-ring steroids/enones that must never be 'fatal', a panel of
impossible or non-isolable structures that must never be 'ok', and a fixed sample
of 5000 ZINC molecules (all purchasable, so none may be 'fatal').

Known limitation: trans-cycloalkenes in rings <=7 are NOT caught, because RDKit
does not retain E/Z stereo on bonds in small rings. This is a stability gate only
- it says nothing about whether a synthetic route exists to a stable but exotic
scaffold. For that, score the final Pareto front with AiZynthFinder.

Standalone use:
    python main/feasibility_gate.py "C1=NC(c2ccnc(Nc3ccc(N4CCOCC4)cc3)c2)=C1"
    python main/feasibility_gate.py --audit results/results_ToolMol_x_y1.yaml --top 100
"""
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

# Gate modes, in increasing strictness. 'off' is the default everywhere so that
# every existing invocation is unaffected unless the flag is passed explicitly.
GATE_MODES = ("off", "fatal", "strict")

# --- functional-group liabilities (SMARTS) -----------------------------------
# (label, SMARTS, severity). Severity 'fatal' is reserved for the programmatic
# ring checks below; labile groups are 'hard' because real drugs contain them.
SMARTS_RULES = [
    ("peroxide",                "[OX2]-[OX2]",                         "hard"),
    ("N-N-N chain (non-azide)", "[NX3,NX2]-[NX3,NX2]-[NX3,NX2]",       "hard"),
    ("geminal diol",            "[CX4]([OX2H])[OX2H]",                 "hard"),
    ("hemiketal/hemiacetal",    "[CX4;!$(C(O)(O)O)]([OX2H])[OX2][#6]", "hard"),
    ("aminal",                  "[CX4]([NX3])[NX3]",                   "hard"),
    ("enol (non-aromatic)",     "[CX3;!a]=[CX3;!a][OX2H]",             "hard"),
    ("enamine-to-ketone taut.", "[CX3;!a]=[CX3;!a][NX3H2]",            "hard"),
    ("orthoester",              "[CX4]([OX2][#6])([OX2][#6])[OX2][#6]","hard"),
    ("acyl halide",             "[CX3](=O)[F,Cl,Br,I]",                "hard"),
    ("thiol-ketone gem",        "[CX4]([SX2H])[SX2H]",                 "hard"),
    ("vinyl/aryl diazo",        "[CX3]=[NX2]=[NX1]",                   "hard"),
    ("nitroso on carbon",       "[#6][NX2]=[OX1]",                     "hard"),
]
_COMPILED = [(l, Chem.MolFromSmarts(s), sev) for l, s, sev in SMARTS_RULES]
_AZIDE = Chem.MolFromSmarts("[NX1]~[NX2]~[NX3]")
_STABLE_ENOL = Chem.MolFromSmarts("[OX2H][CX3]=[CX3][CX3]=[OX1]")


# --- ring strain / antiaromaticity (programmatic over the SSSR) --------------
def _ring_flags(mol):
    """Strain and antiaromaticity. Done programmatically rather than in SMARTS
    because the relevant facts (endocyclic bond orders per ring, ring size) are
    relational and awkward to express as a single pattern."""
    flags = []
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        n = len(ring)
        ring_set = set(ring)
        endo = [b for b in mol.GetBonds()
                if b.GetBeginAtomIdx() in ring_set and b.GetEndAtomIdx() in ring_set]
        if len(endo) < n:
            continue
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        n_double = sum(1 for b in endo if b.GetBondType() == Chem.BondType.DOUBLE)
        n_triple = sum(1 for b in endo if b.GetBondType() == Chem.BondType.TRIPLE)
        n_allene = sum(1 for i in ring
                       if mol.GetAtomWithIdx(i).GetDegree() == 2
                       and sum(1 for b in mol.GetAtomWithIdx(i).GetBonds()
                               if b.GetBondType() == Chem.BondType.DOUBLE) == 2)
        if n_triple and n <= 7:
            flags.append((f"endocyclic triple bond in {n}-ring", "fatal"))
        if n_allene and n <= 8:
            flags.append((f"allene in {n}-ring", "fatal"))
        if aromatic:
            continue                      # rdkit already validated the aromatic system
        if n == 4 and n_double >= 2:
            flags.append(("antiaromatic 4-ring (cyclobutadiene/azete type)", "fatal"))
        elif n == 3 and n_double >= 2:
            flags.append(("3-ring with 2 double bonds", "fatal"))
        elif n == 5 and n_double >= 3:
            flags.append(("5-ring with 3 endocyclic double bonds", "fatal"))
        elif n == 3 and n_double == 1:
            flags.append(("cyclopropene/azirine (strained, reactive)", "hard"))
        elif n == 4 and n_double == 1:
            flags.append(("cyclobutene (strained)", "hard"))
    return flags


def _anti_bredt(mol):
    """Bredt's rule: a double bond at the bridgehead of a small BRIDGED bicycle.

    A bridgehead is an atom on two SSSR rings that share three or more atoms (a
    shared bridge plus both bridgeheads). Rings that share only one bond are
    merely FUSED - their junction atoms may freely carry a double bond (cholesterol,
    testosterone, every Wieland-Miescher-type enone), and rings sharing one atom are
    spiro. An earlier version treated any atom on two rings as a bridgehead and so
    rejected steroids and ~1% of ZINC.

    For a bridged pair, Fawcett's S is the sum of the three bridge lengths,
    S = r1 + r2 - shared - 2. A bridgehead alkene is isolable only for S >= 7
    (bicyclo[3.3.1]non-1-ene, S = 7, is a real compound); S <= 6 (bicyclo[2.2.2]oct-1-ene,
    bicyclo[2.2.1]hept-1-ene) is not. The S >= 7 cut is the heuristic part. It is
    evaluated per pair of SSSR rings, so very large polycyclic cages are judged by
    their smallest bridged sub-bicycle only."""
    out = []
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    seen = set()
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            shared = rings[i] & rings[j]
            if len(shared) < 3:
                continue                      # fused (2 shared) or spiro (1): not bridged
            s_num = len(rings[i]) + len(rings[j]) - len(shared) - 2
            if s_num >= 7:
                continue
            union = rings[i] | rings[j]
            for idx in shared:
                atom = mol.GetAtomWithIdx(idx)
                if idx in seen or atom.GetIsAromatic():
                    continue
                if sum(1 for n in atom.GetNeighbors() if n.GetIdx() in union) < 3:
                    continue                  # a bridge atom, not a bridgehead
                if any(b.GetBondType() == Chem.BondType.DOUBLE and not b.GetIsAromatic()
                       and b.GetOtherAtomIdx(idx) in union for b in atom.GetBonds()):
                    seen.add(idx)
                    out.append((f"anti-Bredt alkene at bridgehead (S={s_num})", "fatal"))
    return out


def _trans_small_ring_alkene(mol):
    """trans alkene in a small ring. Fires only when E/Z stereo survived parsing,
    which for rings <=7 it usually does not - see the module docstring."""
    out = []
    ri = mol.GetRingInfo()
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DOUBLE or b.GetIsAromatic():
            continue
        if b.GetStereo() != Chem.BondStereo.STEREOE:
            continue
        sizes = [len(r) for r in ri.BondRings() if b.GetIdx() in r]
        if sizes and min(sizes) <= 7:
            out.append((f"trans alkene in {min(sizes)}-ring", "fatal"))
    return out


def check(smiles):
    """Return (verdict, reasons). verdict: 'ok' | 'hard' | 'fatal' | 'invalid'."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid", ["unparseable SMILES"]
    reasons = []
    reasons += _ring_flags(mol)
    reasons += _anti_bredt(mol)
    reasons += _trans_small_ring_alkene(mol)
    azide_atoms = {i for m in mol.GetSubstructMatches(_AZIDE) for i in m}
    for label, patt, sev in _COMPILED:
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue
        if label.startswith("N-N-N") and all(set(m) <= azide_atoms for m in matches):
            continue                      # a genuine azide is fine
        if label.startswith("enol") and mol.HasSubstructMatch(_STABLE_ENOL):
            continue                      # vinylogous acid / 1,3-dicarbonyl enol is stable
        reasons.append((label, sev))
    if any(sev == "fatal" for _, sev in reasons):
        return "fatal", [r for r, _ in reasons]
    if reasons:
        return "hard", [r for r, _ in reasons]
    return "ok", []


def rejects(smiles, mode):
    """(bool, reasons) - whether `mode` rejects this molecule.

    'off'    rejects nothing (current behaviour of every existing run).
    'fatal'  rejects verdict 'fatal' and 'invalid'.
    'strict' also rejects verdict 'hard'.
    """
    if mode == "off":
        return False, []
    verdict, reasons = check(smiles)
    if verdict in ("fatal", "invalid"):
        return True, reasons
    if mode == "strict" and verdict == "hard":
        return True, reasons
    return False, reasons


def is_feasible(smiles, mode="fatal"):
    return not rejects(smiles, mode)[0]


def penalty(smiles, fatal=1.0, hard=0.25):
    """Additive penalty for use as an objective term. Folding this into Phi
    redefines the objective, so top-1/top-10/AUC stop being comparable with
    previously-completed runs - prefer the sanitize-time gate."""
    v, _ = check(smiles)
    return {"ok": 0.0, "hard": hard, "fatal": fatal, "invalid": fatal}[v]


def audit_results_yaml(path, top=None):
    """Audit a results_*.yaml written by Oracle.save_result ({smiles: [score, idx]}).
    Returns [(smiles, score, verdict, reasons)] sorted by score descending."""
    import yaml
    with open(path) as f:
        buf = yaml.safe_load(f) or {}
    rows = sorted(((s, v[0]) for s, v in buf.items()), key=lambda kv: -kv[1])
    if top:
        rows = rows[:top]
    out, counts = [], {"ok": 0, "hard": 0, "fatal": 0, "invalid": 0}
    for smi, score in rows:
        v, r = check(smi)
        counts[v] += 1
        out.append((smi, score, v, r))
    n = max(len(rows), 1)
    print(f"{path}: {len(rows)} molecules | "
          + " | ".join(f"{k}={counts[k]} ({counts[k] / n:.1%})" for k in counts))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("smiles", nargs="*")
    ap.add_argument("--audit", metavar="RESULTS_YAML",
                    help="audit a results_*.yaml instead of scoring SMILES")
    ap.add_argument("--top", type=int, default=None,
                    help="with --audit, only the top-N molecules by score")
    ap.add_argument("--show", choices=["all", "flagged"], default="flagged")
    a = ap.parse_args()
    if a.audit:
        for smi, score, v, r in audit_results_yaml(a.audit, a.top):
            if a.show == "all" or v != "ok":
                print(f"  {score:6.3f}  {v:7s} {'; '.join(r)[:60]:60s} {smi}")
    for s in a.smiles:
        v, r = check(s)
        print(f"{v:7s} {'; '.join(r):50s} {s}")
