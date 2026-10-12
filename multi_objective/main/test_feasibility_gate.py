"""Validation panel for main/feasibility_gate.py. Run from multi_objective/:

    python main/test_feasibility_gate.py

Real drugs and natural products must not be called 'fatal'; impossible or
non-isolable structures must not be called 'ok'. Re-run this after touching any
threshold in feasibility_gate.py - the gate is only useful if its false-positive
rate on real chemistry stays at zero.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feasibility_gate import check

# (name, smiles, strictest verdict allowed)
MUST_PASS = [
    ("aspirin",           "CC(=O)Oc1ccccc1C(=O)O",                                      "ok"),
    ("imatinib",          "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1","ok"),
    ("sunitinib",         "CCN(CC)CCNC(=O)c1c(C)[nH]c(/C=C2\\C(=O)Nc3ccc(F)cc32)c1C",   "ok"),
    ("penicillin G",      "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",    "ok"),
    ("morphine",          "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",   "ok"),
    ("camphor",           "CC1(C)C2CCC1(C)C(=O)C2",                                     "ok"),
    ("cubane",            "C12C3C4C1C5C2C3C45",                                         "ok"),
    ("caffeine",          "Cn1c(=O)c2c(ncn2C)n(C)c1=O",                                 "ok"),
    ("vitamin C",         "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O",                             "ok"),
    ("norbornene",        "C1CC2CC1C=C2",                                               "ok"),
    ("trans-cyclooctene", "C1CCC/C=C/CC1",                                              "ok"),
    ("beta-lactam core",  "O=C1CCN1",                                                   "ok"),
    ("azetidine",         "C1CNC1",                                                      "ok"),
    ("oxetane",           "C1COC1",                                                      "ok"),
    ("pyrrole",           "c1cc[nH]c1",                                                  "ok"),
    ("furan",             "c1ccoc1",                                                     "ok"),
    ("phenyl azide",      "[N-]=[N+]=Nc1ccccc1",                                         "ok"),
    ("artemisinin",       "C[C@@H]1CC[C@H]2[C@@H](C)C(=O)O[C@@H]3O[C@@]4(C)CC[C@@H]1[C@@]23OO4", "hard"),
    # Regression: a double bond on a ring-FUSION atom is not a Bredt violation. The
    # first anti-Bredt rule rejected all of these (and ~1% of ZINC) as 'fatal'.
    ("cholesterol",       "CC(C)CCC[C@@H](C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C", "ok"),
    ("testosterone",      "C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=CC(=O)CC[C@]34C", "ok"),
    ("progesterone",      "CC(=O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CCC4=CC(=O)CC[C@]34C)C", "ok"),
    ("Wieland-Miescher",  "CC12CCC(=O)C=C1CCCC2=O",                                     "ok"),
    ("bicyclo[2.2.2]oct-2-ene", "C1CC2CCC1C=C2",                                       "ok"),
    ("bicyclo[3.3.1]non-1-ene", "C1CC2CCCC(=C1)C2",                                    "ok"),  # S=7, isolable
    ("fused 5-6 enone (ZINC)", "C=C(C)CN(C)C(=O)CN1CC2=C(C1=O)[C@@H](c1ccc(Cl)cc1Cl)NC(=O)N2CC", "ok"),
    ("indene",            "C1=Cc2ccccc2C1",                                             "ok"),
]

# (name, smiles, verdict required)
MUST_FAIL = [
    ("azete on scaffold", "C1=NC(c2ccnc(Nc3ccc(N4CCOCC4)cc3)c2)=C1", "fatal"),
    ("cyclobutadiene",    "C1=CC=C1",                                "fatal"),
    ("bare azete",        "C1=NC=C1",                                "fatal"),
    ("oxete",             "C1=COC=1",                                "fatal"),
    ("cyclopropyne",      "C1#CC1",                                  "fatal"),
    ("benzyne",           "C1#CC=CC=C1",                             "fatal"),
    ("anti-Bredt [5,5]",  "C1CC2=CCC1C2",                            "fatal"),
    ("bicyclo[2.2.2]oct-1-ene", "C1CC2CCC1=CC2",                     "fatal"),  # alkene AT the bridgehead
    ("cyclic allene",     "C1CCC=C=CC1",                             "fatal"),
    ("1,2-dioxetane",     "C1COO1",                                  "hard"),
    ("geminal diol",      "OC(O)c1ccccc1",                           "hard"),
    ("triazane",          "NNNc1ccccc1",                             "hard"),
]

RANK = {"ok": 0, "hard": 1, "fatal": 2, "invalid": 3}

# Every molecule in ZINC is a real, purchasable compound, so none may be 'fatal'. A fixed
# sample catches false positives the hand-picked panel above can't think of. 'hard' is
# allowed (aminals etc. are real); only 'fatal'/'invalid' fail. Skipped if data/zinc.tab
# is absent. Run from multi_objective/ (single_objective/ ships the same file).
ZINC_SAMPLE, ZINC_SEED = 5000, 0
ZINC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "zinc.tab")


def zinc_sample():
    import csv
    import random
    if not os.path.exists(ZINC_PATH):
        return None
    with open(ZINC_PATH, newline="") as f:
        smiles = [row[0] for row in csv.reader(f, delimiter="\t")][1:]   # row 0 is the header
    return random.Random(ZINC_SEED).sample(smiles, min(ZINC_SAMPLE, len(smiles)))


def main():
    failures = []
    print("MUST PASS (real chemistry - never 'fatal'):")
    for name, smi, worst in MUST_PASS:
        v, r = check(smi)
        bad = RANK[v] > RANK[worst]
        print(f"  {'FAIL' if bad else 'pass'}  {name:20s} {v:7s} {'; '.join(r)[:48]}")
        if bad:
            failures.append(f"{name}: got '{v}', expected at most '{worst}'")
    print("\nMUST FAIL (impossible / non-isolable - never 'ok'):")
    for name, smi, need in MUST_FAIL:
        v, r = check(smi)
        bad = RANK[v] < RANK[need]
        print(f"  {'FAIL' if bad else 'pass'}  {name:20s} {v:7s} {'; '.join(r)[:48]}")
        if bad:
            failures.append(f"{name}: got '{v}', expected at least '{need}'")
    sample = zinc_sample()
    if sample is None:
        print(f"\nZINC sample: SKIPPED ({os.path.normpath(ZINC_PATH)} not found)")
    else:
        flagged = [(smi, check(smi)) for smi in sample]
        flagged = [(smi, v, r) for smi, (v, r) in flagged if v in ("fatal", "invalid")]
        print(f"\nZINC sample ({len(sample)} real molecules, seed {ZINC_SEED}): "
              f"{len(flagged)} fatal/invalid (must be 0)")
        for smi, v, r in flagged[:10]:
            print(f"  FAIL  {v:7s} {'; '.join(r)[:48]}  {smi}")
            failures.append(f"ZINC molecule called '{v}': {smi}")
    print()
    if failures:
        for f in failures:
            print("FAILURE:", f)
        return 1
    print(f"all {len(MUST_PASS) + len(MUST_FAIL)} panel molecules"
          + (" and the ZINC sample" if sample is not None else "") + " behaved as expected")
    print("known gap: trans-cycloalkenes in rings <=7 are not caught "
          "(rdkit drops E/Z stereo in small rings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
