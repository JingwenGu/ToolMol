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
    ("bicyclo[2.2.2]ene", "C1CC2CCC1C=C2",                           "fatal"),
    ("cyclic allene",     "C1CCC=C=CC1",                             "fatal"),
    ("1,2-dioxetane",     "C1COO1",                                  "hard"),
    ("geminal diol",      "OC(O)c1ccccc1",                           "hard"),
    ("triazane",          "NNNc1ccccc1",                             "hard"),
]

RANK = {"ok": 0, "hard": 1, "fatal": 2, "invalid": 3}


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
    print()
    if failures:
        for f in failures:
            print("FAILURE:", f)
        return 1
    print(f"all {len(MUST_PASS) + len(MUST_FAIL)} panel molecules behaved as expected")
    print("known gap: trans-cycloalkenes in rings <=7 are not caught "
          "(rdkit drops E/Z stereo in small rings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
