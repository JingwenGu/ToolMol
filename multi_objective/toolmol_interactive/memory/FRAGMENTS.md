# Fragment ledger — what this JNK3 oracle actually rewards

Distillate of `runs/*/oracle_ledger.jsonl` joined against the motif each offspring was
built to carry. jnk3 values are the oracle's own output, so these are measurements, not
predictions.

## The headline result (run seed1, generation 1: 10 blind single-motif probes)

jnk3 does **not** track "is there a kinase hinge heterocycle present". It tracks
**how many (hetero)aryl–NH–(hetero)aryl linkages the molecule has.**

| jnk3 | product | what it carries |
|------|---------|-----------------|
| 0.10 | `Cc1nn(-c2ccccc2)c(C)c1NC(=O)CNc1ccc(-c2ccnc(N)n2)cc1` | anilide NH + aniline NH + biaryl aminopyrimidine → **2 Ar-NH-Ar links** |
| 0.08 | `O=C(NCc1ccccc1)Nc1ccc(Nc2ncccn2)cc1` | urea anilide NH + pyrimidinylamino NH → **2 links** |
| 0.07 | `CC(=O)NCc1cccc(Nc2ncccn2)c1` | pyrimidinylamino NH → **1 link**, MW only 241 |
| 0.07 | `CS(=O)(=O)N1CC(C(=O)Nc2ccc(-c3ccn[nH]3)cc2)C1` | anilide NH + pendant pyrazole → **1 link + azole** |
| 0.06 | `Cc1nn(-c2ccccc2)c(C)c1NC(=O)Cc1c[nH]c2ncccc12` | anilide NH + 7-azaindole |
| 0.03 | `Cc1nn(-c2ccccc2)c(C)c1NC(=O)Cc1ccnc(N)n1` | anilide NH + *bare* aminopyrimidine (no aryl bridge) |
| 0.03 | `CS(=O)(=O)N[C@H]1CCN(c2ccc(-c3nc4ccccc4[nH]3)c(F)c2)C1=O` | benzimidazole biaryl, no NH link |
| 0.02 | `CS(=O)(=O)N1CC(c2ccc3[nH]ncc3c2)C1` | indazole alone |
| 0.01 | `CS(=O)(=O)N1CC(c2ccc3c(c2)[nH]c2ccccc23)C1` | carbazole alone |

## Verdicts

| motif | fragment | n | jnk3 | verdict |
|-------|----------|---|------|---------|
| B 2-anilinopyrimidine | `[1*]Nc1ncccn1` | 2 | 0.07, 0.08 | **best value** — big jnk3 per Dalton, keeps QED ≥ 0.67 |
| M 4-(2-aminopyrimidin-4-yl)anilino | `[1*]Nc1ccc(-c2ccnc(N)n2)cc1` | 1 | 0.10 | **highest jnk3**, but +185 Da ⇒ QED 0.45. Bad Phi. |
| pyrazolylphenyl-anilide | `[1*]C(=O)Nc1ccc(-c2ccn[nH]2)cc1` | 1 | 0.07 | **best Phi partner** — jnk3 0.07 *and* QED 0.875 |
| D 7-azaindol-3-yl | `[1*]c1c[nH]c2ncccc12` | 1 | 0.06 | moderate; costs QED (0.59) |
| A 2-aminopyrimidin-4-yl (bare) | `[1*]c1ccnc(N)n1` | 1 | 0.03 | weak on its own — needs the aryl-NH bridge |
| I benzimidazol-2-yl | `[1*]c1nc2ccccc2[nH]1` | 1 | 0.03 | weak |
| C 1H-indazol-5-yl | `[1*]c1ccc2[nH]ncc2c1` | 1 | 0.02 | **dead on its own** |
| K carbazol-3-yl | `[1*]c1ccc2c(c1)[nH]c1ccccc12` | 1 | 0.01 | **dead on its own** |

## Confirmed-dead on a minimal scaffold
A lone fused aromatic core with no NH linker: **indazole, carbazole**. Matched-scaffold
probes on `CS(=O)(=O)N1CC(X)C1` gave 0.02 and 0.01 — i.e. indistinguishable from random
ZINC. Planarity and "looks like a kinase core" are not what the model is reading.

## Confirmed-live
`Ar-NH-Ar'` where Ar' is an electron-poor azine (pyrimidin-2-yl above all), and
`Ar-C(=O)-NH-Ar'` anilides. Stacking two such links raises jnk3 further (0.07 → 0.08-0.10).

## The Phi trap (important)
jnk3 moves in steps of ~0.03; QED moves in steps of ~0.2. Chasing jnk3 by bolting on mass
**loses**: the 0.10-jnk3 molecule scored Phi 1.404, while a 241 Da molecule with jnk3 0.07
scored Phi 1.829. **Rule: install the NH-azine link, keep MW ≤ ~340 and QED ≥ 0.80.**

---

# Generation 2 (seed1): 10 designed offspring, all scored. This is the real SAR.

| jnk3 | QED | SA | Phi | molecule | design |
|------|-----|----|-----|----------|--------|
| **0.14** | 0.874 | 1.831 | **1.921** | `CC(=O)Nc1ccc(Nc2ccccn2)c(F)c1` | acetanilide + **pyridin-2-ylamino**, ortho-F |
| 0.13 | 0.772 | 1.649 | 1.830 | `O=C(Nc1ccc(Nc2ncccn2)cc1)c1ccccc1` | benzanilide + pyrimidin-2-ylamino |
| 0.10 | 0.845 | 1.742 | 1.862 | `CC(=O)Nc1ccc(Nc2ncccn2)cc1` | acetanilide + pyrimidin-2-ylamino, para |
| 0.10 | 0.845 | 1.818 | 1.854 | `CC(=O)Nc1cccc(Nc2ncccn2)c1` | same, **meta** |
| 0.10 | 0.688 | 1.956 | 1.682 | `O=C(Nc1ccncc1)Nc1ccc(Nc2ncccn2)cc1` | **three** NH links |
| 0.05 | 0.771 | 1.751 | 1.737 | `CC(=O)Nc1ccc(Nc2ncnc3ccccc23)cc1` | quinazolin-4-ylamino (fused) |
| 0.05 | 0.846 | 1.916 | 1.794 | `O=C(Nc1ccc(-c2ccn[nH]2)cc1)C1CC1` | pyrazolyl **biaryl**, no NH-azine |
| 0.03 | 0.761 | 1.723 | 1.711 | `O=C(Nc1ccc(-c2ccn[nH]2)cc1)c1ccccc1` | pyrazolyl biaryl |
| 0.01 | 0.764 | 1.738 | 1.692 | `CC(=O)Nc1ccc(Nc2nc3ccccc3s2)cc1` | benzothiazol-2-ylamino (fused) |
| **0.00** | 0.896 | 2.097 | 1.774 | `CC(=O)Nc1ccc(N(C)c2ncccn2)cc1` | **N-methyl control** |

## Five rules, each from a controlled comparison

1. **The arylamine N–H is required, absolutely.** The N-methyl control is the *identical*
   molecule to the 0.10 acetanilide-pyrimidinylamine except for one hydrogen, and it scores
   **0.00**. Topology alone buys nothing - it is the donor. **Never methylate these NHs.**
   (It does raise QED to 0.896, so it is a tempting and completely worthless move.)
2. **Keep the azine monocyclic.** Fusing a benzo ring onto it destroys activity:
   pyrimidin-2-ylamino 0.10 → quinazolin-4-ylamino 0.05; and benzothiazol-2-ylamino 0.01.
   This is counterintuitive - the anilinoquinazoline is the textbook kinase motif - but this
   oracle does not reward it. Do not spend budget on fused azines again.
3. **2-aminopyridine ≥ 2-aminopyrimidine** (0.14 vs 0.10) and it is cheaper in TPSA, so QED
   is better too. Pyridin-2-ylamino is the best link found.
4. **The anilide NH must be conjugated to its ring.** Direct anilide 0.10 vs CH2-separated
   amide 0.07 on otherwise matched molecules. Meta and para are equivalent (0.10 / 0.10).
5. **A third NH link does not pay** (0.10, same as two) and costs 0.16 of QED. Two is the
   plateau. The acyl cap is nearly free: acetyl 0.10 vs benzoyl 0.13, but benzoyl costs QED.

## The scaffold to start from in any future run

    CC(=O)Nc1ccc(Nc2ccccn2)c(F)c1      jnk3 0.14  QED 0.874  SA 1.83  Phi 1.921

An acetanilide and a 2-aminopyridine linked para across a benzene, one ortho fluorine,
MW 245. Build this (or its close analogues) early rather than rediscovering it: it is
reachable in two `replace_substructure` calls from almost any ZINC molecule that has an
aryl ring with two disposable substituents.

---

# Generation 3 + final call (seed1). This closes the SAR.

| jnk3 | QED | SA | Phi | molecule | variable |
|------|-----|----|-----|----------|----------|
| **0.41** | 0.891 | 1.791 | **2.213** | `Cc1ccnc(Nc2ccc(NC(=O)C3CC3)cc2)c1` | cyclopropyl cap + 4-Me pyridine, **no F** |
| 0.28 | 0.889 | 1.933 | 2.066 | `CC(=O)Nc1ccc(Nc2cc(C)ccn2)c(F)c1` | acetyl + **4-Me pyridine** + F |
| 0.25 | 0.880 | 1.688 | 2.053 | `O=C(Nc1ccc(Nc2ccccn2)cc1)C1CC1` | **cyclopropyl cap**, no F |
| 0.23 | 0.755 | 1.999 | 1.874 | `Fc1cc(Nc2ccccn2)ccc1Nc1ccccn1` | **bis**-pyridin-2-ylamino |
| 0.20 | 0.897 | 1.869 | 2.000 | `O=C(Nc1ccc(Nc2ccccn2)c(F)c1)C1CC1` | cyclopropyl cap + F |
| 0.20 | 0.894 | 1.973 | 1.986 | `CC(=O)Nc1ccc(Nc2ccc(F)cn2)c(F)c1` | 5-F on the pyridine |
| 0.15 | 0.847 | 1.633 | 1.927 | `CC(=O)Nc1ccc(Nc2ccccn2)cc1` | **reference**: acetyl, pyridine, no F |
| 0.12 | 0.885 | 1.869 | 1.909 | `CCC(=O)Nc1ccc(Nc2ccccn2)c(F)c1` | propanamide cap |
| 0.11 | 0.872 | 1.939 | 1.877 | `CC(=O)Nc1ccc(Nc2ncccn2)c(F)c1` | pyrimidine + F |
| **0.00** | 0.847 | 1.664 | 1.774 | `CC(=O)Nc1ccc(Nc2cccnc2)cc1` | **pyridin-3-ylamino** |
| 0.01 | 0.857 | 1.849 | 1.772 | `CC(=O)Nc1ccc(Nc2nccs2)cc1` | **monocyclic** thiazol-2-ylamino |

## The pharmacophore, stated exactly

    aryl – N(H) – azine      with the azine MONOCYCLIC and its ring N ORTHO to that N–H

Each clause is load-bearing, and each was broken on a matched pair:

| break it by | jnk3 |
|---|---|
| nothing (reference) | 0.15 |
| methylating the N–H | **0.00** |
| moving the ring N from ortho to meta (pyridin-3-yl) | **0.00** |
| fusing a benzo ring on (quinazolin-4-yl) | 0.05 |
| using thiazole instead of pyridine, monocyclic | 0.01 |
| using pyrimidine instead of pyridine | 0.10 |

**Correction to an earlier inference:** I had read benzothiazole's failure (0.01) as being
about *fusion*, since quinazoline also failed. The monocyclic 2-aminothiazole also scored
0.01, so the thiazole ring itself is dead and fusion was never the explanation there. Fusion
does still hurt independently (quinazoline 0.05 vs pyrimidine 0.10).

## Decorations, measured as deltas from the 0.15 reference

| change | Δ jnk3 | Δ QED | worth it? |
|---|---|---|---|
| acetyl → **cyclopropanecarboxamide** | **+0.10** | +0.03 | **yes, free** |
| add **4-methyl** to the pyridine | **+0.13** | +0.04 | **yes, free** |
| add 5-fluoro to the pyridine | +0.06 | — | yes |
| acetyl → benzoyl | +0.03 (pyrimidine series) | −0.07 | no |
| acetyl → propanamide | −0.03 | +0.04 | no |
| add **fluorine to the central ring** | **−0.01 to −0.05** | ~0 | **no — this was a red herring** |
| second aryl-NH-pyridine link (bis) | +0.08 | −0.09 | roughly a wash |
| third NH link | 0 | −0.16 | no |

The two "free" decorations compose: cyclopropyl cap **and** 4-methylpyridine, with the
fluorine dropped, took jnk3 from 0.15 to **0.41** while QED went 0.847 → 0.891 and SA
1.633 → 1.791. That combination was built as the 50th and last oracle call.

## Start here next time

    O=C(C1CC1)Nc1ccc(Nc2cc(C)ccn2)cc1     jnk3 0.41  QED 0.891  SA 1.79  Phi 2.213
    (= Cc1ccnc(Nc2ccc(NC(=O)C3CC3)cc2)c1)

N-{4-[(4-methylpyridin-2-yl)amino]phenyl}cyclopropanecarboxamide, MW 253. Two cuts from
almost any ZINC molecule with a disubstituted benzene. Do not re-probe chemotypes.

**Untested directions for pushing past 0.41:** larger 4-alkyl on the pyridine (ethyl,
isopropyl, cyclopropyl); a second pyridine methyl (3,4- or 4,6-dimethyl); cyclobutane- or
cyclopentanecarboxamide caps; 4-methyl on *both* links of the bis-arylamine; and replacing
the anilide with a second methylpyridin-2-ylamino.

---

# Run 2 (seed 2), generation 1: SAR around the run-1 winner. 10 designs, all scored.

Episode 1 rebuilt the run-1 winner from a fresh ZINC host and it scored **0.410 / 0.891 /
1.791 → Phi 2.2134, identical to run 1 to four decimals**. The oracle is deterministic and
the memory transfers exactly. Rebuilding it costs one call and anchors the population with
a Phi-2.21 parent from generation 1 — worth it, since a generation gives no feedback.

| jnk3 | QED | SA | Phi | molecule | variable vs winner |
|------|-----|----|-----|----------|--------------------|
| **0.43** | 0.874 | 1.984 | 2.195 | `O=C(Nc1ccc(Nc2cc(C3CC3)ccn2)cc1)C1CC1` | 4-**cyclopropyl** |
| 0.42 | 0.884 | 1.822 | **2.213** | `COc1ccnc(Nc2ccc(NC(=O)C3CC3)cc2)c1` | 4-**methoxy** |
| 0.41 | 0.891 | 1.791 | **2.213** | `Cc1ccnc(Nc2ccc(NC(=O)C3CC3)cc2)c1` | *(the winner, reproduced)* |
| 0.41 | 0.894 | 1.820 | **2.213** | `Cc1ccnc(Nc2ccc(NC(=O)C3CCC3)cc2)c1` | **cyclobutane**carboxamide cap |
| 0.41 | 0.886 | 1.847 | 2.202 | `Cc1ccnc(Nc2ccc(NC(=O)C3CCCC3)cc2)c1` | **cyclopentane**carboxamide cap |
| 0.36 | 0.878 | 1.855 | 2.143 | `CCc1ccnc(Nc2ccc(NC(=O)C3CC3)cc2)c1` | 4-ethyl |
| 0.34 | 0.738 | 2.013 | 1.966 | `Cc1ccnc(Nc2ccc(Nc3cc(C)ccn3)cc2)c1` | **bis**, both 4-methylated |
| 0.32 | 0.884 | 1.895 | 2.105 | `CC(=O)Nc1ccc(Nc2cc(C(C)C)ccn2)cc1` | 4-isopropyl + acetyl cap |
| **0.14** | 0.891 | 1.782 | 1.944 | `Cc1ccc(Nc2ccc(NC(=O)C3CC3)cc2)nc1` | **5-methyl** instead of 4-methyl |
| 0.06 | 0.694 | 2.536 | 1.584 | `Cc1cc(C)nc(Nc2ccc3c(c2)sc(=S)n3C)c1` | link on a **fused-thione** aryl |

## Four more rules

6. **The pyridine 4-position is position-critical but substituent-insensitive.** Moving the
   methyl from C4 to C5 costs **0.27** (0.41 → 0.14). But at C4, methyl 0.41 ≈ cyclopropyl
   0.43 ≈ methoxy 0.42 > ethyl 0.36 > isopropyl 0.32. Methoxy matching methyl rules out an
   electronic explanation — C4 wants *something compact*, and linear/bulky alkyls are worse.
7. **Acyl cap: any cycloalkyl, ring size irrelevant.** cyclopropyl 0.41 = cyclobutyl 0.41 =
   cyclopentyl 0.41. All beat acetyl. So pick the cap for QED/SA, not activity —
   cyclobutanecarboxamide gave the best QED of the run (0.894).
8. **The bis-arylamine is a dead end for Phi.** Two 4-methylpyridin-2-ylamino links reach
   jnk3 0.34 (up from 0.15 unmethylated) but QED falls to 0.738 → Phi 1.966. The anilide cap
   beats a second azine link. Stop testing bis.
9. **The pharmacophore is context-dependent.** Installed on an aryl ring fused into a
   thiazole-2-thione it scored **0.06** despite having the link. It needs a plain,
   unremarkable benzene — do not try to graft it onto a fused heterocyclic host.

## Current frontier
Three molecules tie at **Phi 2.2134**; best jnk3 is 0.43. Phi is QED/SA-limited again
(QED caps ~0.89, SA floor ~1.79 ⇒ ceiling ≈ 1.80 + jnk3). Pushing past 0.43 is the only
real lever.

---

# Run 2, generation 2: the plateau breaks, and two of my own rules were wrong.

| jnk3 | QED | SA | Phi | molecule | variable |
|------|-----|----|-----|----------|----------|
| **0.50** | 0.891 | 1.856 | **2.296** | `Cc1ccnc(Nc2cccc(NC(=O)C3CC3)c2)c1` | **anilide META, not para** |
| 0.44 | 0.896 | 1.915 | 2.234 | `Cc1ccnc(Nc2ccc(NC(=O)C3CC3)c(C)c2)c1` | **methyl on the central ring** |
| 0.42 | 0.898 | 1.865 | 2.222 | `O=C(Nc1ccc(Nc2cc(Cl)ccn2)cc1)C1CC1` | 4-Cl |
| 0.42 | 0.858 | 2.017 | 2.165 | `O=C(Nc1ccc(Nc2cc(C3CCC3)ccn2)cc1)C1CC1` | 4-cyclobutyl |
| 0.41 | **0.905** | 2.049 | 2.199 | `Cc1ccnc(Nc2ccc(NC(=O)C3COC3)cc2)c1` | oxetane-3-carboxamide cap |
| 0.19 | 0.894 | 2.057 | 1.967 | `Cc1ccnc(Nc2ccc(NC(=O)C3CC3)cn2)c1` | central ring → pyridine |
| 0.11 | 0.902 | 2.027 | 1.898 | `Cc1ccnc(Nc2ccc(NC(=O)C3CC3)cc2)c1F` | **3-F on the pyridine** |
| 0.08 | 0.896 | 1.873 | 1.879 | `Cc1cc(C)nc(Nc2ccc(NC(=O)C3CC3)cc2)c1` | **4,6-dimethyl** |
| 0.08 | 0.863 | 2.084 | 1.822 | `Cc1cc(C2CC2)cc(Nc2ccc(NC(=O)C3CC3)cc2)n1` | 4-cPr + **6-Me** |
| 0.05 | 0.896 | 1.942 | 1.841 | `Cc1ccnc(Nc2ccc(NC(=O)C3CC3)cc2)c1C` | **3,4-dimethyl** |

## Rules 10-14

10. **Pyridine C3 and C6 must be bare.** Any substituent on either collapses activity:
    3-Me 0.05, 3-F 0.11, 6-Me 0.08, 4-cPr+6-Me 0.08 — all from 0.41. C3 is adjacent to the
    bridging N-H and C6 is adjacent to the ring N, so the hinge donor/acceptor pair needs
    both of its neighbours clear. **Only C4 is productive; C5 is harmful (0.14); C3/C6 are
    fatal.** This is now a complete position scan of the ring.
11. **The anilide belongs META to the arylamine, not para: 0.50 vs 0.41.** This is the single
    biggest gain found in run 2 and it *overturns a rule I wrote in run 1.* I had measured
    meta ≈ para (0.10 vs 0.10) and concluded placement was free — but that comparison was
    made at jnk3 ≈ 0.10, where a 0.09 difference is invisible. **Never generalize a null
    result from a low-signal regime; re-test geometry once activity is high.**
12. **Substituting the central ring is fine — fluorine specifically is not.** A methyl ortho
    to the anilide gave +0.03 (0.41 → 0.44). In run 1 a *fluorine* there cost 0.01-0.05 and I
    wrongly generalized that to "keep the central ring bare".
13. **The central ring must be a benzene.** Making it a pyridine costs 0.22 (0.41 → 0.19),
    even though it improves LogP.
14. **C4 is completely substituent-insensitive among compact groups**: Me 0.41, Cl 0.42,
    OMe 0.42, cPr 0.43, cBu 0.42. Pick it for QED/SA. Likewise the cap: oxetan-3-yl 0.41 =
    cyclopropyl = cyclobutyl = cyclopentyl, and oxetane gives the best QED seen (0.905).

## Two corrected generalizations — worth remembering as a failure mode
Both of my wrong rules came from the same mistake: generalizing from a single substituent
(fluorine) or from a comparison made when the signal was near zero. When a probe comes back
null, check whether the assay had the resolution to see the effect before writing a rule.

---

# Run 2, generation 3: geometry scan completed, and additivity fails.

| jnk3 | QED | SA | Phi | molecule | design |
|------|-----|----|-----|----------|--------|
| **0.52** | 0.889 | 1.951 | **2.303** | `O=C(Nc1cccc(Nc2cc(Cl)ccn2)c1)C1CCC1` | meta, **bare centre**, 4-Cl, cyclobutyl |
| 0.47 | 0.892 | 1.985 | 2.253 | `Cc1ccc(Nc2cc(Cl)ccn2)cc1NC(=O)C1CC1` | meta + central Me + 4-Cl |
| 0.46 | 0.886 | 1.928 | 2.243 | `COc1ccc(Nc2cc(C)ccn2)cc1NC(=O)C1CC1` | meta + central OMe |
| 0.44 | 0.896 | 1.914 | 2.235 | `Cc1ccnc(Nc2ccc(C)c(NC(=O)C3CC3)c2)c1` | meta + central Me at C4 |
| 0.43 | 0.889 | 1.939 | 2.215 | `Cc1ccnc(Nc2ccc(C)c(NC(=O)C3CCC3)c2)c1` | meta + central Me + cyclobutyl |
| 0.41 | 0.892 | 1.945 | 2.197 | `Cc1ccnc(Nc2ccc(Cl)c(NC(=O)C3CC3)c2)c1` | meta + central **Cl** |
| 0.35 | 0.896 | 2.018 | 2.133 | `Cc1cc(NC(=O)C2CC2)cc(Nc2cc(C)ccn2)c1` | meta + central Me at **C5** |
| 0.34 | 0.896 | 1.927 | 2.133 | `Cc1ccnc(Nc2cc(NC(=O)C3CC3)ccc2C)c1` | meta + central Me at **C6** |
| 0.32 | 0.896 | 1.987 | 2.106 | `Cc1ccnc(Nc2cccc(NC(=O)C3CC3)c2C)c1` | meta + central Me at **C2** |
| 0.31 | 0.891 | 1.909 | 2.100 | `Cc1ccnc(Nc2ccccc2NC(=O)C2CC2)c1` | **ortho** geometry |

## Rules 15-17

15. **Geometry scan complete: meta (0.50) > para (0.41) > ortho (0.31).** The 1,3
    relationship between the arylamine and the anilide is optimal; ortho is worst, probably
    because the carbonyl chelates the adjacent N-H and quenches the donor.
16. **The two gains are ANTAGONISTIC, not additive.** A central-ring methyl is worth +0.03
    on the *para* core (0.41 → 0.44) but **−0.06 on the meta core** (0.50 → 0.44). The full
    methyl scan of the meta central ring - C2 0.32, C4 0.44, C5 0.35, C6 0.34 - is *below*
    the bare meta ring at 0.50 in every position. **On the meta core the central ring must
    be bare.** Do not assume effects measured on one scaffold transfer to a better one;
    re-measure them there.
17. **4-Cl is marginally the best C4 group** (0.42 vs 0.41 para; 0.52 vs 0.50 meta) and also
    has the best QED of the set. Cap: all of cyclopropyl / cyclobutyl / cyclopentyl /
    oxetan-3-yl are activity-identical; cyclobutyl has the best QED (0.894), cyclopropyl the
    best SA (1.79). Use cyclobutyl unless SA is binding.

## Complete position map (both rings)

                     pyridine                          central benzene
    C3  fatal (0.05-0.11)                 arylamine ── anilide: meta best, para -0.09,
    C4  productive, substituent-agnostic                        ortho -0.19
    C5  harmful (0.14)                    any substituent anywhere: harmful on the
    C6  fatal (0.08)                                      meta core

## START HERE in any future run

    O=C(Nc1cccc(Nc2cc(Cl)ccn2)c1)C1CCC1     jnk3 0.52  QED 0.889  SA 1.95  Phi 2.3031

N-{3-[(4-chloropyridin-2-yl)amino]phenyl}cyclobutanecarboxamide, MW 302. Reachable in ONE
tool call from most ZINC molecules - see STRATEGY.md §2 "one-call pharmacophore graft".

**Untested after two runs:** 4-Br/4-CN/4-CF3 at pyridine C4; 5- or 6-membered azines other
than pyridine with the N ortho and a C4 substituent (pyridazin-3-yl, pyrimidin-4-yl);
replacing the anilide carbonyl with a sulfonyl or thioamide; a meta-core molecule with a
*second* ring fused onto the cap; and whether anything at all beats jnk3 0.52 on this
chemotype - two runs have now plateaued in the 0.50-0.52 band.

---

# Run 3 (seed 3): the azine/linker scans, a full decomposition, and C4 corrected again.

## Generation 1 — alternative azines and linkers (baseline for these: ~0.50)

| jnk3 | molecule | variable |
|------|----------|----------|
| 0.44 | `Cc1cc(Nc2cccc(NC(=O)C3CCC3)c2)ncn1` | pyrimidin-4-yl (2nd N meta to donor) |
| 0.40 | `Cc1ccnc(Nc2cccc(NC(=S)C3CCC3)c2)c1` | **thioamide** linker |
| 0.37 | `Cc1ccnc(Nc2cccc(NS(=O)(=O)C3CC3)c2)c1` | **sulfonamide** linker |
| 0.37 | `Cc1ccnc(Nc2cccc(NC(=O)NC3CCC3)c2)c1` | **urea** linker |
| 0.36 | `Cc1ccnc(Nc2cccc(C(=O)NC3CCC3)c2)c1` | **reversed** amide |
| 0.43 | `Cc1ccnc(Nc2cccc(NC(=O)c3ccccn3)c2)c1` | picolinamide cap (2nd ortho-N azine) |
| 0.29 | `Cc1cnc(Nc2cccc(NC(=O)C3CCC3)c2)cn1` | pyrazin-2-yl (2nd N para) |
| 0.25 | `Cc1cnnc(Nc2cccc(NC(=O)C3CCC3)c2)c1` | pyridazin-3-yl (2nd N adjacent) |
| 0.23 | `Cc1ccnc(Nc2cccc(NC(=O)C3CCC3)c2)n1` | pyrimidin-2-yl (N's flanking) |

**Rule 18 — exactly one ring nitrogen, ortho to the donor.** Every diazine loses, and the
closer the second nitrogen sits to the hinge the worse: meta 0.44 > para 0.29 > adjacent
0.25 > flanking 0.23. The azine scan is complete; pyridine is uniquely right.

**Rule 19 — the anilide is optimal but not required.** Every linker variant survives at
0.36-0.40 against ~0.50. So the linker is a real contributor (~0.13) rather than a pure
spacer, but nothing about it is load-bearing. A second ortho-N azine on the *cap* adds
nothing (0.43): the motif is not additive, wherever it is placed.

## Generation 2 — the decomposition (this is the most useful block in this file)

| jnk3 | what was removed or blocked | cost |
|------|------------------------------|------|
| 0.53 | *(reference: 4-Cl, meta, cyclopropyl)* | — |
| 0.42 | the acyl cap entirely (free aniline) | **−0.11** |
| 0.36 | the anilide N–H, methylated | **−0.17** |
| 0.04 | the azine **ring nitrogen** (plain diarylamine) | **−0.49** |
| 0.00 | the arylamine N–H, methylated *(run 1)* | **−0.53** |

**Rule 20 — the pharmacophore is two atoms.** A free arylamine N–H and a ring nitrogen
ortho to it. Remove either and activity is gone (0.00 / 0.04). Everything else — the
anilide, the cap, the geometry, the C4 substituent — is a modifier worth 0.1-0.2 each.

Also from generation 2: a remote ring nitrogen in the **central** ring still costs 0.17
(0.33), so that ring must be a benzene throughout, not just near its attachments. A
quaternary cap attachment carbon costs 0.10. The bis-arylamine loses on the meta core too
(0.38). The N-methyl anilide posts the highest QED ever recorded here, 0.932 — and 0.36 jnk3.

## Generation 3 — the C4 series, and a third correction

| jnk3 | C4 substituent | | jnk3 | C4 substituent |
|------|----------------|---|------|----------------|
| **0.60** | **Br** (cyclopropyl cap) | | 0.53-0.54 | OMe |
| 0.59 | Br (cyclobutyl cap) | | 0.50-0.53 | Cl |
| **0.58** | **CN** (cyclobutyl cap) | | 0.53-0.54 | F |
| 0.53 | CN (oxetane cap) | | 0.49 | Me |

**Rule 21 — C4 is NOT substituent-insensitive, and this is the third time a rule of mine
formed at low resolution turned out to be wrong.** Run 2 measured Me/Cl/OMe/cPr/cBu all
within 0.41-0.43 on the *para* core and I wrote "substituent-agnostic, pick it for QED".
On the meta core at double the activity the same position spreads 0.49 → 0.60. Bromine and
nitrile — the two largest/most polarizable groups tried — are the best, and the series had
**not converged when the budget ran out**.

## START HERE in any future run

    O=C(Nc1cccc(Nc2cc(Br)ccn2)c1)C1CC1    jnk3 0.60  QED 0.892  SA 2.01  Phi 2.3799

N-{3-[(4-bromopyridin-2-yl)amino]phenyl}cyclopropanecarboxamide, MW 346. One tool call from
most hosts. Runner-up and a good second parent: the 4-CN cyclobutyl analogue, Phi 2.366.

**Untested, and this is now the live lead:** the C4 series is still climbing. Try **iodine,
SMe, SCN, ethynyl, nitro, and acetylenic/extended EW groups at C4**, and Br or CN combined
with the oxetane cap (best QED, 0.911). Nothing else on this chemotype is open — both rings
and the linker are fully mapped.

---

# Run 4 (seed 4): C4 fully characterised, the ceiling broken, and a new conflict.

## Generation 1 — ten C4 substituents, every electronic class (cap held constant)

| jnk3 | C4 group | class | | jnk3 | C4 group | class |
|------|----------|-------|---|------|----------|-------|
| 0.61 | CF3 | strong withdrawer | | 0.57 | I | weak withdrawer, huge |
| 0.61 | NHMe | strong donor | | 0.56 | SMe | donor, polarisable |
| 0.60 | Br *(ref)* | weak withdrawer | | 0.56 | C(=O)CH3 | withdrawer |
| 0.59 | C≡CH | withdrawer, linear | | 0.53 | CO2Me | withdrawer |
| 0.58 | C≡C-CH3 | withdrawer, linear | | 0.51 | SO2Me | strong withdrawer |

**Rule 22 — C4 reads occupancy, not electronics.** Trifluoromethyl and methylamino are at
opposite electronic extremes and both score 0.61; the spread *within* the withdrawing groups
alone (0.51-0.61) is as large as the spread between withdrawers and donors. Against methyl
0.49 and fluorine 0.53, what every good group has in common is simply being bigger. Stop
treating that position as an electronic handle.

## Generations 2-3 — a flat aromatic at C4 breaks the activity ceiling

| jnk3 | QED | Phi | C4 ring |
|------|-----|-----|---------|
| **0.74** | 0.746 | 2.344 | pyridazin-4-yl |
| **0.74** | 0.708 | 2.311 | 1-methylpyrazol-4-yl *(cyclopentyl cap)* |
| 0.72 | 0.746 | 2.333 | pyrimidin-5-yl |
| 0.71 | 0.747 | 2.300 | isoxazol-4-yl |
| 0.71 | 0.740 | 2.312 | pyridin-3-yl *(oxetane cap)* |
| 0.70 | 0.623 | 2.220 | **phenyl** *(the original finding, cyclopentyl cap)* |
| 0.70 | 0.747-0.749 | 2.303 | oxazol-5-yl, 1-methylimidazol-4-yl |
| 0.68 | 0.734 | 2.299 | pyridin-4-yl |
| 0.66 | 0.749 | 2.262 | 1-methyl-1,2,3-triazol-4-yl |
| 0.62 | 0.887 | **2.391** | *CF3 — best Phi of the project* |

**Rule 23 — an aryl at C4 is worth +0.12 to +0.14 of jnk3.** Ten compact substituents capped
at 0.61; eleven aromatic rings in the same position run 0.66-0.74. The size ceiling I inferred
in run 3 was an artefact of only ever trying compact groups — the third time an apparent
ceiling turned out to be the edge of what I had sampled.

**Rule 24 — and it is a wash on Phi, for a reason worth knowing.** *Every* aryl-at-C4 molecule
lands at QED 0.71-0.75 no matter how polar the ring: phenyl (LogP > 5) and methyltriazolyl
(LogP ~2.5) differ by two and a half log units and score 0.623 and 0.749. The penalty is not
lipophilicity — it is the **third aromatic ring itself**, which QED's aromatic-ring term
penalises directly. So +0.13 of activity buys −0.14 of QED, plus a worse SA, and Phi does not
move. Polarising the ring is not a fix; nothing within the aryl series is.

## Best molecules after four runs

    O=C(Nc1cccc(Nc2cc(C(F)(F)F)ccn2)c1)C1CC1   jnk3 0.62 QED 0.887 SA 2.05  **Phi 2.3906**
    O=C(Nc1cccc(Nc2cc(Br)ccn2)c1)C1CC1         jnk3 0.60 QED 0.892 SA 2.01   Phi 2.3799
    O=C(Nc1cccc(Nc2cc(-c3ccnnc3)ccn2)c1)C1CC1  jnk3 0.74 QED 0.746 SA 2.28   Phi 2.3440

**The live lead for run 5.** The two objectives are now in direct conflict at C4: activity
wants a flat ring there, QED forbids a third aromatic ring. The question is whether anything
delivers aryl-like bulk *without* being aromatic — a bicyclo[1.1.1]pentyl, cubyl, cyclohexenyl,
spiro or fused saturated system, an isopropenyl or a 1,3-butadienyl. A cyclopropyl at C4 only
reached 0.57 and a cyclobutyl was never tried, so the saturated-bulk series is barely sampled.
If one of them reaches 0.70 while keeping the aromatic count at two, Phi clears 2.45.
