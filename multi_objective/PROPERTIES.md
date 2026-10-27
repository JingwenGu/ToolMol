# The ten C-MuMO properties

Written 2026-10-26, moved into the arm 2026-10-27. Implements section 5 of
`../../CMUMO_PORT_DESIGN.md`. **Oracles only** — no GA, no agent, no run.

## Where things are

```
cmumo_oracles/properties.py   the batched + cached evaluator (runs in the toolmol venv)
cmumo_oracles/admet_worker.py persistent ADMET-AI worker (runs in ../.venv-admet)
cmumo_oracles/paths.py        every path that points outside the committed tree
calibration/                  calibrate_m.py and the sweep's results
oracle/ -> ../assets/oracle   drd2_current.pkl, fpscores.pkl  (linked, gitignored)
data/   -> ../assets/data     zinc.tab                        (linked, gitignored)
```

Two environments, because they cannot be one: the GA is python 3.9 and admet-ai needs
torch. They talk over a pipe. `../assets/property_cache.jsonl` is a cache, not a record —
safe to delete, rebuilt at ~4 ms/molecule.

## Check it works

```bash
cd AI4S/cmumo/cmumo_objectives_base
python setup_assets.py                                    # once, after a clone
../../RSI/.venv-toolmol39/bin/python -m cmumo_oracles.properties
```

That runs `self_check()`, which asserts the specs still match the paper's Table 2 and that
every property discriminates across four very different molecules. **Run it before any
run.** The reason is the first trap below.

## Four traps, each of which cost something to find

**1. `tdc.Oracle('DRD2')` returns 0.0 when its model fails to load.** TDC's download gave a
199-byte **403 HTML page**, the unpickle raised `invalid load key, '<'`, and the oracle
answered `0.0` anyway. A constant objective does not crash a run — it silently removes an
axis, and the run looks healthy. The fix was copying the real 35 MB pickle from
`MOLLEO/multi_objective/oracle/`. The guard is `self_check()`, which fails on any axis
whose spread across the probe set is below 1e-6. Same family as the credit-exhaustion 400.

**2. chemprop/lightning print to stdout, which is the worker's protocol channel.** The
symptom is the client failing to parse the handshake, not an obvious crash. The worker now
dups fd 1 to a private handle and points fd 1 at fd 2; the client sends the worker's stderr
to `admet_worker.log`. Anything added to the worker must use `send()`, never `print`.

**3. ADMET-AI emits a `*_drugbank_approved_percentile` twin for every property.** Those are
ranks against DrugBank, not predictions. `admet_worker.WANT` names raw columns only.

**4. A one-molecule batch predicts nothing.** chemprop drops the last batch when it has
size 1 ("to avoid issues with batch normalization"). The warm-up uses two molecules. Worth
remembering if anything ever calls `predict` with a single SMILES.

## The property table, measured on ZINC

200 random ZINC-250K molecules, seed 0. "compliant" = already on the good side of C-MuMO's
Th_p, i.e. the fraction that would sit on the plateau if that property were a maintain axis.

| prop | dir | Th_p | D_p | min | median | max | compliant | source |
|---|---|---|---|---|---|---|---|---|
| AMP | + | 0.80 | 0.10 | 0.072 | 0.920 | 0.993 | 71.5% | ADMET-AI `PAMPA_NCATS` |
| BBBP | + | 0.80 | 0.10 | 0.039 | 0.838 | 0.991 | 54.5% | ADMET-AI `BBB_Martins` |
| CARC | − | 0.20 | 0.20 | 0.014 | 0.136 | 0.470 | 71.0% | ADMET-AI `Carcinogens_Lagunin` |
| DRD2 | + | 0.40 | 0.10 | 0.000 | 0.002 | 0.096 | **0.0%** | tdc `DRD2` |
| HIA | + | 0.40 | 0.10 | 0.005 | 1.000 | 1.000 | **91.0%** | ADMET-AI `HIA_Hou` |
| LIV | − | 0.90 | 0.10 | 0.000 | 0.712 | 0.992 | 76.0% | ADMET-AI `DILI` |
| MUT | − | 0.20 | 0.10 | 0.014 | 0.273 | 0.970 | 40.0% | ADMET-AI `AMES` |
| PlogP | + | 1.50 | 1.00 | −6.657 | 0.573 | 3.381 | 27.5% | rdkit, normalized |
| QED | + | 0.90 | 0.10 | 0.352 | 0.768 | 0.941 | **5.0%** | rdkit |
| hERG | − | 0.30 | 0.20 | 0.010 | 0.563 | 0.981 | 20.0% | ADMET-AI `hERG` |

**What this says about improve-vs-maintain.** In C-MuMO, Th_p is a *per-instance
classifier* — it decides, for one input molecule, whether a property gets improved or
maintained. Pooling the run (design section 1) turns it into a population-level target,
and the occupancy column is then the thing that decides whether an axis can be a maintain
axis at all:

- Usable as **maintain**: HIA (91%), LIV (76%), AMP (71.5%), CARC (71%), BBBP (54.5%).
- Not usable as maintain: **QED (5%)** and **DRD2 (0%)** would penalise essentially the
  whole seed pool, which is not what "maintain" means. They are improve axes or nothing.
- **HIA is near-inert**: median 1.000, and a spread of 0.0017 across the self-check probes.
  It saturates for drug-like input, so it buys an axis and contributes almost no signal.
  `self_check()` now warns below a spread of 0.01.
- **DRD2 on random ZINC is a needle in a haystack** (median 0.002, max 0.096 over 200).
  As an improve axis it behaves like jnk3 does in the other arms, which is worth knowing
  before reading any early generation as failure.

## Both open items resolved against the PDF  [2026-10-26]

Checked against the published PDF (arXiv:2505.23987), Table 2 and section 2. **No spec
changed - both were already right.**

**The threshold/tolerance table matches exactly.** Table 2's Delta row
(0.1 0.1 0.2 0.1 0.2 0.1 0.1 0.1 1.0 0.1) and Theta row
(0.8 0.8 0.2 0.4 0.3 0.4 0.9 0.2 1.5 0.9) over
AMP BBBP CARC DRD2 hERG HIA LIV MUT PlogP QED are what `SPECS` holds, and the direction
row maps as AMP+ BBBP+ CARC- DRD2+ hERG- HIA+ LIV- MUT- PlogP+, with QED+ stated in prose.

**LIV is genuinely lower-is-better with Theta 0.9**, not a transcription artifact. It looks
trivially satisfiable because the thresholds are the 60th percentile of *their* 256K
training pairs, where DILI predictions skew high.

**PlogP is the normalized form.** The paper adopts "the official implementation from You
et al. (2018)" for DRD2 and PlogP, which is GCPN's `reward_penalized_log_p` - the
normalized version with exactly the three mean/std constants we use. One immaterial
deviation: GCPN takes the largest ring from networkx `cycle_basis`, we take it from RDKit's
SSSR; these differ only on some fused polycyclics.

**And one thing the check established that is worth more than either.** The "60th
percentile" calibration does not transfer. If it did, every property would be ~40%
compliant on any drug-like pool; measured on 1000 ZINC molecules, compliance runs from 0%
(DRD2) to 88% (HIA), and only MUT lands at 40%. So the occupancy column above is a fact
about ZINC, not about their thresholds - which is exactly why the improve/maintain
assignment had to be measured rather than read off the paper.

## One version difference to record with any result

C-MuMO used ADMET-AI as published in 2024 (admet-ai 1.x, chemprop 1.x). This is **admet-ai
2.0.1 with chemprop 2.3.1**, so the checkpoints are not guaranteed identical to theirs.
Our numbers are internally consistent and are not comparable to values printed in their
paper, which the pooled design already rules out for other reasons.

## Calibration

`calibrate_m.py` runs the M / C sweep (readouts 2 and 3 of design section 2) against a
ZINC pool. No agent calls, no GA.

```bash
../../RSI/.venv-toolmol39/bin/python calibration/calibrate_m.py --n 1000           # full sweep
../../RSI/.venv-toolmol39/bin/python calibration/calibrate_m.py --n 1000 --chosen  # chosen only
```

Results and the decisions they produced are in `../../CMUMO_PORT_DESIGN.md` section 2;
raw data in `calibration/calibration.json` (sweep) and `calibration/chosen_config.json`. Short version:
**C = 5** and insensitive, **alpha = 1** for BBBP/AMP/MUT/PlogP, **alpha = 0.25** for
CARC and LIV (D_p = D_p makes them inert), **HIA dropped**, and the population cap is a
safety valve at 40 rather than a tuning knob - fronts came out at 6-10 of 95, not the
80-95 that was feared.

Two things to carry into any result: a small maintain share of Var(Phi) is ambiguous
between "compliant and flat" and "too soft to bite", which is why the script reports mean
and active fraction per axis; and DRD2 carries a variance share of 0.001 on a ZINC seed
pool, so it is a cold-start needle objective exactly like jnk3.

## Throughput

4 ms/molecule in-process for a batch of 100; 35 ms/molecule end-to-end for 200 including
the ~5 s model load. The load is paid once per run because the worker is persistent. With
the cache, a re-run pays nothing for molecules already seen.
