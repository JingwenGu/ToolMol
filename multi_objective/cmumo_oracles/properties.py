"""The ten C-MuMO properties, as a batched + cached evaluator for the GA side.

Runs in RSI/.venv-toolmol39 (python 3.9) alongside the GA. Eight of the ten properties
come from ADMET-AI, which cannot live in that env, so they arrive over a pipe from
`admet_worker.py` running in cmumo/.venv-admet. DRD2 comes from tdc, and QED and PlogP are
computed with rdkit.

Three things this module exists to prevent:

1. **Per-molecule neural-net calls.** `Oracle.evaluate()` is called on the whole population
   every generation by the parent sampler, and again by the population cap, so an
   un-batched un-cached ADMET-AI call would be the dominant cost of a run. `prefetch()`
   takes a list; `get()` is then a dict lookup.
2. **Silent constant objectives.** `tdc.Oracle('DRD2')` swallows a failed model load and
   returns **0.0 for every molecule** - we hit exactly this: TDC's download returned a
   403 HTML page, the unpickle raised, and the oracle still answered 0.0. A constant
   objective does not crash a run, it quietly removes an axis. `self_check()` asserts
   every property discriminates on a known pair and must be called before any run.
3. **Reading a percentile as a prediction.** ADMET-AI emits a
   *_drugbank_approved_percentile twin for every property. Those are ranks against
   DrugBank, not predictions. `admet_worker.WANT` selects raw columns only.

Cache: in-memory dict keyed on canonical SMILES, optionally backed by an append-only
jsonl so a re-run or a resume pays nothing for molecules already seen.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, QED, RDConfig

RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

from cmumo_oracles import paths

_HERE = os.path.dirname(os.path.abspath(__file__))

ADMET_PYTHON = paths.ADMET_PYTHON
ADMET_WORKER = os.path.join(_HERE, "admet_worker.py")


# --- the property table ------------------------------------------------------------
# direction: "+" = higher is better, "-" = lower is better.
# theta: C-MuMO's pharmaceutical threshold Th_p (60th percentile of their training set).
# delta: C-MuMO's tolerance D_p.
# source: "admet" (over the pipe) | "tdc" | "rdkit".
#
# VERIFIED 2026-10-26 against the published PDF, Table 2 (arXiv:2505.23987). Every theta
# and delta below matches that table exactly, and the direction row maps as
# AMP+ BBBP+ CARC- DRD2+ hERG- HIA+ LIV- MUT- PlogP+ (QED+ stated in prose, section 2).
#
# LIV is NOT a transcription artifact: it really is lower-is-better with theta 0.9. That
# looks trivially satisfiable because the thresholds are the 60th percentile of THEIR 256K
# training pairs, and DILI predictions skew high there. The rule does not transfer to ZINC
# at all - if it did, every property would be ~40% compliant on any drug-like pool, and
# measured compliance on 1000 ZINC molecules ranges from 0% (DRD2) to 88% (HIA), with only
# MUT landing at 40%. Which is why occupancy had to be measured; see cmumo/README.md.
SPECS = {
    "AMP":   {"direction": "+", "theta": 0.8, "delta": 0.1, "source": "admet"},
    "BBBP":  {"direction": "+", "theta": 0.8, "delta": 0.1, "source": "admet"},
    "CARC":  {"direction": "-", "theta": 0.2, "delta": 0.2, "source": "admet"},
    "DRD2":  {"direction": "+", "theta": 0.4, "delta": 0.1, "source": "tdc"},
    "hERG":  {"direction": "-", "theta": 0.3, "delta": 0.2, "source": "admet"},
    "HIA":   {"direction": "+", "theta": 0.4, "delta": 0.1, "source": "admet"},
    "LIV":   {"direction": "-", "theta": 0.9, "delta": 0.1, "source": "admet"},
    "MUT":   {"direction": "-", "theta": 0.2, "delta": 0.1, "source": "admet"},
    "PlogP": {"direction": "+", "theta": 1.5, "delta": 1.0, "source": "rdkit"},
    "QED":   {"direction": "+", "theta": 0.9, "delta": 0.1, "source": "rdkit"},
}

ALL_PROPERTIES = sorted(SPECS)


# Table 2 of the published PDF, transcribed and frozen. SPECS is the live table and this is
# the reference it is checked against, so a later edit to a theta or a direction cannot
# drift away from the paper unnoticed. Asserted by self_check().
PDF_TABLE2 = {
    #        direction, theta, delta
    "AMP":   ("+", 0.8, 0.1),
    "BBBP":  ("+", 0.8, 0.1),
    "CARC":  ("-", 0.2, 0.2),
    "DRD2":  ("+", 0.4, 0.1),
    "hERG":  ("-", 0.3, 0.2),
    "HIA":   ("+", 0.4, 0.1),
    "LIV":   ("-", 0.9, 0.1),
    "MUT":   ("-", 0.2, 0.1),
    "PlogP": ("+", 1.5, 1.0),
    "QED":   ("+", 0.9, 0.1),
}


def check_against_paper():
    """Every theta, delta and direction against Table 2. Raises on any drift."""
    bad = []
    for name, (direction, theta, delta) in PDF_TABLE2.items():
        spec = SPECS.get(name)
        if spec is None:
            bad.append(f"{name}: missing from SPECS")
            continue
        for field, ref in (("direction", direction), ("theta", theta), ("delta", delta)):
            if spec[field] != ref:
                bad.append(f"{name}.{field}: SPECS has {spec[field]!r}, paper has {ref!r}")
    if bad:
        raise RuntimeError(
            "SPECS no longer matches C-MuMO Table 2 (arXiv:2505.23987):\n  "
            + "\n  ".join(bad))
    return True


# --- rdkit-computed properties -----------------------------------------------------
# RESOLVED 2026-10-26. The paper states it adopts "the official implementation from You
# et al. (2018) for computing DRD2 and PlogP scores", and describes PlogP as "logP
# penalized by synthetic accessibility score and number of large rings". That
# implementation is GCPN's `reward_penalized_log_p`, which is the NORMALIZED form using
# exactly the three mean/std constants below - so this matches the cited code, and the
# scale the calibrated theta/M live on is the right one.
#
# One deviation, immaterial for most molecules but worth knowing: GCPN finds the largest
# ring with networkx `cycle_basis` on the molecular graph, while `_largest_ring_excess`
# below uses RDKit's SSSR (`GetRingInfo().AtomRings()`). The two agree except on some
# fused polycyclics, where the chosen ring set can differ.
_LOGP_MEAN, _LOGP_STD = 2.4570953396190123, 1.434324401111988
_SA_MEAN, _SA_STD = -3.0525811293166134, 0.8335207024513095
_CYCLE_MEAN, _CYCLE_STD = -0.0485696876403053, 0.2860212110245455


def _largest_ring_excess(mol):
    rings = mol.GetRingInfo().AtomRings()
    if not rings:
        return 0
    return max(0, max(len(r) for r in rings) - 6)


def penalized_logp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return -100.0
    logp = (Crippen.MolLogP(mol) - _LOGP_MEAN) / _LOGP_STD
    sa = (-sascorer.calculateScore(mol) - _SA_MEAN) / _SA_STD
    cycle = (-_largest_ring_excess(mol) - _CYCLE_MEAN) / _CYCLE_STD
    return float(logp + sa + cycle)


def qed(smi):
    mol = Chem.MolFromSmiles(smi)
    return 0.0 if mol is None else float(QED.qed(mol))


# --- the ADMET-AI client -----------------------------------------------------------
class AdmetClient:
    """Persistent worker over a pipe. Lazily started, so importing this module costs
    nothing and a run that touches no ADMET property never loads torch."""

    def __init__(self, python=ADMET_PYTHON, worker=ADMET_WORKER, log_path=None):
        self.python = python
        self.worker = worker
        self.log_path = log_path or paths.WORKER_LOG
        self.proc = None
        self._log = None

    def start(self):
        if self.proc is not None:
            return
        if not os.path.exists(self.python):
            raise FileNotFoundError(
                f"ADMET-AI interpreter not found at {self.python!r}. Run "
                f"`python setup_assets.py` for the exact command, or set "
                f"CMUMO_ADMET_PYTHON.")
        # The worker points its own fd 1 at fd 2, so everything chemprop prints lands in
        # this log and only protocol lines come back on stdout.
        self._log = open(self.log_path, "a")
        self.proc = subprocess.Popen(
            [self.python, "-u", self.worker],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, bufsize=1)
        ready = self.proc.stdout.readline()
        if not ready:
            raise RuntimeError(
                f"ADMET-AI worker died before signalling ready; see {self.log_path}")
        payload = json.loads(ready)
        if not payload.get("ready"):
            raise RuntimeError(f"unexpected worker handshake: {payload!r}")

    def predict(self, smis):
        """smis -> {smi: {property: value}}. One round trip for the whole list."""
        if not smis:
            return {}
        self.start()
        self.proc.stdin.write(json.dumps({"smiles": list(smis)}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("ADMET-AI worker closed the pipe mid-request")
        out = json.loads(line)
        if not out.get("ok"):
            raise RuntimeError(f"ADMET-AI worker error: {out.get('error')}")
        return out["values"]

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        finally:
            self.proc = None
            if self._log is not None:
                self._log.close()
                self._log = None


# --- the evaluator -----------------------------------------------------------------
class PropertyEvaluator:
    """Batched, cached access to the ten properties.

    Usage from the GA:

        ev = PropertyEvaluator(["QED", "hERG", "MUT"], cache_path=".../cache.jsonl")
        ev.self_check()                 # once, before the run
        ev.prefetch(population_smiles)  # once per generation, before any get()
        ev.get(smi)                     # dict lookup, no model call
    """

    def __init__(self, properties=None, cache_path=None, admet_client=None):
        self.properties = list(properties or ALL_PROPERTIES)
        unknown = [p for p in self.properties if p not in SPECS]
        if unknown:
            raise KeyError(f"unknown properties {unknown}; known: {ALL_PROPERTIES}")
        self.cache = {}
        self.cache_path = cache_path
        self.admet = admet_client or AdmetClient()
        self._drd2 = None
        self._needs_admet = any(SPECS[p]["source"] == "admet" for p in self.properties)
        if cache_path and os.path.exists(cache_path):
            self._load_cache(cache_path)

    # -- cache
    def _load_cache(self, path):
        n = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # a truncated last line from a killed run
                self.cache[rec["smiles"]] = rec["values"]
                n += 1
        print(f"PropertyEvaluator: loaded {n} cached molecule(s) from {path}", flush=True)

    def _append_cache(self, smi, values):
        if not self.cache_path:
            return
        with open(self.cache_path, "a") as f:
            f.write(json.dumps({"smiles": smi, "values": values}) + "\n")

    # -- the tdc oracle, built lazily so importing never downloads anything
    def _drd2_oracle(self):
        if self._drd2 is None:
            import tdc
            self._drd2 = tdc.Oracle(name="DRD2")
        return self._drd2

    @staticmethod
    def canonical(smi):
        mol = Chem.MolFromSmiles(smi)
        return None if mol is None else Chem.MolToSmiles(mol)

    def _complete(self, can):
        """True when the cache holds every property THIS evaluator was asked for.

        Not merely "is it cached": the cache file is shared across runs, and a run
        configured for one property set writes entries that lack the others. Treating a
        partial entry as a hit raised KeyError on the first run with a different
        combination, which is how this check came to exist.
        """
        cached = self.cache.get(can)
        return cached is not None and all(p in cached for p in self.properties)

    def prefetch(self, smis):
        """Compute and cache every requested property for every molecule whose cache entry
        is missing or incomplete. One ADMET-AI round trip for the whole batch."""
        todo = []
        for smi in smis:
            can = self.canonical(smi)
            if can is None or self._complete(can):
                continue
            if can not in todo:
                todo.append(can)
        if not todo:
            return

        admet_values = {}
        if self._needs_admet and any(
                SPECS[p]["source"] == "admet" and p not in (self.cache.get(can) or {})
                for can in todo for p in self.properties):
            admet_values = self.admet.predict(todo)

        for can in todo:
            # Merge rather than replace: an entry written by a run with a different
            # property set keeps whatever it already had.
            values = dict(self.cache.get(can) or {})
            for p in self.properties:
                if p in values:
                    continue
                src = SPECS[p]["source"]
                if src == "admet":
                    values[p] = float(admet_values[can][p])
                elif src == "rdkit":
                    values[p] = penalized_logp(can) if p == "PlogP" else qed(can)
                elif src == "tdc":
                    values[p] = float(self._drd2_oracle()(can))
                else:
                    raise AssertionError(f"unhandled source {src!r} for {p!r}")
            self.cache[can] = values
            self._append_cache(can, values)

    def get(self, smi):
        """The property dict for one molecule. Falls back to a one-molecule prefetch if
        the caller forgot to batch, so correctness never depends on the hook being wired
        - but a run that hits this path repeatedly is paying per-molecule and the log line
        says so."""
        can = self.canonical(smi)
        if can is None:
            return None
        if not self._complete(can):
            print(f"PropertyEvaluator: cache miss on {can} - prefetch() was not called "
                  f"for this molecule", flush=True)
            self.prefetch([can])
        return self.cache.get(can)

    # -- the guard
    def self_check(self, verbose=True):
        """Assert every requested property discriminates. Raises on a constant axis.

        The pairs below are chosen so each property should differ substantially between
        them; the point is not accuracy, it is that a loaded model answers at all. A
        silently-constant objective is the failure this catches - see the module docstring.
        """
        probes = [
            "OC(CCN1CCC(O)(CC1)c1ccc(Cl)cc1)c1ccc(F)cc1",   # haloperidol: DRD2 active
            "CCO",                                            # ethanol: tiny, inactive
            "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1",       # a drug-like ZINC molecule
            "c1ccc2c(c1)c1ccc3cccc4c3c1c1c2cccc41",          # a PAH: fused, unsynthesizable-ish
        ]
        check_against_paper()
        self.prefetch(probes)
        rows = [self.cache[self.canonical(s)] for s in probes]
        flat = []
        for p in self.properties:
            vals = [r[p] for r in rows]
            spread = max(vals) - min(vals)
            if verbose:
                print(f"  self_check {p:>6}: " + " ".join(f"{v:8.4f}" for v in vals)
                      + f"   spread={spread:.4f}", flush=True)
            if spread < 1e-6:
                flat.append(p)
            elif spread < 0.01:
                # Not a failure, but worth refusing to discover mid-run: an axis this flat
                # carries almost no signal and, if its theta is already satisfied by
                # everything, contributes a constant that cancels in k^Phi. HIA_Hou is the
                # known case - it saturates near 1.0 for drug-like input.
                print(f"  self_check WARNING: {p} spread {spread:.4f} over very different "
                      f"molecules - near-constant, check it is worth an axis", flush=True)
        if flat:
            raise RuntimeError(
                f"constant property/properties {flat} across {len(probes)} very different "
                f"molecules. A model failed to load and is answering a constant - do not "
                f"run. (tdc.Oracle swallows a failed load and returns 0.0; check "
                f"cmumo/oracle/*.pkl is a real pickle and not a 403 HTML page.)")
        return True

    def close(self):
        self.admet.close()


if __name__ == "__main__":
    ev = PropertyEvaluator(cache_path=paths.PROPERTY_CACHE)
    print("properties:", ev.properties)
    ev.self_check()
    print("OK - all properties discriminate")
    ev.close()
