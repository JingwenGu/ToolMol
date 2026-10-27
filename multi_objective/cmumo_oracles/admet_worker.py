"""ADMET-AI prediction worker. Runs in cmumo/.venv-admet (python 3.12), NOT in the GA's
.venv-toolmol39 (python 3.9) - admet-ai needs torch and chemprop 2.x, which that env
cannot hold.

Protocol: one JSON object per line on stdin, one JSON object per line on stdout.

**stdout is claimed by the protocol and nothing else.** chemprop/lightning print progress
bars and tips to stdout, which corrupts the stream - the first symptom is the caller
failing to parse the handshake. So fd 1 is duplicated to a private handle for protocol
writes and then pointed at fd 2, which the caller sends to a log file.

    in   {"smiles": ["CCO", "c1ccccc1"]}
    out  {"ok": true, "values": {"CCO": {"hERG": 0.01, ...}, ...}}
    out  {"ok": false, "error": "..."}

The model is loaded once, at startup, and the "ready" line is printed only after that -
so the caller can block on it and never pay the load cost again. Per-call subprocesses
would pay ~5s of model loading per batch, which is why this is a persistent worker rather
than a script.

Only the properties in WANT are returned. ADMET-AI emits 52 raw columns plus a
*_drugbank_approved_percentile twin for each; the percentile columns are a ranking against
DrugBank, not a prediction, and must never be read as one.
"""
import json
import os
import sys

# The eight C-MuMO properties that come from ADMET-AI, in C-MuMO's own abbreviations.
# QED comes from rdkit and PlogP/DRD2 from the GA-side env, so they are not here.
WANT = {
    "AMP": "PAMPA_NCATS",
    "BBBP": "BBB_Martins",
    "CARC": "Carcinogens_Lagunin",
    "hERG": "hERG",
    "HIA": "HIA_Hou",
    "LIV": "DILI",
    "MUT": "AMES",
}


def main():
    # Claim fd 1 for the protocol before importing anything that might print to it.
    protocol = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(obj):
        protocol.write(json.dumps(obj) + "\n")
        protocol.flush()

    from admet_ai import ADMETModel

    model = ADMETModel()
    # Warm two molecules so the first real batch is not also the first forward pass.
    # One molecule trips chemprop's "dropping last batch" path and predicts nothing.
    model.predict(smiles=["CCO", "c1ccccc1"])
    send({"ready": True, "properties": sorted(WANT)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            smis = list(req["smiles"])
            if not smis:
                out = {"ok": True, "values": {}}
            else:
                df = model.predict(smiles=smis)
                # predict() returns one row per input in input order; index is the SMILES
                # as passed, so align on position rather than trusting the index.
                values = {}
                for i, smi in enumerate(smis):
                    row = df.iloc[i]
                    values[smi] = {name: float(row[col]) for name, col in WANT.items()}
                out = {"ok": True, "values": values}
        except Exception as e:  # noqa: BLE001 - the caller decides what a failure means
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        send(out)


if __name__ == "__main__":
    main()
