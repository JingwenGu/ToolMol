"""Every path that points outside the committed tree, in one place.

The arm is meant to be committable on its own, so nothing heavy lives inside it: model
weights, the ZINC pool, the ADMET-AI virtualenv and the property cache all sit in an
assets directory beside it. `setup_assets.py` links the two cwd-relative ones into place,
because tdc resolves `oracle/fpscores.pkl` and its model downloads relative to the
process's cwd and offers no way to configure that.

The arm is `multi_objective/` inside the repository, matching every other arm branch of
github.com/JingwenGu/ToolMol, and the assets directory is a sibling of the repo root.

Override any of these with the matching environment variable when the layout differs.
"""
import os

# This file is <repo>/multi_objective/cmumo_oracles/paths.py, so:
ARM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # multi_objective/
REPO_ROOT = os.path.dirname(ARM_ROOT)                                   # the git root
WORKSPACE = os.path.dirname(REPO_ROOT)                                  # beside the repo

# The invariant that matters: assets live OUTSIDE the repository, so nothing heavy can be
# committed by accident. The default is a sibling of the repo root.
ASSETS = os.environ.get("CMUMO_ASSETS", os.path.join(WORKSPACE, "assets"))

# python 3.12 + admet-ai + torch. Cannot be the GA's own env (python 3.9), and also sits
# outside the repo.
ADMET_PYTHON = os.environ.get(
    "CMUMO_ADMET_PYTHON",
    os.path.join(WORKSPACE, ".venv-admet", "bin", "python"))

# Append-only {smiles, values} cache. A cache, not a record: safe to delete, and rebuilt
# on demand at ~4 ms/molecule.
PROPERTY_CACHE = os.environ.get(
    "CMUMO_PROPERTY_CACHE", os.path.join(ASSETS, "property_cache.jsonl"))

# chemprop/lightning chatter from the worker. Outside the arm so a run leaves the
# committed tree untouched; the arm's .gitignore would catch it anyway.
WORKER_LOG = os.environ.get("CMUMO_WORKER_LOG", os.path.join(ASSETS, "admet_worker.log"))

# What setup_assets.py links into the arm, and what tdc expects to find at ./<name>.
LINKED = {
    "oracle": os.path.join(ASSETS, "oracle"),
    "data": os.path.join(ASSETS, "data"),
}

# Where the ZINC pool lands once linked.
ZINC = os.path.join(ARM_ROOT, "data", "zinc.tab")
