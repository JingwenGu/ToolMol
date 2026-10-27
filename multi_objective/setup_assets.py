"""Put the heavy, uncommitted pieces where the arm expects them. Run once after a clone.

Nothing heavy lives in the committed tree: model weights, the ZINC pool, the property
cache and the ADMET-AI virtualenv all sit in an assets directory beside it. Two of those
have to appear *inside* the arm at runtime anyway, because tdc resolves
`oracle/fpscores.pkl` and its model downloads relative to the process's cwd with no way to
configure it (`tdc/utils/load.py:546`, `tdc/chem_utils/oracle/oracle.py:346`). So this
script symlinks them in, and `.gitignore` keeps the links out of git.

    python setup_assets.py          # link what exists, report what does not
    python setup_assets.py --check  # report only, create nothing

Needs no third-party package, so it runs under any interpreter.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmumo_oracles import paths  # noqa: E402

# What each missing piece is, and how to get it back. Sizes are approximate.
REQUIRED = [
    ("oracle/drd2_current.pkl", "35 MB", True,
     "copy from MOLLEO/multi_objective/oracle/drd2_current.pkl. Do NOT rely on tdc's "
     "own download: it returns a 403 HTML page, and tdc.Oracle then answers 0.0 for "
     "every molecule instead of raising (see PROPERTIES.md)."),
    ("oracle/fpscores.pkl", "8.6 MB", True,
     "copy from MOLLEO/multi_objective/oracle/fpscores.pkl. tdc's SA oracle opens this "
     "path literally, relative to the cwd."),
    ("oracle/jnk3_current.pkl", "10 MB", False,
     "copy from MOLLEO/multi_objective/oracle/jnk3_current.pkl. Only the molleo_* "
     "methods need it; the C-MuMO arm does not."),
    ("data/zinc.tab", "11 MB", True,
     "tdc writes it on first use (MolGen(name='ZINC')), or copy it from any existing "
     "arm's data/ directory."),
]


def _status(path):
    return "ok" if os.path.exists(path) else "MISSING"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, create nothing")
    args = ap.parse_args()

    print(f"arm:    {paths.ARM_ROOT}")
    print(f"assets: {paths.ASSETS}")
    print()

    problems = []

    # 1. The two cwd-relative directories, as symlinks into assets.
    for name, target in sorted(paths.LINKED.items()):
        link = os.path.join(paths.ARM_ROOT, name)
        if os.path.islink(link):
            current = os.path.realpath(link)
            if current == os.path.realpath(target):
                print(f"  link {name:<8} -> {target}  (already correct)")
                continue
            print(f"  link {name:<8} points at {current}, expected {target}")
            if not args.check:
                os.unlink(link)
            else:
                problems.append(f"{name} points at the wrong target")
                continue
        elif os.path.isdir(link):
            # A real directory, not a link: someone put files here directly. Leave it.
            print(f"  dir  {name:<8} exists as a real directory - left alone")
            continue

        if not os.path.isdir(target):
            if args.check:
                problems.append(f"{target} does not exist")
                print(f"  link {name:<8} CANNOT be made: {target} does not exist")
                continue
            os.makedirs(target, exist_ok=True)
            print(f"  made {target}")
        if not args.check:
            os.symlink(target, link)
            print(f"  link {name:<8} -> {target}  (created)")

    # 2. The files themselves.
    print()
    for rel, size, required, how in REQUIRED:
        full = os.path.join(paths.ASSETS, rel)
        state = _status(full)
        tag = "" if required else "  (optional)"
        print(f"  {state:>7}  {rel:<26} {size:>7}{tag}")
        if state == "MISSING" and required:
            problems.append(f"{rel}: {how}")

    # 3. The ADMET-AI interpreter.
    print()
    if os.path.exists(paths.ADMET_PYTHON):
        print(f"       ok  ADMET-AI python        {paths.ADMET_PYTHON}")
    else:
        print(f"  MISSING  ADMET-AI python        {paths.ADMET_PYTHON}")
        venv = os.path.dirname(os.path.dirname(paths.ADMET_PYTHON))
        problems.append(
            "the ADMET-AI environment. It cannot share the GA's env (python 3.9, no "
            "torch):\n      python3.12 -m venv " + venv
            + "\n      " + os.path.join(venv, "bin", "pip") + " install admet-ai")

    # 4. Somewhere for run output to go.
    results = os.path.join(paths.ARM_ROOT, "results")
    if not os.path.isdir(results) and not args.check:
        os.makedirs(results)
        print(f"\n  made {results}")

    print()
    if problems:
        print(f"{len(problems)} thing(s) still needed:\n")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}\n")
        print("The property cache is NOT in this list: it is a cache, rebuilt on demand "
              "at ~4 ms/molecule.")
        return 1
    print("ready. Verify the oracles with:\n"
          "  <toolmol venv>/bin/python -m cmumo_oracles.properties")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
