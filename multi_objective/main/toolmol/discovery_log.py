"""Append-only record of every molecule a run considered.

Without this a finished run leaves almost nothing. `save_result` dumps
`storing_buffer | mol_buffer`, which holds only molecules that reached a Pareto front, and
only as `[Phi, index]` - no per-objective breakdown. Everything else the agent proposed is
scored during front selection through `obj.raw` and then dropped on the floor. So the
question "what did the agent actually generate, and how good was it" has no answer after
the fact, and the FA-style filtered metric, which is defined over the whole generated pool,
cannot be computed at all.

One line per distinct molecule, at first sighting:

    {"gen": 3, "smi": "...", "phi": 2.61, "on_front": true, "rejected": false,
     "origin": "llm", "raw": {"QED": 0.73, ...}, "scaled": {"QED": 0.73, ...}}

`rejected` marks a molecule the violator pre-filter dropped before front selection. Those
are logged deliberately: a proposal killed by a constraint is exactly what you want to
count afterwards, and it never reaches the front or the results file.

`origin` is always 'llm' in this arm: every molecule is either the seed population or an
agent edit of one. The field exists because `cmumo_objectives_opfs` also tags 'injected'
and 'injected+llm', so a reader can compare logs from the two arms without special cases.

Crash-safe by construction: the file is opened in append mode and flushed per generation,
so a killed run keeps everything up to its last completed generation.

First sighting only. A molecule that persists on the front for twenty generations is
written once, with the generation it first appeared; `pareto_snapshots_*.yaml` is where
front composition over time lives.
"""
import json
import os


class DiscoveryLog:

    def __init__(self, path, objectives=None):
        self.path = path
        # A sidecar naming the axes. Without it a reader has to infer which axis was a
        # plateaued maintain one from the value distribution, which is guesswork.
        if objectives:
            meta = {
                "improve": [o.name for o in objectives
                            if getattr(o, "kind", None) != "maintain"],
                "maintain": [o.name for o in objectives
                             if getattr(o, "kind", None) == "maintain"],
                "maintain_params": {
                    o.name: {"theta": o.theta, "delta": o.delta, "M": o.M, "C": o.C}
                    for o in objectives if getattr(o, "kind", None) == "maintain"},
            }
            with open(os.path.splitext(path)[0] + ".meta.json", "w") as f:
                json.dump(meta, f, indent=1)
        self.seen = set()
        # Resume support: a re-run or a --resume_from into the same output directory must
        # not duplicate lines, and must not re-announce molecules as newly discovered.
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.seen.add(json.loads(line)["smi"])
                    except (ValueError, KeyError):
                        continue  # a truncated last line from a killed run
            if self.seen:
                print(f"DiscoveryLog: {len(self.seen)} molecule(s) already recorded in "
                      f"{path}", flush=True)
        self._fh = open(path, "a")

    def record_generation(self, generation, candidates, objectives, front_smis,
                          origins=None, rejected=None):
        """Log every candidate not seen before. `candidates` is the full set that went into
        front selection this generation: the previous front plus this generation's offspring."""
        front = set(front_smis)
        origins = origins or {}
        rejected = rejected or set()
        n_new = 0
        for smi in candidates:
            if smi in self.seen:
                continue
            self.seen.add(smi)
            raw, scaled, phi = {}, {}, 0.0
            for obj in objectives:
                r = obj.raw(smi)
                s = obj.rescaled(r)
                raw[obj.name] = r
                scaled[obj.name] = s
                phi += s
            self._fh.write(json.dumps({
                "gen": generation,
                "smi": smi,
                "phi": phi,
                "on_front": smi in front,
                "rejected": smi in rejected,
                "origin": origins.get(smi, "llm"),
                "raw": raw,
                "scaled": scaled,
            }) + "\n")
            n_new += 1
        if n_new:
            self._fh.flush()
        return n_new

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
