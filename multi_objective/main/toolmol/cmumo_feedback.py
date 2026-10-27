"""What the agent is told: the instruction, and the per-objective breakdown of a parent.

Design section 4. The agent previously saw one scalar - `Score: 2.37` - which is Phi, the
sum. With maintain axes in play that is unusable: a molecule can lose 0.5 of Phi to a hERG
violation and the agent cannot tell that from a molecule that is merely mediocre on QED.

Two separate things, and the distinction matters:

- **The instruction** names each property, its direction and its target. In the pooled
  design (section 1) improve-vs-maintain is fixed per combination, not per input molecule,
  so this is **constant for a whole run** - there is no per-instance task to restate.
- **The breakdown** is the parent's actual values against those targets, and that does
  change every episode.

The breakdown names the *violated constraint*, not just the vector. The agent already
games the cheapest axis blindly; handing it raw numbers lets it do so deliberately, and
saying "hERG 0.41, maintain <= 0.30, violated by 0.11" points it at the binding one.
"""


def _target_phrase(obj):
    spec_dir = "higher is better" if obj.direction == "+" else "lower is better"
    if getattr(obj, "kind", None) == "maintain":
        rel = ">=" if obj.direction == "+" else "<="
        return (f"{obj.name}: MAINTAIN {rel} {obj.theta:g} "
                f"(tolerance {obj.M:g}; {spec_dir})")
    return f"{obj.name}: IMPROVE, {spec_dir}"


def goal_text(objectives):
    """The run's instruction. Constant per run - see the module docstring."""
    improve = [o for o in objectives if getattr(o, "kind", None) != "maintain"]
    maintain = [o for o in objectives if getattr(o, "kind", None) == "maintain"]
    parts = []
    if improve:
        parts.append("improve " + ", ".join(o.name for o in improve))
    if maintain:
        parts.append("while keeping " + ", ".join(o.name for o in maintain)
                     + " within their acceptable ranges")
    head = " ".join(parts) if parts else "improve the configured objectives"
    lines = "\n".join("  - " + _target_phrase(o) for o in objectives)
    return (f"{head}.\nThe objectives, with their targets:\n{lines}\n"
            f"A MAINTAIN property scores full marks anywhere inside its range, so pushing "
            f"it further than that buys nothing - spend the modifications on the IMPROVE "
            f"properties instead")


def breakdown(objectives, smi):
    """Per-objective lines for one molecule: raw value, target, and what is wrong."""
    lines = []
    total = 0.0
    for obj in objectives:
        raw = obj.raw(smi)
        f = obj.rescaled(raw)
        total += f
        if getattr(obj, "kind", None) == "maintain":
            d = obj.distance(raw)
            rel = ">=" if obj.direction == "+" else "<="
            if d <= obj.M:
                status = f"OK (maintain {rel} {obj.theta:g})"
            else:
                status = (f"VIOLATED by {d - obj.M:.3f} "
                          f"(maintain {rel} {obj.theta:g}, tolerance {obj.M:g})")
        else:
            status = "improve" + (" - lower is better" if obj.direction == "-" else "")
        lines.append(f"    {obj.name}: {raw:.3f}  [score {f:.3f}]  {status}")
    worst = _worst_violation(objectives, smi)
    tail = f"\n    binding constraint: {worst}" if worst else ""
    return f"  Phi {total:.3f} of {len(objectives)}.0\n" + "\n".join(lines) + tail


def _worst_violation(objectives, smi):
    """The maintain axis furthest past its threshold, in units of M, or None if all are
    compliant. This is the line the agent should act on first."""
    worst, worst_v = None, 0.0
    for obj in objectives:
        if getattr(obj, "kind", None) != "maintain":
            continue
        v = obj.violation(obj.raw(smi))
        if v > worst_v:
            worst, worst_v = obj, v
    if worst is None:
        return None
    rel = ">=" if worst.direction == "+" else "<="
    return (f"{worst.name} is the worst violation ({worst_v:.1f}x its tolerance; "
            f"needs {rel} {worst.theta:g})")
