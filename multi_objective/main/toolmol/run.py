from __future__ import print_function

import json
import os

import numpy as np
import yaml
from rdkit import Chem, rdBase
rdBase.DisableLog('rdApp.error')

import main.toolmol.crossover as co
import main.toolmol.mutate as mu
from main.pareto_optimizer import BaseOptimizer
from main.toolmol.oracle import ToolMolOracle
from main.toolmol.agent import ToolMolAgent
from main.toolmol.sampling import make_mating_pool_phi
from main.toolmol import cmumo_feedback, objectives as cmumo_objectives
from main.toolmol.discovery_log import DiscoveryLog

SNAPSHOT_EVERY = 10  # record the current Pareto front every k generations


def _goal_description(args, objectives=None):
    """C-MuMO's instruction, built from the objective objects so direction and target come
    from the property table rather than from which CLI list a name landed in. Constant for
    a whole run: the pooled design fixes improve-vs-maintain per combination, so there is
    no per-instance task to restate (see cmumo_feedback)."""
    if objectives:
        return cmumo_feedback.goal_text(objectives)
    parts = []
    if args.max_obj:
        parts.append("maximize " + ", ".join(args.max_obj) + " score" + ("s" if len(args.max_obj) > 1 else ""))
    if args.min_obj:
        parts.append("minimize " + ", ".join(args.min_obj) + " score" + ("s" if len(args.min_obj) > 1 else ""))
    return "; ".join(parts)


def _apply_population_floor(oracle, front, candidates, floor):
    """Top a collapsed front back up to `floor` with the next-best molecules by Phi.

    A front of one leaves the GA able only to cross that molecule with itself, so the
    search is over. It is reachable because compliant molecules tie at 1 on every maintain
    axis, so dominance is decided by the improve axes alone and one molecule that is best
    on both dominates the whole population. Measured over 40 draws of 95 molecules it never
    happened, so this is a safety valve; in `cmumo_objectives_opfs` an injected beam-search
    winner - strong by construction - makes it much easier to hit.

    Off by default (floor 0), because it changes the inherited GA's survival semantics.
    The warning below fires whenever it would have helped."""
    if len(front) >= 2:
        return front
    if not floor or floor < 2:
        print(f"  WARNING: the Pareto front collapsed to {len(front)} molecule(s). The GA "
              f"can only cross it with itself from here, so expect no further progress. "
              f"Pass --population_floor (e.g. 4) to top the front up with the next-best "
              f"molecules by Phi.", flush=True)
        return front
    in_front = {Chem.MolToSmiles(m) for m in front}
    rest = sorted((c for c in dict.fromkeys(candidates) if c not in in_front),
                  key=lambda smi: oracle.evaluate(smi), reverse=True)
    added = [Chem.MolFromSmiles(smi) for smi in rest[:max(0, floor - len(front))]]
    added = [m for m in added if m is not None]
    if added:
        print(f"  population floor: front was {len(front)}, topped up with {len(added)} "
              f"molecule(s) by Phi to {len(front) + len(added)}", flush=True)
    return list(front) + added


def _prefilter_violators(oracle, mols, reject_at):
    """Drop offspring whose maintain axes are past reject_at * M before the Pareto front is
    computed (design section 3, mitigation 2). This is what collapses the effective front
    dimension to |improve set|: survivors all tie at 1 on the maintain axes, so dominance
    is decided by the improve axes alone. Keeps whole-front survival untouched."""
    maintain = [o for o in getattr(oracle, "objectives", []) if getattr(o, "kind", None) == "maintain"]
    if not maintain or reject_at is None:
        return mols, 0
    kept, dropped = [], 0
    for mol in mols:
        smi = Chem.MolToSmiles(mol)
        if any(o.violation(o.raw(smi)) > reject_at for o in maintain):
            dropped += 1
        else:
            kept.append(mol)
    # Never hand an empty population to select_pareto_front: early generations on a raw
    # ZINC seed can legitimately have every molecule violating something.
    if not kept:
        print(f"  pre-filter would have dropped all {len(mols)} molecules - keeping them "
              f"all this generation instead", flush=True)
        return mols, 0
    return kept, dropped


def _cap_front(oracle, mols, cap):
    """Truncate to the `cap` highest-Phi molecules BEFORE they are scored. Placement is the
    whole point: oracle(...) is what charges front admissions against the budget, so
    truncating the already-scored list would save nothing. oracle.evaluate() is free (it
    recomputes objectives without touching mol_buffer, which is why the sampler can call it
    every generation), so Phi is available here at no budget cost.

    Calibration put fronts at 6-10 of 95, so this is a safety valve and not a tuning knob."""
    if cap is None or len(mols) <= cap:
        return mols, 0
    scored = sorted(mols, key=lambda m: oracle.evaluate(Chem.MolToSmiles(m)), reverse=True)
    return scored[:cap], len(mols) - cap


class GB_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "toolmol"
        # BaseOptimizer.__init__/reset hardcode the base pareto_optimizer.Oracle; ToolMol
        # needs the generically-rescaled ToolMolOracle instead.
        self.oracle = ToolMolOracle(args=self.args)

        self.mol_lm = None
        if args.mol_lm == "ToolMol":
            extra_body = json.loads(args.llm_extra_body) if args.llm_extra_body else None
            self.mol_lm = ToolMolAgent(model=args.llm_model, base_url=args.llm_base_url,
                                       api_key_env=args.llm_api_key_env, extra_body=extra_body,
                                       backend=args.llm_backend, device_map=args.llm_device_map,
                                       max_new_tokens=args.llm_max_new_tokens,
                                       system_prompt_suffix=getattr(args, "llm_system_prompt_suffix", None),
                                       few_shot_file=getattr(args, "few_shot_file", None))
            self.mol_lm.goal_description = _goal_description(args)

    def reset(self):
        del self.oracle
        self.oracle = ToolMolOracle(args=self.args)

    def _optimize(self, config):

        self.oracle.assign_evaluator(self.args)
        self.mol_lm.max_steps = config["max_steps"]
        # AFTER assign_evaluator, not in __init__: the objectives do not exist yet there, so
        # _goal_description silently fell back to listing args.max_obj ("maximize QED,
        # PlogP, BBBP scores") instead of the C-MuMO instruction with its targets.
        self.mol_lm.goal_description = _goal_description(args=self.args,
                                                         objectives=self.oracle.objectives)
        print(f"goal:\n{self.mol_lm.goal_description}", flush=True)
        # Feedback on the agent's OWN intermediate molecules, not just the two parents.
        self.mol_lm.breakdown_fn = lambda smi: cmumo_feedback.breakdown(
            self.oracle.objectives, smi)

        if getattr(self.args, "resume_from", None):
            # Reconstruct the true starting population by re-running select_pareto_front()
            # over every molecule the prior run ever scored, rather than a naive top-N by
            # summed score - population_mol at any generation is always exactly the
            # non-dominated front over the full history (dominance is monotonic under the
            # union), so this recovers the exact live population, not an approximation.
            # Loading the saved buffer into mol_buffer first means the re-scoring call
            # below hits the cache instead of spending fresh oracle budget on molecules we
            # already have scores for.
            with open(self.args.resume_from) as f:
                saved = yaml.safe_load(f)
            print(f"resuming from {self.args.resume_from}: {len(saved)} saved molecules", flush=True)
            self.oracle.mol_buffer = dict(saved)
            saved_mol = [Chem.MolFromSmiles(smi) for smi in saved]
            saved_mol = [m for m in saved_mol if m is not None]
            population_mol = self.oracle.select_pareto_front([Chem.MolToSmiles(mol) for mol in saved_mol])
            population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
            print(f"resumed population: {len(population_mol)} molecules on the reconstructed "
                  f"Pareto front, {len(self.oracle)} oracle calls carried over", flush=True)

            # The generation-10 snapshot history (main/pareto_optimizer.py's
            # record_snapshot/save_snapshots) lives only in-memory (self.oracle.snapshots)
            # and gets overwritten by save_snapshots' 'w'-mode open - so without this, each
            # new chain link starts that dict empty and clobbers the previous link's
            # snapshot file the first time it saves. save_result and save_snapshots always
            # write to the same task-label-derived filenames (results_<label>.yaml /
            # pareto_snapshots_<label>.yaml in the same directory), so the snapshot file
            # sitting alongside resume_from's results file - if any - is the prior link's
            # history to carry forward.
            resume_dir = os.path.dirname(self.args.resume_from)
            resume_basename = os.path.basename(self.args.resume_from)
            if resume_basename.startswith('results_'):
                snapshot_path = os.path.join(resume_dir, 'pareto_snapshots_' + resume_basename[len('results_'):])
                if os.path.exists(snapshot_path):
                    with open(snapshot_path) as f:
                        self.oracle.snapshots = yaml.safe_load(f) or {}
                    print(f"Also resumed {len(self.oracle.snapshots)} snapshot(s) from {snapshot_path}", flush=True)
        else:
            if self.smi_file is not None:
                starting_population = self.all_smiles[:config["population_size"]]
            else:
                starting_population = np.random.choice(self.all_smiles, config["population_size"])

            population_smiles = starting_population
            population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
            # One batched oracle round trip for the seed population.
            cmumo_objectives.prefetch([Chem.MolToSmiles(m) for m in population_mol if m])
            population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])

        # Everything the run ever considers, not just what reaches a front - see
        # discovery_log.py for why that distinction loses almost the whole run.
        self.discovery_log = DiscoveryLog(
            os.path.join(self.args.output_dir,
                         'discovered_' + str(self.oracle.task_label) + '.jsonl'),
            objectives=self.oracle.objectives)

        patience = 0
        generation = 0

        while True:
            generation += 1

            # `>= 1`, not `> 1`. With `> 1` a front that collapses to a single molecule
            # skipped BOTH this and the improvement check below, so patience never
            # incremented while the budget had also stopped growing (only front members are
            # charged, and the front was not changing) - the run span forever. Observed at
            # 4878 generations with population=1 in the arm that has injection.
            if len(self.oracle) >= 1:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())])
            else:
                old_score = 0

            population_smi_list = [Chem.MolToSmiles(mol) for mol in population_mol]
            smi_to_score = dict(zip(population_smi_list, population_scores))


            pairs = make_mating_pool_phi(population_mol, population_smi_list, self.oracle,
                                          k=config["k"], offspring_size=config["offspring_size"])
            print(f"generation {generation}: population={len(population_mol)}, "
                  f"oracle_calls={len(self.oracle)}, generating {len(pairs)} offspring...", flush=True)
            offspring_mol = []
            for pair_idx, (m0, m1) in enumerate(pairs):
                m0_smi, m1_smi = Chem.MolToSmiles(m0), Chem.MolToSmiles(m1)
                print(f"  generation {generation} pair {pair_idx + 1}/{len(pairs)}: starting edit_pair", flush=True)
                objs = self.oracle.objectives
                child_mol = self.mol_lm.edit_pair(
                    m0, smi_to_score[m0_smi],
                    m1, smi_to_score[m1_smi],
                    config["mutation_rate"],
                    breakdown1=cmumo_feedback.breakdown(objs, m0_smi),
                    breakdown2=cmumo_feedback.breakdown(objs, m1_smi))
                offspring_mol.append(child_mol)
                print(f"  generation {generation} pair {pair_idx + 1}/{len(pairs)}: edit_pair done", flush=True)


            # add new_population
            population_mol += offspring_mol
            population_mol = self.sanitize(population_mol)
            # Batch every candidate before anything reads a property off it.
            cmumo_objectives.prefetch([Chem.MolToSmiles(m) for m in population_mol])
            # Design section 3: reject hard violators, then cap, then select the whole
            # non-dominated front exactly as before. Both steps run BEFORE the oracle call
            # that charges front admissions.
            # Snapshot the candidate set BEFORE the pre-filter. Molecules rejected for
            # violating a constraint are exactly the ones worth analysing afterwards, and
            # logging only survivors would erase them.
            all_candidates = [Chem.MolToSmiles(mol) for mol in population_mol]
            population_mol, n_dropped = _prefilter_violators(
                self.oracle, population_mol, getattr(self.args, "reject_at", None))
            population_mol, n_capped = _cap_front(
                self.oracle, population_mol, getattr(self.args, "front_cap", None))
            if n_dropped or n_capped:
                print(f"  pre-front: dropped {n_dropped} violator(s), capped {n_capped}, "
                      f"{len(population_mol)} molecule(s) into select_pareto_front", flush=True)
            # Pareto optimal set
            candidates = [Chem.MolToSmiles(mol) for mol in population_mol]
            self.oracle.clean_buffer()
            population_mol = self.oracle.select_pareto_front(candidates)
            population_mol = _apply_population_floor(
                self.oracle, population_mol, candidates,
                getattr(self.args, "population_floor", 0))
            # Record before anything is discarded. Every raw() here is a cache hit: the
            # whole candidate set went through prefetch a few lines above.
            n_new = self.discovery_log.record_generation(
                generation, all_candidates, self.oracle.objectives,
                [Chem.MolToSmiles(m) for m in population_mol],
                rejected=set(all_candidates) - set(candidates))
            print(f"  discovered {n_new} new molecule(s) this generation "
                  f"({len(self.discovery_log.seen)} total)", flush=True)
            # stats
            population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]


            if generation % SNAPSHOT_EVERY == 0:
                self.oracle.record_snapshot(generation)
                self.oracle.save_snapshots(self.oracle.task_label)

            ### early stopping - matches the paper's Algorithm 1: the oracle-budget check
            ### happens once per generation (after a full batch of offspring completes and
            ### the Pareto front is recomputed), not mid-generation, so the final generation
            ### may slightly overshoot max_oracle_calls.
            if len(self.oracle) >= 1:
                self.sort_buffer()
                new_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())])
                if (new_score - old_score) < 1e-3:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        self.oracle.record_snapshot(generation)
                        self.oracle.save_snapshots(self.oracle.task_label)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

                old_score = new_score

            if self.finish:
                self.oracle.record_snapshot(generation)
                self.oracle.save_snapshots(self.oracle.task_label)
                break
