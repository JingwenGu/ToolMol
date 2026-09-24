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

SNAPSHOT_EVERY = 10  # record the current Pareto front every k generations


def _goal_description(args):
    parts = []
    if args.max_obj:
        parts.append("maximize " + ", ".join(args.max_obj) + " score" + ("s" if len(args.max_obj) > 1 else ""))
    if args.min_obj:
        parts.append("minimize " + ", ".join(args.min_obj) + " score" + ("s" if len(args.min_obj) > 1 else ""))
    return "; ".join(parts)


class GB_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "toolmol"
        # BaseOptimizer.__init__/reset hardcode the base pareto_optimizer.Oracle; ToolMol
        # needs the generically-rescaled ToolMolOracle instead.
        self.oracle = ToolMolOracle(args=self.args)

        # Products of the current generation, for duplicate detection before scoring (see
        # _is_duplicate). Initialised here so the lookup is safe if called before the first
        # generation's loop assigns it.
        self._generation_products = set()

        self.mol_lm = None
        if args.mol_lm == "ToolMol":
            extra_body = json.loads(args.llm_extra_body) if args.llm_extra_body else None
            self.mol_lm = ToolMolAgent(model=args.llm_model, base_url=args.llm_base_url,
                                       api_key_env=args.llm_api_key_env, extra_body=extra_body,
                                       backend=args.llm_backend, device_map=args.llm_device_map,
                                       max_new_tokens=args.llm_max_new_tokens,
                                       system_prompt_suffix=getattr(args, "llm_system_prompt_suffix", None),
                                       few_shot_file=getattr(args, "few_shot_file", None),
                                       io_dir=getattr(args, "llm_io_dir", None),
                                       domain_brief_file=getattr(args, "domain_brief", None))
            self.mol_lm.goal_description = _goal_description(args)
            self.mol_lm.score_detail = self._score_detail
            self.mol_lm.is_duplicate = self._is_duplicate
            self.mol_lm.population_summary = self._population_summary

    # How many of the run's best molecules to show the agent each episode. Enough to convey what
    # the population looks like and which ideas are already taken; small enough that the block
    # costs well under 200 tokens on a prompt that already carries two full atom tables.
    POPULATION_SHOWN = 8

    def _population_summary(self):
        """Compact view of what the run has scored so far, for the opening prompt.

        Shows the best molecules by Phi with their binding term, because that is the objective
        the agent cannot otherwise reason about and the one carrying the headroom. The point is
        not only to avoid duplicates - the duplicate check already catches those after the fact -
        but to let the agent see which structural ideas are already represented, so it can choose
        to differ from them deliberately rather than rediscovering them.
        """
        scored = self.oracle.all_molecules()
        if not scored:
            return ""
        rows = sorted(scored.items(), key=lambda kv: -kv[1][0])[:self.POPULATION_SHOWN]
        lines = [f"Molecules already evaluated in this run ({len(scored)} total, best "
                 f"{len(rows)} shown). Your product must differ from every one of them; where "
                 f"you can, differ from a molecule in exactly one respect so the comparison is "
                 f"informative:"]
        for smi, (phi, _idx) in rows:
            jnk3 = ""
            for obj in self.oracle.objectives:
                if obj.name not in ('qed', 'sa'):
                    jnk3 = f"  {obj.name} {obj.raw(smi):.2f}"
                    break
            lines.append(f"  {phi:.3f}{jnk3}  {smi}")
        return "\n".join(lines)

    def _is_duplicate(self, smi):
        """Has this molecule already been produced or scored this run?

        Two sources, and both are needed. all_molecules() merges storing_buffer (flushed by
        clean_buffer each generation) with the current mol_buffer, covering everything the oracle
        has scored. But offspring are not scored until their generation ends, so that alone misses
        the case where two episodes of the *same* generation converge - which happened in
        iteration 2, episodes 17 and 23, with the gate staying silent. _generation_products closes
        that window by recording each offspring as it is produced.
        """
        return smi in self.oracle.all_molecules() or smi in self._generation_products

    def _score_detail(self, smi):
        """Break Phi down per objective, showing each raw value and its rescaled contribution.

        Phi is a sum over objectives rescaled to [0,1], so a single number cannot say which
        objective is lagging. In practice QED and SA saturate near their ceilings early while
        the binding term stays near zero, and a model shown only the sum will keep optimising
        the two it can infer from the properties block and ignore the one carrying all the
        remaining headroom. Spelling the split out costs a few tokens per episode.
        """
        parts = []
        for obj in self.oracle.objectives:
            raw = obj.raw(smi)
            parts.append(f"{obj.name} {raw:.3g} -> {obj.rescaled(raw):.3f}")
        return "; ".join(parts)

    def reset(self):
        del self.oracle
        self.oracle = ToolMolOracle(args=self.args)

    def _optimize(self, config):

        self.oracle.assign_evaluator(self.args)
        self.mol_lm.max_steps = config["max_steps"]

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
            population_scores = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])

        patience = 0
        generation = 0

        while True:
            generation += 1

            if len(self.oracle) > 1:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())])
            else:
                old_score = 0

            population_smi_list = [Chem.MolToSmiles(mol) for mol in population_mol]
            smi_to_score = dict(zip(population_smi_list, population_scores))

            pairs = make_mating_pool_phi(population_mol, population_smi_list, self.oracle,
                                          k=config["k"], offspring_size=config["offspring_size"],
                                          share_radius=config.get("share_radius", 0.0),
                                          max_parent_share=config.get("max_parent_share", 1.0))
            # len(self.oracle) is len(mol_buffer) - the current Pareto front - not the budget
            # consumed, which is the deduplicated union tracked by all_molecules(). Printing the
            # former as "oracle_calls" made the log look like the run had barely started.
            print(f"generation {generation}: population={len(population_mol)}, "
                  f"oracle_calls={len(self.oracle.all_molecules())}/{self.args.max_oracle_calls}, "
                  f"generating {len(pairs)} offspring...", flush=True)
            offspring_mol = []
            # Offspring are not scored until the generation ends, so the oracle's buffers cannot
            # tell an episode that an earlier episode of this same generation already produced the
            # molecule it is about to return. Recording each product here closes that window; see
            # _is_duplicate. Reset per generation because everything from prior generations has by
            # then been scored and is visible through the oracle.
            self._generation_products = set()
            for pair_idx, (m0, m1) in enumerate(pairs):
                print(f"  generation {generation} pair {pair_idx + 1}/{len(pairs)}: starting edit_pair", flush=True)
                child = self.mol_lm.edit_pair(
                    m0, smi_to_score[Chem.MolToSmiles(m0)],
                    m1, smi_to_score[Chem.MolToSmiles(m1)],
                    config["mutation_rate"])
                offspring_mol.append(child)
                if child is not None:
                    try:
                        self._generation_products.add(Chem.MolToSmiles(child))
                    except (ValueError, RuntimeError):
                        pass
                print(f"  generation {generation} pair {pair_idx + 1}/{len(pairs)}: edit_pair done", flush=True)

            # add new_population
            population_mol += offspring_mol
            population_mol = self.sanitize(population_mol)
            # Pareto optimal set
            self.oracle.clean_buffer()
            population_mol = self.oracle.select_pareto_front([Chem.MolToSmiles(mol) for mol in population_mol])
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
            if len(self.oracle) > 1:
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
