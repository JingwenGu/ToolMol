from __future__ import print_function

import json
import os
from collections import Counter

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
from main.toolmol.online_fewshot import OnlineFewShotBuffer

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

        self.mol_lm = None
        self.online_buffer = None
        # SMILES -> 'injected' (a beam-search-discovered molecule, never yet touched by an LLM
        # edit) | 'injected+llm' (descends from an injected molecule via at least one further
        # LLM edit) | 'llm' (no beam-search ancestry at all - includes both the original random
        # starting population and every purely LLM-derived molecule). Approximate for crossover:
        # an offspring is tagged 'injected+llm' if EITHER parent carries injected ancestry, even
        # if that specific episode never actually called crossover_molecules against parent2 -
        # agent.py's edit_pair doesn't currently report which tools an episode used, so this
        # errs toward over-attributing injected ancestry rather than under-attributing it.
        self._molecule_origin = {}
        if args.mol_lm == "ToolMol":
            if getattr(args, "few_shot_file", None) and getattr(args, "online_fewshot", False):
                raise ValueError("--few_shot_file and --online_fewshot are mutually exclusive - "
                                  "the online buffer drives self.mol_lm.few_shot_messages itself, "
                                  "so a static file would just be overwritten on the first update.")
            extra_body = json.loads(args.llm_extra_body) if args.llm_extra_body else None
            self.mol_lm = ToolMolAgent(model=args.llm_model, base_url=args.llm_base_url,
                                       api_key_env=args.llm_api_key_env, extra_body=extra_body,
                                       backend=args.llm_backend, device_map=args.llm_device_map,
                                       max_new_tokens=args.llm_max_new_tokens,
                                       system_prompt_suffix=getattr(args, "llm_system_prompt_suffix", None),
                                       few_shot_file=getattr(args, "few_shot_file", None))
            self.mol_lm.goal_description = _goal_description(args)

            if getattr(args, "online_fewshot", False):
                self.online_buffer = OnlineFewShotBuffer(
                    buffer_size=args.online_fewshot_buffer_size,
                    update_every=args.online_fewshot_update_every,
                    sample_size=args.online_fewshot_sample_size,
                    beam_width=args.online_fewshot_beam,
                    workers=args.online_fewshot_workers,
                    llm_model=args.llm_model,
                    api_key_env=args.llm_api_key_env,
                    goal_description=self.mol_lm.goal_description,
                    diversity_evaluator=self.oracle.diversity_evaluator,
                    diversity_floor_frac=args.online_fewshot_diversity_floor,
                    signature_max_repeats=args.online_fewshot_max_repeats,
                    seed_examples_file=getattr(args, "online_fewshot_seed_file", None),
                    state_file=os.path.join(args.output_dir, "online_fewshot_buffer.json"),
                    load_buffer_file=getattr(args, "online_fewshot_load_buffer", None),
                )

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

            if self.online_buffer is not None:
                self.online_buffer.update_population_diversity(population_smi_list)

            pairs = make_mating_pool_phi(population_mol, population_smi_list, self.oracle,
                                          k=config["k"], offspring_size=config["offspring_size"])
            print(f"generation {generation}: population={len(population_mol)}, "
                  f"oracle_calls={len(self.oracle)}, generating {len(pairs)} offspring...", flush=True)
            offspring_mol = []
            for pair_idx, (m0, m1) in enumerate(pairs):
                m0_smi, m1_smi = Chem.MolToSmiles(m0), Chem.MolToSmiles(m1)
                if self.online_buffer is not None:
                    # record_episode first: at a window boundary it may block until the previous
                    # beam-search cycle finishes, and this episode should then see the buffer
                    # that cycle produced. Sync still happens before edit_pair, not during it,
                    # so an update never lands mid-call.
                    self.online_buffer.record_episode(m0_smi, m1_smi)
                    self.mol_lm.few_shot_messages = self.online_buffer.few_shot_messages
                print(f"  generation {generation} pair {pair_idx + 1}/{len(pairs)}: starting edit_pair", flush=True)
                child_mol = self.mol_lm.edit_pair(
                    m0, smi_to_score[m0_smi],
                    m1, smi_to_score[m1_smi],
                    config["mutation_rate"])
                offspring_mol.append(child_mol)
                print(f"  generation {generation} pair {pair_idx + 1}/{len(pairs)}: edit_pair done", flush=True)

                # edit_pair can legitimately return None (no successful tool call and the
                # Graph-GA fallback also failed) - sanitize() filters these out further down,
                # but origin-tagging has to guard against it directly since it runs first.
                child_smi = Chem.MolToSmiles(child_mol) if child_mol is not None else None
                if child_smi is not None and child_smi != m0_smi:
                    parent_origins = {self._molecule_origin.get(m0_smi, 'llm'),
                                       self._molecule_origin.get(m1_smi, 'llm')}
                    if 'injected' in parent_origins or 'injected+llm' in parent_origins:
                        self._molecule_origin[child_smi] = 'injected+llm'
                    else:
                        self._molecule_origin.setdefault(child_smi, 'llm')

            # add new_population
            population_mol += offspring_mol
            if self.online_buffer is not None and getattr(self.args, "online_fewshot_inject", False):
                # Fold in whichever beam-search-discovered molecules the background buffer has
                # found since the last drain, so they compete for a population slot alongside
                # the LLM's own offspring rather than only ever serving as few-shot text. Since
                # a cycle can span several generations at beam=30, this drains into whichever
                # generation happens to be running when the cycle completes, not necessarily
                # "the next" one in a literal sense.
                injected_smi = self.online_buffer.drain_injections()
                injected_mol = [m for m in (Chem.MolFromSmiles(s) for s in injected_smi) if m is not None]
                if injected_mol:
                    print(f"  injecting {len(injected_mol)} beam-search-discovered molecule(s) "
                          f"into the population", flush=True)
                    for smi in injected_smi:
                        self._molecule_origin.setdefault(smi, 'injected')
                population_mol += injected_mol
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

            if self.online_buffer is not None and getattr(self.args, "online_fewshot_inject", False):
                origin_counts = Counter(
                    self._molecule_origin.get(Chem.MolToSmiles(mol), 'llm') for mol in population_mol)
                print(f"  generation {generation} Pareto front origin: "
                      f"injected(untouched)={origin_counts['injected']}, "
                      f"injected+llm={origin_counts['injected+llm']}, "
                      f"llm-only={origin_counts['llm']}", flush=True)

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
