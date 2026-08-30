import numpy as np
from rdkit import Chem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from main.pareto_optimizer import Oracle
from main.toolmol.objectives import build_objectives


class ToolMolOracle(Oracle):
    """Generic per-objective rescaling, replacing the base Oracle's SA-hardcoded special case.

    Phi(m) = sum(objective.rescaled(raw) for objective in objectives) exactly matches ToolMol's
    paper (Section 3.1): each objective individually scaled to [0,1], then summed, used both as
    the per-molecule stored score and as the parent-sampling fitness.
    """

    def assign_evaluator(self, args):
        self.objectives = build_objectives(list(self.max_obj) + list(self.min_obj))
        # keep these populated too, in case any inherited code path still reads them
        n_max = len(self.max_obj)
        self.max_evaluator = [obj.evaluator for obj in self.objectives[:n_max]]
        self.min_evaluator = [obj.evaluator for obj in self.objectives[n_max:]]

    def evaluate(self, smi):
        return sum(obj.rescaled(obj.raw(smi)) for obj in self.objectives)

    def select_pareto_front(self, smiles_lst):
        if type(smiles_lst) != list:
            print('Smiles should be in the list format.')
            return None
        # pymoo's NonDominatedSorting minimizes by convention; every objective here is
        # rescaled to [0,1] with 1=best, so (1 - rescaled) uniformly gives the "lower is
        # better" value pymoo expects, for max- and min-type objectives alike.
        score_list = [[1.0 - obj.rescaled(obj.raw(smi)) for obj in self.objectives] for smi in smiles_lst]
        score_array = np.array(score_list)
        nds = NonDominatedSorting().do(score_array, only_non_dominated_front=True)
        pareto_front = np.array(smiles_lst)[nds]
        return [Chem.MolFromSmiles(smi) for smi in list(pareto_front)]
