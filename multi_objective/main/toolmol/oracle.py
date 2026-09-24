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

    # Reference molecule with hand-checked values, used to catch a silently broken objective.
    # TDC's Oracle swallows exceptions from its scorers and returns 0.0, so a corrupt data file
    # turns an objective into a constant rather than raising: the run then optimises fewer
    # objectives than it reports, looking healthy throughout. This happened for real - a
    # CRLF-mangled fpscores.pkl made SA return 0.0 for every molecule - and nothing in the run
    # output would have revealed it.
    _SANITY_SMILES = 'CC(=O)Nc1ccc(Nc2ccccc2F)c(C)c1'
    _SANITY_EXPECTED = {'jnk3': 0.04, 'qed': 0.8779, 'sa': 1.6698}

    def assign_evaluator(self, args):
        self.objectives = build_objectives(list(self.max_obj) + list(self.min_obj))
        self._check_objectives()
        # keep these populated too, in case any inherited code path still reads them
        n_max = len(self.max_obj)
        self.max_evaluator = [obj.evaluator for obj in self.objectives[:n_max]]
        self.min_evaluator = [obj.evaluator for obj in self.objectives[n_max:]]

    def _check_objectives(self):
        """Fail loudly at startup if an objective is not actually working.

        Checked against a reference molecule rather than by catching exceptions, because the
        failure mode is a silent 0.0, not a traceback. Tolerances are loose - this is looking for
        a dead objective, not guarding against a library version bumping a value slightly.
        """
        broken = []
        for obj in self.objectives:
            expected = self._SANITY_EXPECTED.get(obj.name)
            if expected is None:
                continue  # no reference value recorded for this objective; nothing to check
            try:
                actual = obj.raw(self._SANITY_SMILES)
            except Exception as e:
                broken.append(f"{obj.name}: raised {type(e).__name__}: {e}")
                continue
            if abs(actual - expected) > max(0.05, 0.1 * abs(expected)):
                broken.append(f"{obj.name}: got {actual:.4g}, expected about {expected:.4g}")
        if broken:
            raise RuntimeError(
                "Objective sanity check failed on " + self._SANITY_SMILES + ":\n  "
                + "\n  ".join(broken)
                + "\nAn objective returning 0.0 is usually a corrupt or missing TDC data file "
                  "(see multi_objective/oracle/); TDC returns 0.0 rather than raising, which "
                  "would silently drop this objective from Phi and from Pareto dominance.")
        print("objective sanity check passed: "
              + ", ".join(f"{o.name}={o.raw(self._SANITY_SMILES):.4g}" for o in self.objectives),
              flush=True)

    def score_smi(self, smi):
        """As the base class, but numbering molecules by a run-global counter.

        The base implementation stores len(mol_buffer)+1 as each molecule's index, and
        clean_buffer() empties mol_buffer every generation - so the index restarts each
        generation and is not a discovery order. That makes a run's trajectory impossible to
        reconstruct from its results file after the fact, which the regret tooling needs.
        """
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        smi = Chem.MolToSmiles(mol)
        if smi not in self.mol_buffer:
            if smi in self.storing_buffer:
                # Seen in an earlier generation and flushed out by clean_buffer. Carry its
                # original index forward rather than issuing a new one, so the number stays a
                # first-discovery order across the whole run.
                self.mol_buffer[smi] = list(self.storing_buffer[smi])
            else:
                self._n_scored = getattr(self, '_n_scored', 0) + 1
                self.mol_buffer[smi] = [float(self.evaluate(smi)), self._n_scored]
        return self.mol_buffer[smi][0]

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
