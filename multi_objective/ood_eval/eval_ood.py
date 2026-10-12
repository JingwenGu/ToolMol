import os, sys, glob, json, yaml, random
import numpy as np
REPO = r"D:\校外学习\Reading\Computer Science\ML papers\AI4S\MOLLEO\multi_objective"
os.chdir(REPO)
from tdc import Oracle
from tdc.generation import MolGen
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem
RDLogger.DisableLog('rdApp.*')

# initial pool: optimize() reseeds from --seed before np.random.choice over ZINC
np.random.seed(1); random.seed(1)
zinc = MolGen(name='ZINC').get_data()['smiles'].tolist()
np.random.seed(1)
init = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in np.random.choice(zinc, 60)]

base = os.path.join(REPO, "main", "toolmol", "results")
RUNS = {
 'baseline':      "task_gpt54_seed1_max200_2026-09-13",
 'static 3-step': "task_gpt54_3stepfewshot_seed1_max200_resumed_2026-09-16",
 'on-policy':     "task_gpt54_onlinefewshot_seeded_beam30_seed1_max200_2026-09-28",
 'inject m=2':    "task_gpt54_onlinefewshot_seeded_beam30_inject_seed1_max200_2026-09-29",
 'inject m=5':    "task_gpt54_onlinefewshot_seeded_beam30_k25_m5_inject_seed1_max200_2026-10-05",
}
sets = {'initial pool': init}
phi = {}
for name, d in RUNS.items():
    y = yaml.safe_load(open(glob.glob(os.path.join(base, d, "results_*.yaml"))[0]))
    items = sorted(y.items(), key=lambda kv: -kv[1][0])
    phi[name] = dict((k, v[0]) for k, v in items)
    sets[name + ' (top-100)'] = [k for k, _ in items[:100]]
    sets[name + ' (top-10)'] = [k for k, _ in items[:10]]
print({k: len(v) for k, v in sets.items()}, flush=True)

gsk, drd, jnk = Oracle(name='gsk3b'), Oracle(name='drd2'), Oracle(name='jnk3')
init_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in init]

def small_ring(m):
    ri = m.GetRingInfo()
    return int(any(len(r) in (3, 4) for r in ri.AtomRings()))
def odd_ring(m):   # azete/azirine-type: 3-4 membered ring containing N/O/S, or any ring-size 3-4 with sp2 atom
    ri = m.GetRingInfo()
    for r in ri.AtomRings():
        if len(r) <= 4 and any(m.GetAtomWithIdx(i).GetSymbol() != 'C' for i in r): return 1
    return 0

cache = {}
def props(s):
    if s in cache: return cache[s]
    m = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
    r = dict(jnk3=float(jnk(s)), gsk3b=float(gsk(s)), drd2=float(drd(s)),
             logp=Crippen.MolLogP(m), mw=Descriptors.MolWt(m), tpsa=rdMolDescriptors.CalcTPSA(m),
             hbd=rdMolDescriptors.CalcNumHBD(m), hba=rdMolDescriptors.CalcNumHBA(m),
             rotb=rdMolDescriptors.CalcNumRotatableBonds(m), rings=rdMolDescriptors.CalcNumRings(m),
             arom=rdMolDescriptors.CalcNumAromaticRings(m), fsp3=rdMolDescriptors.CalcFractionCSP3(m),
             heavy=m.GetNumHeavyAtoms(), small_ring=small_ring(m), hetero_small_ring=odd_ring(m),
             sim_init=max(DataStructs.BulkTanimotoSimilarity(fp, init_fps)))
    cache[s] = r
    return r

out = {}
for k, smis in sets.items():
    rows = [props(s) for s in smis]
    out[k] = {key: [r[key] for r in rows] for key in rows[0]}
    out[k]['smiles'] = smis
    print(k, 'done', flush=True)
json.dump(out, open(os.path.join(os.environ['TEMP'], 'ood', 'ood_raw.json'), 'w'))
print("saved")
