import json, os, yaml, glob, numpy as np
d = json.load(open(os.path.join(os.environ['TEMP'], 'ood', 'ood_raw.json')))
init = set(d['initial pool']['smiles'])
base = r"D:\校外学习\Reading\Computer Science\ML papers\AI4S\MOLLEO\multi_objective\main\toolmol\results"
for n, dd in {'baseline':"task_gpt54_seed1_max200_2026-09-13",'inject m=5':"task_gpt54_onlinefewshot_seeded_beam30_k25_m5_inject_seed1_max200_2026-10-05"}.items():
    y = yaml.safe_load(open(glob.glob(os.path.join(base, dd, "results_*.yaml"))[0]))
    print(n, 'init-pool molecules found in run file:', len(init & set(y)), 'of 60 (file has', len(y), ')')
cols = ['jnk3','gsk3b','drd2','logp','mw','tpsa','rotb','arom','fsp3','sim_init','small_ring','hetero_small_ring']
print('%-26s' % 'set', *['%9s' % c[:9] for c in cols])
for k, v in d.items():
    print('%-26s' % k, *['%9.3f' % np.mean(v[c]) for c in cols])
print('\nfraction with gsk3b>0.5 / drd2>0.5 / max-sim-to-init<0.4')
for k, v in d.items():
    print('%-26s' % k, '%.2f %.2f %.2f' % (np.mean(np.array(v['gsk3b'])>0.5), np.mean(np.array(v['drd2'])>0.5), np.mean(np.array(v['sim_init'])<0.4)))
print('\ndrd2 max / 90th pct:')
for k, v in d.items(): print('%-26s' % k, '%.3f %.3f' % (max(v['drd2']), np.percentile(v['drd2'], 90)))
# correlations across all top-100 molecules pooled
allv = {c: np.concatenate([d[k][c] for k in d if 'top-100' in k]) for c in cols}
print('\ncorr within pooled top-100s:')
for c in ['gsk3b','drd2','logp']: print(c, 'vs jnk3 r=%.2f' % np.corrcoef(allv['jnk3'], allv[c])[0,1])
