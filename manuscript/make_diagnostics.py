"""Generate manuscript diagnostics from completed, sealed trajectory outputs."""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/sections'
def read(p): return json.loads(p.read_text())
def table(name,caption,columns,header,rows):
 text=['\\begin{table*}[t]','\\caption{'+caption+'}','\\label{tab:'+name+'}', '\\centering\\small\\begin{tabular}{'+columns+'}','\\toprule',header+r' \\',r'\midrule']
 text += [' & '.join(map(str,r))+r' \\' for r in rows]
 text += [r'\bottomrule\end{tabular}',r'\end{table*}']
 (OUT/(name+'.tex')).write_text('\n'.join(text)+'\n')
main=read(ROOT/'artifacts/lifecycle-v25/analysis.json')
comp=read(ROOT/'artifacts/lifecycle-v26-coordinate/analysis.json')
assert main['completed']==400 and comp['completed']==200
names={'quorum':'Quorum','clipping':'Clipping','periodic':'Periodic','forced_periodic':'Forced periodic','finetune':'Fine-tune','forced_finetune':'Forced fine-tune','fineprune':'Fine-Pruning','forced_fineprune':'Forced Fine-Pruning','nad':'NAD-style','coordinate_median':'Median'}
datasets={'femnist':'FEMNIST','fashion_mnist':'Fashion-MNIST'}
# Symmetrical main comparison with clean accuracy and admission reference columns.
rows=[]
for ds in datasets:
 for arm in list(names)[:8]:
  def group(cond): return next(g for g in main['complete_seed_groups'] if (g['dataset'],g['condition'],g['trigger'],g['arm'])==(ds,cond,'patch',arm))
  stop,clean=group('attack_stop'),group('clean')
  rows.append([datasets[ds],names[arm]]+[f"{100*g[m]['mean']:.2f}" for g,m in [(clean,'final_accuracy'),(stop,'final_accuracy'),(stop,'endpoint_asr'),(stop,'mean_after_stop_asr'),(stop,'mean_post_onset_asr')]]+[f"{clean['repair_admissions']['mean']:.1f}",f"{stop['repair_admissions']['mean']:.1f}"])
table('stop','All eight policies under patch-trigger attacker-stop poisoning, with clean-training references. Accuracy and ASR are percentages; admissions are means out of four scheduled attempts (zero for non-repair policies). Each mean uses five seeds.','llrrrrrrr','Dataset & Policy & Clean acc. & Attack acc. & End ASR & Stop ASR & Onset ASR & Clean adm. & Attack adm.',rows)
# Temporal phases, same five seeds per row, no pooling repeated controls.
rows=[]; phase_data=[]
for ds in datasets:
 for cond in ['attack_stop','continuous']:
  rs=[r for r in main['records'] if (r['dataset'],r['condition'],r['trigger'],r['arm'])==(ds,cond,'patch','periodic')]
  assert len(rs)==5
  vals=[]
  for r in rs:
   path=ROOT/'results/lifecycle-v25'/r['id']; assert (path/'completion.json').exists()
   h=read(path/'summary.json')['targeted_attack_success_history']; assert len(h)==100
   vals.append([h[98],h[99],h[50],h[75]]+[h[t]-h[t-1] for t in [25,50,75]])
  means=np.mean(vals,axis=0)*100
  rows.append([datasets[ds],'Stop' if cond=='attack_stop' else 'Continuous']+[f'{v:.2f}' for v in means])
  phase_data.append(dict(dataset=ds,condition=cond,seed_values=vals,means_percent=means.tolist()))
table('phases','Periodic repair with the patch trigger: five-seed mean ASR at selected rounds (percent) and one-round changes after repairs (percentage points). Positive changes indicate rebound in committed ASR.','llrrrrrrr',r'Dataset & Schedule & $A_{99}$ & $A_{100}$ & $A_{51}$ & $A_{76}$ & $A_{26}-A_{25}$ & $A_{51}-A_{50}$ & $A_{76}-A_{75}$',rows)
# Gate diagnostics: count each private attempt once, across both panels; reasons overlap.
gate_rows=[]; candidate_rows=[]; details={}
for arm,analysis,folder in [('periodic',main,'lifecycle-v25'),('finetune',main,'lifecycle-v25'),('fineprune',main,'lifecycle-v25'),('nad',comp,'lifecycle-v26-coordinate')]:
 counts=dict(attempts=0,rejected=0,loss=0,accuracy=0,patch=0,collapse=0)
 for ds in datasets:
  rejected=[]
  for r in analysis['records']:
   if r['arm']!=arm or r['dataset']!=ds: continue
   path=ROOT/'results'/folder/r['id']; assert (path/'completion.json').exists()
   observations=read(path/'offline-evaluation.json'); assert len(observations)==4
   for o in observations:
    payload=read(path/'online'/f"round-{o['round']:03d}"/'decision.json')
    decision=payload['decision']; panels=list(decision['panels'].values())
    assert bool(decision['passed'])==bool(o['admitted'])
    counts['attempts']+=1
    if o['admitted']: continue
    counts['rejected']+=1
    counts['loss']+=any(p['clean_loss_check']['mean_loss_change']>0.02+2*p['clean_loss_check']['standard_error'] for p in panels)
    counts['accuracy']+=any(p['clean_accuracy_delta'] < -0.02-1e-12 for p in panels)
    counts['patch']+=any(p['worst_patch_loss_delta']>1e-12 for p in panels)
    counts['collapse']+=any(p['constant_prediction'] for p in panels)
    if r['condition']=='clean': continue
    m=o['models']; assert m['source']['model_hash']==m['committed']['model_hash']
    rejected.append([m['source']['trigger_rate'],m['candidate']['trigger_rate'],m['source']['accuracy'],m['candidate']['accuracy']])
  arr=np.asarray(rejected)
  if len(arr):
   strong=int(((arr[:,0]>=.5)&(arr[:,1]<=.1)).sum())
   preserved=int(((arr[:,0]>=.5)&(arr[:,1]<=.1)&(arr[:,3]>=arr[:,2]-.02)).sum())
   candidate_rows.append([datasets[ds],names[arm],len(arr)]+[f'{v:.2f}' for v in arr.mean(axis=0)*100]+[strong,preserved])
 assert counts['attempts']==200
 gate_rows.append([names[arm]]+list(counts.values()))
 details[arm]=counts
table('gates','Private-admission decisions across clean and attacked conditions. A criterion is counted once per rejected attempt when either panel violates it. Counts overlap. Main-comparison controls are used once; NAD uses the companion comparison.','lrrrrrrr','Policy & Attempts & Rejected & Clean loss & Accuracy & Patch loss & Collapse', [r[:3]+r[3:] for r in gate_rows])
# seven columns actually: policy + six counters
p=OUT/'gates.tex';p.write_text(p.read_text().replace('{lrrrrrrr}','{lrrrrrr}'))
table('candidates','Rejected repairs in attacked conditions, pooled descriptively over both triggers, both schedules, and four interventions per trajectory. Source is also the retained committed model. ASR and accuracy are percentages. Strong reduction counts source ASR $\\geq50\\%$ and candidate ASR $\\leq10\\%$; the last column additionally requires a clean-accuracy loss of at most two percentage points.','llrrrrrrr','Dataset & Policy & Rejected & Source ASR & Cand. ASR & Source acc. & Cand. acc. & Strong & Utility kept',candidate_rows)
(ROOT/'manuscript/data/diagnostic_summary.json').write_text(json.dumps(dict(phases=phase_data,gates=details,candidates=candidate_rows),indent=2))
print(json.dumps(dict(phases=phase_data,gates=details,candidates=candidate_rows),indent=2))
