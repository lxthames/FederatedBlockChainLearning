"""Generate manuscript material from completed, frozen follow-up seed groups."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
GPU_JOB=ROOT/'artifacts/trust-extension-v1-gpu'
JOB=GPU_JOB if (GPU_JOB/'frozen-inputs.json').exists() else ROOT/'artifacts/trust-extension-v1'
PAPER=ROOT/'Details/ACM_TRUST_2027'
GENERATED=PAPER/'generated'


def main():
    GENERATED.mkdir(parents=True,exist_ok=True)
    analysis_path=JOB/'five-seed-analysis.json'
    groups=json.loads(analysis_path.read_text())['complete_groups'] if analysis_path.exists() else []
    rows=[]
    for group in groups:
        if group['seeds']!=[13,29,47,71,101]:
            raise ValueError('incomplete group in confirmatory analysis')
        for metric,value in group['metrics'].items():
            rows.append({'family':group['group'][0],'dataset':group['group'][1],
                'method':group['group'][2],'attack':group['group'][3],
                'recovery_policy':group['group'][4],'validator_behavior':group['group'][5],
                'validator_corruption':group['group'][6],'metric':metric,'mean':value['mean'],
                'ci_low':value['bootstrap_ci95'][0],'ci_high':value['bootstrap_ci95'][1]})
    fields=['family','dataset','method','attack','recovery_policy','validator_behavior',
        'validator_corruption','metric','mean','ci_low','ci_high']
    with (GENERATED/'extension-results.csv').open('w',newline='',encoding='utf-8') as out:
        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    tex=[r'\paragraph{Follow-up evidence status.}']
    if not groups:
        tex.append('No complete five-seed follow-up group is available. FedVal, FEMNIST, adaptive-attack and recovery-control results remain pending.')
    else:
        tex.append(f'{len(groups)} follow-up groups have complete five-seed summaries. Full completion, failed cells and protocol-version compatibility still require review.')
    names={'fedavg':'FedAvg','fltrust':'FLTrust','fedval':'FedVal','validation_only':'Validation only','full':'Full extension'}
    attacks={'none':'Clean','model_replacement':'Replacement','low_and_slow':'Gradual'}
    for dataset in ('fashion_mnist','femnist'):
        selected=[g for g in groups if g['group'][0]=='matched_baselines' and g['group'][1]==dataset]
        # Do not publish an apparently complete comparison when any method/condition is missing.
        if len(selected)!=15:continue
        label='Fashion-MNIST' if dataset=='fashion_mnist' else 'FEMNIST'
        tex.extend([r'\begin{table}[t]\centering\small',
            '\\caption{'+label+' follow-up utility. Brackets give seed-bootstrap 95\\% intervals; five seeds give limited precision.}',
            r'\begin{tabular}{lll}\toprule Condition & Method & Utility [95\% CI]\\\midrule'])
        for group in sorted(selected,key=lambda g:(list(attacks).index(g['group'][3]),list(names).index(g['group'][2]))):
            metric=group['metrics']['final_utility'];lo,hi=metric['bootstrap_ci95']
            tex.append(f"{attacks[group['group'][3]]} & {names[group['group'][2]]} & {metric['mean']:.3f} [{lo:.3f}, {hi:.3f}]\\\\")
        tex.extend([r'\bottomrule\end{tabular}\end{table}'])
    (GENERATED/'extension_results.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')


if __name__=='__main__':main()
