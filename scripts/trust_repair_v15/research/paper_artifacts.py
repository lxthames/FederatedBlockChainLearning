"""Generate manuscript tables from complete evidence groups; never invent missing values."""
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from .results import aggregate_summaries
from .analysis import bootstrap_mean_ci, holm_adjust
from .trust_matrix import SEEDS, Trust2027Matrix

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'Details/ACM_TRUST_2027'
GENERATED=PAPER/'generated'


def tex(text):
    return str(text).replace('_',r'\_').replace('%',r'\%')


def paired_statistics(summaries):
    """Paired exact sign-flip tests on seed-level method differences."""
    groups={}
    for s in summaries:
        c=dict(s['configuration'])
        for key in ('experiment_id','seed','defense','device','rpc_url','deployment_path'):
            c.pop(key,None)
        key=json.dumps(c,sort_keys=True)
        groups.setdefault(key,{}).setdefault(s['defense'],{})[s['seed']]=s
    rows=[]
    for key,methods in groups.items():
        proposed=methods.get('two_layer',{})
        if set(proposed)!=set(SEEDS): continue
        for name,baseline in methods.items():
            if name=='two_layer' or set(baseline)!=set(SEEDS): continue
            if any(proposed[s]['data_fingerprint']!=baseline[s]['data_fingerprint'] or
                   proposed[s]['test_fingerprint']!=baseline[s]['test_fingerprint'] or
                   proposed[s]['source_hash']!=baseline[s]['source_hash'] for s in SEEDS):
                raise ValueError('unmatched data in paired comparison')
            differences=np.array([proposed[s]['final_utility']-baseline[s]['final_utility'] for s in SEEDS])
            permutations=np.array(list(itertools.product((-1,1),repeat=len(SEEDS))))
            null=np.abs((permutations*differences).mean(1))
            p=float(np.mean(null>=abs(differences.mean())-1e-12))
            ci=bootstrap_mean_ci(differences.tolist(),seed=20261019)
            rows.append({'setting':json.loads(key),'baseline':name,'paired_difference_mean':ci.mean,
                         'paired_ci_lower':ci.lower,'paired_ci_upper':ci.upper,'p_value':p,
                         'paired_standardized_effect':float(differences.mean()/differences.std(ddof=1)) if differences.std(ddof=1)>0 else None})
    for row,adjusted in zip(rows,holm_adjust([r['p_value'] for r in rows])):
        row['holm_p_value']=adjusted
    return rows


def main():
    GENERATED.mkdir(parents=True,exist_ok=True)
    paths=[]
    incomplete=[]
    configs=Trust2027Matrix().core_confirmation()
    for config in configs:
        path=ROOT/'results/trust-2027/core_confirmation'/config.experiment_id/'summary.json'
        if path.exists():
            summary=json.loads(path.read_text())
            if summary.get('protocol_version')=='trust-v2' and summary.get('evidence_tier')=='confirmatory' and summary.get('status')=='completed':
                paths.append(path)
        else: incomplete.append(config.experiment_id)
    by_group={}
    for path in paths:
        s=json.loads(path.read_text())
        by_group.setdefault(s['config_group_hash'],[]).append(path)
    rows=[]
    included=[]
    for group in by_group.values():
        seeds={json.loads(p.read_text())['seed'] for p in group}
        if seeds==set(SEEDS):
            rows.extend(aggregate_summaries(group,True))
            included.extend(json.loads(p.read_text()) for p in group)
    (GENERATED/'core-analysis.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (GENERATED/'paired-comparisons.json').write_text(json.dumps(paired_statistics(included),indent=2),encoding='utf-8')
    status={'core_runs_completed':len(paths),'core_runs_expected':len(configs),
            'complete_comparison_cells':len(rows),'missing_core_experiments':incomplete,
            'sift_provenance_available':(ROOT/'data/sift-provenance.json').exists(),
            'submission_ready':False,'reason':'Author claim audit, SIFT provenance and visual QA required in addition to completed experiments'}
    (PAPER/'evidence-status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    (GENERATED/'evidence_status.tex').write_text(
        f'As of the latest artifact generation, {len(paths)} of {len(configs)} core seed-level runs have completed. '
        f'{len(rows)} comparison cells contain all five confirmatory seeds. '
        'Incomplete cells are excluded from numerical estimates.\n',encoding='utf-8')
    if not rows:
        table='\\paragraph{Matched robustness comparison.} Confirmatory results are pending; no ranking or robustness estimate is available.\n'
    else:
        table='\\begin{longtable}{llllr}\n\\caption{Complete core cells: held-out final utility (mean and bootstrap 95\\% CI).}\\\\\n\\toprule Dataset & Method & Attack & Seeds & Utility \\\\\n\\midrule\n'
        for row in rows:
            table+='{} & {} & {} & {} & {:.3f} [{:.3f}, {:.3f}] \\\\\n'.format(tex(row['dataset']),tex(row['defense']),tex(row['attack']),row['seed_count'],row['final_utility_mean'],row['final_utility_ci_lower'],row['final_utility_ci_upper'])
        table+='\\bottomrule\n\\end{longtable}\n'
        # A separate CSV carries all cells; use a compact figure for the paper
        # to avoid overflowing the conference page budget as results accumulate.
        table='\\paragraph{Matched robustness comparison.} Complete-cell numerical results are exported in the companion artifact; the final paper figure requires editorial selection before submission.\n'
        with (GENERATED/'core-results.csv').open('w',newline='',encoding='utf-8') as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (GENERATED/'core_results.tex').write_text(table,encoding='utf-8')
    chain_path=ROOT/'results/trust-live-evm-final/chain-summary.json'
    if chain_path.exists():
        chain=json.loads(chain_path.read_text())
        table=(r'\begin{table}[t]\centering\caption{Measured local-EVM implementation check (five synthetic rounds). These are not robustness results.}'+'\n'+
               r'\begin{tabular}{lr}\toprule Measurement & Value \\ \midrule'+'\n'+
               f'Transactions (including setup) & {chain["transaction_count"]} \\\\\n'+
               f'Total gas used & {chain["gas_used"]:,} \\\\\n'+
               f'Total calldata bytes & {chain["input_bytes"]:,} \\\\\n'+
               r'\bottomrule\end{tabular}\label{tab:contract}\end{table}'+'\n')
    else: table='Local-EVM execution evidence is pending.\n'
    (GENERATED/'contract_costs.tex').write_text(table,encoding='utf-8')
    systems_path=ROOT/'results/trust-systems-final/systems-summary.json'
    if systems_path.exists():
        systems=json.loads(systems_path.read_text())
        systems_table=(r'\begin{table}[t]\centering\caption{Complete five-round synthetic development traces replayed on the local EVM. Totals include deployment and enrollment; each population has one trace.}'+'\n'+
            r'\begin{tabular}{rrr}\toprule Clients & Transactions & Total gas \\ \midrule'+'\n')
        for row in systems:
            systems_table+=f'{row["num_clients"]} & {row["transaction_count"]:,} & {row["gas_used"]:,} \\\\\n'
        systems_table+=r'\bottomrule\end{tabular}\label{tab:scaling}\end{table}'+'\n'
    else:
        systems_table='The 20/50/80-client local-EVM trace replay is pending.\n'
    (GENERATED/'systems_costs.tex').write_text(systems_table,encoding='utf-8')
    # A standalone plot is created only when complete confirmatory groups exist.
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from .trust_matrix import ATTACKS,METHODS
        fig,axes=plt.subplots(1,2,figsize=(10,3.3),sharey=True)
        for dataset,ax in zip(('fashion_mnist','cifar10'),axes):
            for method_index,method in enumerate(METHODS):
                selected=[r for r in rows if r['dataset']==dataset and r['defense']==method]
                if not selected: continue
                positions=np.array([ATTACKS.index(r['attack']) for r in selected])+0.09*(method_index-2.5)
                means=np.array([r['final_utility_mean'] for r in selected])
                errors=np.array([[r['final_utility_mean']-r['final_utility_ci_lower'] for r in selected],
                                 [r['final_utility_ci_upper']-r['final_utility_mean'] for r in selected]])
                ax.errorbar(positions,means,yerr=errors,fmt='o',capsize=2,label=method,markersize=3)
            ax.set_title(dataset.replace('_',' ').title())
            ax.set_xticks(range(len(ATTACKS)),[a.replace('_',' ') for a in ATTACKS],rotation=25,ha='right')
            ax.set_ylim(0,1); ax.grid(axis='y',alpha=0.2)
        axes[0].set_ylabel('Held-out accuracy')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.08))
        fig.tight_layout(); fig.savefig(PAPER/'figures/core_accuracy.pdf',bbox_inches='tight'); plt.close(fig)
        (GENERATED/'core_results.tex').write_text(
            r'\begin{figure*}[t]\centering\includegraphics[width=0.95\textwidth]{figures/core_accuracy.pdf}'+'\n'+
            r'\caption{Matched core evaluation. Points show five-seed mean held-out accuracy; bars show bootstrap 95\% confidence intervals. Missing cells are incomplete or held, not zero-valued.}'+'\n'+
            r'\Description{Two dataset panels comparing complete five-seed groups across five attack conditions.}'+'\n'+
            r'\end{figure*}'+'\n',encoding='utf-8')

if __name__=='__main__': main()
