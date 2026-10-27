"""Descriptive clean-run audit; no held-out labels or attacker identities read."""
import json
from pathlib import Path
import numpy as np
from scripts.data_preparation import run_comparison_v8 as v8
from .femnist import dump,digest

ROOT=Path(__file__).resolve().parents[2]
JOB=ROOT/'artifacts/bias-v9-pilot'


def audit(folder,datafolder):
    summary=json.loads((folder/'summary.json').read_text())
    if summary['attack']!='none':raise ValueError('Diagnosis requires clean controls')
    split=json.loads((folder/'splits.json').read_text())
    y=np.load(datafolder/'labels.npy',allow_pickle=False)
    classes=int(y.max())+1
    def hist(indices):
        h=np.bincount(y[indices],minlength=classes).astype(float)
        return h/h.sum()
    train=[hist(ids) for ids in split['train']]
    validation=[hist(ids) for ids in split['validation']]
    overlap=[];approval=[];negative_score_approved=0;rounds=[]
    for line in (folder/'events.jsonl').read_text().splitlines():
        event=json.loads(line);p=event['payload']
        if event['event_type']=='loss_vote':
            overlap.append(float(np.minimum(train[p['candidate']],validation[p['validator']]).sum()))
            approval.append(int(p['approve']))
            negative_score_approved+=int(p['approve'] and p['loss_score']<0)
        if event['event_type']=='round_completed':rounds.append(p)
    a=np.asarray(approval);o=np.asarray(overlap)
    low,high=np.quantile(o,[.25,.75])
    trust=np.asarray(list(summary['final_trust'].values()),dtype=float)
    threshold=summary['configuration']['minimum_trust']
    return {'dataset':summary['dataset'],'seed':summary['seed'],
            'final_utility':summary['final_utility'],
            'benign_update_exclusion_rate':sum(len(p['participants'])-len(p['accepted_client_ids']) for p in rounds)/sum(len(p['participants']) for p in rounds),
            'vote_count':len(a),'approval_rate':float(a.mean()),
            'class_overlap_mean':float(o.mean()),
            'low_overlap_approval':float(a[o<=low].mean()),'high_overlap_approval':float(a[o>=high].mean()),
            'overlap_approval_correlation':float(np.corrcoef(o,a)[0,1]) if o.std()>0 and a.std()>0 else None,
            'negative_rank_score_approvals':negative_score_approved,
            'clients_below_validator_trust_threshold':int((trust<threshold).sum()),
            'minimum_final_trust':float(trust.min()),
            'inputs':{str(p.relative_to(ROOT)):digest(p) for p in
                (folder/'summary.json',folder/'events.jsonl',folder/'splits.json',datafolder/'labels.npy')},
            'interpretation':'Descriptive correlated votes, not independent samples or causal evidence. Negative rank score includes norm penalty.'}


def run():
    records=[]
    for dataset in ('fashion_mnist','femnist'):
        for seed in v8.SEEDS:
            if dataset=='fashion_mnist':
                folder=next(p for m,c,p in v8.reuse_cells() if m=='rank_trust' and c.seed==seed and c.attack=='none')
            else:folder=v8.OUT/v8.config(dataset,'rank_trust','none',seed).experiment_id
            records.append(audit(folder,v8.v7.DATA/f'{dataset}-{seed}'))
    dump(JOB/'diagnosis.json',{'runs':records,'publication_ready':False,
        'mathematical_issue':'A fixed top-80% rank discards a tail even when all submitted updates are benign. Local label support changes the objective being ranked; correlated quorum decisions do not remove this bias.'})
    lines=['# Clean-validation bias diagnosis','',
           '| Dataset | Seed | Benign-update exclusion | Low-overlap approval | High-overlap approval | Overlap correlation | Below trust threshold |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['dataset']} | {r['seed']} | {100*r['benign_update_exclusion_rate']:.1f}% | {100*r['low_overlap_approval']:.1f}% | {100*r['high_overlap_approval']:.1f}% | {r['overlap_approval_correlation']:.3f} | {r['clients_below_validator_trust_threshold']} |")
    lines+=['','Votes share models, data and rounds; these correlations establish no causal or significance claim.',
            'Only training/private-validation labels were read. See diagnosis.json for artifact hashes.',
            'Benign-update exclusion counts all clients in clean runs. Earlier honest-exclusion summaries counted only clients outside the simulator potential-attacker role; those denominators differ.']
    (JOB/'diagnosis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':run()
