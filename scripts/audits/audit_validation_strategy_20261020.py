"""Bounded reproductions; never modifies frozen experiment sources/results."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from scripts.trust_bias_v9.loss_voting import LossCommittee
from scripts.trust_bias_v9.research.config import ExperimentConfig
from scripts.trust_bias_v9.research.contracts import ClientUpdate
from scripts.trust_bias_v9.research.runner_v2 import BenchmarkRunner
from scripts.trust_bias_v9.research.splits import DataSplit, matched_test_shards
from scripts.data_preparation.run_comparison_v8 import paired_row

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/validation-strategy-audit-20261020'


def withholding():
    c = ExperimentConfig(experiment_id='audit-withholding', dataset='synthetic',
        num_clients=4, participation_rate=1., rounds=4, seed=11,
        defense='two_layer', vote_rule='rank_loss', ranking_keep_fraction=1.,
        ablation='validation_only', malicious_fraction=0.,
        malicious_validator_fraction=.5, validator_behavior='withhold',
        device='cpu', num_classes=2, server_samples_per_class=2,
        candidate_clip_ratio=2.)
    rng = np.random.default_rng(71)
    x = rng.normal(size=(160,3)).astype('float32')
    y = (x[:,0] > 0).astype('int64')
    summary = BenchmarkRunner(c, OUT/'withholding').run_arrays(x,y,lambda:nn.Linear(3,2))
    trace = json.loads((OUT/'withholding/chain-trace.json').read_text())
    corrupt = set(summary['malicious_validator_ids'])
    honest = set(range(4))-corrupt
    third = trace[3]
    fourth = trace[4]
    assert all(summary['final_trust'][str(i)] < c.minimum_trust for i in honest)
    assert all(summary['final_trust'][str(i)] > c.initial_trust for i in corrupt)
    assert len(third['candidates']) == 4 and not fourth['candidates']
    assert all(v['approve'] for row in third['candidates'] for v in row['votes'])
    return {'poisoning_clients':summary['malicious_client_ids'],
        'withholding_validators':sorted(corrupt),'honest_validators':sorted(honest),
        'final_trust':summary['final_trust'],
        'round_3_accepted':third['accepted_client_ids'],
        'round_4_candidate_count':len(fourth['candidates']),
        'round_4_accepted':fourth['accepted_client_ids'],
        'unsupported_rounds':summary['unsupported_rounds'],
        'all_received_round_3_votes_approve':True}


def moving_test_groups():
    # Training is fixed; repartition only the same 80 validation examples.
    y = np.array([0]*10+[1]*10+[0]*40+[1]*40)
    ty = np.array([0]*100+[1]*100)
    train = [np.arange(10),np.arange(10,20)]
    private = [np.arange(20,60),np.arange(60,100)]
    mixed = [np.r_[20:40,60:80],np.r_[40:60,80:100]]
    out = {}
    for name,validation in [('private',private),('redistributed',mixed)]:
        split = DataSplit(train,validation,np.array([],dtype=int),np.array([],dtype=int))
        groups = matched_test_shards(y,ty,split,11)
        # Predictions are unchanged and always label 0.
        utilities = [float(np.mean(ty[g]==0)) for g in groups]
        out[name] = {'client_utilities':utilities,'worst':min(utilities),
            'lower_decile':float(np.quantile(utilities,.1)),
            'test_counts':[len(g) for g in groups]}
    assert out['private']['worst']==0 and out['redistributed']['worst']>0
    out['unchanged_global_accuracy'] = .5
    return out


def relative_ranking():
    c=ExperimentConfig(experiment_id='audit-rank',dataset='synthetic',num_clients=6,
        participation_rate=1.,defense='two_layer',vote_rule='rank_loss',
        ablation='validation_only',ranking_norm_penalty=0.,device='cpu')
    updates=[ClientUpdate(i,{'weight':torch.ones(2,1)*.01},1) for i in range(5)]
    committee=LossCommittee(SimpleNamespace(config=c),nn.Linear(1,2,bias=False),
        np.ones((2,1)),np.array([0,1]),SimpleNamespace(validation=[np.arange(2)]*6),updates)
    results={}
    for condition,changes in [('all_helpful',[-.5,-.4,-.3,-.2,-.1]),
                              ('all_harmful',[.1,.2,.3,.4,.5])]:
        committee.pair=lambda validator,client:(np.ones(2),np.ones(2)+changes[client])
        votes=[committee.vote(5,i,False)[0].approve for i in range(5)]
        results[condition]={'loss_changes':changes,'approvals':votes}
    assert results['all_helpful']['approvals']==[True,True,True,True,False]
    assert results['all_harmful']['approvals']==[True,True,True,True,False]
    return results


def weak_pairing_check():
    common=dict(dataset='fashion_mnist',seed=11,attack='none',final_utility=.7,
        participants=[[0,1]],data_fingerprint='same-arrays',test_fingerprint='same-test',
        unsupported_rounds=0,honest_validator_penalty_amount=0,honest_update_exclusion_rate=0.)
    baseline=dict(common,method='fedavg',train_split_digest='split-A')
    candidate=dict(common,method='probe',train_split_digest='split-B')
    row=paired_row([baseline,candidate],candidate)
    assert row['matched']
    return {'different_training_split_digests':True,'reported_matched':row['matched'],
        'scope':'Synthetic metadata demonstrates omitted split binding, not a known mismatch in V8.'}


def main():
    if OUT.exists():
        raise FileExistsError(f'Preserve prior audit output: {OUT}')
    OUT.mkdir(parents=True)
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        report={'withholding':withholding(),'evaluation_group_drift':moving_test_groups(),
                'rank_is_relative':relative_ranking(),'pairing_check':weak_pairing_check()}
    paths=[Path(__file__),ROOT/'scripts/trust_bias_v9/research/runner_v2.py',
           ROOT/'scripts/trust_bias_v9/research/splits.py',ROOT/'scripts/trust_bias_v9/research/defenses.py',
           ROOT/'scripts/trust_bias_v9/loss_voting.py',ROOT/'scripts/data_preparation/run_comparison_v8.py']
    report['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report['scope']='Small deterministic reproductions; no efficacy claim; frozen results unchanged.'
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
