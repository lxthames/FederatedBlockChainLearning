import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from scripts.trust_extension.fedval import score_weights, FedVal
from scripts.trust_extension.adaptive import constrained_replacement
from scripts.trust_extension.research.config import ExperimentConfig
from scripts.trust_extension.research.runner_v2 import BenchmarkRunner
from scripts.trust_extension.research.splits import make_splits, matched_test_shards


def cfg(**changes):
    return ExperimentConfig(**(dict(experiment_id='extension-test', dataset='synthetic',
        num_clients=4, participation_rate=1, rounds=4, seed=7, device='cpu',
        server_samples_per_class=2) | changes))


def arrays():
    rng = np.random.default_rng(5)
    y = np.tile([0,1],100)
    return (rng.normal(size=(200,2))+y[:,None]).astype(np.float32), y


def test_fedval_hand_computed_weights_and_identical_losses():
    assert np.allclose(score_weights([[1,1],[3,3]],3), [2/3,1/3])
    assert np.allclose(score_weights(np.ones((4,3)),3), np.full(4,0.25))
    assert score_weights([[0.1,0.2],[1,1],[10,10]],3)[0] > score_weights([[0.1,0.2],[1,1],[10,10]],3)[2]


@pytest.mark.parametrize('losses', [[[float('nan')]], [[-1]], [1,2]])
def test_fedval_rejects_invalid_loss_matrix(losses):
    with pytest.raises(ValueError): score_weights(losses,3)


def test_fedval_requires_class_coverage(tmp_path):
    x,y=arrays(); runner=BenchmarkRunner(cfg(),tmp_path)
    with pytest.raises(ValueError,match='every task label'):
        FedVal(runner,x[y==0],y[y==0],2)


def test_fedval_executes_without_test_data_access(tmp_path):
    x,y=arrays()
    traces=[]
    for name, test_labels in [('first',y),('second',1-y)]:
        c=cfg(defense='fedval',evaluation_batch_size=3)
        result=BenchmarkRunner(c,tmp_path/name).run_arrays(x,y,lambda:nn.Linear(2,2),(x+2,test_labels))
        assert result['status']=='completed'
        traces.append(json.loads((tmp_path/name/'chain-trace.json').read_text()))
    assert traces[0]==traces[1]


def test_scheduled_recovery_is_matched_and_penalties_persist(tmp_path):
    x,y=arrays(); results={}; traces={}
    for policy in ('continue','scheduled_plain','scheduled_accountable'):
        c=cfg(defense='two_layer',recovery_policy=policy,intervention_round=2,
              intervention_checkpoint=1, minimum_validation_delta=-100,
              minimum_trust=0,trust_floor=0,trust_penalty=0,trust_reward=0)
        results[policy]=BenchmarkRunner(c,tmp_path/policy).run_arrays(x,y,lambda:nn.Linear(2,2))
        traces[policy]=json.loads((tmp_path/policy/'chain-trace.json').read_text())
    assert len({traces[p][2]['model_hash'] for p in traces})==1
    assert results['continue']['recovery_count']==0
    assert results['scheduled_plain']['recovery_count']==1
    assert results['scheduled_accountable']['recovery_count']==1
    assert results['scheduled_plain']['honest_validator_penalty_amount']==0
    assert results['scheduled_accountable']['honest_validator_penalty_amount']>0
    assert traces['scheduled_plain'][2]['recovery']['restored_version']==1
    assert len(traces['scheduled_plain'])==5  # History survives restoration.


def test_natural_split_rejects_overlaps_and_preserves_test_clients(tmp_path):
    data={'seed':7,'writers':['a','b','c','d'],'reference_writers':['e'],
          'train':[[0],[1],[2],[3]],'validation':[[4],[5],[6],[7]],
          'reference':[8,9],'test_clients':[[0],[1],[2],[3]]}
    path=tmp_path/'splits.json';path.write_text(json.dumps(data))
    c=cfg(natural_splits_path=str(path))
    split=make_splits(np.arange(10)%2,c,external_test=True)
    assert [a.tolist() for a in matched_test_shards(np.arange(10)%2,np.array([0,1,0,1]),split,7)]==[[0],[1],[2],[3]]
    data['validation'][0]=[0];path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='overlap'):
        make_splits(np.arange(10)%2,c,external_test=True)


def test_false_reports_require_recovery_quorum(tmp_path):
    x,y=arrays()
    runner=BenchmarkRunner(cfg(num_clients=10,defense='two_layer',rounds=8,
        malicious_validator_fraction=0.1,validator_behavior='false_recovery',
        minimum_validation_delta=-100),tmp_path)
    runner._score=lambda *args:1.0
    summary=runner.run_arrays(x,y,lambda:nn.Linear(2,2))
    assert summary['recovery_count']==0
    trace=json.loads((tmp_path/'chain-trace.json').read_text())
    assert all(row.get('recovery') is None for row in trace[1:])


def test_adaptive_update_obeys_proxy_constraint_and_norm_budget(tmp_path):
    x,y=arrays(); c=cfg()
    runner=BenchmarkRunner(c,tmp_path); torch.manual_seed(7); model=nn.Linear(2,2)
    honest={k:torch.ones_like(v)*0.01 for k,v in model.state_dict().items()}
    poisoned={k:torch.ones_like(v)*2 for k,v in model.state_dict().items()}
    delta,record=constrained_replacement(runner,model,honest,poisoned,x,y)
    actual=float(torch.cat([v.flatten() for v in delta.values()]).norm())
    assert actual<=record['norm_bound']+1e-6
    assert record['proxy_clean_accuracy']>=record['proxy_baseline']-c.adaptive_clean_tolerance


def test_femnist_provenance_and_natural_test_correspondence():
    root=Path(__file__).resolve().parents[1]/'data/femnist-tff'
    if not (root/'provenance.json').exists():pytest.skip('download not complete')
    from scripts.trust_extension.femnist import digest
    provenance=json.loads((root/'provenance.json').read_text())
    assert provenance['classes']==62
    for row in provenance['seeds']:
        folder=root/f"seed-{row['seed']}"
        for name,expected in row['files'].items():assert digest(folder/name)==expected
        data=json.loads((folder/'natural-splits.json').read_text())
        assert not set(data['writers'])&set(data['reference_writers'])
        assert len(data['writers'])==50
        labels=np.load(folder/'labels.npy');test_labels=np.load(folder/'test_labels.npy')
        split=make_splits(labels,cfg(seed=row['seed'],num_clients=50,natural_splits_path=str(folder/'natural-splits.json')),True)
        assert len(matched_test_shards(labels,test_labels,split,row['seed']))==50
        import hashlib
        features=np.load(folder/'features.npy');test=np.load(folder/'test_features.npy')
        hashes=[hashlib.sha256(image.tobytes()).digest() for image in features]
        test_hashes={hashlib.sha256(image.tobytes()).digest() for image in test}
        assert len(hashes)==len(set(hashes))
        assert not set(hashes)&test_hashes
    assert provenance['archive_sha256']==provenance['expected_archive_sha256']


def test_femnist_trigger_uses_ink_not_white_background(tmp_path):
    runner=BenchmarkRunner(cfg(dataset='femnist'),tmp_path)
    original=np.ones((2,1,28,28),dtype=np.float32)
    triggered=runner._apply_trigger(original)
    assert np.all(triggered[:,:,-2:,-2:]==0)
    assert np.all(original==1)
