import copy
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from scripts.trust_challenge_v14 import study
from scripts.trust_challenge_v14.challenges import challenge_bank, collapse_score, passes, ChallengeRound
from scripts.trust_challenge_v14.research.runner_v2 import BenchmarkRunner
from scripts.trust_security_v13.research.runner_v2 import BenchmarkRunner as OldRunner
from scripts.trust_security_v13.research.config import ExperimentConfig as OldConfig


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        yield


def small(arm='current',attack='none'):
    return replace(study.config('fashion_mnist',arm,11,attack),dataset='synthetic',num_classes=2,
        num_clients=5,participation_rate=1.,rounds=3,attack_start_round=2,natural_splits_path=None,
        device='cpu',server_samples_per_class=2,batch_size=16,dirichlet_alpha=100.)


def arrays():
    x=np.random.default_rng(71).random((200,1,8,8),dtype=np.float32)
    return x,(x[:,0,0,0]>.5).astype('int64')


def factory():return nn.Sequential(nn.Flatten(),nn.Linear(64,2))


def run(c,path,arm,thresholds=None,eval_labels=None):
    runner=BenchmarkRunner(c,path);runner.challenge_arm=arm;runner.challenge_thresholds=thresholds
    x,y=arrays()
    return runner.run_arrays(x,y,factory,None if eval_labels is None else (x[:20],eval_labels))


def test_matrix_is_finite_clean_first_with_passive_calibration():
    rows=study.manifest()
    assert len(rows)==60 and len({r['config']['experiment_id'] for r in rows})==60
    assert all(r['arm']=='clip_only' and r['attack']=='none' for r in rows[:6])
    assert all(r['attack']=='none' for r in rows[:30])
    assert all(r['attack']=='model_replacement' for r in rows[30:])


def test_bank_is_deterministic_bounded_and_does_not_mutate_inputs_or_global_rng():
    x,_=arrays();original=x[:16].copy();np.random.seed(42);expected=np.random.random()
    np.random.seed(42);bank=challenge_bank(original,11,1,3)
    assert np.random.random()==expected
    assert bank.shape==(80,1,8,8) and bank.min()>=0 and bank.max()<=1
    assert np.array_equal(original,x[:16]) and np.array_equal(bank[:16],original)
    assert np.array_equal(bank,challenge_bank(original,11,1,3))
    assert not np.array_equal(bank,challenge_bank(original,11,2,3))


def test_score_detects_perturbation_specific_concentration_and_subtracts_clean_shift():
    parent=np.full((5,16,2),.5,dtype='float32')
    candidate=parent.copy();candidate[1,:,0]=.9;candidate[1,:,1]=.1
    assert collapse_score(parent,parent)==0
    assert collapse_score(parent,candidate)==pytest.approx(.4)
    uniform_shift=parent.copy();uniform_shift[:,:,0]=.8;uniform_shift[:,:,1]=.2
    assert collapse_score(parent,uniform_shift)==pytest.approx(0)
    assert passes(.2,.2) and not passes(.21,.2)
    with pytest.raises(ValueError):passes(float('nan'),.2)


@pytest.mark.parametrize('attack',['none','model_replacement'])
def test_passive_calibration_matches_parent_and_changes_no_models(tmp_path,attack):
    c=small('clip_only',attack);new=run(c,tmp_path/'new','calibrate')
    old=OldConfig(**dict(c.to_dict(),protocol_version='trust-security-v13'))
    x,y=arrays();OldRunner(old,tmp_path/'old').run_arrays(x,y,factory)
    hashes=lambda name:[r['model_hash'] for r in study.read(tmp_path/name/'chain-trace.json')]
    assert hashes('new')==hashes('old')
    assert new['validation_work']['challenge_example_evaluations']>0


@pytest.mark.parametrize('arm',['current','perturbation','aggregate','combined'])
def test_clean_attack_pre_onset_match_and_challenge_work(tmp_path,arm):
    t=dict(candidate=.02,aggregate=.02)
    summaries=[]
    for attack in ('none','model_replacement'):
        c=small(arm,attack);folder=tmp_path/attack
        s=run(c,folder,arm,t);summaries.append(dict(study.parent.enrich(s,folder),arm=arm))
    assert study.parent.condition_pair(*summaries)['matched']
    if arm=='current':assert 'challenge_example_evaluations' not in summaries[0]['validation_work']
    else:assert summaries[0]['validation_work']['challenge_example_evaluations']>0


def test_evaluation_labels_do_not_influence_challenge_or_training(tmp_path):
    t=dict(candidate=.02,aggregate=.02);c=small('combined')
    for name,labels in [('zero',np.zeros(20,dtype='int64')),('one',np.ones(20,dtype='int64'))]:
        run(c,tmp_path/name,'combined',t,labels)
    traces=[study.read(tmp_path/n/'chain-trace.json') for n in ('zero','one')]
    assert [r['model_hash'] for r in traces[0]]==[r['model_hash'] for r in traces[1]]


def test_aggregate_checks_actual_result_not_nominal_probe_and_blocks_without_penalty(tmp_path):
    from scripts.trust_challenge_v14.research.contracts import AggregationResult
    from scripts.trust_challenge_v14.research.runner_helpers import _apply_delta
    model=factory();parent_state=copy.deepcopy(model.state_dict())
    runner=BenchmarkRunner(small(),tmp_path);runner.challenge_arm='aggregate'
    runner.challenge_thresholds=dict(candidate=.01,aggregate=.01)
    x,y=arrays();split=SimpleNamespace(validation=[np.arange(16)]*5)
    engine=ChallengeRound(runner,model,x,y,split,1,[],list(range(5)))
    delta={k:torch.ones_like(v)*.1 for k,v in model.state_dict().items()}
    evaluated=[]
    def fake(candidate,validator,stage,client=None):
        evaluated.append(copy.deepcopy(candidate.state_dict()));return False,True
    engine.evaluate=fake
    result,blocked=engine.aggregate(AggregationResult(delta,(1,2)))
    assert blocked and not result.accepted_client_ids and set(result.rejected_client_ids)=={1,2}
    assert all(torch.count_nonzero(v)==0 for v in result.state.values())
    for k in parent_state:
        assert torch.equal(model.state_dict()[k],parent_state[k])
        assert torch.equal(evaluated[0][k],parent_state[k]+delta[k])


def test_missing_calibration_fails_closed(tmp_path):
    c=small('perturbation')
    with pytest.raises(ValueError,match='frozen clean calibration'):run(c,tmp_path,'perturbation')


def test_gate_requires_all_six_clean_cells_and_limits_vetoes(monkeypatch):
    monkeypatch.setattr(study.parent,'method_pair',lambda a,b:dict(matched=True))
    rows=[]
    for d in study.DATASETS:
        for s in study.SEEDS:
            for arm in ('clip_only','combined'):
                rows.append(dict(dataset=d,seed=s,arm=arm,attack='none',final_utility=.7,
                    unsupported_rounds=0,honest_validator_penalty_amount=0))
    assert study.gate(rows,'combined')['passed']
    assert not study.gate(rows[:-1],'combined')['passed']
    rows[-1]['unsupported_rounds']=4
    assert not study.gate(rows,'combined')['passed']
    rows[-1]['unsupported_rounds']=0;rows[-1]['final_utility']=.64
    assert not study.gate(rows,'combined')['passed']


def test_seal_binds_external_challenge_settings_and_detects_mutation(tmp_path):
    c=small('clip_only');run(c,tmp_path,'calibrate')
    settings=dict(arm='calibrate',thresholds=None,calibration_sha256=None)
    study.dump(tmp_path/'challenge-settings.json',settings)
    study.dump(tmp_path/'diagnostic-inputs.json',study.parent.EMPTY_BINDING)
    study.dump(tmp_path/'completion.json',{n:study.digest(tmp_path/n) for n in study.FILES})
    study.verify_cell(tmp_path,c,settings)
    with pytest.raises(ValueError,match='challenge binding'):
        study.verify_cell(tmp_path,c,dict(settings,thresholds={'candidate':1.}))
    study.dump(tmp_path/'challenge-settings.json',dict(settings,arm='combined'))
    with pytest.raises(ValueError,match='seal'):
        study.verify_cell(tmp_path,c,settings)
