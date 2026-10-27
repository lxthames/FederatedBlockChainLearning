import copy
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from scripts.trust_repair_v15 import study
from scripts.trust_repair_v15.repair import (
    RepairRound, behavior, challenge_bank, transformations, weighted_result, choose_repair)
from scripts.trust_repair_v15.research.contracts import ClientUpdate
from scripts.trust_repair_v15.research.runner_v2 import BenchmarkRunner
from scripts.trust_challenge_v14.research.runner_v2 import BenchmarkRunner as PreviousRunner
from scripts.trust_challenge_v14.research.config import ExperimentConfig as PreviousConfig


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        yield


def small(arm='clip_only', attack='none'):
    return replace(study.config('fashion_mnist',arm,11,attack), dataset='synthetic',num_classes=2,
        num_clients=20,participation_rate=.25,rounds=3,attack_start_round=2,
        natural_splits_path=None,device='cpu',server_samples_per_class=2,
        batch_size=16,dirichlet_alpha=100.)


def arrays():
    x=np.random.default_rng(71).random((1000,1,8,8),dtype=np.float32)
    return x,(x[:,0,0,0]>.5).astype('int64')


def factory():
    return nn.Sequential(nn.Flatten(),nn.Linear(64,2))


def limits(value=10.):
    return {stage:{k:value for k in ('absolute','incremental','clean_excess')} for stage in ('screen','verify')}


def run(c,path,arm,thresholds=None,eval_labels=None):
    runner=BenchmarkRunner(c,path)
    runner.challenge_arm,runner.challenge_thresholds=arm,thresholds
    x,y=arrays()
    return runner.run_arrays(x,y,factory,None if eval_labels is None else (x[:20],eval_labels))


def test_manifest_is_114_unique_100_round_cells_with_clean_first_and_matching_controls():
    rows=study.manifest()
    assert len(rows)==114 and len({r['config']['experiment_id'] for r in rows})==114
    assert all(r['config']['rounds']==100 for r in rows)
    assert all(r['role']=='calibration' and r['attack']=='none' for r in rows[:6])
    assert all(r['attack']=='none' for r in rows[:42])
    assert all(r['attack']!='none' for r in rows[42:])
    for arm in ('clip_only','aggregate','repair'):
        assert study.config('femnist',arm,11).ranking_keep_fraction==1.
        assert study.config('femnist',arm,11).trust_updates=='fixed'


def test_bank_spans_boundaries_is_deterministic_and_does_not_mutate_rng_or_inputs():
    image=np.full((2,1,28,28),.3,dtype='float32')
    original=image.copy()
    specs=transformations(28,28)
    assert len(specs)==147
    assert (2,0.,0,0) in specs and (2,1.,26,26) in specs and (4,.5,24,24) in specs
    np.random.seed(42);expected=np.random.random();np.random.seed(42)
    bank=challenge_bank(image,11,1,3)
    assert np.random.random()==expected
    assert bank.shape==(298,1,28,28) and 0<=bank.min()<=bank.max()<=1
    assert np.array_equal(image,original) and np.array_equal(bank,challenge_bank(image,11,1,3))


def test_absolute_score_exposes_parent_behavior_when_incremental_score_is_zero():
    parent=np.full((3,4,2),.5)
    parent[1,:,0]=.95;parent[1,:,1]=.05
    metrics,_,_=behavior(parent,parent)
    assert metrics['absolute']==pytest.approx(.45)
    assert metrics['incremental']==0
    shifted=np.full_like(parent,.5);shifted[:,:,0]=.8;shifted[:,:,1]=.2
    assert behavior(shifted,shifted)[0]['absolute']==0
    bad=parent.copy();bad[0,0,0]=np.nan
    with pytest.raises(ValueError):behavior(parent,bad)


def updates():
    return [ClientUpdate(i,{'weight':torch.tensor([float(i+1)])},i+1) for i in range(3)]


def test_omission_renormalizes_weights_and_selection_requires_real_improvement():
    result=weighted_result(updates(),omitted=2)
    assert result.state['weight'].item()==pytest.approx(5/3)
    assert result.accepted_client_ids==(0,1) and result.rejected_client_ids==(2,)
    original=dict(risk=2.,passed=False)
    assert choose_repair(original,[(0,dict(risk=.1,passed=False)),(1,dict(risk=2.,passed=True))]) is None
    assert choose_repair(original,[(0,dict(risk=.4,passed=True)),(1,dict(risk=.2,passed=True))])[0]==1


@pytest.mark.parametrize('arm',['clip_only','calibrate'])
def test_passive_or_disabled_validation_matches_previous_clipping_model_trajectory(tmp_path,arm):
    c=small()
    run(c,tmp_path/'new',arm)
    old=PreviousConfig(**dict(c.to_dict(),protocol_version='trust-challenge-v14'))
    x,y=arrays();PreviousRunner(old,tmp_path/'old').run_arrays(x,y,factory)
    trace=lambda name:[r['model_hash'] for r in study.read(tmp_path/name/'chain-trace.json')]
    assert trace('new')==trace('old')


@pytest.mark.parametrize('arm',['aggregate','repair'])
def test_clean_attack_pre_onset_match_and_disjoint_noncontributor_panels(tmp_path,monkeypatch,arm):
    records=[]
    for attack in ('none','model_replacement'):
        c=small(arm,attack);folder=tmp_path/attack
        summary=run(c,folder,arm,limits())
        records.append(study.enrich(summary,folder,arm))
        panels={}
        contributors={r['round']:set(r['accepted_client_ids']) for r in study.read(folder/'chain-trace.json') if r.get('round',0)>0}
        for line in (folder/'events.jsonl').read_text().splitlines():
            e=study.json.loads(line)
            if e['event_type']=='repair_panel':
                p=e['payload'];panels[e['round_index'],p['stage']]=set(p['members'])
        for rnd in (1,2,3):
            screen,verify=panels[rnd,'screen'],panels[rnd,'verify']
            assert len(screen)==len(verify)==5 and not screen&verify
            assert not (screen|verify)&contributors[rnd]
        assert summary['validation_work']['repair_example_evaluations']>0
    assert study.parent.condition_pair(*records)['matched']
    monkeypatch.setattr(study,'JOB',tmp_path/'report')
    study.analyze(records,dict(status='test'))
    report=study.read(study.JOB/'analysis.json')
    assert len(report['runs'])==2
    assert report['condition_pairs'][0]['matched']
    assert report['runs'][0]['artifact_bytes']>0
    assert report['runs'][0]['aggregate_holds']==0
    assert not any(group['complete'] for group in report['groups'])


def engine(tmp_path,arm='repair'):
    runner=BenchmarkRunner(small(arm),tmp_path)
    runner.challenge_arm,runner.challenge_thresholds=arm,limits()
    us=updates()
    obj=RepairRound(runner,factory(),*arrays(),SimpleNamespace(validation=[]),2,us,list(range(20)))
    return obj,weighted_result(us)


@pytest.mark.parametrize('verification_pass',[True,False])
def test_repair_selects_by_screen_only_and_verifies_once_without_retry(tmp_path,verification_pass):
    obj,full=engine(tmp_path)
    calls=[]
    def evaluate(result,stage,omitted=None):
        calls.append((stage,omitted))
        if omitted is None:return dict(passed=False,risk=2.)
        if stage=='verify':return dict(passed=verification_pass,risk=.1)
        return dict(passed=omitted in (0,1),risk=.1 if omitted==0 else .2)
    obj.evaluate=evaluate;obj.archive=lambda *a,**kw:None
    repaired,blocked=obj.aggregate(full)
    assert calls==[('screen',None),('verify',None),('screen',0),('screen',1),('screen',2),('verify',0)]
    if verification_pass:
        assert not blocked and repaired.accepted_client_ids==(1,2)
        assert torch.equal(repaired.state['weight'],weighted_result(updates(),0).state['weight'])
    else:
        assert blocked and not repaired.accepted_client_ids
        assert repaired.rejected_client_ids==(0,1,2)
        assert torch.count_nonzero(repaired.state['weight'])==0


def test_verification_only_failure_holds_without_search(tmp_path):
    obj,result=engine(tmp_path)
    calls=[]
    def evaluate(result,stage,omitted=None):
        calls.append((stage,omitted));return dict(passed=stage=='screen',risk=1.)
    obj.evaluate=evaluate;obj.archive=lambda *a,**kw:None
    held,blocked=obj.aggregate(result)
    assert blocked and not held.accepted_client_ids
    assert calls==[('screen',None),('verify',None)]


def test_evaluation_labels_cannot_influence_training_or_repair_decisions(tmp_path):
    c=small('repair')
    for name,labels in [('zero',np.zeros(20,dtype='int64')),('one',np.ones(20,dtype='int64'))]:
        run(c,tmp_path/name,'repair',limits(),labels)
    hashes=lambda name:[r['model_hash'] for r in study.read(tmp_path/name/'chain-trace.json')]
    assert hashes('zero')==hashes('one')


def test_gate_requires_every_clean_cell_and_limits_holds(monkeypatch):
    monkeypatch.setattr(study.parent,'method_pair',lambda a,b:dict(matched=True))
    rows=[]
    for d in study.DATASETS:
        for s in study.SEEDS:
            for arm in ('clip_only','repair'):
                rows.append(dict(dataset=d,seed=s,arm=arm,attack='none',final_utility=.7,
                    unsupported_rounds=0,honest_validator_penalty_amount=0))
    assert study.gate(rows,'repair')['passed']
    assert not study.gate(rows[:-1],'repair')['passed']
    rows[-1]['unsupported_rounds']=4
    assert not study.gate(rows,'repair')['passed']


def test_missing_calibration_fails_closed_and_seal_binds_npz_and_settings(tmp_path):
    with pytest.raises(ValueError,match='calibration'):
        run(small('repair'),tmp_path/'bad','repair')
    folder=tmp_path/'good';c=small()
    run(c,folder,'clip_only')
    settings=dict(arm='clip_only')
    study.dump(folder/'repair-settings.json',settings)
    study.dump(folder/'diagnostic-inputs.json',study.parent.EMPTY_BINDING)
    behavior_dir=folder/'behavior';behavior_dir.mkdir()
    np.savez_compressed(behavior_dir/'test.npz',value=np.array([1.]))
    study.dump(folder/'completion.json',study.sealed_files(folder))
    study.verify_cell(folder,c,settings)
    np.savez_compressed(behavior_dir/'test.npz',value=np.array([2.]))
    with pytest.raises(ValueError,match='seal'):
        study.verify_cell(folder,c,settings)
