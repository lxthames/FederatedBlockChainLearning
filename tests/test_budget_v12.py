from dataclasses import replace
import copy
import json
import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from scripts.trust_budget_v12 import study
from scripts.trust_budget_v12.research.runner_v2 import BenchmarkRunner
from scripts.trust_budget_v12.research.contracts import ValidationVote
from scripts.trust_budget_v12.loss_voting import LossCommittee
from scripts.trust_screen_v11 import study as v11
from scripts.trust_screen_v11.research.runner_v2 import BenchmarkRunner as V11Runner


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        yield


def small(arm='fedavg',budget='epoch',rounds=2):
    return replace(study.config('default',budget,arm,11),dataset='synthetic',num_classes=2,
        num_clients=5,participation_rate=1.,rounds=rounds,natural_splits_path=None,
        device='cpu',server_samples_per_class=2,batch_size=8,dirichlet_alpha=.5)


def arrays():
    x=np.random.default_rng(71).normal(size=(200,3)).astype('float32')
    return x,(x[:,0]>0).astype('int64')


def run_cell(c,path,binding=None,test_labels=None):
    binding=binding or {'mass_schedule':None,'source':None}
    runner=BenchmarkRunner(c,path,mass_schedule=binding['mass_schedule'],mass_source_binding=binding['source'])
    study.dump(path/'diagnostic-inputs.json',binding)
    x,y=arrays()
    return runner.run_arrays(x,y,lambda:nn.Linear(3,2),None if test_labels is None else (x[:20],test_labels))


def test_finite_manifest_and_dependency_order():
    rows=study.manifest()
    assert len(rows)==60 and len({r['config']['experiment_id'] for r in rows})==60
    assert all(r['model']=='default' for r in rows[:30])
    assert all(r['config']['attack']=='none' and r['config']['trust_updates']=='fixed' for r in rows)
    seen=set()
    for row in rows:
        if row['dependency']:assert row['dependency'] in seen
        seen.add(row['config']['experiment_id'])
    for kwargs in ({'local_steps':0},{'local_steps':True},{'local_training_budget':'unknown'},
                   {'local_training_budget':'fixed_steps','local_epochs':2},
                   {'aggregation_mass_policy':'nominal'}):
        with pytest.raises(ValueError):replace(small(),**kwargs)


def test_backbone_shapes_and_default_parity():
    for model in study.MODELS:
        torch.manual_seed(11)
        m=study.create_model('femnist',(1,28,28),62,model)
        assert m(torch.zeros(3,1,28,28)).shape==(3,62)
        assert not list(m.buffers())
        if model=='default':
            torch.manual_seed(11)
            old=v11.create_model('femnist',(1,28,28),62)
            assert all(torch.equal(v,old.state_dict()[k]) for k,v in m.state_dict().items())
            assert sum(p.numel() for p in m.parameters())==213630
        else:assert sum(p.numel() for p in m.parameters())>213630


@pytest.mark.parametrize('arm,old_arm',[('fedavg','fedavg'),('clip_only','clip_only'),('scaled_nominal','scaled_quorum')])
def test_epoch_controls_reproduce_v11_training_and_models(tmp_path,arm,old_arm):
    c=small(arm);s=run_cell(c,tmp_path/'new')
    old_data={k:v for k,v in c.to_dict().items() if k not in ('local_training_budget','local_steps','aggregation_mass_policy')}
    old_data.update(protocol_version='trust-screen-v11')
    old=v11.ExperimentConfig(**old_data)
    x,y=arrays();o=V11Runner(old,tmp_path/'old').run_arrays(x,y,lambda:nn.Linear(3,2))
    traces=[study.read(tmp_path/n/'chain-trace.json') for n in ('new','old')]
    assert [r['model_hash'] for r in traces[0]]==[r['model_hash'] for r in traces[1]]
    assert s['final_accuracy']==o['final_accuracy']


def test_fixed_steps_exact_even_when_loader_restarts(tmp_path):
    c=replace(small(budget='fixed_steps'),local_steps=5)
    runner=BenchmarkRunner(c,tmp_path)
    x,y=arrays();m=nn.Linear(3,2)
    runner._train(m,x[:17],y[:17],1,0)
    assert runner.local_training_work=={'optimizer_steps':5,'processed_examples':33}


def test_four_step_prefix_matches_a_four_batch_epoch(tmp_path):
    x,y=arrays();a=nn.Linear(3,2);b=copy.deepcopy(a)
    epoch=BenchmarkRunner(small(),tmp_path/'epoch')
    fixed=BenchmarkRunner(small(budget='fixed_steps'),tmp_path/'fixed')
    epoch._train(a,x[:32],y[:32],1,0);fixed._train(b,x[:32],y[:32],1,0)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    assert fixed.local_training_work['optimizer_steps']==4


def test_mass_policies_have_exact_single_round_coefficients(tmp_path,monkeypatch):
    def train(self,model,x,y,rnd,client,poison=False):
        with torch.no_grad():
            for p in model.parameters():p.add_(client+1.)
    def vote(self,validator,client,corrupt,forced=None):
        return ValidationVote(validator,client,0.,self.runner.config.ranking_keep_fraction==1. or client==0,False,'test'),{}
    monkeypatch.setattr(BenchmarkRunner,'_train',train)
    monkeypatch.setattr(LossCommittee,'vote',vote)
    x,y=arrays()
    for arm in ('scaled_nominal','scaled_renormalized','clip_damping_replay'):
        c=small(arm,rounds=1)
        binding={'mass_schedule':[.25],'source':{'test':'bound'}} if arm=='clip_damping_replay' else {'mass_schedule':None,'source':None}
        runner=BenchmarkRunner(c,tmp_path/arm,mass_schedule=binding['mass_schedule'],mass_source_binding=binding['source'])
        model=nn.Linear(3,2);initial={k:v.clone() for k,v in model.state_dict().items()}
        s=runner.run_arrays(x,y,lambda:model)
        splits=study.read(tmp_path/arm/'splits.json');sizes=[len(ids) for ids in splits['train']]
        if arm=='scaled_nominal':expected=sizes[0]/sum(sizes)
        elif arm=='scaled_renormalized':expected=1.
        else:expected=.25*sum(n*(i+1) for i,n in enumerate(sizes))/sum(sizes)
        for k,v in model.state_dict().items():assert torch.allclose(v-initial[k],torch.full_like(v,expected),atol=2e-7)


@pytest.mark.parametrize('bad',[None,[],[1.],[float('nan'),.5],[1.1,.5]])
def test_replay_rejects_missing_or_invalid_schedule(tmp_path,bad):
    with pytest.raises(ValueError):BenchmarkRunner(small('clip_damping_replay'),tmp_path,mass_schedule=bad,mass_source_binding={'source':'x'})


def test_declared_interventions_do_not_hide_other_changes(tmp_path):
    a=run_cell(small(),tmp_path/'epoch')
    b=run_cell(small(budget='fixed_steps'),tmp_path/'fixed')
    assert study.compare_intervention(b,a,('local_training_budget',))['matched']
    assert not study.compare_intervention(b,a)['matched']
    b=copy.deepcopy(b);b['configuration']['learning_rate']*=2
    assert not study.compare_intervention(b,a,('local_training_budget',))['matched']
    b=copy.deepcopy(a);b['configuration']['model']='wide_cnn'
    b['training_binding']=study.digest_value({k:b['configuration'][k] for k in study.TRAINING_FIELDS})
    b['initial_model_hash']='declared-new-shape'
    assert study.compare_intervention(b,a,('model',))['matched']
    b['participants']=[]
    assert not study.compare_intervention(b,a,('model',))['matched']


def test_replay_derivation_and_seals_fail_on_source_tampering(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'OUT',tmp_path)
    c=small('scaled_nominal');c=replace(c,experiment_id='source')
    monkeypatch.setattr(study,'config',lambda *args:c)
    run_cell(c,tmp_path/'source');study.seal(tmp_path/'source')
    row={'arm':'clip_damping_replay','dependency':'source','model':'default','budget':'epoch','seed':11}
    binding=study.replay_inputs(row)
    assert len(binding['mass_schedule'])==2
    assert binding['source']['completion_sha256']==study.digest(tmp_path/'source'/'completion.json')
    with (tmp_path/'source'/'events.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='digest mismatch'):study.replay_inputs(row)


def test_heldout_labels_do_not_affect_fixed_step_models(tmp_path):
    x,y=arrays();c=small('scaled_renormalized',budget='fixed_steps')
    run_cell(c,tmp_path/'a',test_labels=y[:20]);run_cell(c,tmp_path/'b',test_labels=1-y[:20])
    traces=[study.read(tmp_path/n/'chain-trace.json') for n in ('a','b')]
    assert [r['model_hash'] for r in traces[0]]==[r['model_hash'] for r in traces[1]]


def test_work_metrics_and_partial_groups(tmp_path,monkeypatch):
    c=small(budget='fixed_steps');s=run_cell(c,tmp_path/'cell')
    assert s['local_training_work']['optimizer_steps']==2*5*4
    metrics=study.trace_metrics(tmp_path/'cell',s)
    assert metrics['mean_accepted_nominal_mass']==pytest.approx(1.)
    monkeypatch.setattr(study,'JOB',tmp_path/'report')
    r=dict(s,arm='fedavg',budget='fixed_steps',model='default',trace_metrics=metrics)
    study.analyze([r],{})
    report=study.read(study.JOB/'analysis.json')
    assert all(not g['complete'] and g['mean_accuracy'] is None for g in report['groups'])


def test_gpu_queue_requires_successful_parent_and_released_lock(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'PARENT',tmp_path)
    study.dump(tmp_path/'status.json',{'status':'running'})
    assert not study.parent_ready()
    state={'status':'completed_with_gaps','completed':72,'held':24,'failed':0,'total':96}
    study.dump(tmp_path/'status.json',state)
    assert study.parent_ready()
    with study.exclusive_lock(tmp_path/'coordinator.lock'):
        assert not study.parent_ready()
    state['failed']=1;study.dump(tmp_path/'status.json',state)
    with pytest.raises(RuntimeError):study.parent_ready()


@pytest.mark.parametrize('arm',['scaled_nominal','scaled_renormalized'])
def test_all_rejected_is_finite_zero_update(tmp_path,monkeypatch,arm):
    def reject(self,validator,client,corrupt,forced=None):
        return ValidationVote(validator,client,0.,False,False,'test'),{}
    monkeypatch.setattr(LossCommittee,'vote',reject)
    c=small(arm,rounds=1);x,y=arrays();model=nn.Linear(3,2)
    before={k:v.clone() for k,v in model.state_dict().items()}
    s=BenchmarkRunner(c,tmp_path).run_arrays(x,y,lambda:model)
    assert s['unsupported_rounds']==1
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_zero_mass_replay_is_noop_with_accepted_clients(tmp_path):
    c=small('clip_damping_replay',rounds=1);x,y=arrays();model=nn.Linear(3,2)
    before={k:v.clone() for k,v in model.state_dict().items()}
    s=BenchmarkRunner(c,tmp_path,mass_schedule=[0.],mass_source_binding={'source':'test'}).run_arrays(x,y,lambda:model)
    assert s['unsupported_rounds']==0
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
