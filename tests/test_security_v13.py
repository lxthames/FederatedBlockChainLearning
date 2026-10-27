from dataclasses import replace
import copy
import numpy as np
import pytest
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from scripts.trust_security_v13 import study
from scripts.trust_security_v13.research.runner_v2 import BenchmarkRunner
from scripts.trust_security_v13.research.attacks import LowAndSlowAttack, AttackContext
from scripts.trust_security_v13.research.contracts import ClientUpdate
from scripts.trust_budget_v12.research.config import ExperimentConfig as V12Config
from scripts.trust_budget_v12.research.runner_v2 import BenchmarkRunner as V12Runner


@pytest.fixture(autouse=True)
def threads():
    with threadpool_limits(limits=2):
        torch.set_num_threads(2)
        yield


def small(arm=study.CANDIDATE,attack='none'):
    return replace(study.config('fashion_mnist',arm,11,attack),dataset='synthetic',num_classes=2,
        num_clients=5,participation_rate=1.,rounds=3,attack_start_round=2,
        natural_splits_path=None,device='cpu',server_samples_per_class=2,batch_size=16)


def arrays():
    x=np.random.default_rng(71).normal(size=(200,3)).astype('float32')
    return x,(x[:,0]>0).astype('int64')


def run_cell(c,path,test_labels=None):
    runner=BenchmarkRunner(c,path);study.dump(path/'diagnostic-inputs.json',study.EMPTY_BINDING)
    x,y=arrays()
    summary=runner.run_arrays(x,y,lambda:nn.Linear(3,2),None if test_labels is None else (x[:20],test_labels))
    return summary,dict(study.enrich(summary,path),arm=study.CANDIDATE if c.vote_rule=='scaled_loss' else c.defense)


def test_manifest_clean_first_finite_and_fixed_protocol():
    rows=study.manifest()
    assert len(rows)==72 and len({r['config']['experiment_id'] for r in rows})==72
    assert all(r['attack']=='none' for r in rows[:24])
    assert all(r['attack']!='none' for r in rows[24:])
    assert all(r['dataset']=='fashion_mnist' for r in rows[:12])
    assert all(r['config']['model']=='wide_cnn' and r['config']['local_epochs']==1 for r in rows)
    c=small()
    for changes in ({'local_training_budget':'fixed_steps'},{'aggregation_mass_policy':'nominal'},
                    {'trust_weighting':'dynamic'},{'rejection_trust_penalty':.02}):
        with pytest.raises(ValueError):replace(c,**changes)


@pytest.mark.parametrize('attack',['none','low_and_slow','model_replacement'])
def test_candidate_reproduces_v12_models(tmp_path,attack):
    c=small(attack=attack);s,_=run_cell(c,tmp_path/'new')
    old=V12Config(**dict(c.to_dict(),protocol_version='trust-budget-v12'))
    x,y=arrays();before=V12Runner(old,tmp_path/'old').run_arrays(x,y,lambda:nn.Linear(3,2))
    traces=[study.read(tmp_path/name/'chain-trace.json') for name in ('new','old')]
    assert [r['model_hash'] for r in traces[0]]==[r['model_hash'] for r in traces[1]]
    assert s['final_accuracy']==before['final_accuracy']


@pytest.mark.parametrize('arm',study.ARMS)
@pytest.mark.parametrize('attack',study.ATTACKS)
def test_attack_pipeline_and_pre_onset_condition_pair(tmp_path,arm,attack):
    _,clean=run_cell(small(arm),tmp_path/'clean')
    summary,attacked=run_cell(small(arm,attack),tmp_path/'attack')
    assert summary['malicious_updates_submitted']>0
    assert study.condition_pair(clean,attacked)['matched']
    assert summary['local_training_work']['optimizer_steps']>0
    if attack=='model_replacement':
        assert attacked['post_onset_mean_asr'] is not None and attacked['endpoint_asr'] is not None
    if arm=='fedval':assert summary['validation_work']['reference_example_evaluations']>0
    if arm==study.CANDIDATE:assert summary['validation_work']['private_example_evaluations']>0


def fake_records(tmp_path):
    # Use real run artifacts for the binding layout; controlled outcomes isolate
    # the analysis rules without treating simulated attacks as effective evidence.
    _,template=run_cell(small(),tmp_path)
    records=[]
    for d in study.DATASETS:
        for seed in study.SEEDS:
            for arm in study.ARMS:
                for attack in ('none',)+study.ATTACKS:
                    row=copy.deepcopy(template)
                    c=replace(small(arm,attack),dataset=d,seed=seed)
                    row.update(configuration=c.to_dict(),dataset=d,seed=seed,attack=attack,arm=arm,
                        training_binding=study.digest_value({k:c.to_dict()[k] for k in study.TRAINING_FIELDS}),
                        final_utility=.7 if attack=='none' else (.65 if arm!=study.CANDIDATE else .68),
                        post_onset_mean_asr=.8 if arm!=study.CANDIDATE else .2,
                        post_onset_mean_clean_trigger_rate=.05,endpoint_asr=.2,
                        unsupported_rounds=0,honest_validator_penalty_amount=0)
                    records.append(row)
    return records


def test_gate_requires_all_six_clean_cells_and_preserves_bounds(tmp_path):
    records=fake_records(tmp_path)
    assert study.candidate_gate(records)['passed']
    row=study.find(records,study.CANDIDATE,'femnist',23)
    row['final_utility']=.649
    assert not study.candidate_gate(records)['passed']
    row['final_utility']=.7;row['unsupported_rounds']=4
    assert not study.candidate_gate(records)['passed']
    records.remove(row)
    assert not study.candidate_gate(records)['passed']


def test_effectiveness_requires_damage_and_condition_integrity(tmp_path):
    records=fake_records(tmp_path)
    assert study.effectiveness(records,'femnist',11,'low_and_slow')['effective']
    assert study.effectiveness(records,'femnist',11,'model_replacement')['effective']
    gradual=study.find(records,'fedavg','femnist',11,'low_and_slow')
    gradual['final_utility']=.7
    assert not study.effectiveness(records,'femnist',11,'low_and_slow')['effective']
    gradual['pre_onset_model_hashes']=['wrong']
    assert study.effectiveness(records,'femnist',11,'low_and_slow')['effective'] is None
    targeted=study.find(records,'fedavg','femnist',11,'model_replacement')
    targeted['post_onset_mean_asr']=.08
    assert not study.effectiveness(records,'femnist',11,'model_replacement')['effective']


def test_condition_pair_rejects_changes_beyond_attack(tmp_path):
    _,clean=run_cell(small(),tmp_path/'clean')
    _,attacked=run_cell(small(attack='model_replacement'),tmp_path/'attack')
    assert study.condition_pair(clean,attacked)['matched']
    attacked['configuration']['loss_standard_errors']=3.
    assert not study.condition_pair(clean,attacked)['matched']


def test_method_pair_rejects_unbound_threat_model_change(tmp_path):
    _,row=run_cell(small(),tmp_path)
    changed=copy.deepcopy(row)
    changed['configuration']['malicious_validator_fraction']=.2
    assert not study.method_pair(changed,row)['matched']


def test_analysis_needs_effective_attacks_all_seeds_and_strong_clean_controls(tmp_path,monkeypatch):
    records=fake_records(tmp_path/'fixture')
    monkeypatch.setattr(study,'JOB',tmp_path/'report')
    study.analyze(records,{})
    report=study.read(study.JOB/'analysis.json')
    screens=[r for r in report['benefit_screens'] if r['reference']=='clip_only']
    assert all(r['interpretable'] and r['descriptive_benefit_screen_passed'] for r in screens)
    row=study.find(records,'fedavg','femnist',11,'low_and_slow');row['final_utility']=.7
    study.analyze(records,{})
    report=study.read(study.JOB/'analysis.json')
    screens=[r for r in report['benefit_screens'] if r['dataset']=='femnist' and r['attack']=='low_and_slow']
    assert all(not r['interpretable'] and r['descriptive_benefit_screen_passed'] is None for r in screens)
    row['final_utility']=.65
    control=study.find(records,'fedval','fashion_mnist',11);control['final_utility']=.5
    study.analyze(records,{})
    report=study.read(study.JOB/'analysis.json')
    assert all(not r['interpretable'] for r in report['benefit_screens'] if r['dataset']=='fashion_mnist' and r['reference']=='fedval')
    row=study.find(records,study.CANDIDATE,'femnist',23,'model_replacement');records.remove(row)
    study.analyze(records,{})
    report=study.read(study.JOB/'analysis.json')
    group=next(g for g in report['groups'] if (g['dataset'],g['arm'],g['attack'])==('femnist',study.CANDIDATE,'model_replacement'))
    assert not group['complete'] and group['mean_accuracy'] is None


def test_enrichment_is_post_onset_and_seals_detect_mutation(tmp_path):
    c=small(attack='model_replacement');summary,enriched=run_cell(c,tmp_path)
    assert enriched['post_onset_mean_asr']==pytest.approx(np.mean(summary['targeted_attack_success_history'][1:]))
    assert 'post_onset_mean_asr' not in summary
    study.seal(tmp_path);study.load_completed(tmp_path,c)
    (tmp_path/'diagnostic-inputs.json').write_text('{}')
    with pytest.raises(ValueError,match='digest mismatch'):study.load_completed(tmp_path,c)


def test_heldout_labels_do_not_change_candidate_model_trace(tmp_path):
    x,y=arrays();c=small(attack='model_replacement')
    run_cell(c,tmp_path/'a',test_labels=y[:20]);run_cell(c,tmp_path/'b',test_labels=1-y[:20])
    traces=[study.read(tmp_path/name/'chain-trace.json') for name in ('a','b')]
    assert [r['model_hash'] for r in traces[0]]==[r['model_hash'] for r in traces[1]]


def test_gradual_attack_is_bounded_against_own_update():
    update=ClientUpdate(0,{'weight':torch.tensor([1.,2.])},1,True)
    attack=LowAndSlowAttack(1.5,10)
    for round_index in range(1,21):
        changed=attack.apply(update,AttackContext(round_index))
        expected=1-2.5*min(round_index/10,1.)
        assert torch.allclose(changed.state['weight'],update.state['weight']*expected)
        assert changed.state['weight'].norm()<=1.5*update.state['weight'].norm()+1e-7
