import json
from dataclasses import replace
import numpy as np
import pytest
import torch
from torch import nn
from scripts.research.config import ExperimentConfig
from scripts.research.splits import make_splits, stream
from scripts.research.validation import LocalValidator, VoteAggregator
from scripts.research.runner import BenchmarkRunner
from scripts.research.detector import TrendDetector, AdjacentDetector


def config(**changes):
    return ExperimentConfig(**(dict(experiment_id='integrity',dataset='synthetic',
        num_clients=4,participation_rate=1.0,rounds=3,device='cpu',seed=7,
        server_samples_per_class=2) | changes))


def data():
    rng=np.random.default_rng(1)
    y=np.tile([0,1],100)
    return (rng.normal(size=(200,2)).astype(np.float32)+y[:,None]).astype(np.float32), y


def factory(): return nn.Linear(2,2)


def test_every_split_is_disjoint_and_complete():
    _,y=data()
    split=make_splits(y,config(dirichlet_alpha=0.1))
    groups=split.train+split.validation+[split.reference,split.test]
    assigned=np.concatenate(groups)
    assert len(assigned)==len(y)
    assert len(set(assigned))==len(y)
    assert all(len(x)>0 for x in split.train+split.validation)


def test_committee_rng_consumption_cannot_change_participation():
    first=stream(13,'participants',4).permutation(50)
    stream(13,'committee',4,2).random(100000)
    assert np.array_equal(first,stream(13,'participants',4).permutation(50))
    assert not np.array_equal(first,stream(13,'committee',4,2).permutation(50))


def test_duplicate_votes_are_rejected():
    vote=LocalValidator(2).vote(0,1.0,1.0,False)
    with pytest.raises(ValueError,match='duplicate'):
        VoteAggregator().decide([vote,vote],2)


def test_changing_external_test_labels_cannot_change_model_or_votes(tmp_path):
    x,y=data()
    c=config(defense='two_layer')
    for folder,test_labels in [('a',y),('b',1-y)]:
        BenchmarkRunner(c,tmp_path/folder).run_arrays(x,y,factory,(x+10,test_labels))
    a=json.loads((tmp_path/'a/chain-trace.json').read_text())
    b=json.loads((tmp_path/'b/chain-trace.json').read_text())
    assert a==b


def test_duplicate_attempts_produce_measured_enrollment_errors(tmp_path):
    x,y=data()
    summary=BenchmarkRunner(config(duplicate_identity_fraction=0.5,
        identity_false_accept_rate=1.0),tmp_path).run_arrays(x,y,factory)
    assert summary['duplicate_attempts']==2
    assert summary['enrollment_false_accepts']==2


def test_existing_run_is_never_overwritten(tmp_path):
    x,y=data()
    BenchmarkRunner(config(),tmp_path).run_arrays(x,y,factory)
    with pytest.raises(FileExistsError): BenchmarkRunner(config(),tmp_path)


def test_detector_resets_after_recovery():
    detector=TrendDetector()
    for score in (1,0.9,0.8,0.7): detector.observe(score)
    assert detector.observe(0.6).detected
    detector.reset()
    assert not detector.observe(0.1).detected
    adjacent=AdjacentDetector()
    adjacent.observe(1)
    assert adjacent.observe(0.8).detected


def test_no_eligible_committee_records_unsupported_round(tmp_path):
    x,y=data()
    summary=BenchmarkRunner(config(defense='two_layer',minimum_stake=30),tmp_path).run_arrays(x,y,factory)
    assert summary['unsupported_rounds']==3
    assert summary['malicious_updates_accepted']==0


def test_recovery_rolls_back_window_and_preserves_versions(tmp_path,monkeypatch):
    from scripts.research.detector import DetectionResult
    monkeypatch.setattr(TrendDetector,'observe',lambda self,score: DetectionResult(True,'test',0.1,-0.1))
    x,y=data()
    summary=BenchmarkRunner(config(defense='two_layer'),tmp_path).run_arrays(x,y,factory)
    trace=json.loads((tmp_path/'chain-trace.json').read_text())
    assert summary['recovery_count']>0
    assert len(trace)==4
    assert trace[1]['recovery']['restored_version']==0
    # Penalties reduce trust below eligibility, so subsequent rounds explicitly
    # record no-op commits rather than inventing a replacement committee.
    assert summary['unsupported_rounds']>0
    assert summary['active_version']==3
    assert summary['wrongful_slashing_count']>0
    assert all(v<=20 for v in summary['final_stakes'].values())


def test_fairness_test_examples_follow_observed_client_class_support():
    from scripts.research.splits import DataSplit,matched_test_shards
    y=np.array([0,0,1,1])
    split=DataSplit([np.array([0]),np.array([2])],
                    [np.array([1]),np.array([3])],np.array([],dtype=int),np.array([],dtype=int))
    test_y=np.array([0,1]*100)
    shards=matched_test_shards(y,test_y,split,7)
    assert np.all(test_y[shards[0]]==0)
    assert np.all(test_y[shards[1]]==1)
    assert sorted(np.concatenate(shards).tolist())==list(range(200))


def test_trust_manifest_has_clean_controls_and_nonzero_identity_attempts():
    from scripts.research.trust_matrix import Trust2027Matrix
    matrix=Trust2027Matrix()
    configs=matrix.core_confirmation()
    assert len(configs)==300
    assert len({c.experiment_id for c in configs})==300
    assert {c.participation_rate for c in configs}=={0.4}
    assert {c.sampling_policy for c in configs}=={'bounded'}
    assert 'none' in {c.attack for c in configs}
    assert all(c.duplicate_identity_fraction>0 for c in matrix.identity_committee_sweep())
    assert {c.num_clients for c in matrix.sift_case_study()}=={4,8}


def test_boba_aggregate_stays_on_estimated_affine_subspace():
    from scripts.research.defenses import BOBADefense
    from scripts.research.contracts import ClientUpdate
    # Honest reference/client points lie on z=0. A far off-plane outlier must
    # not re-enter through averaging the original vectors after projection.
    reference=torch.tensor([[0.,0.,0.],[1.,0.,0.]])
    updates=[ClientUpdate(i,{'w':torch.tensor(value)},1) for i,value in enumerate(
        [[0.,0.,0.],[1.,0.,0.],[0.5,0.,0.],[0.5,0.,100.]])]
    result=BOBADefense(1,reference).aggregate({},updates,None)
    assert abs(result.state['w'][2].item())<1e-6


def test_bounded_schedule_is_matched_and_respects_byzantine_budget(tmp_path):
    x,y=data()
    for defense in ('fedavg','coordinate_median'):
        BenchmarkRunner(config(defense=defense,malicious_fraction=0.25,
            participation_rate=0.75,sampling_policy='bounded'),tmp_path/defense).run_arrays(x,y,factory)
    rounds=[]
    for defense in ('fedavg','coordinate_median'):
        events=[json.loads(line) for line in (tmp_path/defense/'events.jsonl').read_text().splitlines()]
        summaries=json.loads((tmp_path/defense/'summary.json').read_text())
        selected=[e['payload']['participants'] for e in events if e['event_type']=='round_completed']
        bad=set(summaries['malicious_client_ids'])
        assert all(len(set(ids)&bad)<=int(len(ids)*0.25) for ids in selected)
        rounds.append(selected)
    assert rounds[0]==rounds[1]
