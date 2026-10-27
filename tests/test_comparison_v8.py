import json
from scripts.data_preparation import run_comparison_v8 as study


def test_fixed_manifest_reuse_and_dataset_settings():
    rows=study.manifest()
    assert len(rows)==44 and len({r['config']['experiment_id'] for r in rows})==44
    assert len(study.reuse_cells())==19
    assert sum(r['config']['dataset']=='femnist' for r in rows)==18
    for row in rows:
        c=row['config']
        assert c==study.config(c['dataset'],row['method'],c['attack'],c['seed']).to_dict()
        assert c['attack_start_round']==11 and c['recovery_penalties_enabled'] is False
        if row['method']=='rank_trust':assert c['ablation']=='validation_only'
        if c['dataset']=='femnist':
            assert c['rounds']==40 and c['num_classes']==62 and c['dirichlet_alpha'] is None
            assert c['natural_splits_path']
        else:assert c['rounds']==30 and row['method'] in study.BASELINES


def test_clean_gate_cannot_borrow_another_dataset_or_seed():
    def row(dataset,method,seed,utility):
        return dict(dataset=dataset,method=method,seed=seed,attack='none',final_utility=utility,
                    unsupported_rounds=0,honest_validator_penalty_amount=0)
    records=[row('femnist','fedavg',11,.7),row('femnist','rank_trust',11,.64),
             row('fashion_mnist','rank_trust',11,.8),row('femnist','rank_trust',17,.8)]
    assert not study.clean_gate(records,'rank_trust',11,'femnist')['passed']
    assert not study.clean_gate(records,'rank_trust',17,'femnist')['passed']
    records[1]['final_utility']=.68
    assert study.clean_gate(records,'rank_trust',11,'femnist')['passed']


def test_comparison_marks_mismatched_artifacts_and_effective_attacks():
    base=dict(dataset='fashion_mnist',method='fedavg',seed=11,attack='none',final_utility=.7,
              unsupported_rounds=0,honest_validator_penalty_amount=0,participants=[[0,1]],
              data_fingerprint='data',test_fingerprint='holdout',post_onset_mean_asr=None,
              post_onset_mean_clean_trigger_rate=.02,honest_update_exclusion_rate=0)
    attacked=dict(base,attack='model_replacement',final_utility=.6,post_onset_mean_asr=.5)
    candidate=dict(attacked,method='rank_trust',post_onset_mean_asr=.2,test_fingerprint='changed')
    p=study.paired_row([base,attacked,candidate],candidate)
    assert not p['matched'] and p['effective_attack_vs_fedavg']
    assert abs(p['asr_reduction_vs_fedavg']-.3)<1e-12


def test_incomplete_groups_have_no_reported_means(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'JOB',tmp_path)
    study.analyze([],{'status':'starting'})
    result=json.loads((tmp_path/'analysis.json').read_text())
    assert len(result['groups'])==21
    assert all(not r['complete'] and r['mean_final_utility'] is None for r in result['groups'])
    assert not result['publication_ready'] and not result['automatic_promotion']
