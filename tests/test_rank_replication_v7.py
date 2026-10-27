from dataclasses import replace
import hashlib
import json
import pytest
from scripts.data_preparation import run_rank_replication_v7 as study


def test_replication_changes_only_seed_and_experiment_id():
    rows = study.manifest()
    assert len(rows)==13
    assert len({r['config']['experiment_id'] for r in rows})==13
    for row in rows[:12]:
        c = row['config']
        expected = replace(study.v6.config(row['method'],c['attack']),
                           seed=c['seed'],experiment_id=c['experiment_id']).to_dict()
        assert c==expected
        assert c['seed'] in (17,23)
    assert all(r['config']['attack']=='none' for r in rows[:4])
    control = rows[-1]['config']
    detector = study.v6.config('fldetector_pp','none').to_dict()
    assert {k for k in control if control[k]!=detector[k]}=={'experiment_id','defense'}


def test_clean_gate_is_per_seed_and_does_not_relax_v6_threshold():
    def row(method,seed,utility):
        return dict(method=method,seed=seed,attack='none',final_utility=utility,
                    unsupported_rounds=0,honest_validator_penalty_amount=0)
    records=[row('fedavg',17,.7),row('rank_trust',17,.649),
             row('fedavg',23,.7),row('rank_trust',23,.8)]
    assert not study.seed_gate(records,'rank_trust',17)['passed']
    assert study.seed_gate(records,'rank_trust',23)['passed']
    assert not study.seed_gate(records,'rank_trust',11)['passed']


def test_cache_requires_completed_status_and_matching_config(tmp_path):
    c=study.v6.config('fedavg','none')
    summary={'status':'completed','config_hash':hashlib.sha256(json.dumps(c.to_dict(),sort_keys=True).encode()).hexdigest()}
    path=tmp_path/'summary.json';path.write_text(json.dumps(summary))
    assert study.load_completed(tmp_path,c)==summary
    with pytest.raises(ValueError):
        study.load_completed(tmp_path,replace(c,seed=17))
    summary['status']='running';path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        study.load_completed(tmp_path,c)


def test_missing_seed_group_cannot_pass_review(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'JOB',tmp_path)
    study.analyze([],{'status':'starting'})
    report=json.loads((tmp_path/'analysis.json').read_text())
    assert not report['publication_ready'] and not report['automatic_promotion']
    assert all(not g['complete_three_seed_group'] and not g['passes_prespecified_review_criteria']
               for g in report['paired_results'])
