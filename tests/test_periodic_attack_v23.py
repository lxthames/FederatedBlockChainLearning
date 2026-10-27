import copy
from dataclasses import replace
import json

import pytest

from scripts.trust_periodic_attack_v23 import analysis,study


def rows():
    result=[]
    for seed in analysis.SEEDS:
        for attack in analysis.ATTACKS:
            for arm in analysis.ARMS:
                proposed=arm=='v20_periodic';target=attack=='model_replacement'
                result.append(dict(seed=seed,attack=attack,arm=arm,final_accuracy=.8,
                    reused=not (proposed and attack!='none'),repair_attempts=4 if proposed else 0,
                    repair_admissions=3 if proposed else 0,
                    mean_post_onset_asr=(.1 if proposed else .9) if target else None,
                    endpoint_asr=(.1 if proposed else .9) if target else None))
    return result


def test_six_new_cells_and_twenty_one_distinct_references():
    manifest=study.manifest();refs=study.references()
    assert len(manifest)==6 and len({r['id'] for r in manifest})==6
    assert [r['attack'] for r in manifest]==['model_replacement']*3+['low_and_slow']*3
    assert all(r['arm']=='v20_periodic' for r in manifest)
    assert len(refs)==21 and len({r['path'] for r in refs})==21
    assert sum(r['kind']=='v22_clean' for r in refs)==9
    assert sum(r['kind']=='v16_control' for r in refs)==12
    for row in manifest:
        candidate=study.config(row['seed'],row['attack'])
        prior=study.base.config(row['seed'],row['attack'],'v20_periodic')
        assert replace(candidate,experiment_id=prior.experiment_id)==prior
    with pytest.raises(ValueError,match='unknown'):study.config(11,'none')


def test_qualification_preserves_per_seed_utility_and_both_asr_targets():
    data=rows();result=analysis.qualify(data)
    assert result['passed'] and result['new_completed']==6 and result['reused_rows']==21
    assert not result['publication_ready'] and not result['automatic_followup']
    for attack,key,value in (('low_and_slow','final_accuracy',.77),
                             ('model_replacement','final_accuracy',.77),
                             ('model_replacement','mean_post_onset_asr',.21),
                             ('model_replacement','endpoint_asr',.21)):
        changed=copy.deepcopy(data)
        row=next(r for r in changed if r['seed']==17 and r['attack']==attack and r['arm']=='v20_periodic')
        row[key]=value
        assert not analysis.qualify(changed)['passed']
    changed=copy.deepcopy(data)
    for r in changed:
        if r['attack']=='model_replacement' and r['arm']=='current':
            r['mean_post_onset_asr']=.25;r['endpoint_asr']=.25
    assert not analysis.qualify(changed)['passed']


def test_missing_new_evidence_is_pending_without_relabeling_reuse():
    references=[r for r in rows() if r['reused']]
    verdict=analysis.qualify(references)
    assert verdict['status']=='running' and verdict['new_completed']==0 and verdict['reused_rows']==21
    assert analysis.clean_gate(references)['passed']
    missing=references[:-1]
    # Missing a clean source cannot be treated as a passed screen.
    clean_missing=[r for r in references if not (r['seed']==11 and r['attack']=='none' and r['arm']=='current')]
    assert not analysis.clean_gate(clean_missing)['passed']
    data=rows();next(r for r in data if r['seed']==11 and r['attack']=='none' and r['arm']=='v20_periodic')['repair_admissions']=0
    assert analysis.qualify(data)['status']=='clean_evidence_rejected'


def test_duplicate_wrong_origin_and_nonfinite_metrics_fail_closed():
    data=rows()
    with pytest.raises(ValueError,match='duplicate'):analysis.qualify(data+[copy.deepcopy(data[0])])
    changed=copy.deepcopy(data);changed[-1]['reused']=True
    with pytest.raises(ValueError,match='accounting'):analysis.qualify(changed)
    changed=copy.deepcopy(data);changed[-1]['endpoint_asr']=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):analysis.qualify(changed)


def test_runner_and_inner_policy_are_direct_frozen_v22_reuse():
    assert study.base.DEPLOYMENT.first_round==25 and study.base.DEPLOYMENT.interval==25
    assert study.base.v20.POLICY.passes*study.base.v20.POLICY.repair_clients*study.base.v20.POLICY.steps_per_client==200
    code=(study.ROOT/'scripts/trust_periodic_attack_v23/study.py').read_text(encoding='utf-8')
    assert "base.OnlineRepair('v20_periodic'" in code
    assert "r['decision']['memory_gate_applied']" in code
    assert 'base.validate_parity(folder,reference' in code
