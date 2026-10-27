from copy import deepcopy
import pytest
from scripts.data_preparation.assess_loss_monitor import assess


def records():
    result=[]
    for v in ('rank_trust','accuracy_guard','loss_quorum'):
        for a,o in (('none',1),('low_and_slow',1),('model_replacement',1),('low_and_slow',21),('model_replacement',21)):
            helpful=v=='loss_quorum' and a!='none'
            result.append(dict(variant=v,attack=a,attack_start_round=o,seed=11,
                final_utility=.55 if helpful else .5,post_onset_mean_utility=.55 if helpful else .5,
                post_onset_mean_asr=(.4 if helpful else .7) if a=='model_replacement' else None,
                recovery_count=int(helpful),unsupported_rounds=0,honest_validator_penalty_amount=0,
                participants=[[1,2,3]],data_fingerprint='train',test_fingerprint='evaluation',
                pre_onset_model_hashes=['same'],pre_onset_alarm_rounds=[],alarm_rounds=[24] if helpful else [],
                recovery_utility_gain_sum=.02 if helpful else 0,recovery_regret_sum=0))
    return result


def test_complete_beneficial_pilot_still_not_publication_ready():
    report=assess(records())
    assert report['complete'] and not report['publication_ready'] and not report['automatic_promotion']
    assert sum(r['worth_replication_review'] for r in report['comparisons'])==4


@pytest.mark.parametrize('change',['missing','duplicate','wrong_seed'])
def test_incomplete_or_duplicate_groups_cannot_be_promoted(change):
    rows=records()
    if change=='missing':rows.pop()
    elif change=='duplicate':rows[-1]=deepcopy(rows[0])
    else:rows[-1]['seed']=17
    report=assess(rows)
    assert not report['complete'] and not report['comparisons']


@pytest.mark.parametrize('field,value',[
    ('participants',[[9,8,7]]),('test_fingerprint','changed'),
    ('pre_onset_model_hashes',['different']),('pre_onset_alarm_rounds',[8]),
    ('honest_validator_penalty_amount',5),('final_utility',.3)])
def test_benefit_cannot_hide_unmatched_or_harmful_run(field,value):
    rows=records();rows[-1][field]=value
    report=assess(rows)
    cell=next(r for r in report['comparisons'] if r['variant']=='loss_quorum'
              and r['attack']=='model_replacement' and r['attack_start_round']==21)
    assert not cell['worth_replication_review']


def test_false_clean_recovery_blocks_all_attack_replication_reviews():
    rows=records()
    next(r for r in rows if r['variant']=='loss_quorum' and r['attack']=='none')['recovery_count']=1
    report=assess(rows)
    assert all(not r['worth_replication_review'] for r in report['comparisons'])


def test_zero_added_benefit_is_reported_as_unsuccessful():
    rows=records()
    for r in rows:
        r.update(final_utility=.5,post_onset_mean_utility=.5,
                 post_onset_mean_asr=.7 if r['attack']=='model_replacement' else None)
    report=assess(rows)
    assert not any(r['worth_replication_review'] for r in report['comparisons'])
    assert 'Reconsider the recovery claim' in report['recommendation']
