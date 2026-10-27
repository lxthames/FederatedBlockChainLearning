from scripts.data_preparation.assess_math_screening import assess


def records():
    rows = []
    for seed in (11,17,23):
        for variant in ('fedavg','paired_loss','scaled_loss','rank_loss'):
            for attack in ('none','low_and_slow','model_replacement'):
                defended = variant != 'fedavg'
                utility = .7 if attack == 'none' else (.65 if defended else .4)
                rows.append(dict(variant=variant,seed=seed,attack=attack,final_utility=utility,
                    development_majority_accuracy=.1,mean_clean_trigger_prediction_rate=.05,
                    mean_targeted_attack_success=.1 if defended else .8))
    return rows


def test_complete_clean_and_effective_attack_groups_required():
    rows=records()
    assert assess(rows)['methods']['paired_loss']['ready_for_reintegration_review']
    rows=[r for r in rows if not (r['variant']=='paired_loss' and r['seed']==23 and r['attack']=='model_replacement')]
    assert not assess(rows)['methods']['paired_loss']['ready_for_reintegration_review']


def test_ineffective_attack_is_not_defense_success():
    rows=records()
    for r in rows:
        if r['variant']=='fedavg' and r['attack']=='low_and_slow':r['final_utility']=.7
    assert not assess(rows)['methods']['scaled_loss']['ready_for_reintegration_review']


def test_clean_loss_and_duplicate_seeds_prevent_promotion():
    rows=records()
    for r in rows:
        if r['variant']=='rank_loss' and r['attack']=='none':r['final_utility']=.4
    assert assess(rows)['methods']['rank_loss']['status']=='failed_clean_screen'
    rows=records()
    rows.append(next(r.copy() for r in rows if r['variant']=='paired_loss' and r['attack']=='none'))
    assert not assess(rows)['methods']['paired_loss']['ready_for_reintegration_review']


def test_high_asr_cannot_pass_on_accuracy_alone():
    rows=records()
    for r in rows:
        if r['variant']=='paired_loss' and r['attack']=='model_replacement':
            r['mean_targeted_attack_success']=.85
            r['final_utility']=.8
    assert not assess(rows)['methods']['paired_loss']['ready_for_reintegration_review']
