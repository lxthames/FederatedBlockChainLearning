"""Qualification only; no training decisions or parameter selection."""
import numpy as np

SEEDS=(11,17,23)
ATTACKS=('none','low_and_slow','model_replacement')
ARMS=('clip_only','current','v19_repair','v20_repair')


def clean_gate(rows):
    clean=[r for r in rows if r['attack']=='none']
    idx={(r['seed'],r['arm']):r for r in clean}
    if len(idx)!=len(clean):raise ValueError('duplicate clean cells')
    expected={(s,a) for s in SEEDS for a in ARMS}
    if set(idx)!=expected:
        return dict(status='pending',passed=False,required=12,completed=len(idx))
    checks=[]
    for seed in SEEDS:
        candidate=idx[seed,'v20_repair'];base=idx[seed,'current']
        delta=candidate['final_accuracy']-base['final_accuracy']
        passed=delta>=-.02-1e-12 and candidate['repair_attempts']==91 and candidate['repair_admissions']>0
        checks.append(dict(seed=seed,accuracy_delta=delta,repair_admissions=candidate['repair_admissions'],
                           repair_attempts=candidate['repair_attempts'],passed=bool(passed)))
    passed=all(c['passed'] for c in checks)
    return dict(status='clean_qualified' if passed else 'clean_rejected',passed=passed,checks=checks,
                schedule_tuning=False,attack_outcomes_used=False)


def qualify(rows):
    gate=clean_gate(rows)
    if gate['status']=='pending':return dict(status='clean_screen_running',passed=False,clean_gate=gate)
    if not gate['passed']:return dict(status='clean_rejected',passed=False,clean_gate=gate,held_attack_cells=24)
    if len(rows)!=36:return dict(status='development_running',passed=False,clean_gate=gate,completed=len(rows),required=36)
    idx={(r['seed'],r['attack'],r['arm']):r for r in rows}
    if len(idx)!=36 or set(idx)!={(s,t,a) for s in SEEDS for t in ATTACKS for a in ARMS}:
        raise ValueError('missing or duplicate development cells')
    checks=[]
    for seed in SEEDS:
        current=lambda t:idx[seed,t,'current']
        proposed=lambda t:idx[seed,t,'v20_repair']
        v=dict(seed=seed,clean_accuracy_delta=proposed('none')['final_accuracy']-current('none')['final_accuracy'],
            gradual_accuracy_delta=proposed('low_and_slow')['final_accuracy']-current('low_and_slow')['final_accuracy'],
            targeted_accuracy_vs_clean=proposed('model_replacement')['final_accuracy']-current('none')['final_accuracy'],
            mean_post_onset_asr=proposed('model_replacement')['mean_post_onset_asr'],
            endpoint_asr=proposed('model_replacement')['endpoint_asr'],
            mean_asr_improvement=current('model_replacement')['mean_post_onset_asr']-proposed('model_replacement')['mean_post_onset_asr'],
            endpoint_asr_improvement=current('model_replacement')['endpoint_asr']-proposed('model_replacement')['endpoint_asr'])
        v['passed']=bool(min(v[k] for k in ('clean_accuracy_delta','gradual_accuracy_delta','targeted_accuracy_vs_clean'))>=-.02-1e-12
            and max(v['mean_post_onset_asr'],v['endpoint_asr'])<=.20+1e-12
            and min(v['mean_asr_improvement'],v['endpoint_asr_improvement'])>=.20-1e-12)
        checks.append(v)
    passed=all(c['passed'] for c in checks)
    return dict(status='development_promising' if passed else 'completed_targets_not_met',passed=passed,
                checks=checks,clean_gate=gate,publication_ready=False,automatic_followup=False,
                independent_confirmation=False)


def extract(summary,online_rows,offline,integrity,events):
    c=summary['configuration'];onset=c['attack_start_round'];attack=c['attack']
    asrs=summary['targeted_attack_success_history']
    targeted=attack=='model_replacement'
    if targeted and len(asrs)!=100:raise ValueError('incomplete targeted trajectory')
    honest_total=honest_rejected=0
    malicious=set(summary['malicious_client_ids'])
    for event in events:
        if event['event_type']!='round_completed':continue
        rnd=event['round_index'];p=event['payload'];accepted=set(p['accepted_client_ids'])
        honest=[i for i in p['participants'] if attack=='none' or rnd<onset or i not in malicious]
        honest_total+=len(honest);honest_rejected+=sum(i not in accepted for i in honest)
    admits=sum(r['decision']['passed'] for r in online_rows)
    panel_cost=sum(r['verification_presentations'] for r in online_rows)
    student=sum(r['work']['training_examples']+r['work']['proposal_examples']+r['work']['diagnostic_examples'] for r in online_rows)
    teacher=sum(r['work'].get('teacher_examples',0) for r in online_rows)
    # Before/post values measured on the same evaluation examples after the verdict.
    reinfection=[]
    by_round={r['round']:r for r in offline}
    if targeted:
        for rnd in range(max(onset,11),101):
            if rnd in by_round and rnd-1 in by_round:
                reinfection.append(dict(round=rnd,pre_repair_trigger_rate=by_round[rnd]['source']['trigger_target_rate'],
                    previous_committed_trigger_rate=by_round[rnd-1]['committed']['trigger_target_rate'],
                    increase=by_round[rnd]['source']['trigger_target_rate']-by_round[rnd-1]['committed']['trigger_target_rate']))
    return dict(seed=c['seed'],attack=attack,arm=integrity['arm'],final_accuracy=summary['final_accuracy'],
        mean_post_onset_asr=float(np.mean(asrs[onset-1:])) if targeted else None,
        endpoint_asr=asrs[-1] if targeted else None,
        lower_decile_client_utility=summary['lower_decile_client_utility'],
        repair_attempts=len(online_rows),repair_admissions=admits,repair_holds=len(online_rows)-admits,
        repair_seconds=sum(r['seconds'] for r in online_rows),
        local_training_work=summary['local_training_work'],validation_work=summary['validation_work'],
        extra_repair_example_presentations=student+teacher+panel_cost,
        extra_teacher_presentations=teacher,extra_panel_presentations=panel_cost,
        extra_repair_transmitted_bytes=sum(r['extra_transmitted_bytes'] for r in online_rows),
        transmitted_bytes=summary['transmitted_bytes'],runtime_seconds=summary['runtime_seconds_total'],
        honest_updates_excluded=honest_rejected,honest_updates_submitted=honest_total,
        honest_update_exclusion_rate=honest_rejected/max(1,honest_total),
        accepted_malicious_weight_history=summary['accepted_malicious_weight_history'],
        mean_post_onset_malicious_weight=summary['mean_post_onset_malicious_weight'],
        malicious_update_acceptance_rate=summary['malicious_update_acceptance_rate'],
        reinfection=reinfection,integrity=integrity)
