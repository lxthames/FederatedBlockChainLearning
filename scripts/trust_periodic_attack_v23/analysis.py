"""Fixed project gates for six new trajectories and 21 reused comparison rows."""
import math

SEEDS=(11,17,23)
ATTACKS=('none','low_and_slow','model_replacement')
ARMS=('clip_only','current','v20_periodic')


def index_rows(rows):
    idx={(r['seed'],r['attack'],r['arm']):r for r in rows}
    expected={(s,t,a) for s in SEEDS for t in ATTACKS for a in ARMS}
    if len(idx)!=len(rows) or not set(idx)<=expected:
        raise ValueError('duplicate or unexpected comparison rows')
    return idx


def clean_gate(rows):
    idx=index_rows(rows);checks=[]
    for seed in SEEDS:
        if (seed,'none','v20_periodic') not in idx or (seed,'none','current') not in idx:
            return dict(passed=False,status='missing_clean_evidence')
        candidate=idx[seed,'none','v20_periodic'];base=idx[seed,'none','current']
        delta=candidate['final_accuracy']-base['final_accuracy']
        passed=(math.isfinite(delta) and delta>=-.02-1e-12 and
                candidate['repair_attempts']==4 and candidate['repair_admissions']>0)
        checks.append(dict(seed=seed,delta=delta,attempts=candidate['repair_attempts'],
                           admissions=candidate['repair_admissions'],passed=bool(passed)))
    return dict(passed=all(c['passed'] for c in checks),checks=checks,
        selected_after_exposed_clean_results=True,independent_confirmation=False)


def qualify(rows):
    idx=index_rows(rows);gate=clean_gate(rows)
    if not gate['passed']:return dict(status='clean_evidence_rejected',passed=False,clean_gate=gate)
    new=[r for r in rows if not r['reused']]
    if len(idx)!=27:return dict(status='running',passed=False,new_completed=len(new),new_required=6,
                               reused_rows=sum(r['reused'] for r in rows),clean_gate=gate)
    if len(new)!=6 or any(r['arm']!='v20_periodic' or r['attack']=='none' for r in new):
        raise ValueError('new and reused evidence accounting differs from frozen study')
    checks=[]
    for seed in SEEDS:
        current=lambda t:idx[seed,t,'current']
        candidate=lambda t:idx[seed,t,'v20_periodic']
        target=candidate('model_replacement');base=current('model_replacement')
        values=dict(seed=seed,
            clean_delta=candidate('none')['final_accuracy']-current('none')['final_accuracy'],
            gradual_delta=candidate('low_and_slow')['final_accuracy']-current('low_and_slow')['final_accuracy'],
            targeted_accuracy_vs_clean=target['final_accuracy']-current('none')['final_accuracy'],
            mean_post_onset_asr=target['mean_post_onset_asr'],endpoint_asr=target['endpoint_asr'],
            mean_asr_improvement=base['mean_post_onset_asr']-target['mean_post_onset_asr'],
            endpoint_asr_improvement=base['endpoint_asr']-target['endpoint_asr'])
        if any(v is None or not math.isfinite(v) for k,v in values.items() if k!='seed'):
            raise ValueError('missing or nonfinite qualification metric')
        values['passed']=bool(min(values[k] for k in ('clean_delta','gradual_delta','targeted_accuracy_vs_clean'))>=-.02-1e-12
            and max(values['mean_post_onset_asr'],values['endpoint_asr'])<=.20+1e-12
            and min(values['mean_asr_improvement'],values['endpoint_asr_improvement'])>=.20-1e-12)
        checks.append(values)
    passed=all(c['passed'] for c in checks)
    return dict(status='development_promising' if passed else 'completed_targets_not_met',passed=passed,
                checks=checks,new_completed=6,reused_rows=21,clean_gate=gate,
                publication_ready=False,independent_confirmation=False,automatic_followup=False)
