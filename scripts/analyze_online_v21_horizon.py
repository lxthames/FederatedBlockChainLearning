"""Descriptive accounting of V21 clean trajectories, never a counterfactual claim."""
from datetime import datetime,timezone
import math
from scripts.trust_online_v21_fixed import study as s


def main():
    s.assert_frozen();status=s.read(s.JOB/'status.json')
    if status['status']!='clean_rejected' or status['completed']!=12 or status['failed']:
        raise ValueError('completed rejected clean screen required')
    rows=s.collect_rows();idx={(r['seed'],r['arm']):r for r in rows};checks=[]
    for seed in s.SEEDS:
        folder=s.OUT/s.config(seed,'none','v20_repair').experiment_id
        s.parent.verify_seal(folder)
        history=s.read(folder/'summary.json')['accuracy_history']
        records=s.read(folder/'offline-repair-evaluation.json')['records']
        repair_changes=[r['committed']['accuracy']-r['source']['accuracy'] for r in records]
        native_changes=[r['source']['accuracy']-history[r['round']-2] for r in records]
        repair_sum=sum(repair_changes);native_sum=sum(native_changes)
        total=history[-1]-history[8]
        if not math.isclose(repair_sum+native_sum,total,abs_tol=1e-12):
            raise ValueError('observed trajectory accounting does not telescope')
        admissions=idx[seed,'v20_repair']['repair_admissions']
        checks.append(dict(seed=seed,final_accuracy=history[-1],admissions=admissions,
            final_delta_vs_quorum=history[-1]-idx[seed,'current']['final_accuracy'],
            sum_immediate_committed_repair_changes=repair_sum,
            mean_immediate_change_per_admitted_repair=repair_sum/admissions,
            positive_immediate_repairs=sum(v>0 for v in repair_changes),
            negative_immediate_repairs=sum(v<0 for v in repair_changes),
            sum_intervening_local_training_and_aggregation_changes=native_sum,
            round9_to_round100_change=total,telescoping_verified=True,
            completion_sha256=s.digest(folder/'completion.json')))
    job=s.ROOT/'artifacts/online-v21-followup'
    s.dump(job/'trajectory-accounting.json',dict(passed=True,checks=checks,
        original_freeze_sha256=s.digest(s.JOB/'frozen-inputs.json'),script_sha256=s.digest(__file__),
        created_utc=datetime.now(timezone.utc).isoformat(),
        caveat='Observed path accounting; changes in later local updates are not a counterfactual causal decomposition.'))
    lines=['# V21 clean trajectory accounting','',
        'The proposed small-negative-step ratchet explanation is contradicted by the observed immediate changes.',
        'Accepted repairs improve immediate clean accuracy on average, while the final trajectory is worse than quorum.','',
        '| Seed | Accepted repairs | Final loss versus quorum | Mean immediate change per accepted repair |',
        '|---|---:|---:|---:|']
    for c in checks:
        lines.append(f"| {c['seed']} | {c['admissions']} | {100*c['final_delta_vs_quorum']:+.2f}pp | {100*c['mean_immediate_change_per_admitted_repair']:+.2f}pp |")
    lines+=['','Observed repair effects plus intervening local-training/aggregation changes telescope exactly',
        'from round 9 to round 100. This is bookkeeping along the repaired trajectory. It does not',
        'identify why subsequent updates undo improvements or establish a neuron/gradient mechanism.',
        'Dividing final loss versus a different trajectory by admissions is not a per-intervention effect.']
    (job/'trajectory-accounting.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':main()
