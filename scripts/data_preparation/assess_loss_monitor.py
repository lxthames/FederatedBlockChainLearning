"""Read-only postprocessing of V5. Never changes a runner or starts training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / 'artifacts/loss-monitor-v5-pilot'
OUT = ROOT / 'results/loss-monitor-v5-pilot'
POLICY = {
    'role': 'exploratory single-seed assessment, specified after pilot launch',
    'max_clean_utility_loss': .05,
    'max_attacked_utility_loss': .05,
    'min_gradual_final_utility_gain': .01,
    'min_targeted_post_onset_asr_reduction': .05,
    'automatic_promotion': False,
    'publication_ready': False,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(records, expected=15):
    """Require unique complete matched cells before suggesting further review."""
    indexed = {}
    errors = []
    for r in records:
        key = (r['variant'], r['attack'], r['attack_start_round'], r['seed'])
        if key in indexed:
            errors.append(f'duplicate cell: {key}')
        indexed[key] = r
    required = {(v,a,o,11) for v in ('rank_trust','accuracy_guard','loss_quorum')
                for a,o in (('none',1),('low_and_slow',1),('model_replacement',1),
                            ('low_and_slow',21),('model_replacement',21))}
    complete = len(records) == expected and set(indexed) == required and not errors
    report = {'complete':complete, 'errors':errors, 'policy':POLICY,
              'publication_ready':False, 'automatic_promotion':False,
              'comparisons':[], 'recommendation':'Wait for complete, valid pilot evidence.'}
    if not complete:
        report['missing_cells'] = [list(k) for k in sorted(required-set(indexed))]
        return report
    for variant in ('accuracy_guard','loss_quorum'):
        clean = indexed[variant,'none',1,11]
        base_clean = indexed['rank_trust','none',1,11]
        clean_effect = clean['final_utility']-base_clean['final_utility']
        clean_ok = (clean_effect >= -POLICY['max_clean_utility_loss']
                    and clean['recovery_count']==0 and clean['unsupported_rounds']<=4
                    and clean['honest_validator_penalty_amount']==0)
        for attack,onset in (('low_and_slow',1),('model_replacement',1),
                             ('low_and_slow',21),('model_replacement',21)):
            r = indexed[variant,attack,onset,11]
            base = indexed['rank_trust',attack,onset,11]
            matched = (r['participants']==base['participants']
                       and r['data_fingerprint']==base['data_fingerprint']
                       and r['test_fingerprint']==base['test_fingerprint'])
            utility_effect = r['final_utility']-base['final_utility']
            asr_effect = (base['post_onset_mean_asr']-r['post_onset_mean_asr']
                          if attack=='model_replacement' else None)
            benefit = (utility_effect>=POLICY['min_gradual_final_utility_gain']
                       if attack=='low_and_slow' else
                       asr_effect is not None and asr_effect>=POLICY['min_targeted_post_onset_asr_reduction'])
            pre_equal = (r['pre_onset_model_hashes']==base['pre_onset_model_hashes'])
            worth_review = bool(clean_ok and matched and pre_equal and benefit
                and utility_effect>=-POLICY['max_attacked_utility_loss']
                and not r['pre_onset_alarm_rounds'] and r['honest_validator_penalty_amount']==0)
            report['comparisons'].append({
                'variant':variant,'attack':attack,'attack_start_round':onset,
                'clean_utility_effect':clean_effect,'clean_screen_passed':clean_ok,
                'matched_participants_and_data':matched,'identical_pre_onset_models':pre_equal,
                'final_utility_effect':utility_effect,'post_onset_asr_reduction':asr_effect,
                'residual_post_onset_asr':r['post_onset_mean_asr'],
                'post_onset_mean_utility_effect':r['post_onset_mean_utility']-base['post_onset_mean_utility'],
                'recovery_count':r['recovery_count'],'alarm_rounds':r['alarm_rounds'],
                'pre_onset_alarm_rounds':r['pre_onset_alarm_rounds'],
                'immediate_restoration_gain_sum':r['recovery_utility_gain_sum'],
                'immediate_restoration_regret_sum':r['recovery_regret_sum'],
                'worth_replication_review':worth_review})
    promising=[c for c in report['comparisons'] if c['worth_replication_review']]
    report['recommendation'] = (
        'Review only the listed promising conditions for a separately frozen replication; single-seed effects remain exploratory.'
        if promising else
        'No recovery condition passes this exploratory screen. Reconsider the recovery claim before further training.')
    return report


def load_records():
    frozen=json.loads((JOB/'frozen-inputs.json').read_text())
    for name,h in {**frozen['source'],**frozen['data']}.items():
        if digest(ROOT/name)!=h:
            raise ValueError(f'Frozen input changed: {name}')
    records=[];inputs={}
    for row in frozen['manifest']:
        c=row['config'];folder=OUT/c['experiment_id'];path=folder/'summary.json'
        if not path.exists():continue
        s=json.loads(path.read_text())
        expected=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        if s['status']!='completed' or s['config_hash']!=expected or s['rounds_completed']!=c['rounds']:
            raise ValueError(f'Invalid summary: {path}')
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        rounds=[e for e in events if e['event_type']=='round_completed']
        if [e['round_index'] for e in rounds]!=list(range(1,c['rounds']+1)):
            raise ValueError(f'Incomplete or duplicate round events: {folder}')
        trace=json.loads((folder/'chain-trace.json').read_text())
        if len(trace)!=c['rounds']+1:
            raise ValueError(f'Incomplete append-only history: {folder}')
        s['participants']=[e['payload']['participants'] for e in rounds]
        s['pre_onset_model_hashes']=[r['model_hash'] for r in trace if r.get('round',0)<c['attack_start_round']]
        records.append(s)
        for name in ('summary.json','events.jsonl','chain-trace.json'):
            inputs[str((folder/name).relative_to(ROOT))]=digest(folder/name)
    return records,inputs


def write_report():
    state=json.loads((JOB/'status.json').read_text())
    records,inputs=load_records()
    report=assess(records)
    report.update(generated_utc=datetime.now(timezone.utc).isoformat(),run_status=state,
                  input_fingerprints=inputs,assessor_sha256=digest(__file__))
    if state.get('failed') or state.get('held') or state.get('status')!='completed':
        report['complete']=False
        report['recommendation']='Execution is incomplete or has failed/held cells. Resolve and report gaps before any replication decision.'
    path=JOB/'assessment.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(report,indent=2,sort_keys=True));temp.replace(path)
    lines=['# V5 pilot assessment','',report['recommendation'],'',
           'Exploratory development seed 11 only. Publication readiness: false.',
           'Effects are relative to matched rank/trust with recovery disabled. Positive utility or ASR reduction favors recovery.','',
           '| Method | Attack | Onset | Utility change (pp) | ASR reduction (pp) | Recoveries | Review replication |',
           '|---|---|---:|---:|---:|---:|---|']
    for r in report['comparisons']:
        asr='n/a' if r['post_onset_asr_reduction'] is None else f"{100*r['post_onset_asr_reduction']:.2f}"
        lines.append(f"| {r['variant']} | {r['attack']} | {r['attack_start_round']} | {100*r['final_utility_effect']:.2f} | {asr} | {r['recovery_count']} | {r['worth_replication_review']} |")
    (JOB/'assessment.md').write_text('\n'.join(lines)+'\n')
    print(report['recommendation'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    deadline=time.monotonic()+4*60*60
    while args.watch:
        state=json.loads((JOB/'status.json').read_text())
        if state['status'] not in ('starting','running'):break
        if time.monotonic()>deadline:raise TimeoutError('Pilot watcher expired after four hours')
        time.sleep(15)
    write_report()


if __name__=='__main__':main()
