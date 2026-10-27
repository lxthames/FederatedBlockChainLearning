"""Read-only post-run audit of frozen V20 outputs; writes separate audit artifacts."""
import json
from datetime import datetime,timezone
from statistics import mean

import torch
from scripts.trust_anchor_v20 import study as s
from scripts.trust_repair_v16_fixed.integrity import state_hash


def main():
    s.assert_frozen()
    status=s.read(s.JOB/'status.json')
    if status['completed']!=6 or status['failed']:
        raise ValueError('V20 is not completely sealed')
    records=[];seals={};checked_steps=0;checked_models=0;frozen_parameters=0
    for row in s.manifest():
        folder=s.OUT/row['id'];s.parent.verify_seal(folder)
        seals[row['id']]=s.digest(folder/'completion.json')
        ref=s.reference_folder(row['seed'],row['condition']);s.parent.verify_seal(ref)
        replay=s.parent.OUT/s.parent.replay_config(row['seed'],row['condition']).experiment_id
        parent=torch.load(replay/'checkpoints/round-100.pt',map_location='cpu',weights_only=True)
        if state_hash(parent['state'])!=parent['model_hash']:raise ValueError('parent hash mismatch')
        for r in s.read(folder/'summary.json')['records']:
            arm=r['arm'];telemetry=s.read(folder/f'{arm}-repair.json')
            if arm=='hard_patch':
                s.baseline_parity(telemetry,r,ref);checked_steps+=len(telemetry['steps'])
            elif telemetry['teacher_hash']!=parent['model_hash']:
                raise ValueError('teacher differs from original parent')
            candidate=torch.load(folder/f'{arm}-candidate.pt',map_location='cpu',weights_only=True)
            committed=torch.load(folder/f'{arm}-committed.pt',map_location='cpu',weights_only=True)
            for model in (candidate,committed):
                if state_hash(model['state'])!=model['model_hash']:raise ValueError('saved model hash mismatch')
                checked_models+=1
            if candidate['model_hash']!=r['raw']['model_hash'] or candidate['model_hash']!=r['decision']['scored_model_hash']:
                raise ValueError('evaluated candidate differs from admitted candidate')
            expected=candidate['model_hash'] if r['decision']['passed'] else parent['model_hash']
            if committed['model_hash']!=expected or committed['model_hash']!=r['committed']['model_hash']:
                raise ValueError('committed model mismatch')
            for name in telemetry.get('frozen_parameter_names',[]):
                if not torch.equal(candidate['state'][name],parent['state'][name]):raise ValueError('frozen layer changed')
                frozen_parameters+=1
            records.append(r)
    if s.analyze(records)!=s.read(s.JOB/'analysis.json'):raise ValueError('analysis differs from sealed records')
    groups=[]
    for arm in s.ARMS:
        clean=[r for r in records if r['arm']==arm and r['condition']=='none']
        attacked=[r for r in records if r['arm']==arm and r['condition']=='model_replacement']
        groups.append(dict(arm=arm,clean_raw_accuracy=mean(r['raw']['accuracy'] for r in clean),
            clean_committed_accuracy=mean(r['committed']['accuracy'] for r in clean),
            clean_raw_delta=mean(r['raw']['accuracy']-r['baseline']['accuracy'] for r in clean),
            clean_admissions=sum(r['decision']['passed'] for r in clean),
            attacked_raw_accuracy=mean(r['raw']['accuracy'] for r in attacked),
            attacked_committed_accuracy=mean(r['committed']['accuracy'] for r in attacked),
            raw_trigger_rate=mean(r['raw']['trigger_target_rate'] for r in attacked),
            committed_trigger_rate=mean(r['committed']['trigger_target_rate'] for r in attacked),
            attack_admissions=sum(r['decision']['passed'] for r in attacked),
            mean_cell_seconds=mean(r['seconds'] for r in clean+attacked),
            raw_lower_decile_client_utility=mean(r['raw']['lower_decile_client_utility'] for r in attacked),
            mean_student_transmissions=mean(r['work']['model_transmissions'] for r in clean+attacked),
            mean_teacher_transmissions=mean(r['work'].get('teacher_transmissions',0) for r in clean+attacked),
            mean_teacher_examples=mean(r['work'].get('teacher_examples',0) for r in clean+attacked)))
    audit=dict(passed=True,status=status,checked_baseline_steps=checked_steps,checked_saved_models=checked_models,
        checked_frozen_parameter_tensors=frozen_parameters,completion_seals=seals,groups=groups,
        freeze_sha256=s.digest(s.JOB/'frozen-inputs.json'),created_utc=datetime.now(timezone.utc).isoformat(),
        analysis_sha256=s.digest(s.JOB/'analysis.json'),audit_script_sha256=s.digest(__file__),publication_ready=False)
    s.dump(s.JOB/'audit.json',audit)
    lines=['# V20 completion audit','',f"Status: {status['status']}; 24/24 cells; {checked_steps} exact baseline optimizer steps; {checked_models} saved model hashes checked.",
        'All completion seals and frozen source bindings verified. All frozen convolution tensors unchanged.','',
        '| Arm | Clean raw accuracy | Clean raw delta | Clean admits | Attacked raw accuracy | Raw trigger rate | Attack admits | Committed trigger rate |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"| {g['arm']} | {100*g['clean_raw_accuracy']:.2f}% | {100*g['clean_raw_delta']:+.2f}pp | {g['clean_admissions']}/3 | {100*g['attacked_raw_accuracy']:.2f}% | {100*g['raw_trigger_rate']:.2f}% | {g['attack_admissions']}/3 | {100*g['committed_trigger_rate']:.2f}% |")
    lines+=['','Means are descriptive across three exposed development seeds. Qualification requires every seed.',
        'Fixed-checkpoint efficacy does not measure online defense, generalization, or acceptance readiness.',
        'The predeclared primary candidate is anchored_suppress. Other arms are ablations; no automatic promotion.']
    (s.JOB/'audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('completion_seals','status')},indent=2))


if __name__=='__main__':main()
