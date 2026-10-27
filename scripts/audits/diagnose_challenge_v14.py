"""Read-only discrimination and veto audit of sealed V14 development traces.

No training, threshold selection, source edits or new experiment launches.
AUROC is descriptive; scores/votes share models, clients and validation data.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from scripts.trust_challenge_v14 import study
from scripts.trust_challenge_v14.research.splits import stream

OUT=study.ROOT/'artifacts/challenge-v14-diagnosis'


def auc(positive,negative):
    """P(score of malicious > score of honest), with half credit for ties."""
    positive=np.asarray(positive,dtype=float);negative=np.sort(np.asarray(negative,dtype=float))
    if not len(positive) or not len(negative):return None
    assert np.isfinite(positive).all() and np.isfinite(negative).all()
    lo=np.searchsorted(negative,positive,side='left');hi=np.searchsorted(negative,positive,side='right')
    return float(np.mean((lo+hi)/(2*len(negative))))


def distribution(values,threshold):
    x=np.asarray(values,dtype=float)
    if not len(x):return dict(count=0)
    return dict(count=len(x),mean=float(x.mean()),median=float(np.median(x)),
        p90=float(np.quantile(x,.9)),p99=float(np.quantile(x,.99)),max=float(x.max()),
        fraction_above_frozen_threshold=float(np.mean(x>threshold)))


def bank_coverage(seed,used,dataset):
    matches=overlaps=0
    for rnd,validator in used:
        rng=stream(seed,'challenge-bank-v14',rnd,validator)
        positions=[(int(rng.integers(27)),int(rng.integers(27))) for _ in range(2)]
        top,left=positions[0 if dataset=='femnist' else 1]
        matches+=int((top,left)==(26,26));overlaps+=int(top>=25 and left>=25)
    return dict(unique_validator_round_banks=len(used),exact_attack_patch_location_color_matches=matches,
        same_color_patch_overlaps_attack_location=overlaps,
        role='post hoc geometric coverage only; no new model evaluation; other perturbations may transfer')


def analyze(folder,calibration):
    c=study.read(folder/'config.json');s=study.read(folder/'summary.json')
    settings=study.read(folder/'challenge-settings.json');arm=settings['arm']
    onset=c['attack_start_round'];malicious=set(s['malicious_client_ids'])
    candidate=defaultdict(list);loss_votes={};aggregate=defaultdict(list);gates={};used=set()
    for line in (folder/'events.jsonl').open(encoding='utf-8'):
        e=json.loads(line);r=e['round_index'];p=e['payload']
        if r<onset:continue
        if e['event_type']=='challenge_vote':
            assert np.isfinite(p['score']) and p['examples']>=2
            if arm!='calibrate':
                assert p['threshold']==calibration['thresholds'][c['dataset']][p['stage']]
                assert p['challenge_approve']==(p['score']<=p['threshold'])
            if p['stage']=='candidate':candidate[(r,p['client'])].append(p)
            else:aggregate[r].append(p)
            used.add((r,p['validator']))
        elif e['event_type']=='loss_vote':loss_votes[(r,p['candidate'],p['validator'])]=p
        elif e['event_type']=='aggregate_challenge':gates[r]=p
    threshold=calibration['thresholds'][c['dataset']]['candidate']
    positive=[];negative=[];proposal_positive=[];proposal_negative=[]
    extra_removals=defaultdict(int);clean_admitted=defaultdict(int);proposals=defaultdict(int)
    for (r,client),votes in candidate.items():
        # Poisoning identities are only labels for post hoc analysis, never detection inputs.
        bad=c['attack']!='none' and client in malicious;label='malicious' if bad else 'honest'
        values=[p['score'] for p in votes]
        (positive if bad else negative).extend(values)
        (proposal_positive if bad else proposal_negative).append(float(np.median(values)))
        ordinary=[loss_votes[(r,client,p['validator'])]['clean_loss_approve'] for p in votes]
        clean_yes=sum(ordinary);both_yes=sum(a and p['challenge_approve'] for a,p in zip(ordinary,votes))
        quorum=10
        assert len(votes)==15
        proposals[label]+=1;clean_admitted[label]+=int(clean_yes>=quorum)
        extra_removals[label]+=int(clean_yes>=quorum and both_yes<quorum)
    reasons=defaultdict(int);round_rows=[]
    for r,votes in sorted(aggregate.items()):
        gate=gates[r];quorum=gate['quorum'];clean_yes=sum(p['clean_approve'] for p in votes)
        challenge_yes=sum(p['challenge_approve'] for p in votes)
        both=sum(p['clean_approve'] and p['challenge_approve'] for p in votes)
        assert both==gate['affirmative'] and len(votes)==15
        if arm!='calibrate':assert gate['blocked']==(both<quorum)
        if both>=quorum:reason='pass'
        elif clean_yes<quorum and challenge_yes>=quorum:reason='clean_loss_alone_fails'
        elif clean_yes>=quorum and challenge_yes<quorum:reason='challenge_alone_fails'
        elif clean_yes<quorum and challenge_yes<quorum:reason='both_individually_fail'
        else:reason='intersection_fails_despite_each_individual_quorum'
        reasons[reason]+=1
        round_rows.append(dict(round=r,reason=reason,blocked=gate['blocked'],clean_yes=clean_yes,
            challenge_yes=challenge_yes,both_yes=both,quorum=quorum,
            mean_score=float(np.mean([p['score'] for p in votes])),
            asr=s['targeted_attack_success_history'][r-1] if c['attack']!='none' else None))
    return dict(experiment=folder.name,dataset=c['dataset'],seed=c['seed'],attack=c['attack'],arm=arm,
        candidate_vote_auc=auc(positive,negative),candidate_proposal_median_auc=auc(proposal_positive,proposal_negative),
        malicious_scores=distribution(positive,threshold),honest_scores=distribution(negative,threshold),
        candidate_counts=dict(proposals),clean_loss_admitted=dict(clean_admitted),
        additional_challenge_rejections=dict(extra_removals),aggregate_reasons=dict(reasons),aggregate_rounds=round_rows,
        coverage=bank_coverage(c['seed'],used,c['dataset']),endpoint_asr=s['targeted_attack_success_history'][-1]
            if c['attack']!='none' else None)


def report(records):
    lines=['# V14 score discrimination diagnosis','',
        'Read-only post hoc audit of frozen development traces. No thresholds or models changed.',
        'All 54 completion seals, frozen source/data/protocol inputs and calibration source digests verified.',
        'AUROC treats larger scores as more suspicious. Repeated votes are dependent; no confidence intervals or significance claim.',
        '', '## Perturbation-only arm: active attack rounds','',
        '| Dataset | Seed | Vote AUROC | Median-proposal AUROC | Malicious votes over threshold | Honest votes over threshold | Extra malicious proposals rejected | Extra honest proposals rejected |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    rows=[r for r in records if r['arm']=='perturbation' and r['attack']!='none']
    for r in rows:
        lines.append('| {} | {} | {:.3f} | {:.3f} | {:.2f}% | {:.2f}% | {} / {} | {} / {} |'.format(
            r['dataset'],r['seed'],r['candidate_vote_auc'],r['candidate_proposal_median_auc'],
            100*r['malicious_scores']['fraction_above_frozen_threshold'],100*r['honest_scores']['fraction_above_frozen_threshold'],
            r['additional_challenge_rejections'].get('malicious',0),r['clean_loss_admitted'].get('malicious',0),
            r['additional_challenge_rejections'].get('honest',0),r['clean_loss_admitted'].get('honest',0)))
    lines+=['','Extra rejections compare the saved clean-loss votes with the combined votes on the SAME proposed updates/model trajectory. They do not estimate the end-to-end accuracy/ASR of another threshold or policy. Median-proposal AUROC is a descriptive summary, not the implemented quorum rule.',
        '', '## Aggregate-only arm: active attack rounds', '',
        '| Dataset | Seed | Pass | Clean loss alone fails | Challenge alone fails | Both fail | Vote intersection fails | Endpoint ASR |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        if r['arm']!='aggregate' or r['attack']=='none':continue
        z=r['aggregate_reasons']
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {:.2f}% |'.format(r['dataset'],r['seed'],
            *(z.get(k,0) for k in ('pass','clean_loss_alone_fails','challenge_alone_fails','both_individually_fail','intersection_fails_despite_each_individual_quorum')),
            100*r['endpoint_asr']))
    lines+=['','## Combined three-seed descriptions','',
        '| Dataset | Mean vote AUROC | Extra malicious removals / clean-loss-admitted | Extra honest removals / clean-loss-admitted | Exact matching patch banks / all used banks |',
        '|---|---:|---:|---:|---:|']
    for d in study.DATASETS:
        cells=[r for r in rows if r['dataset']==d]
        lines.append('| {} | {:.3f} | {} / {} | {} / {} | {} / {} |'.format(d,
            float(np.mean([r['candidate_vote_auc'] for r in cells])),
            sum(r['additional_challenge_rejections'].get('malicious',0) for r in cells),
            sum(r['clean_loss_admitted'].get('malicious',0) for r in cells),
            sum(r['additional_challenge_rejections'].get('honest',0) for r in cells),
            sum(r['clean_loss_admitted'].get('honest',0) for r in cells),
            sum(r['coverage']['exact_attack_patch_location_color_matches'] for r in cells),
            sum(r['coverage']['unique_validator_round_banks'] for r in cells)))
    lines+=['','Reasons above use the frozen voting predicates and observed proposed aggregate. A veto holds the complete round update. A clean-loss-driven veto alone is not evidence that the new perturbation score detected a backdoor.',
        '', '## Interpretation and limits', '',
        'The score records only the largest positive INCREASE in perturbation-induced class concentration relative to the current parent. It cannot by itself certify absence of a backdoor already present in that parent. Saturated predictions, small nominal candidate steps, averaging over 16 private examples, and incomplete spatial coverage are plausible sensitivity limits. Logs contain only the maximum scalar score, not per-class/per-transformation probabilities, so they cannot identify which of these dominates.',
        'Challenge calibration pools early and late clean clipping trajectories. Calibration and defense trajectories differ; the nominal 0.5% tail has no guaranteed false-positive rate on either clean defense trajectories or attack-affected honest updates.',
        'Random spatial challenges need not reproduce the injected corner trigger. Geometric exact-color/location matches are included in diagnosis.json; this does not prove that a matching patch changes any prediction or that nonmatching challenges are useless.',
        'Do not infer that a lower threshold will solve the problem from this analysis. A useful next bounded diagnostic would evaluate a predeclared spatially comprehensive bank, save per-class/per-transformation probability changes, and distinguish incremental anomaly from absolute trigger sensitivity. Known-trigger evaluation belongs only in a separate oracle diagnostic. Any deployable redesign must avoid using attacker identities or evaluation ASR as decision inputs.',
        '', '## Reproduction', '',
        'Run .venv-trust/Scripts/python.exe -m scripts.audits.diagnose_challenge_v14.',
        'Machine-readable per-seed score distributions, aggregate decisions, and coverage: artifacts/challenge-v14-diagnosis/diagnosis.json.']
    return '\n'.join(lines)+'\n'


def main():
    frozen=study.read(study.JOB/'frozen-inputs.json')
    assert study.inputs()==frozen,'frozen inputs changed'
    state=study.read(study.JOB/'status.json');assert state['completed']==54 and state['failed']==0 and state['held']==6
    calibration=study.read(study.JOB/'calibration.json');caldigest=study.digest(study.JOB/'calibration.json')
    records=[];verified=0
    for row in frozen['manifest']:
        folder=study.OUT/row['config']['experiment_id']
        if not (folder/'completion.json').exists():continue
        settings=study.read(folder/'challenge-settings.json')
        study.verify_cell(folder,study.ExperimentConfig(**row['config']),settings);verified+=1
        if settings['arm']=='calibrate':
            assert calibration['source_completion_digests'][folder.name]==study.digest(folder/'completion.json')
        else:
            assert settings['calibration_sha256']==caldigest
            assert settings['thresholds']==calibration['thresholds'][row['dataset']]
        if settings['arm'] not in ('current','clip_only'):records.append(analyze(folder,calibration))
    assert verified==54
    OUT.mkdir(parents=True,exist_ok=True)
    study.dump(OUT/'diagnosis.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        verified_cells=verified,frozen_inputs_verified=True,calibration_sha256=caldigest,
        role='descriptive post hoc exposed-seed audit; no threshold tuning',records=records))
    (OUT/'diagnosis.md').write_text(report(records),encoding='utf-8')
    print(report(records))


if __name__=='__main__':main()
