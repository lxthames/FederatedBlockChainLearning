"""Read-only diagnosis of sealed V11 clean traces; never trains or edits V11.

Run from repository root with python -m scripts.audits.diagnose_screen_v11.
The snapshot only includes cells sealed when enumeration starts. All rates are
descriptive: repeated votes and rounds are not independent statistical samples.
"""
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from scripts.trust_screen_v11 import study


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/screen-v11-diagnosis'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def mean(values):
    return float(np.mean(values)) if len(values) else None


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None


def analyze(folder):
    c = read(folder / 'config.json')
    summary = study.load_completed(folder, study.ExperimentConfig(**c))
    split = read(folder / 'splits.json')
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
    rounds = [e for e in events if e['event_type'] == 'round_completed']
    votes = [e for e in events if e['event_type'] == 'loss_vote']
    fedval = {e['round_index']: e['payload'] for e in events if e['event_type'] == 'fedval_weights'}
    clipping = {e['round_index']: e['payload'] for e in events if e['event_type'] == 'candidate_clipping'}
    labels = np.load(study.DATA / f"{c['dataset']}-{c['seed']}" / 'labels.npy')
    train = [np.bincount(labels[ids], minlength=c['num_classes']) for ids in split['train']]
    val = [np.bincount(labels[ids], minlength=c['num_classes']) for ids in split['validation']]
    sizes = np.array([len(ids) for ids in split['train']])
    train_dist = np.array([counts/counts.sum() for counts in train])
    global_dist = np.sum(train, axis=0)/sizes.sum()
    client_rows = [dict(client=i, train_count=int(sizes[i]), validation_count=int(sum(val[i])),
                        validation_classes=int(np.count_nonzero(val[i])),
                        local_global_label_tv=float(np.abs(val[i]/sum(val[i])-global_dist).sum()/2),
                        submitted=0, accepted=0, nominal_mass=0., actual_weight=0.) for i in range(len(sizes))]
    previous_trust = {str(i): c['initial_trust'] for i in range(len(sizes))}
    round_rows = []
    for event in rounds:
        r, p = event['round_index'], event['payload']
        participants = p['participants']
        mask = np.array([i in p['accepted_client_ids'] for i in participants])
        nominal = sizes[participants]/sizes[participants].sum()
        mass = float(nominal[mask].sum())
        if r in fedval:
            weights = np.array(fedval[r]['weights'])
        else:
            weights = nominal * mask
            if c['trust_weighting'] == 'dynamic':
                weights *= np.array([max(c['probation_weight_floor'], previous_trust[str(i)]) for i in participants])
            if weights.sum() and c['vote_rule'] != 'scaled_loss':
                weights /= weights.sum()
        assert np.isfinite(weights).all() and (weights >= 0).all()
        assert np.array_equal(weights > 0, mask)
        expected_sum = mass if c['vote_rule'] == 'scaled_loss' else 1.
        assert np.isclose(weights.sum(), expected_sum)
        normalized = weights/weights.sum() if weights.sum() else weights
        actual_labels = normalized @ train_dist[participants]
        offered_labels = nominal @ train_dist[participants]
        row = dict(round=r, accuracy=p['accuracy'], accepted_count=int(mask.sum()),
                   accepted_nominal_mass=mass, applied_coefficient_sum=float(weights.sum()),
                   effective_client_count=float(1/(normalized@normalized)),
                   sample_weighted_effective_clients=float(1/(nominal@nominal)),
                   max_normalized_weight=float(normalized.max()),
                   label_tv_from_offered=float(np.abs(actual_labels-offered_labels).sum()/2),
                   eligible_count=len(p['validator_eligible_ids']))
        if r in fedval:
            row.update(s2=fedval[r]['s2'], reference_macro_loss=fedval[r]['reference_macro_loss'])
        if r in clipping:
            scales=np.array(clipping[r]['scales'])
            row.update(clipped_count=int((scales < 1).sum()), minimum_clip_scale=float(scales.min()))
        round_rows.append(row)
        for i, a, w in zip(participants, nominal, weights):
            client_rows[i]['submitted'] += 1
            client_rows[i]['accepted'] += int(i in p['accepted_client_ids'])
            client_rows[i]['nominal_mass'] += float(a)
            client_rows[i]['actual_weight'] += float(w)
        if p['trust_after_round'] is not None:
            previous_trust = p['trust_after_round']
    rejection_rates=[]
    for row in client_rows:
        row['rejection_rate'] = 1-row['accepted']/row['submitted']
        row['influence_to_nominal_ratio'] = row['actual_weight']/row['nominal_mass']
        rejection_rates.append(row['rejection_rate'])
    # Quantile comparisons are descriptions of this trace, not randomized effects.
    by_size=sorted(client_rows, key=lambda row: row['train_count'])
    size_groups={}
    for name, group in [('smallest_10', by_size[:10]), ('largest_10', by_size[-10:])]:
        size_groups[name] = dict(mean_train_count=mean([x['train_count'] for x in group]),
                                rejection_rate=1-sum(x['accepted'] for x in group)/sum(x['submitted'] for x in group))
    vote_groups=defaultdict(list)
    disagreements=defaultdict(list)
    rejected_helpful=approved_harmful=harmful=0
    for event in votes:
        p=event['payload']
        i, v=p['candidate'],p['validator']
        overlap=float(train_dist[i][val[v]>0].sum())
        vote_groups['coverage_below_50pct' if overlap < .5 else 'coverage_at_least_50pct'].append(p['approve'])
        disagreements[(event['round_index'],i)].append(p['approve'])
        rejected_helpful += int(not p['approve'] and not p['locally_harmful'])
        approved_harmful += int(p['approve'] and p['locally_harmful'])
        harmful += int(p['locally_harmful'])
    phases={}
    for start in range(0,len(round_rows),10):
        rows=round_rows[start:start+10]
        phases[f'{start+1}-{start+len(rows)}']={key:mean([x[key] for x in rows]) for key in
            ('accepted_nominal_mass','applied_coefficient_sum','effective_client_count','label_tv_from_offered')}
        phases[f'{start+1}-{start+len(rows)}']['endpoint_accuracy']=rows[-1]['accuracy']
    metrics=dict(final_accuracy=summary['final_accuracy'],
        benign_exclusion=summary['benign_update_exclusion_rate'],
        mean_accepted_nominal_mass=mean([x['accepted_nominal_mass'] for x in round_rows]),
        sum_applied_coefficients=sum(x['applied_coefficient_sum'] for x in round_rows),
        mean_effective_clients=mean([x['effective_client_count'] for x in round_rows]),
        mean_nominal_effective_clients=mean([x['sample_weighted_effective_clients'] for x in round_rows]),
        mean_label_tv_from_offered=mean([x['label_tv_from_offered'] for x in round_rows]),
        size_rejection_correlation=corr(sizes,rejection_rates),size_groups=size_groups,
        validation_class_median=float(np.median([x['validation_classes'] for x in client_rows])),
        validation_sample_median=float(np.median([x['validation_count'] for x in client_rows])),
        local_global_label_tv_mean=mean([x['local_global_label_tv'] for x in client_rows]),
        vote_count=len(votes),rejected_locally_helpful_votes=rejected_helpful,
        approved_locally_harmful_votes=approved_harmful,locally_harmful_votes=harmful,
        mixed_vote_candidate_fraction=mean([0 < sum(v) < len(v) for v in disagreements.values()]),
        vote_coverage_groups={k:dict(count=len(v),approval_rate=mean(v)) for k,v in vote_groups.items()},
        unsupported_rounds=summary['unsupported_rounds'],
        inconclusive=sum(summary['inconclusive_counts'].values()),
        unavailable=sum(summary['committee_unavailable_counts'].values()),
        abstentions=sum(e['event_type']=='loss_vote_abstention' for e in events),
        clipped_proposals=sum(x.get('clipped_count',0) for x in round_rows),
        reference_class_counts=np.bincount(labels[split['reference']],minlength=c['num_classes']).tolist())
    if fedval:
        metrics.update(s2_min=min(x['s2'] for x in fedval.values()),s2_max=max(x['s2'] for x in fedval.values()))
    base=folder.parent / f"v11-{c['dataset']}-fedavg-none-s{c['seed']}"
    baseline=study.load_completed(base,study.config(c['dataset'],'fedavg',c['seed']))
    pair=study.compare_pair(summary,baseline)
    assert pair['matched'], pair
    metrics['accuracy_delta_from_fedavg']=summary['final_accuracy']-baseline['final_accuracy']
    return dict(experiment=folder.name,dataset=c['dataset'],seed=c['seed'],
                arm=folder.name.split(f"{c['dataset']}-")[1].split('-none-')[0],
                completion_seal=read(folder/'completion.json'),pairing=pair,
                metrics=metrics,phases=phases,rounds=round_rows,clients=client_rows)


def main():
    assert study.inputs() == read(study.JOB/'frozen-inputs.json'), 'V11 frozen inputs changed'
    folders=sorted(p.parent for p in study.OUT.glob('*-none-*/completion.json'))
    records=[analyze(p) for p in folders]
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),
                study_status_at_analysis=read(study.JOB/'status.json'),
                sealed_clean_cells=len(records),frozen_inputs_verified=True,
                interpretation='Post hoc descriptive diagnosis; training labels only for distribution diagnostics; no retraining or threshold tuning.',
                records=records)
    by_key={(r['dataset'],r['seed'],r['arm']):r for r in records}
    for seed in study.SEEDS:
        key=('femnist',seed,'scaled_quorum')
        if key not in by_key:
            continue
        def trace(arm):
            path=study.OUT/f'v11-femnist-{arm}-none-s{seed}'/'chain-trace.json'
            return [x for x in read(path) if x['kind']=='round']
        scaled,control=trace('scaled_quorum'),trace('clip_only')
        by_key[key]['initial_divergence_check'] = dict(
            same_round_1_model=scaled[0]['model_hash']==control[0]['model_hash'],
            same_round_2_candidate_hashes=[(x['client_id'],x['candidate_hash']) for x in scaled[1]['candidates']]
                ==[(x['client_id'],x['candidate_hash']) for x in control[1]['candidates']],
            same_round_2_model=scaled[1]['model_hash']==control[1]['model_hash'])
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'diagnosis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    make_figure(by_key)
    make_report(report,by_key)
    for r in records:
        m=r['metrics']
        print(r['experiment'],json.dumps({k:m[k] for k in ('final_accuracy','mean_accepted_nominal_mass',
              'mean_effective_clients','size_rejection_correlation','clipped_proposals')}))
    print('Saved',OUT/'diagnosis.json')


def make_figure(records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    colors={'fedavg':'#222222','clip_only':'#0072B2','rank_no_penalty':'#009E73',
            'scaled_quorum':'#D55E00','fedval':'#CC79A7'}
    for arm,color in colors.items():
        rows=records[('femnist',11,arm)]['rounds']
        axes[0,0].plot([x['round'] for x in rows],[100*x['accuracy'] for x in rows],label=arm,color=color)
    axes[0,0].set(xlabel='Round',ylabel='Development accuracy (%)',title='Learning stalls under scaled quorum')
    axes[0,0].legend(fontsize=8)
    for seed,style in [(11,'-'),(17,'--')]:
        if ('femnist',seed,'scaled_quorum') not in records:
            continue
        record=records[('femnist',seed,'scaled_quorum')]
        axes[0,1].plot([x['round'] for x in record['rounds']],
                      [100*x['accepted_nominal_mass'] for x in record['rounds']],style,label=f'Seed {seed}')
        axes[1,0].scatter([x['train_count'] for x in record['clients']],
                          [100*x['rejection_rate'] for x in record['clients']],alpha=.7,s=22,label=f'Seed {seed}')
    axes[0,1].axhline(100,color='black',linewidth=.8)
    axes[0,1].set(xlabel='Round',ylabel='Applied coefficient sum (%)',title='Rejected weights remain unused')
    axes[0,1].legend(fontsize=8)
    axes[1,0].set(xlabel='Client training examples',ylabel='Updates rejected (%)',title='Scaled quorum disproportionately rejects larger clients')
    axes[1,0].legend(fontsize=8)
    for arm in ('fedavg','fedval'):
        rows=records[('femnist',11,arm)]['rounds']
        axes[1,1].plot([x['round'] for x in rows],[x['effective_client_count'] for x in rows],label=arm,color=colors[arm])
    axes[1,1].set(xlabel='Round',ylabel='Effective client count (1 / sum weights²)',title='FedVal concentrates aggregation weights')
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.18)
    fig.suptitle('FEMNIST clean development diagnosis — seed 11 unless labeled; descriptive traces',fontsize=12)
    fig.savefig(OUT/'diagnosis.png',dpi=180)
    fig.savefig(OUT/'diagnosis.pdf')
    plt.close(fig)


def make_report(report,records):
    lines=['# V11 failure diagnosis', '',f"Snapshot: {report['created_utc']}. {report['sealed_clean_cells']} sealed clean runs analyzed.",
        'All included completion digests, frozen source/data/policy hashes, and comparisons to matched FedAvg passed.',
        'This is a post hoc development diagnosis. Training and validation labels describe distribution differences; no official test data or new training runs were used.',
        '', '## Observed clean outcomes', '', '| FEMNIST arm | Seed 11 accuracy | Seed 17 accuracy | Seed 23 accuracy |', '|---|---:|---:|---:|']
    for arm in study.CLEAN_ARMS:
        values=[f"{100*records[('femnist',s,arm)]['metrics']['final_accuracy']:.2f}%" if ('femnist',s,arm) in records else 'Pending in this snapshot' for s in study.SEEDS]
        lines.append(f"| {arm} | {' | '.join(values)} |")
    lines += ['', '## 1. Private validation produces selective rejection under writer heterogeneity', '',
        'For seed 11, the median validator has 31 validation examples covering 19 of 62 labels. Seed 17 has medians 32 and 20.5. These are small writer-specific samples of a different objective from pooled learning.',
        'Under scaled quorum, seed-11 approval is 41.1% when a validator’s observed labels cover less than half the candidate’s training-label mass, versus 83.7% otherwise. Seed 17: 40.3% versus 86.0%. These pooled vote associations are not independent significance tests or causal identification of label coverage.',
        'Committees return mixed votes on 87.8% and 86.6% of proposals, respectively. Ten approvals out of fifteen are required. Honest disagreement is being converted into a hard participation filter.',
        '', '| Scaled quorum diagnostic | Seed 11 | Seed 17 |', '|---|---:|---:|']
    for label,func in [
        ('Largest ten clients: rejection rate',lambda m:100*m['size_groups']['largest_10']['rejection_rate']),
        ('Smallest ten clients: rejection rate',lambda m:100*m['size_groups']['smallest_10']['rejection_rate']),
        ('Mean accepted nominal weight (%)',lambda m:100*m['mean_accepted_nominal_mass']),
        ('Client size–rejection correlation',lambda m:m['size_rejection_correlation'])]:
        vals=[f"{func(records[('femnist',s,'scaled_quorum')]['metrics']):.3f}" for s in (11,17)]
        lines.append(f"| {label} | {' | '.join(vals)} |")
    lines += ['',
        'This is more specific than a generic claim that ranking rejects useful updates: larger-data clients lose much more influence. One local epoch also means larger clients take more SGD steps (batch size 16), so sample size, label breadth, local training length and update geometry are confounded. The trace does not establish which causes the size association.',
        '', '## 2. Scaled quorum adds systematic damping to selective filtering', '',
        'The implementation forms a sample-weighted average of accepted clipped updates and multiplies it once by their nominal sample mass. This equals the intended sum of accepted nominal contributions; inspection found no extra division or double scaling.',
        'On seed 11, the coefficient sum averages 0.627, and only 0.541 over rounds 1–10. Across 40 rounds it sums to 25.07 (seed 17: 27.15). These are weight totals, NOT equivalent FedAvg rounds or measurements of update-vector norm: directions and trajectories differ.',
        'Seed 11 is still at 4.66% accuracy after round 20 and ends at 18.08%; clipping-only reaches 19.22% and 57.97%. Damping and altered client composition are both present. Existing runs change them jointly, so their separate contributions to the accuracy collapse remain unmeasured.',
        'For both seeds 11 and 17, scaled quorum and clipping-only have identical round-1 model hashes and round-2 candidate hashes. Their models diverge after round-2 filtering. This ties initial divergence to the aggregation decision rather than mismatched training inputs.',
        '', '## 3. Making the probe smaller does not remove validation mismatch', '',
        'The rule tests mean(d) <= 2 SE(d), with zero additive tolerance. In a locally linear regime, replacing an update by a times that update scales both mean(d) and SE(d) by a positive a. The inequality approximately stays the same. Smaller proposals therefore do not automatically become acceptable; their direction relative to each validator’s objective still matters. Nonlinear effects can change the decision.',
        'The 2-SE allowance is a heuristic, not a calibrated global-utility or Byzantine-safety guarantee. It accepts 3,047 locally loss-increasing votes out of 12,000 on seed 11; rejection of every local loss increase would not itself be a valid fix.',
        '', '## 4. Removing penalties leaves the main selection rule in place', '',
        'Rank without penalties reaches 48.46% versus FedAvg 58.03% on seed 11, and 46.04% versus 50.25% on seed 17. Seed 11 still rejects 50.6% of the largest ten clients’ proposals, versus 17.4% of the smallest ten. Rewards remain enabled and final trust ranges from 0.54 to 0.94, so the change does not eliminate weight feedback.',
        'Only 8 of 12,000 votes on seed 11 reject a locally helpful or neutral proposal. Thus mandatory rejection of locally helpful updates is not the dominant observed vote-level symptom here. Full local updates often harm another writer’s private loss even though pooled averaging learns: clipping-only records 8,768 locally harmful full-step probes while reaching 57.97%. Local loss harm must not be equated with maliciousness or global harm.',
        'The completed V10 interventions offer stronger controlled evidence: redistributing the same validation pool improves FEMNIST mean accuracy by 4.00 points; fixing all trust channels improves private-ranking accuracy by 2.43 points. Fixed-trust private ranking still trails clipping-only by 4.39 points. Trust amplifies the problem in those FEMNIST runs, but does not fully explain it.',
        '', '## 5. FedVal has a different failure signature', '',
        'The paper-based adapter uses all 62 labels with ten reference examples each. Missing reference classes are ruled out. The reference writers differ from enrolled writers; the effect of that writer shift is not established by these logs.',
        'Its adaptive exponent reaches 63 on seed 11 and 68 on seed 17. It sets roughly 46% of clean proposals to zero on seed 11; normalized weights have an effective client count of 7.75 instead of FedAvg’s 17.24 (out of 20 offered). Seed 17: 8.19 versus 17.58. This is aggressive concentration, not the lost coefficient mass seen in scaled quorum: FedVal weights still sum to one.',
        'Unlike scaled quorum, this adapter more often excludes smaller-data clients. Do not merge their failure mechanisms. Nor do these results prove a defect in published FedVal: the audited adapter has not reproduced author accuracy, and the per-candidate class-loss matrices needed to replay alternate exponents were not archived.',
        '', '## Ruled out and unresolved', '',
        'All analyzed runs have zero unsupported rounds, missing-committee events, insufficient-response decisions and abstentions. Scaled quorum has fixed trust. There is no evidence here of an attack, committee liveness failure, or a GPU execution error. Clipping alone preserves clean utility closely; its application rate rises on failed trajectories but that correlation does not establish clipping as the initiating cause.',
        'The clean pilots stop at 40 FEMNIST rounds and the controls are still improving. These results establish poor learning within the fixed budget, not an asymptotic inability to converge. Development seeds have been repeatedly examined; this diagnosis is not independent confirmation.',
        '', '## Smallest informative follow-up', '',
        'Prespecify a bounded matched clean comparison before further changes: (1) existing clipping-only; (2) clipping-only multiplied by the frozen per-round accepted-mass schedule from scaled quorum, as a diagnostic damping-only control; (3) the same scaled voting rule with accepted weights renormalized; (4) existing nominal-mass scaled quorum. Keep training, seeds, rounds and data fixed. Online votes in arm 3 will evolve with its model, so report this as a policy intervention rather than identical-update replay.',
        'This separates whether damping alone can reproduce the stall from whether selective filtering remains harmful without damping. Equalizing local SGD steps is a separate follow-up for the client-size association. Do not tune the quorum or SE threshold until these explanations are distinguished. Any successful clean redesign still needs effective attack evaluation.',
        '', '## Reproduction', '',
        'Run `.venv-trust/Scripts/python.exe -W error::RuntimeWarning -m scripts.audits.diagnose_screen_v11`.',
        'Machine-readable evidence, per-round reconstructed weights and per-client rejection rates: `artifacts/screen-v11-diagnosis/diagnosis.json`.',
        'Figure: `artifacts/screen-v11-diagnosis/diagnosis.png` and `.pdf`.',
        'Frozen V11 sources/results are unchanged. No new training or attack experiment was launched by this diagnosis.', '']
    (OUT/'diagnosis.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    main()
