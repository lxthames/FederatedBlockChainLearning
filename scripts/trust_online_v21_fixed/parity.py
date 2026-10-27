"""Reconstruct historical accepted malicious sample mass from recorded events."""
import json


def historical_weights(folder,summary):
    counts={};completed={}
    for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines():
        event=json.loads(line);rnd=event['round_index'];p=event['payload']
        if event['event_type']=='local_update_diagnostic':
            key=(rnd,p['client_id'])
            if key in counts or not isinstance(p['sample_count'],int) or p['sample_count']<1:
                raise ValueError('invalid or duplicate historical sample count')
            counts[key]=p['sample_count']
        if event['event_type']=='round_completed':
            if rnd in completed:raise ValueError('duplicate historical completed round')
            completed[rnd]=p
    rounds=summary['configuration']['rounds']
    if set(completed)!=set(range(1,rounds+1)):raise ValueError('incomplete historical round events')
    malicious=set(summary['malicious_client_ids']);c=summary['configuration'];weights=[]
    for rnd in range(1,rounds+1):
        p=completed[rnd];accepted=p['accepted_client_ids']
        if len(set(accepted))!=len(accepted):raise ValueError('duplicate accepted client')
        if any((rnd,i) not in counts for i in accepted):raise ValueError('accepted client lacks sample count')
        active=c['attack']!='none' and rnd>=c['attack_start_round']
        mass=sum(counts[rnd,i] for i in accepted)
        bad=sum(counts[rnd,i] for i in accepted if active and i in malicious)
        weights.append(bad/max(1,mass))
    # Existing telemetry, when present, must agree rather than bypass reconstruction.
    if 'accepted_malicious_weight_history' in summary and weights!=summary['accepted_malicious_weight_history']:
        raise ValueError('historical weight telemetry differs from reconstructed events')
    return weights
