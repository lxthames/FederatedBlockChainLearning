"""Bounded development replication using the unchanged, frozen V6 implementation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '2')
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from scripts.trust_literature_v6 import study as v6
from scripts.trust_literature_v6.femnist import digest, dump
from scripts.trust_literature_v6.runtime import exclusive_lock
from scripts.trust_literature_v6.research.config import ExperimentConfig
from scripts.trust_literature_v6.research.models import create_model
from scripts.trust_literature_v6.research.runner_v2 import BenchmarkRunner

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT/'artifacts/rank-replication-v7'
OUT = ROOT/'results/rank-replication-v7'
DATA = ROOT/'data/trust-revision-v3-development'
POLICY = ROOT/'docs/research/RANK_REPLICATION_V7.md'
SEEDS = (11, 17, 23)
ATTACKS = ('none', 'low_and_slow', 'model_replacement')


def manifest():
    rows = []
    # All clean replication cells precede attack cells; unchanged per-seed V6 gate.
    for attack in ATTACKS:
        for seed in (17, 23):
            for method in ('fedavg', 'rank_trust'):
                c = replace(v6.config(method, attack), seed=seed,
                            experiment_id=f'v7-{method}-{attack}-s{seed}')
                rows.append({'method': method, 'config': c.to_dict()})
    c = replace(v6.config('coordinate_median', 'none'),
                experiment_id='v7-coordinate_median-none-s11')
    rows.append({'method': 'coordinate_median', 'config': c.to_dict()})
    return rows


def source_hashes():
    return dict(v6.hashes(), **{str(Path(__file__).relative_to(ROOT)): digest(__file__)})


def seed_gate(records, method, seed):
    return v6.clean_gate([r for r in records if r['seed'] == seed], method)


def load_completed(folder, c):
    summary = json.loads((folder/'summary.json').read_text())
    expected = hashlib.sha256(json.dumps(c.to_dict(), sort_keys=True).encode()).hexdigest()
    if summary.get('status') != 'completed' or summary.get('config_hash') != expected:
        raise ValueError(f'Invalid completed cell: {folder}')
    return summary


def reuse_spec():
    rows = [(m, a) for a in ATTACKS for m in ('fedavg', 'rank_trust')]
    return rows + [('fldetector_pp', 'none')]


def frozen_inputs():
    old = json.loads((v6.JOB/'frozen-inputs.json').read_text())
    if old['source'] != v6.hashes():
        raise ValueError('Original V6 sources changed')
    data = {}
    for seed in SEEDS:
        folder = DATA/f'fashion_mnist-{seed}'
        provenance = json.loads((folder/'provenance.json').read_text())
        for name, expected in provenance['files'].items():
            if digest(folder/name) != expected:
                raise ValueError(f'Data changed: {folder/name}')
            data[str((folder/name).relative_to(ROOT))] = expected
        data[str((folder/'provenance.json').relative_to(ROOT))] = digest(folder/'provenance.json')
    reused = {}
    for method, attack in reuse_spec():
        folder = v6.OUT/v6.config(method, attack).experiment_id
        load_completed(folder, v6.config(method, attack))
        for name in ('summary.json', 'events.jsonl', 'chain-trace.json'):
            reused[str((folder/name).relative_to(ROOT))] = digest(folder/name)
    return {'source': source_hashes(), 'data': data, 'reused': reused,
            'manifest': manifest(), 'policy_sha256': digest(POLICY),
            'automatic_confirmation': False, 'publication_ready': False}


def analyze(records, state):
    effects = []
    for attack in ATTACKS:
        pairs = []
        for seed in SEEDS:
            group = {r['method']: r for r in records if r['seed'] == seed and r['attack'] == attack}
            if not {'fedavg', 'rank_trust'} <= group.keys():
                continue
            b, r = group['fedavg'], group['rank_trust']
            clean = next((x for x in records if x['method']=='fedavg' and x['seed']==seed and x['attack']=='none'), None)
            matched = all(b[k] == r[k] for k in ('participants', 'data_fingerprint', 'test_fingerprint'))
            asr = b['post_onset_mean_asr']-r['post_onset_mean_asr'] if attack=='model_replacement' else None
            valid = (clean['final_utility']-b['final_utility']>=.01 if attack=='low_and_slow' else
                     b['post_onset_mean_asr']-clean['post_onset_mean_clean_trigger_rate']>=.10 if attack=='model_replacement' else True)
            pairs.append({'seed': seed, 'matched': matched, 'attack_effective': bool(valid),
                          'utility_effect': r['final_utility']-b['final_utility'], 'asr_reduction': asr,
                          'clean_gate': seed_gate(records, 'rank_trust', seed),
                          'recoveries': r['recovery_count'],
                          'honest_penalty': r['honest_validator_penalty_amount']})
        complete = len(pairs)==3
        eligible = complete and all(p['matched'] and p['attack_effective'] and p['clean_gate']['passed']
                                   and p['utility_effect']>=-.05 and p['honest_penalty']==0 for p in pairs)
        benefit = complete and (np.mean([p['asr_reduction'] for p in pairs])>=.05 if attack=='model_replacement'
                                else np.mean([p['utility_effect'] for p in pairs])>=.01 if attack=='low_and_slow' else True)
        effects.append({'attack': attack, 'pairs': pairs, 'complete_three_seed_group': complete,
                        'passes_prespecified_review_criteria': bool(eligible and benefit)})
    med = next((r for r in records if r['method']=='coordinate_median'), None)
    det = next((r for r in records if r['method']=='fldetector_pp'), None)
    diagnostic = None
    if med and det:
        diagnostic = {'median_utility': med['final_utility'], 'detector_utility': det['final_utility'],
                      'detector_minus_median': det['final_utility']-med['final_utility'],
                      'quarantined': det['quarantined'],
                      'matched': all(med[k]==det[k] for k in ('participants','data_fingerprint','test_fingerprint')),
                      'identical_preonset_model_hashes': med['pre_onset_model_hashes']==det['pre_onset_model_hashes'],
                      'scope': 'single-seed clean aggregation control; no attacker-detection claim'}
    report = {'status': state, 'paired_results': effects, 'median_diagnostic': diagnostic,
              'runs': [{k:r.get(k) for k in ('method','seed','attack','final_utility','post_onset_mean_asr',
                        'recovery_count','honest_update_exclusion_rate','runtime_seconds_total')} for r in records],
              'publication_ready': False, 'automatic_promotion': False,
              'limitation': 'Reused, search-exposed development seeds and overlapping training pools. No confirmatory inference.'}
    dump(JOB/'analysis.json', report)
    lines = ['# Rank/trust development replication', '', report['limitation'], '',
             '| Method | Seed | Condition | Final utility | Mean targeted ASR |',
             '|---|---:|---|---:|---:|']
    for r in records:
        asr = 'n/a' if r['post_onset_mean_asr'] is None else f"{100*r['post_onset_mean_asr']:.1f}%"
        lines.append(f"| {r['method']} | {r['seed']} | {r['attack']} | {100*r['final_utility']:.1f}% | {asr} |")
    lines += ['', 'Criteria and missing cells: see analysis.json. No automatic confirmatory launch.', '',
              'Median diagnostic: '+json.dumps(diagnostic)]
    (JOB/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run():
    with exclusive_lock(JOB/'coordinator.lock'), threadpool_limits(limits=2):
        frozen = frozen_inputs()
        path = JOB/'frozen-inputs.json'
        if path.exists() and json.loads(path.read_text()) != frozen:
            raise ValueError('Frozen replication inputs changed')
        dump(path, frozen)
        state = {'status':'starting', 'pid':os.getpid(), 'completed':0, 'failed':0, 'held':0,
                 'reused':7, 'total_new':13, 'publication_ready':False,
                 'started_utc':datetime.now(timezone.utc).isoformat()}
        dump(JOB/'status.json', state)
        records = []
        try:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required')
            torch.set_num_threads(2)
            dump(JOB/'environment.json', {'python':platform.python_version(), 'torch':torch.__version__,
                 'numpy':np.__version__, 'gpu':torch.cuda.get_device_name(0)})
            for method, attack in reuse_spec():
                # Read original artifacts without rewriting their enriched summaries.
                s = load_completed(v6.OUT/v6.config(method, attack).experiment_id, v6.config(method, attack))
                records.append(s)
            for row in frozen['manifest']:
                if source_hashes()!=frozen['source'] or digest(POLICY)!=frozen['policy_sha256']:
                    raise ValueError('Frozen source/policy changed')
                c = ExperimentConfig(**row['config']); method = row['method']
                state.update(status='running', current=c.experiment_id, updated_utc=datetime.now(timezone.utc).isoformat())
                dump(JOB/'status.json',state)
                folder = OUT/c.experiment_id
                if c.attack!='none' and not seed_gate(records,method,c.seed)['passed']:
                    dump(folder/'held.json',seed_gate(records,method,c.seed)); state['held']+=1
                    analyze(records,state); dump(JOB/'status.json',state); continue
                try:
                    if (folder/'summary.json').exists():
                        s = load_completed(folder,c)
                    else:
                        if folder.exists() and any(folder.iterdir()):
                            folder.rename(folder.with_name(folder.name+f'.interrupted-{time.time_ns()}'))
                        datafolder = DATA/f'fashion_mnist-{c.seed}'
                        for p,h in frozen['data'].items():
                            if Path(p).parent == datafolder.relative_to(ROOT) and digest(ROOT/p)!=h:
                                raise ValueError('Frozen data changed')
                        x,y,ex,ey = [np.load(datafolder/f'{n}.npy',allow_pickle=False) for n in
                                     ('features','labels','development_features','development_labels')]
                        s = BenchmarkRunner(c,folder).run_arrays(x,y,lambda:create_model(c.dataset,x.shape[1:],10),(ex,ey))
                    records.append(v6.enrich(folder,s,method)); state['completed']+=1
                except Exception:
                    state['failed']+=1
                    with (JOB/'failures.jsonl').open('a',encoding='utf-8') as handle:
                        handle.write(json.dumps({'experiment_id':c.experiment_id,'error':traceback.format_exc()})+'\n')
                analyze(records,state); dump(JOB/'status.json',state)
            state['status'] = 'completed_with_gaps' if state['failed'] or state['held'] else 'completed'
        except BaseException:
            state.update(status='failed',error=traceback.format_exc())
            raise
        finally:
            state['finished_utc']=datetime.now(timezone.utc).isoformat()
            dump(JOB/'status.json',state); analyze(records,state)


if __name__=='__main__':
    run()
