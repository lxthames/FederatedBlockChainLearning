"""Finite 42-cell mechanism study. Default command only prints the manifest."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '2')

import argparse
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

from .pairing import compare_pair
from .research.config import ExperimentConfig
from .research.models import create_model
from .research.runner_v2 import BenchmarkRunner
from .runtime import exclusive_lock

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data/trust-revision-v3-development'
JOB = ROOT/'artifacts/mechanism-v10'
OUT = ROOT/'results/mechanism-v10'
POLICY = ROOT/'docs/research/MECHANISM_V10.md'
SEEDS = (11, 17, 23)
ARMS = ('rank_private_dynamic', 'rank_redistributed_dynamic',
        'rank_private_fixed', 'rank_redistributed_fixed',
        'rank_private_weights_only', 'fedavg', 'clip_only')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    temp.replace(path)


def config(dataset, arm, seed):
    if dataset not in {'fashion_mnist', 'femnist'} or arm not in ARMS:
        raise ValueError('unknown mechanism-study cell')
    femnist = dataset == 'femnist'
    fixed = arm.endswith('_fixed') or arm in {'fedavg', 'clip_only'}
    return ExperimentConfig(experiment_id=f'v10-{dataset}-{arm}-s{seed}',
        dataset=dataset, num_clients=50, participation_rate=.4, malicious_fraction=.3,
        dirichlet_alpha=None if femnist else .5, seed=seed, rounds=40 if femnist else 30,
        attack='none', attack_start_round=11, num_classes=62 if femnist else 10,
        batch_size=16 if femnist else 64, learning_rate=.05 if femnist else .01,
        optimizer_momentum=.9, device='cuda', sampling_policy='bounded',
        defense='fedavg' if arm=='fedavg' else 'two_layer', vote_rule='rank_loss',
        ablation='validation_only', candidate_clip_ratio=2.,
        natural_splits_path=str(DATA/f'{dataset}-{seed}/natural-splits.json') if femnist else None,
        validation_distribution='redistributed' if 'redistributed' in arm else 'private',
        trust_updates='fixed' if fixed else 'dynamic',
        trust_weighting='fixed' if fixed else 'dynamic',
        trust_eligibility='fixed' if fixed or arm.endswith('weights_only') else 'dynamic',
        # Explicit historical diagnostic, never the default policy of the runner.
        rejection_trust_policy='none' if fixed else 'rank_rejection',
        ranking_keep_fraction=1. if arm=='clip_only' else .8)


def manifest():
    return [{'arm': arm, 'config': config(dataset, arm, seed).to_dict()}
            for dataset in ('fashion_mnist', 'femnist') for seed in SEEDS for arm in ARMS]


def inputs():
    data = {}
    for dataset in ('fashion_mnist', 'femnist'):
        for seed in SEEDS:
            folder = DATA/f'{dataset}-{seed}'
            provenance = json.loads((folder/'provenance.json').read_text())
            names = ['features.npy', 'labels.npy', 'development_features.npy', 'development_labels.npy']
            if dataset=='femnist': names.append('natural-splits.json')
            for name in names:
                actual = digest(folder/name)
                if provenance['files'].get(name) != actual:
                    raise ValueError(f'Data provenance mismatch: {folder/name}')
                data[str((folder/name).relative_to(ROOT))] = actual
            data[str((folder/'provenance.json').relative_to(ROOT))] = digest(folder/'provenance.json')
    sources = {str(p.relative_to(ROOT)): digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))}
    return {'manifest': manifest(), 'data': data, 'source': sources,
            'policy_sha256': digest(POLICY), 'evaluation_role': 'search-exposed development',
            'automatic_confirmation': False, 'publication_ready': False}


def freeze():
    frozen = inputs()
    path = JOB/'frozen-inputs.json'
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError('V10 frozen inputs changed; use a new study version')
    dump(path, frozen)
    return frozen


FILES = ('config.json', 'summary.json', 'splits.json', 'events.jsonl', 'chain-trace.json')


def seal(folder):
    dump(folder/'completion.json', {name: digest(folder/name) for name in FILES})


def load_completed(folder, c):
    saved = json.loads((folder/'completion.json').read_text())
    if saved != {name: digest(folder/name) for name in FILES}:
        raise ValueError('Completed artifact hash mismatch')
    summary = json.loads((folder/'summary.json').read_text())
    expected = hashlib.sha256(json.dumps(c.to_dict(), sort_keys=True).encode()).hexdigest()
    if summary.get('config_hash') != expected or summary.get('status') != 'completed':
        raise ValueError('Completed configuration/status mismatch')
    return summary


def analyze(records, state):
    comparisons = []
    for row in records:
        baseline = next((r for r in records if r['arm']=='fedavg' and
                         r['dataset']==row['dataset'] and r['seed']==row['seed']), None)
        if baseline:
            pairing = compare_pair(row, baseline, allow_validation_redistribution=True)
            comparisons.append({'arm':row['arm'], 'dataset':row['dataset'], 'seed':row['seed'],
                **pairing, 'utility_difference':row['final_utility']-baseline['final_utility'] if pairing['matched'] else None})
    contrasts = []
    planned = {
        'private_total_trust':('rank_private_dynamic','rank_private_fixed'),
        'redistributed_total_trust':('rank_redistributed_dynamic','rank_redistributed_fixed'),
        'redistribution_dynamic':('rank_redistributed_dynamic','rank_private_dynamic'),
        'redistribution_fixed':('rank_redistributed_fixed','rank_private_fixed'),
        'eligibility_under_dynamic_weights':('rank_private_dynamic','rank_private_weights_only'),
        'weights_updates_under_fixed_eligibility':('rank_private_weights_only','rank_private_fixed'),
    }
    for dataset in ('fashion_mnist','femnist'):
        for seed in SEEDS:
            cells = {r['arm']:r for r in records if r['dataset']==dataset and r['seed']==seed}
            effects = {}
            for name,(left,right) in planned.items():
                if left not in cells or right not in cells: continue
                a,b=cells[left],cells[right]
                pair=compare_pair(a,b,allow_validation_redistribution=True)
                effect=a['final_utility']-b['final_utility'] if pair['matched'] else None
                effects[name]=effect
                contrasts.append({'dataset':dataset,'seed':seed,'contrast':name,**pair,
                    'utility_difference':effect,
                    'lower_decile_difference':a['lower_decile_client_utility']-b['lower_decile_client_utility'] if pair['matched'] else None,
                    'benign_exclusion_difference':a['benign_update_exclusion_rate']-b['benign_update_exclusion_rate'] if pair['matched'] else None})
            keys=('private_total_trust','redistributed_total_trust','redistribution_dynamic','redistribution_fixed')
            if all(effects.get(k) is not None for k in keys):
                contrasts.append({'dataset':dataset,'seed':seed,'contrast':'validation_by_trust_interaction',
                    'matched':True,'utility_difference':effects[keys[0]]-effects[keys[1]]})
    contrast_groups=[]
    for dataset in ('fashion_mnist','femnist'):
        for name in (*planned, 'validation_by_trust_interaction'):
            rows=[r for r in contrasts if r['dataset']==dataset and r['contrast']==name and r['matched']]
            complete=sorted(r['seed'] for r in rows)==list(SEEDS)
            contrast_groups.append({'dataset':dataset,'contrast':name,'complete':complete,
                'seeds':[r['seed'] for r in rows],
                'mean_utility_difference':float(np.mean([r['utility_difference'] for r in rows])) if complete else None})
    compact = [{key:r[key] for key in ('arm', 'dataset', 'seed', 'final_utility',
        'lower_decile_client_utility', 'mean_client_utility', 'worst_client_utility',
        'unsupported_rounds', 'final_trust', 'inconclusive_counts', 'quality_rejection_counts',
        'committee_unavailable_counts', 'client_eligible_round_counts',
        'benign_update_exclusion_rate', 'rank_vote_counts')} for r in records]
    dump(JOB/'analysis.json', {'status':state, 'runs':compact, 'comparisons':comparisons,
        'contrasts':contrasts,'contrast_groups':contrast_groups,
        'publication_ready':False, 'automatic_promotion':False,
        'limitations':['Development only; all diagnostic cells reported, including clean harm.',
                       'Redistribution changes data access and is not a private deployment solution.',
                       'Rank penalties are an explicit historical diagnostic, not evidence of misconduct.']})


def run():
    with exclusive_lock(JOB/'coordinator.lock'), threadpool_limits(limits=2):
        path = JOB/'frozen-inputs.json'
        if not path.exists(): raise ValueError('Run --freeze before --run')
        frozen = json.loads(path.read_text())
        if inputs() != frozen: raise ValueError('Frozen source, policy or data changed')
        if not torch.cuda.is_available(): raise RuntimeError('Study requires CUDA; tests use separate CPU configurations')
        torch.set_num_threads(2)
        dump(JOB/'environment.json', {'python':platform.python_version(), 'torch':torch.__version__,
             'numpy':np.__version__, 'gpu':torch.cuda.get_device_name(0)})
        state = {'status':'running', 'completed':0, 'failed':0, 'total':len(frozen['manifest']),
                 'pid':os.getpid(), 'publication_ready':False, 'started_utc':datetime.now(timezone.utc).isoformat()}
        records = []
        dump(JOB/'status.json', state)
        try:
            for row in frozen['manifest']:
                if inputs() != frozen: raise ValueError('Frozen inputs changed during study')
                c = ExperimentConfig(**row['config'])
                folder = OUT/c.experiment_id
                state['current'] = c.experiment_id
                dump(JOB/'status.json', state)
                try:
                    if (folder/'completion.json').exists():
                        summary = load_completed(folder, c)
                    else:
                        if folder.exists():
                            # Both paths are fixed children of this version's result root.
                            target = folder.with_name(folder.name+f'.interrupted-{time.time_ns()}')
                            if folder.resolve().parent != OUT.resolve() or target.resolve().parent != OUT.resolve():
                                raise ValueError('Interrupted result path escapes V10 output root')
                            folder.rename(target)
                        data = DATA/f'{c.dataset}-{c.seed}'
                        x,y,ex,ey = [np.load(data/f'{name}.npy', allow_pickle=False) for name in
                                    ('features','labels','development_features','development_labels')]
                        summary = BenchmarkRunner(c, folder).run_arrays(x,y,
                            lambda:create_model(c.dataset,x.shape[1:],c.num_classes,c.model),(ex,ey))
                        if inputs() != frozen: raise ValueError('Inputs changed while cell was running')
                        seal(folder)
                    records.append(dict(summary, arm=row['arm']))
                    state['completed'] += 1
                except Exception:
                    state['failed'] += 1
                    with (JOB/'failures.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps({'experiment_id':c.experiment_id,'error':traceback.format_exc()})+'\n')
                analyze(records, state)
                dump(JOB/'status.json', state)
            state['status'] = 'completed_with_gaps' if state['failed'] else 'completed'
        except BaseException:
            state.update(status='failed', error=traceback.format_exc())
            raise
        finally:
            state['finished_utc'] = datetime.now(timezone.utc).isoformat()
            dump(JOB/'status.json', state)
            analyze(records, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--freeze', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run: run()
    elif args.freeze:
        freeze()
        print(f'Frozen {len(manifest())} development cells at {JOB}; training not started.')
    else: print(json.dumps(manifest(), indent=2))
