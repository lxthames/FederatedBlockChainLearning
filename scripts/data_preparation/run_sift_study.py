"""Run the new SIFT1M case study after exact-label preparation passes verification.

Runs outside the frozen research module directory so adding this coordinator
does not change the code identity of the ongoing classification study.
"""
import argparse
from dataclasses import replace
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np

from .sift import atomic_json,sha256,ROOT,DATA
from scripts.research.cli import _execute_config
from scripts.research.trust_matrix import Trust2027Matrix,DEVELOPMENT_SEEDS,SEEDS
from scripts.research.results import aggregate_summaries

JOB=ROOT/'artifacts/sift1m-study'
OUTPUT=ROOT/'results/trust-2027/sift1m-case-study'


def recorded_files_match(directory,records):
    directory=Path(directory).resolve()
    if not records: raise ValueError('empty recorded-file manifest')
    for name,expected in records.items():
        if Path(name).name!=name or not isinstance(expected,str):
            raise ValueError('invalid provenance file entry')
        path=directory/name
        if not path.is_file() or sha256(path)!=expected:
            raise ValueError(f'dataset checksum mismatch: {name}')


def verify_dataset(directory):
    directory=Path(directory)
    provenance=json.loads((directory/'sift-provenance.json').read_text(encoding='utf-8'))
    if provenance.get('dataset_name')!='SIFT1M' or provenance.get('query_target_correspondence_verified') is not True:
        raise ValueError('verified SIFT1M correspondence is required')
    if provenance.get('neighbors_per_query')!=1000 or provenance.get('database_size')!=1000000:
        raise ValueError('case study requires top-1000 targets against the 1M database')
    recorded_files_match(directory,provenance['generated_files'])
    if sha256(directory/'raw/sift.tar.gz')!=provenance['source_archive_sha256']:
        raise ValueError('source archive checksum mismatch')
    for name,item in provenance['source_arrays'].items():
        recorded_files_match(directory/'raw',{name:item['sha256']})
    train=np.load(directory/'SIFT1B_train.npy',mmap_mode='r')
    test=np.load(directory/'SIFT_test_query.npy',mmap_mode='r')
    if train.shape!=(provenance['training_query_count'],128) or test.shape!=(10000,128):
        raise ValueError('source/query shapes do not match provenance')
    for population in (4,8):
        for suffix,rows in (('',len(train)),('test',len(test))):
            targets=np.load(directory/f'gt_{population}M_client_counts_skewed{suffix}.npy',mmap_mode='r')
            if targets.shape!=(rows,population) or np.any(targets<0) or not np.all(targets.sum(1)==1000):
                raise ValueError('invalid routing-target mass or correspondence')
    return provenance


def source_digest():
    return hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'scripts/research').glob('*.py')))).hexdigest()


def configs(device):
    full=[]
    for config in Trust2027Matrix().sift_case_study():
        identity=f'sift1m-n{config.num_clients}-{config.model}-{config.defense}-{config.attack}-s{config.seed}'
        full.append(replace(config,experiment_id=identity,device=device))
    development=[]
    for population in (4,8):
        for model in ('emh_dnn','mlp'):
            for attack in ('none','sign_flip','low_and_slow'):
                template=next(c for c in full if c.num_clients==population and c.model==model
                              and c.defense=='fedavg' and c.attack==attack)
                for seed in DEVELOPMENT_SEEDS:
                    development.append(replace(template,seed=seed,rounds=20,evidence_tier='development',
                        experiment_id=f'sift1m-development-n{population}-{model}-{attack}-s{seed}'))
    return development,full


def attack_gates(summaries):
    gates={}
    for population in (4,8):
        for model in ('emh_dnn','mlp'):
            rows=[s for s in summaries if s['num_clients']==population and s['model']==model]
            clean={s['seed']:s['final_utility'] for s in rows if s['attack']=='none'}
            gates[f'{population}:{model}:none']={'passed':len(clean)==3,'seed_count':len(clean)}
            for attack in ('sign_flip','low_and_slow'):
                attacked={s['seed']:s['final_utility'] for s in rows if s['attack']==attack}
                differences=[clean[s]-attacked[s] for s in DEVELOPMENT_SEEDS if s in clean and s in attacked]
                gates[f'{population}:{model}:{attack}']={
                    'passed':len(differences)==3 and float(np.mean(differences))>0.005,
                    'seed_count':len(differences),'utility_loss_by_seed':differences,
                    'criterion':'three matched development seeds; mean clean-minus-attacked Recall@1000 > 0.005'}
    return gates


def run(device='cpu'):
    JOB.mkdir(parents=True,exist_ok=True)
    state={'status':'running','pid':os.getpid(),'phase':'waiting_for_verified_labels',
        'completed':0,'failed':0,'held':0,'dataset':'SIFT1M','device':device,
        'development_total':36,'case_study_total':180,'publication_ready':False,
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    frozen_source=source_digest()
    def update(**values):
        state.update(values); state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(JOB/'status.json',state)
    def execute(config,stage):
        update(phase=stage,current_experiment=config.experiment_id)
        output=OUTPUT/stage/config.experiment_id
        try:
            if source_digest()!=frozen_source:
                raise RuntimeError('research code changed during study; refusing to mix implementations')
            if (output/'summary.json').exists():
                result=json.loads((output/'summary.json').read_text())
                expected=hashlib.sha256(json.dumps(config.to_dict(),sort_keys=True).encode()).hexdigest()
                if result.get('config_hash')!=expected or result.get('source_hash')!=frozen_source:
                    raise RuntimeError('completed case-study run does not match frozen inputs/code')
            else:
                if output.exists() and any(output.iterdir()):
                    output.rename(output.with_name(output.name+f'.interrupted-{time.time_ns()}'))
                result=_execute_config(config,output,DATA)
                atomic_json(output/'dataset-identity.json',{
                    'dataset_name':'SIFT1M','dataset_provenance_sha256':sha256(DATA/'sift-provenance.json'),
                    'database_size':1000000,'neighbors_per_query':1000,
                    'actual_poisoning_client_fraction':len(result['malicious_client_ids'])/config.num_clients,
                    'timing_environment':'CPU study concurrent with a separate classification job; wall-time overhead is exploratory'})
            state['completed']+=1
            return result
        except Exception:
            state['failed']+=1
            with (JOB/'failures.jsonl').open('a',encoding='utf-8') as out:
                out.write(json.dumps({'experiment_id':config.experiment_id,'stage':stage,
                                     'error':traceback.format_exc()})+'\n')
            return None
        finally: update()
    try:
        update()
        while not (DATA/'sift-provenance.json').exists():
            prep_path=ROOT/'artifacts/sift-data-preparation/status.json'
            if prep_path.exists():
                prep=json.loads(prep_path.read_text())
                update(label_preparation_phase=prep.get('phase'),label_preparation_status=prep.get('status'))
                if prep.get('status')=='failed': raise RuntimeError('label preparation failed; see its status and logs')
            time.sleep(5)
        update(phase='verify_label_provenance')
        provenance=verify_dataset(DATA)
        atomic_json(JOB/'dataset-provenance.json',provenance)
        development,full=configs(device)
        for name,cells in (('development',development),('case-study',full)):
            manifest=JOB/f'{name}-manifest.json'
            expected=[c.to_dict() for c in cells]
            if manifest.exists() and json.loads(manifest.read_text())!=expected:
                raise ValueError('existing SIFT study manifest differs; use a new study version')
            atomic_json(manifest,expected)
        atomic_json(JOB/'frozen-inputs.json',{'research_code_sha256':frozen_source,
            'emh_model_definition_sha256':sha256(ROOT/'scripts/core/models.py'),
            'coordinator_sha256':sha256(__file__),'dataset_provenance_sha256':sha256(DATA/'sift-provenance.json'),
            'manifest_sha256':{name:sha256(JOB/name) for name in ('development-manifest.json','case-study-manifest.json')}})
        screening=[]
        for config in development:
            result=execute(config,'development')
            if result: screening.append(result)
        gates=attack_gates(screening)
        atomic_json(JOB/'attack-validation.json',gates)
        held=[]
        for config in full:
            key=f'{config.num_clients}:{config.model}:{config.attack}'
            if not gates[key]['passed']:
                held.append({'experiment_id':config.experiment_id,'reason':'development control incomplete or attack ineffective'})
                state['held']=len(held); atomic_json(JOB/'held-experiments.json',held); update()
                continue
            execute(config,'case-study')
        complete={}
        for path in (OUTPUT/'case-study').glob('*/summary.json'):
            summary=json.loads(path.read_text())
            complete.setdefault((summary['config_group_hash'],summary['source_hash']),[]).append(path)
        rows=[]
        for group in complete.values():
            if {json.loads(p.read_text())['seed'] for p in group}==set(SEEDS):
                rows.extend(aggregate_summaries(group,require_confirmatory_seeds=True))
        atomic_json(JOB/'five-seed-analysis.json',rows)
        update(status='completed_with_gaps' if state['failed'] or held else 'completed',phase='finished',
               complete_five_seed_cells=len(rows),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    except BaseException:
        update(status='failed',error=traceback.format_exc())
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','auto','cuda'),default='cpu')
    args=parser.parse_args(); run(args.device)

if __name__=='__main__': main()
