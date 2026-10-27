"""Reserve new FEMNIST writers and audit exposed replication data."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scripts.trust_challenge_v14.femnist import digest,dump

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/lifecycle-v25'
SEEDS=(4103,4211,4327,4441,4549)


def historical_exposure():
    writers=set();files={}
    for p in (ROOT/'data').rglob('natural-splits.json'):
        if DATA in p.parents: continue
        d=json.loads(p.read_text())
        writers.update(d.get('writers',[]));writers.update(d.get('reference_writers',[]))
        files[str(p.relative_to(ROOT))]=digest(p)
    seeds=set();summary_count=0
    for p in (ROOT/'results').rglob('summary.json'):
        if 'lifecycle-v25' in p.parts: continue
        d=json.loads(p.read_text())
        seeds.add(d.get('seed',d.get('configuration',{}).get('seed')));summary_count+=1
    if seeds.intersection(SEEDS): raise ValueError('planned seeds already occur in historical summaries')
    return writers,dict(split_sources=files,excluded_writer_ids=sorted(writers),
        historical_summary_count=summary_count,previous_seeds=sorted(s for s in seeds if isinstance(s,int)),
        new_seeds=list(SEEDS))


def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    if (DATA/'provenance.json').exists():
        return json.loads((DATA/'provenance.json').read_text())
    excluded,audit=historical_exposure()
    raw=ROOT/'data/femnist-tff'
    raw_hashes={name:digest(raw/name) for name in ('fed_emnist_train.h5','fed_emnist_test.h5')}
    # Check the prepared raw files against the existing official-source record.
    prior=json.loads((raw/'provenance.json').read_text())
    if raw_hashes!=prior['raw_files']: raise ValueError('official FEMNIST raw data changed')
    provenance=dict(exposure_audit=audit,raw_femnist_hashes=raw_hashes,
        fashion_scope='Fresh seeds and client partitions, but previously exposed official data; replication only.',
        femnist_scope='All enrolled and reference writers disjoint from historical natural-split writer IDs and across new seeds.',
        files={},seeds=[])
    with h5py.File(raw/'fed_emnist_train.h5','r') as tr, h5py.File(raw/'fed_emnist_test.h5','r') as te:
        tr,te=tr['examples'],te['examples']
        for seed in SEEDS:
            rng=np.random.default_rng(np.random.SeedSequence([seed,2501]))
            candidates=sorted(w for w in tr if w in te and w not in excluded and len(tr[w]['label'])>=50 and len(te[w]['label'])>=2)
            order=rng.permutation(candidates).tolist()
            writers=order[:50];reserve=order[50:]
            if len(writers)!=50: raise ValueError('not enough held-out writers')
            txs=[np.asarray(te[w]['pixels'],np.float32)[:,None] for w in writers]
            tys=[np.asarray(te[w]['label'],np.int64) for w in writers]
            tx,ty=np.concatenate(txs),np.concatenate(tys)
            seen={hashlib.sha256(image.tobytes()).digest() for image in tx}
            rx=[];ry=[];counts=np.zeros(62,int);reference_writers=[]
            for writer in reserve:
                used=False
                for image,label in zip(np.asarray(tr[writer]['pixels'],np.float32)[:,None],np.asarray(tr[writer]['label'],np.int64)):
                    h=hashlib.sha256(image.tobytes()).digest()
                    if counts[label]<10 and h not in seen:
                        rx.append(image);ry.append(label);seen.add(h);counts[label]+=1;used=True
                if used: reference_writers.append(writer)
                if np.all(counts==10): break
            if not np.all(counts==10): raise ValueError('missing reference class support')
            xs=[np.asarray(rx)];ys=[np.asarray(ry,np.int64)]
            split=dict(seed=seed,writers=writers,reference_writers=reference_writers,reference=list(range(len(ry))),train=[],validation=[],test_clients=[])
            offset=len(ry);test_offset=0
            for client,w in enumerate(writers):
                x=np.asarray(tr[w]['pixels'],np.float32)[:,None];y=np.asarray(tr[w]['label'],np.int64)
                keep=[]
                for i,image in enumerate(x):
                    h=hashlib.sha256(image.tobytes()).digest()
                    if h not in seen: keep.append(i);seen.add(h)
                x,y=x[keep],y[keep]
                if len(y)<10: raise ValueError('writer too small after deduplication')
                ids=rng.permutation(len(y));n=max(2,int(.2*len(y)))
                split['validation'].append((ids[:n]+offset).tolist())
                split['train'].append((ids[n:]+offset).tolist())
                split['test_clients'].append(list(range(test_offset,test_offset+len(tys[client]))))
                xs.append(x);ys.append(y);offset+=len(y);test_offset+=len(tys[client])
            folder=DATA/f'femnist-{seed}';folder.mkdir(exist_ok=True)
            for name,arr in dict(features=np.concatenate(xs),labels=np.concatenate(ys),test_features=tx,test_labels=ty).items():
                np.save(folder/f'{name}.npy',arr,allow_pickle=False)
            dump(folder/'natural-splits.json',split)
            excluded.update(writers);excluded.update(reference_writers)
            provenance['seeds'].append(dict(seed=seed,writers=writers,reference_writers=reference_writers))
    # Bind the exact existing official Fashion-MNIST arrays without modifying them.
    from torchvision.datasets import FashionMNIST
    for train,name in ((True,'train'),(False,'test')):
        ds=FashionMNIST(ROOT/'data',train=train,download=False)
        for suffix,arr in (('features',ds.data.numpy()[:,None].astype(np.float32)/255.),('labels',ds.targets.numpy().astype(np.int64))):
            np.save(DATA/f'fashion-{name}-{suffix}.npy',arr,allow_pickle=False)
    provenance['files']={str(p.relative_to(DATA)):digest(p) for p in DATA.rglob('*') if p.is_file() and p.name!='provenance.json'}
    dump(DATA/'provenance.json',provenance)
    return provenance


def load(dataset,seed):
    if dataset=='femnist':
        return [np.load(DATA/f'femnist-{seed}'/f'{name}.npy',allow_pickle=False) for name in ('features','labels','test_features','test_labels')]
    return [np.load(DATA/f'fashion-{part}-{name}.npy',allow_pickle=False) for part in ('train','test') for name in ('features','labels')]
