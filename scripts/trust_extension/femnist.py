"""Official TFF distribution of LEAF-derived 62-class writer-partitioned EMNIST."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import requests
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
URL = 'https://storage.googleapis.com/tff-datasets-public/fed_emnist.tar.bz2'
ARCHIVE_SHA256 = 'fe1ed5a502cea3a952eb105920bff8cffb32836b5173cb18a57a32c3606f3ea0'
SOURCE_DOC = 'https://github.com/google-parfait/tensorflow-federated/blob/a484b84e053dca1a1d6afa29b86a3cd5ece9dd9d/tensorflow_federated/python/simulation/datasets/emnist.py'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    temporary.replace(path)


def download(destination, status):
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / 'fed_emnist.tar.bz2'
    if not archive.exists():
        part = destination / 'fed_emnist.tar.bz2.part'
        with requests.get(URL, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            expected = int(response.headers['Content-Length'])
            done, last = 0, 0
            with part.open('wb') as handle:
                for chunk in response.iter_content(1024 * 1024):
                    handle.write(chunk)
                    done += len(chunk)
                    if time.monotonic() - last > 3:
                        dump(status, {'status': 'downloading', 'pid': os.getpid(),
                                      'downloaded_bytes': done, 'expected_bytes': expected})
                        last = time.monotonic()
            if done != expected:
                raise ValueError('incomplete TFF archive')
            part.replace(archive)
    if digest(archive) != ARCHIVE_SHA256:
        raise ValueError('TFF archive does not match SHA256 published in the official loader')
    files = []
    # Only these two expected regular-file basenames may be materialized.
    allowed = {'fed_emnist_train.h5', 'fed_emnist_test.h5'}
    with tarfile.open(archive, 'r:bz2') as bundle:
        for member in bundle:
            name = Path(member.name).name
            if name not in allowed or not member.isfile():
                continue
            target = destination / name
            if not target.exists():
                with bundle.extractfile(member) as source, target.with_suffix('.part').open('wb') as output:
                    shutil.copyfileobj(source, output)
                target.with_suffix('.part').replace(target)
            files.append(name)
    if set(files) != allowed:
        raise ValueError(f'expected full 62-class HDF5 files; found {files}')
    return archive


def prepare_seed(destination, seed, clients=50, reference_per_class=10):
    """Keep official writer train/test correspondence; reserve different writers."""
    train_path = destination / 'fed_emnist_train.h5'
    test_path = destination / 'fed_emnist_test.h5'
    with h5py.File(train_path, 'r') as train, h5py.File(test_path, 'r') as test:
        train, test = train['examples'], test['examples']
        writers = sorted(w for w in train if w in test and len(train[w]['label']) >= 20
                         and len(test[w]['label']) >= 2)
        # Writer selection depends only on IDs and sample availability, never utility.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 202710]))
        writers = rng.permutation(writers).tolist()
        if len(writers) <= clients:
            raise ValueError('insufficient natural writers')
        selected, reserve = writers[:clients], writers[clients:]
        test_hashes = {hashlib.sha256(np.asarray(image,dtype=np.float32).tobytes()).digest()
                       for writer in selected for image in test[writer]['pixels']}
        # Keep official test examples; purge matching pixels from all training/reference data.
        seen = set(test_hashes)
        removed_duplicates = 0
        reference_x, reference_y, counts, reference_writers = [], [], np.zeros(62, int), []
        for writer in reserve:
            labels = np.asarray(train[writer]['label'], dtype=np.int64)
            pixels = np.asarray(train[writer]['pixels'], dtype=np.float32)
            used = False
            for index in rng.permutation(len(labels)):
                label = int(labels[index])
                image_hash = hashlib.sha256(pixels[index].tobytes()).digest()
                if counts[label] < reference_per_class and image_hash not in seen:
                    reference_x.append(pixels[index]); reference_y.append(label)
                    counts[label] += 1; used = True; seen.add(image_hash)
            if used:
                reference_writers.append(writer)
            if np.all(counts == reference_per_class):
                break
        if not np.all(counts == reference_per_class):
            raise ValueError('reference writers do not support all 62 labels')
        xs, ys, txs, tys = [np.asarray(reference_x)[:, None]], [np.asarray(reference_y)], [], []
        split = {'reference': list(range(len(reference_y))), 'train': [], 'validation': [],
                 'test': [], 'test_clients': [], 'writers': selected,
                 'reference_writers': reference_writers, 'seed': seed}
        offset, test_offset = len(reference_y), 0
        for client, writer in enumerate(selected):
            x = np.asarray(train[writer]['pixels'], dtype=np.float32)[:, None]
            y = np.asarray(train[writer]['label'], dtype=np.int64)
            keep=[]
            for index, image in enumerate(x):
                image_hash=hashlib.sha256(image.tobytes()).digest()
                if image_hash in seen:
                    removed_duplicates += 1
                else:
                    keep.append(index);seen.add(image_hash)
            x,y=x[keep],y[keep]
            if len(y)<5:
                raise ValueError(f'writer {writer} has insufficient data after pixel deduplication')
            test_x = np.asarray(test[writer]['pixels'], dtype=np.float32)[:, None]
            test_y = np.asarray(test[writer]['label'], dtype=np.int64)
            order = np.random.default_rng(np.random.SeedSequence([seed, 811, client])).permutation(len(y))
            count = max(1, int(len(y)*0.2))
            split['validation'].append((order[:count]+offset).tolist())
            split['train'].append((order[count:]+offset).tolist())
            split['test_clients'].append(list(range(test_offset, test_offset+len(test_y))))
            xs.append(x); ys.append(y); txs.append(test_x); tys.append(test_y)
            offset += len(y); test_offset += len(test_y)
        output = destination / f'seed-{seed}'
        output.mkdir(exist_ok=True)
        for name, array in {'features': np.concatenate(xs), 'labels': np.concatenate(ys),
                            'test_features': np.concatenate(txs), 'test_labels': np.concatenate(tys)}.items():
            if not np.isfinite(array).all():
                raise ValueError('nonfinite FEMNIST values')
            np.save(output / f'{name}.npy', array, allow_pickle=False)
        dump(output/'natural-splits.json', split)
        return {'seed': seed, 'writers': selected, 'reference_writers': reference_writers,
                'removed_training_pixel_duplicates': removed_duplicates,
                'test_unique_pixel_count':len(test_hashes),
                'training_pool_examples': offset, 'test_examples': test_offset,
                'files': {p.name: digest(p) for p in sorted(output.iterdir())}}


def prepare(destination=None):
    destination = Path(destination or ROOT/'data/femnist-tff')
    status = ROOT/'artifacts/femnist-preparation/status.json'
    archive = download(destination, status)
    dump(status, {'status': 'preparing', 'pid': os.getpid()})
    seeds = [3, 5, 7, 13, 29, 47, 71, 101]
    rows = [prepare_seed(destination, seed) for seed in seeds]
    provenance = {'dataset': 'FEMNIST-TFF-62', 'source': URL,
        'source_description': 'Official historical TFF HDF5 distribution, derived from LEAF; writer-based clients',
        'source_docs': SOURCE_DOC, 'expected_archive_sha256':ARCHIVE_SHA256,
        'archive_sha256': digest(archive), 'raw_files': {p.name: digest(p) for p in destination.glob('*.h5')},
        'preprocessing': 'Official float32 [0,1] inverse-background pixels; NCHW channel added; globally deduplicate training/reference pixels and remove pixels matching official tests before splitting. Official tests retained; no test labels used for deduplication.',
        'classes': 62, 'clients_per_seed': 50, 'seeds': rows,
        'preparation_source_sha256': digest(__file__)}
    dump(destination/'provenance.json', provenance)
    dump(status, {'status': 'completed', 'pid': os.getpid(), 'dataset': provenance['dataset'],
                  'seed_groups': len(rows), 'provenance': str(destination/'provenance.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination')
    args = parser.parse_args()
    try:
        prepare(args.destination)
    except BaseException:
        import traceback
        dump(ROOT/'artifacts/femnist-preparation/status.json',
             {'status': 'failed', 'error': traceback.format_exc()})
        raise
