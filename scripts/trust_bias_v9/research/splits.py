"""Disjoint index-based splits, independent streams and content fingerprints."""
from dataclasses import dataclass
import hashlib
import numpy as np
from .datasets import partition_indices


def stream(seed: int, domain: str, *indices: int) -> np.random.Generator:
    digest = hashlib.sha256(domain.encode()).digest()
    domain_id = int.from_bytes(digest[:4], 'little')
    return np.random.default_rng(np.random.SeedSequence([seed, domain_id, *indices]))


def fingerprint(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


@dataclass
class DataSplit:
    train: list[np.ndarray]
    validation: list[np.ndarray]
    reference: np.ndarray
    test: np.ndarray

    def manifest(self):
        return {key: ([x.tolist() for x in value] if isinstance(value, list)
                      else value.tolist()) for key, value in vars(self).items()}


def make_standard_splits(labels, config, external_test=False):
    classes = labels if labels.ndim == 1 else labels.argmax(axis=1)
    rng = stream(config.seed, 'global-splits')
    test, reference, pool = [], [], []
    for label in np.unique(classes):
        indices = rng.permutation(np.flatnonzero(classes == label))
        count = 0 if external_test else max(1, int(len(indices) * config.test_fraction))
        test.extend(indices[:count])
        remaining = indices[count:]
        root_count = min(config.server_samples_per_class, len(remaining) // 5)
        reference.extend(remaining[:root_count])
        pool.extend(remaining[root_count:])
    pool = np.asarray(pool, dtype=np.int64)
    if len(pool) < 2 * config.num_clients:
        raise ValueError('insufficient samples for disjoint client train/validation splits')
    partitions = partition_indices(classes[pool], config.num_clients,
                                   config.dirichlet_alpha, config.seed)
    # Repair empty/tiny shards deterministically without dropping examples.
    buckets = [list(x) for x in partitions]
    for bucket in buckets:
        while len(bucket) < 2:
            donor = max(buckets, key=len)
            bucket.append(donor.pop())
    train, validation = [], []
    for client, bucket in enumerate(buckets):
        values = stream(config.seed, 'client-splits', client).permutation(pool[bucket])
        count = min(len(values) - 1, max(1, int(len(values) * config.validation_fraction)))
        validation.append(values[:count])
        train.append(values[count:])
    return DataSplit(train, validation, np.asarray(reference, dtype=np.int64),
                     np.asarray(test, dtype=np.int64))


def matched_test_shards(train_labels, test_labels, split, seed):
    """Allocate held-out examples according to the observed client class masses."""
    if hasattr(split, 'natural_test_clients'):
        groups = split.natural_test_clients
        if not np.array_equal(np.sort(np.concatenate(groups)), np.arange(len(test_labels))):
            raise ValueError('natural test shards overlap or omit examples')
        return groups
    train_classes=train_labels if train_labels.ndim==1 else train_labels.argmax(1)
    test_classes=test_labels if test_labels.ndim==1 else test_labels.argmax(1)
    buckets=[[] for _ in split.train]
    for label in np.unique(test_classes):
        masses=np.array([np.count_nonzero(train_classes[np.concatenate((t,v))]==label)
                         for t,v in zip(split.train,split.validation)],dtype=float)
        if masses.sum()==0: masses[:]=1
        rng=stream(seed,'fairness-test',int(label))
        indices=rng.permutation(np.flatnonzero(test_classes==label))
        counts=rng.multinomial(len(indices),masses/masses.sum())
        offset=0
        for bucket,count in zip(buckets,counts):
            bucket.extend(indices[offset:offset+count].tolist()); offset+=count
    return [np.array(x,dtype=np.int64) for x in buckets]


def make_splits(labels, config, external_test=False):
    if not config.natural_splits_path:
        return make_standard_splits(labels, config, external_test)
    import json
    from pathlib import Path
    if not external_test:
        raise ValueError('natural splits require official external test arrays')
    data = json.loads(Path(config.natural_splits_path).read_text())
    if data['seed'] != config.seed or len(data['train']) != config.num_clients:
        raise ValueError('natural split seed/client mismatch')
    groups = [np.asarray(g, dtype=np.int64) for g in data['train']+data['validation']+[data['reference']]]
    all_indices = np.concatenate(groups)
    if any(not len(g) for g in groups) or len(all_indices) != len(labels) or not np.array_equal(np.sort(all_indices),np.arange(len(labels))):
        raise ValueError('natural splits overlap, omit data, or contain invalid indices')
    if set(data['writers']) & set(data['reference_writers']):
        raise ValueError('reference writer overlaps enrolled writers')
    split = DataSplit(groups[:config.num_clients],groups[config.num_clients:2*config.num_clients],groups[-1],np.array([],dtype=np.int64))
    split.natural_test_clients = [np.asarray(g,dtype=np.int64) for g in data['test_clients']]
    return split
