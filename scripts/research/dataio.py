"""Dataset loading boundaries for classification and routing studies."""

from pathlib import Path

import numpy as np

from .config import ExperimentConfig


def load_sift_arrays(
    feature_path: str | Path,
    target_path: str | Path,
    limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.load(feature_path, mmap_mode="r")
    targets = np.load(target_path, mmap_mode="r")
    if len(features) != len(targets):
        raise ValueError('SIFT features and targets must have identical row counts')
    count = min(len(features), len(targets))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        count = min(count, limit)
    selected_features = np.asarray(features[:count], dtype=np.float32)
    if np.issubdtype(features.dtype, np.integer):
        selected_features /= np.iinfo(features.dtype).max
    selected_targets = np.asarray(targets[:count], dtype=np.float32)
    return selected_features, selected_targets


def load_experiment_arrays(
    config: ExperimentConfig, data_root: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    root = Path(data_root)
    if config.dataset == "synthetic":
        rng = np.random.default_rng(config.seed)
        labels = rng.integers(0, config.num_classes, size=config.synthetic_samples)
        centers = rng.normal(size=(config.num_classes, config.input_dim)).astype(np.float32)
        features = centers[labels] + rng.normal(
            scale=0.5, size=(config.synthetic_samples, config.input_dim)
        ).astype(np.float32)
        return features, labels.astype(np.int64)
    if config.dataset == "sift_routing":
        return load_sift_arrays(
            root / "SIFT1B_train.npy",
            root / f"gt_{config.num_clients}M_client_counts_skewed.npy",
        )
    if config.dataset in {"fashion_mnist", "cifar10"}:
        try:
            from torchvision import datasets
        except ImportError as error:
            raise RuntimeError(
                "torchvision is required for Fashion-MNIST and CIFAR-10"
            ) from error
        if config.dataset == "fashion_mnist":
            dataset = datasets.FashionMNIST(root=root, train=True, download=True)
            features = dataset.data.numpy().astype(np.float32)[:, None, :, :] / 255.0
            labels = dataset.targets.numpy().astype(np.int64)
        else:
            dataset = datasets.CIFAR10(root=root, train=True, download=True)
            features = dataset.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
            labels = np.asarray(dataset.targets, dtype=np.int64)
        return features, labels
    raise ValueError(f"unsupported dataset: {config.dataset}")


def load_experiment_data(config, data_root):
    """Load official held-out classification tests; retain API above for legacy callers."""
    features, labels = load_experiment_arrays(config, data_root)
    if config.max_train_samples is not None:
        indices = np.random.default_rng(config.seed).permutation(len(labels))[:config.max_train_samples]
        features, labels = features[indices], labels[indices]
    test = None
    if config.dataset in {'fashion_mnist','cifar10'}:
        from torchvision import datasets
        if config.dataset == 'fashion_mnist':
            dataset = datasets.FashionMNIST(root=data_root, train=False, download=True)
            tx = dataset.data.numpy().astype(np.float32)[:,None,:,:]/255.0
            ty = dataset.targets.numpy().astype(np.int64)
        else:
            dataset = datasets.CIFAR10(root=data_root, train=False, download=True)
            tx = dataset.data.astype(np.float32).transpose(0,3,1,2)/255.0
            ty = np.asarray(dataset.targets,dtype=np.int64)
        test = (tx,ty)
    elif config.dataset == 'sift_routing':
        # Test-query normalization and source correspondence require external
        # provenance before the supporting case study can be executed.
        audit_path = Path(data_root)/'sift-provenance.json'
        if not audit_path.exists():
            raise ValueError('SIFT provenance missing: supply sift-provenance.json before case-study execution')
        import json
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if not audit.get('query_target_correspondence_verified'):
            raise ValueError('SIFT query/target correspondence is not verified')
        test_x = np.load(Path(data_root)/'SIFT_test_query.npy').astype(np.float32)
        test_y = np.load(Path(data_root)/f'gt_{config.num_clients}M_client_counts_skewedtest.npy').astype(np.float32)
        scale = audit.get('test_feature_divisor')
        if not isinstance(scale,(float,int)) or scale <= 0 or len(test_x) != len(test_y):
            raise ValueError('invalid SIFT test normalization or row count')
        test = (test_x/scale,test_y)
    return features, labels, test
