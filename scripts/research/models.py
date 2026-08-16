"""Small reproducible model factories for benchmark tasks."""

import math

import torch.nn as nn


def create_model(dataset: str, input_shape: tuple[int, ...], outputs: int) -> nn.Module:
    if dataset == "fashion_mnist":
        return nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, outputs),
        )
    if dataset == "cifar10":
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, outputs),
        )
    if dataset in {"synthetic", "sift_routing"}:
        inputs = math.prod(input_shape)
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(inputs, 64),
            nn.ReLU(),
            nn.Linear(64, outputs),
        )
    raise ValueError(f"unsupported model dataset: {dataset}")
