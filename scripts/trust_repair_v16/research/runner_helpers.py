"""Deterministic array-based federated benchmark runner."""

import copy
from dataclasses import asdict
import math
import platform
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .attacks import (
    ALIEAttack,
    AttackContext,
    GaussianAttack,
    LowAndSlowAttack,
    ModelReplacementAttack,
    SignFlipAttack,
)
from .accountability import AccountabilityManager
from .config import ExperimentConfig
from .committee import CommitteePolicy
from .contracts import ClientUpdate, VoteDecision
from .datasets import partition_indices
from .defenses import (
    CoordinateMedianDefense,
    AggregationContext,
    BOBADefense,
    FedAvgDefense,
    FLTrustDefense,
    MultiKrumDefense,
    TrimmedMeanDefense,
    TwoLayerDefense,
    flatten_update,
)
from .events import EventLogger
from .identity import IdentityOracle
from .ledger import AppendOnlyRecoveryLedger
from .detector import TrendDetector
from .validation import LocalValidator, VoteAggregator


ModelFactory = Callable[[], nn.Module]


class _NoAttack:
    def apply(self, update: ClientUpdate, context: AttackContext) -> ClientUpdate:
        return update


def _attack(name: str):
    attacks = {
        "none": _NoAttack,
        "sign_flip": SignFlipAttack,
        "gaussian": GaussianAttack,
        "model_replacement": ModelReplacementAttack,
        "alie": ALIEAttack,
        "low_and_slow": LowAndSlowAttack,
    }
    try:
        return attacks[name]()
    except KeyError as error:
        raise ValueError(f"unknown attack: {name}") from error


def _defense(
    name: str,
    malicious_count: int,
    trusted_update: dict[str, torch.Tensor] | None = None,
    server_vectors: torch.Tensor | None = None,
):
    if name == "fedavg":
        return FedAvgDefense()
    if name == "coordinate_median":
        return CoordinateMedianDefense()
    if name == "trimmed_mean":
        return TrimmedMeanDefense()
    if name == "multi_krum":
        return MultiKrumDefense(byzantine_count=malicious_count)
    if name == "two_layer":
        return TwoLayerDefense()
    if name == "fltrust":
        if trusted_update is None:
            raise ValueError("FLTrust requires a trusted root update")
        return FLTrustDefense(trusted_update)
    if name == "boba":
        if server_vectors is None:
            raise ValueError("BOBA requires class-wise trusted server vectors")
        return BOBADefense(malicious_count, server_vectors)
    raise ValueError(f"unknown or context-dependent defense: {name}")


def _state_delta(local: nn.Module, global_model: nn.Module) -> dict[str, torch.Tensor]:
    global_state = global_model.state_dict()
    return {
        name: value.detach().cpu() - global_state[name].detach().cpu()
        if value.is_floating_point()
        else value.detach().cpu().clone()
        for name, value in local.state_dict().items()
    }


def _apply_delta(model: nn.Module, delta: dict[str, torch.Tensor]) -> None:
    from ..integrity import checked_apply_delta
    checked_apply_delta(model, delta)


def _update_bytes(update: ClientUpdate) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in update.state.values()
        if isinstance(value, torch.Tensor)
    )


