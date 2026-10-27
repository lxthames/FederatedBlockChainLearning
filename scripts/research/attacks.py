"""Update-level poisoning attacks with explicit, reproducible contexts."""

from dataclasses import dataclass
from typing import Mapping

import torch

from .contracts import ClientUpdate


@dataclass(frozen=True)
class AttackContext:
    round_index: int
    seed: int = 0
    participant_count: int = 1
    malicious_count: int = 1
    peer_updates: tuple[ClientUpdate, ...] = ()


def _replace_state(update: ClientUpdate, state: Mapping[str, object], metadata=None) -> ClientUpdate:
    return ClientUpdate(
        client_id=update.client_id,
        state=state,
        sample_count=update.sample_count,
        is_malicious=update.is_malicious,
        metadata=update.metadata if metadata is None else metadata,
    )


class SignFlipAttack:
    def apply(self, update: ClientUpdate, context: AttackContext) -> ClientUpdate:
        state = {
            name: -value if isinstance(value, torch.Tensor) and value.is_floating_point() else value
            for name, value in update.state.items()
        }
        return _replace_state(update, state)


class ModelReplacementAttack:
    def apply(self, update: ClientUpdate, context: AttackContext) -> ClientUpdate:
        if context.malicious_count < 1:
            raise ValueError("malicious_count must be positive")
        scale = context.participant_count / context.malicious_count
        state = {
            name: value * scale
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value
            for name, value in update.state.items()
        }
        return _replace_state(update, state)


class ALIEAttack:
    def __init__(self, z_max: float = 1.0):
        self.z_max = z_max

    def apply(self, update: ClientUpdate, context: AttackContext) -> ClientUpdate:
        if len(context.peer_updates) < 2:
            # ALIE is defined from peer statistics.  A layered trust policy can
            # legitimately leave fewer than two honest updates in a round; in
            # that case the attacker cannot construct the perturbation.  Keep
            # the submitted local update and record the operational shortfall
            # so the run remains auditable instead of aborting mid-trajectory.
            return _replace_state(update, update.state, {
                **update.metadata,
                'attack_applied': False,
                'attack_reason': 'insufficient_reference_peers',
                'reference_peer_count': len(context.peer_updates),
            })
        state: dict[str, object] = {}
        for name, original in update.state.items():
            if not isinstance(original, torch.Tensor) or not original.is_floating_point():
                state[name] = original
                continue
            peers = torch.stack([peer.state[name] for peer in context.peer_updates])
            state[name] = peers.mean(dim=0) + self.z_max * peers.std(dim=0, unbiased=False)
        return _replace_state(update, state, {
            **update.metadata,
            'attack_applied': True,
            'reference_peer_count': len(context.peer_updates),
        })


class LowAndSlowAttack:
    def __init__(self, max_scale: float = 1.5, ramp_rounds: int = 10):
        if max_scale < 1:
            raise ValueError("max_scale must be at least 1")
        if ramp_rounds < 1:
            raise ValueError("ramp_rounds must be positive")
        self.max_scale = max_scale
        self.ramp_rounds = ramp_rounds

    def apply(self, update: ClientUpdate, context: AttackContext) -> ClientUpdate:
        progress = min(max(context.round_index, 0) / self.ramp_rounds, 1.0)
        # Bounded interpolation toward a sign-reversed update; at full ramp it
        # actively opposes descent, rather than amplifying an honest update.
        scale = 1.0 - (1.0 + self.max_scale) * progress
        state = {
            name: value * scale
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value
            for name, value in update.state.items()
        }
        return _replace_state(update, state)


class GaussianAttack:
    def __init__(self, stddev: float = 0.1):
        if stddev <= 0:
            raise ValueError("stddev must be positive")
        self.stddev = stddev

    def apply(self, update: ClientUpdate, context: AttackContext) -> ClientUpdate:
        generator = torch.Generator().manual_seed(
            context.seed * 1_000_003 + context.round_index * 1_009 + update.client_id
        )
        state: dict[str, object] = {}
        for name, value in update.state.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                noise = torch.randn(
                    value.shape, generator=generator, dtype=value.dtype, device=value.device
                )
                state[name] = value + self.stddev * noise
            else:
                state[name] = value
        return _replace_state(update, state)
