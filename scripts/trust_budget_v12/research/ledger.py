"""Append-only model audit ledger with application-layer recovery."""

import copy
from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping

from .contracts import RecoveryRecord, StateDict


@dataclass(frozen=True)
class LedgerEntry:
    sequence: int
    kind: str
    payload: Mapping[str, Any]


class AppendOnlyRecoveryLedger:
    def __init__(self):
        self.entries: list[LedgerEntry] = []
        self._checkpoints: dict[int, StateDict] = {}

    def checkpoint(self, version: int, state: StateDict) -> None:
        if version < 0:
            raise ValueError("version cannot be negative")
        self._checkpoints[version] = copy.deepcopy(dict(state))

    def _append(self, kind: str, payload: Mapping[str, Any]) -> LedgerEntry:
        entry = LedgerEntry(len(self.entries), kind, copy.deepcopy(dict(payload)))
        self.entries.append(entry)
        return entry

    def append_model_commit(self, version: int, metadata: Mapping[str, Any]) -> LedgerEntry:
        return self._append("model_commit", {"version": version, "metadata": dict(metadata)})

    def recover(
        self,
        rejected_version: int,
        restored_version: int,
        trigger: str,
        affirmative_signers: tuple[int, ...],
    ) -> tuple[StateDict, RecoveryRecord]:
        if restored_version not in self._checkpoints:
            raise KeyError(f"no checkpoint for version {restored_version}")
        record = RecoveryRecord(
            rejected_version,
            restored_version,
            trigger,
            tuple(sorted(affirmative_signers)),
        )
        self._append("recovery", asdict(record))
        return copy.deepcopy(self._checkpoints[restored_version]), record

    def serialized_size_bytes(self) -> int:
        encoded = json.dumps(
            [asdict(entry) for entry in self.entries], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return len(encoded)
