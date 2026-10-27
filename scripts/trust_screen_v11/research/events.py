"""Structured JSONL experiment-event logging."""

import json
from pathlib import Path
from typing import Any, Mapping


class EventLogger:
    def __init__(self, path: str | Path, experiment_id: str, seed: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.seed = seed
        self._sequence = 0

    def log(self, event_type: str, round_index: int, payload: Mapping[str, Any]) -> None:
        event = {
            "sequence": self._sequence,
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "round_index": round_index,
            "event_type": event_type,
            "payload": dict(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        self._sequence += 1
