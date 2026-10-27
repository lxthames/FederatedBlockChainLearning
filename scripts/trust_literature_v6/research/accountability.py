"""Vote-attributed economic penalties with wrongful-slashing accounting."""

from collections.abc import Mapping

from .contracts import SlashingResult


class AccountabilityManager:
    def __init__(self, penalty: float):
        if penalty <= 0:
            raise ValueError("penalty must be positive")
        self.penalty = penalty
        self._balances: dict[int, float] = {}

    def initialize(self, balances: Mapping[int, float]) -> None:
        if any(value < 0 for value in balances.values()):
            raise ValueError("balances cannot be negative")
        self._balances = dict(balances)

    def balance(self, client_id: int) -> float:
        return self._balances.get(client_id, 0.0)

    def slash_affirmative_signers(
        self, affirmative_signers: tuple[int, ...], malicious_client_ids: set[int]
    ) -> SlashingResult:
        penalized = tuple(sorted(set(affirmative_signers)))
        for client_id in penalized:
            self._balances[client_id] = max(0.0, self.balance(client_id) - self.penalty)
        wrongful = sum(client_id not in malicious_client_ids for client_id in penalized)
        return SlashingResult(penalized, wrongful)
