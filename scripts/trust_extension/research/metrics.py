"""Security metrics with distinct acceptance, detection, and attack outcomes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RoundSecurityMetrics:
    round_index: int
    malicious_submitted: int
    malicious_accepted: int
    targeted_attack_success: float
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0


class SecurityMetricsTracker:
    def __init__(self):
        self.rounds: list[RoundSecurityMetrics] = []

    def add_round(self, metrics: RoundSecurityMetrics) -> None:
        if metrics.malicious_accepted > metrics.malicious_submitted:
            raise ValueError("accepted malicious updates cannot exceed submitted updates")
        self.rounds.append(metrics)

    def summary(self) -> dict[str, float]:
        if not self.rounds:
            raise ValueError("no metrics have been recorded")
        submitted = sum(item.malicious_submitted for item in self.rounds)
        accepted = sum(item.malicious_accepted for item in self.rounds)
        tp = sum(item.true_positives for item in self.rounds)
        fp = sum(item.false_positives for item in self.rounds)
        fn = sum(item.false_negatives for item in self.rounds)
        tn = sum(item.true_negatives for item in self.rounds)
        return {
            "malicious_update_acceptance_rate": accepted / submitted if submitted else 0.0,
            "mean_targeted_attack_success": sum(
                item.targeted_attack_success for item in self.rounds
            )
            / len(self.rounds),
            "detection_true_positive_rate": tp / (tp + fn) if tp + fn else 0.0,
            "detection_false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        }
