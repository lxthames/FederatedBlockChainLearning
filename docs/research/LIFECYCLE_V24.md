# V24 lifecycle continuation study

Predeclared 22 October 2026 for the ACM TRUST full article. This experiment
tests whether a repair that lowers attack success at an intervention remains
useful after later federated training. It is a development study using exposed
FEMNIST seeds 11, 17 and 23; it does not provide independent confirmation.

The 27-cell matrix crosses three conditions, three seeds and three arms:

| Condition | Attack schedule |
|---|---|
| clean | No poisoning |
| attack_stop | Model replacement from rounds 11–50, then attacker stops |
| continuous | Model replacement from rounds 11–100 |

| Arm | Definition |
|---|---|
| quorum | Existing current quorum with no online repair |
| periodic | V20 anchored repair after rounds 25, 50, 75 and 100; original private verifier controls admission |
| forced | Same repair candidate and schedule; candidate is committed regardless of the private verifier decision |

Forced admission is a diagnostic intervention. It estimates the effect of
candidate quality separately from admission selection and is not a deployable
policy. All arms use the same initialization, local training, participation,
model, data partitions and repair budget within a seed and condition.

Primary outcomes are mean committed trigger rate over rounds 11–100, clean
accuracy trajectory, final accuracy, attack-stop versus continued-poisoning
behavior, repair admission, and attack-rate rebound after repair. Endpoint rate
at round 100 is reported separately because round 100 is a scheduled repair.
For attack-stop runs, the report also includes means during the attack and after
the attacker stops.

The runner is copied into `scripts/trust_lifecycle_v24/runner.py`. Its only
training change is a frozen `attack_stop_round` condition. Existing V20–V23
source files and sealed outputs are not modified. Each cell writes events,
model hashes, repair candidates, committed states, an offline repair evaluation,
final evaluation, and a completion seal under `results/lifecycle-v24/`.

The study does not use attack labels, trigger inputs or evaluation outcomes for
admission. Repair providers and private validators remain honest and correctly
labelled, including when a client is malicious in ordinary local training. The
trigger is in the existing patch bank and the development data is exposed.
These assumptions must remain in the paper's limitations.
