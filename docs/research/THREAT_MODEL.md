# Threat Model

## Protected workflow

The study evaluates cross-silo federated learning in which enrolled clients train locally, a rotating committee validates candidate updates on private local validation partitions, and an append-only simulated ledger records decisions and recovery actions.

The learning task may be classification or vector-search routing. Local data and local validation examples are never pooled into a global validation dataset. This is a data-locality property, not a formal privacy guarantee.

## Trusted components

- The experiment driver, cryptographic hash function, configuration manifest, and event log are trusted.
- Identity uniqueness is supplied by an external enrollment oracle. The framework does not implement Proof-of-Personhood.
- The simulator knows attacker labels only to inject attacks and score detector errors. Committee selection and honest validation do not receive those labels.
- FLTrust and BOBA receive trusted server reference data because those baselines require it. Their trust assumptions must be reported separately.

## Adversaries

- Unique but compromised clients may submit poisoned updates after legitimate enrollment.
- Duplicate identities may pass or fail the external oracle according to configured false-accept and false-reject rates.
- Poisoning clients and corrupted validators are selected independently.
- Corrupted validators may collude by approving malicious candidates and rejecting honest candidates.
- Attacks include sign flip, Gaussian noise, ALIE, model replacement, and a ten-round low-and-slow scaling attack.

The study varies poisoning-client fraction, validator-capture fraction, identity-oracle errors, client population, participation, and data heterogeneity. It does not claim protection beyond the tested ranges.

## Security outcomes

The benchmark reports malicious updates submitted and accepted, downstream targeted attack success where a targeted attack is defined, detector TPR/FPR, detection delay, recovery count, recovery regret, and wrongful slashing. An accepted malicious update is not itself counted as downstream attack success.

## Ledger and recovery semantics

Model recovery never deletes or rewrites confirmed entries. The simulator restores an application-layer model checkpoint and appends a recovery record containing the rejected version, restored version, trigger, and affirmative signers. Only affirmative signers are eligible for penalties.

## Out of scope

The study does not provide formal privacy, Sybil-resistant enrollment, production blockchain finality, smart-contract gas measurements, secure aggregation, differential privacy, or a convergence proof.
