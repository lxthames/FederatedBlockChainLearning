# Preregistered Experiment Protocol

> Superseded for the ACM TRUST 2027 revision by `scripts.research.trust_matrix`
> and `TRUST_2027_RUNBOOK.md`. The approved v2 core has six methods, clean
> controls, 40% participation, disjoint splits, and five confirmatory seeds.
> Retain this earlier journal protocol for provenance; do not mix its manifests
> or pilot outputs into the TRUST confirmatory analysis.

## Primary research questions

1. How do non-IID data and scale change clean utility, attack impact, and detector errors?
2. When do identity errors and committee capture defeat layered validation?
3. Does application-layer recovery reduce cumulative damage, and at what wrongful-slashing cost?
4. What computation, communication, and ledger-growth overhead does each defense add?

## Frozen factors

- Seeds: 13, 29, 47, 71, and 101.
- Core datasets: Fashion-MNIST and CIFAR-10.
- Core attacks: sign flip, ALIE, model replacement, and low-and-slow poisoning.
- Defenses: FedAvg, coordinate median, trimmed mean, Multi-Krum, FLTrust, BOBA, and the corrected two-layer reference.
- Default population: 50 clients with 20% participation, 30% poisoning clients, Dirichlet alpha 0.5, 100 rounds, and one local epoch.
- Scale: 20, 50, and 80 clients.
- Heterogeneity: IID and Dirichlet alpha 1.0, 0.5, and 0.1.
- Poisoning-client fraction: 10%, 30%, and 50%.
- Identity FAR: 0%, 5%, 10%, and 20%; identity FRR: 0%, 5%, and 10%.
- Corrupted-validator fraction: 0%, 33%, and 50%.

`scripts.research.matrix.ExperimentMatrix` is the executable source of truth. Any deviation must be documented before viewing confirmatory outcomes.

## Staging

Screening uses seeds 13, 29, and 47 to find runtime failures, ineffective attack implementations, and metric defects. It is not used for confirmatory significance claims. Core and sensitivity studies use all five seeds with matched data partitions across defenses.

## Metrics and statistics

Report clean utility, final utility, area under the learning curve, attack-induced utility loss, targeted ASR when applicable, malicious-update acceptance, detector TPR/FPR, detection delay, recovery count and regret, worst-client utility, wrongful slashing, runtime, transmitted bytes, and ledger bytes.

Calculate statistics on seed-level run summaries. Report bootstrap 95% confidence intervals and paired effect sizes. Apply Holm correction to preregistered pairwise p-values. Do not treat rounds from one run as independent replicates.

## Baseline constraints

Multi-Krum cells must satisfy its Byzantine bound. FLTrust requires a trusted root update. BOBA requires class-wise trusted server vectors and enough participating clients for its class subspace. Infeasible cells are reported as assumption failures, not silently modified algorithms.
