# Claim-to-Evidence Matrix

The original source is available in `Details/GlobecomManuscript/main.tex`.
The revised framework manuscript and issue tracker are in `Details/ACM_TRUST_2027/`.
The restrictions below remain in force until the v2 evidence is complete.

| Prior claim | Status | Required evidence or revision |
|---|---|---|
| PoP prevents Sybil identities | Withdraw | Replace with an external identity-oracle assumption and FAR/FNR sensitivity results. |
| Rollback reverses poisoning | Replace | Describe application-layer checkpoint restoration plus an append-only recovery transaction. |
| ASR falls from 100% to 0% | Withdraw | Report malicious-update acceptance separately from task-level attack success over five matched seeds. |
| Private committee validation is unbiased under non-IID data | Unproven | Report detector FPR/FNR, worst-client utility, and heterogeneity sweeps at IID and Dirichlet alpha 1.0, 0.5, and 0.1. |
| Framework scales to practical FL populations | Unproven | Complete 20-, 50-, and 80-client sweeps with runtime, transmitted bytes, and ledger growth. |
| Framework outperforms prior defenses | Unproven | Reproduce FedAvg, coordinate median, trimmed mean, Multi-Krum, FLTrust, and BOBA under matched configurations and state differing trust assumptions. |
| Local validation is privacy-preserving | Narrow | State only that validation data is not centrally pooled; make no formal privacy claim. |
| Integrated framework is fundamentally novel | Withdraw | Position the contribution as a reproducible stress-test methodology and failure-boundary study. |

No withdrawn or unproven claim may appear in the abstract, conclusion, highlights, or figure captions.
