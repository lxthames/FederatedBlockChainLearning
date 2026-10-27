# Focused literature assessment

Primary pages checked 19 October 2026. This is a targeted relevance review, not
a systematic review or certification of baseline implementations. Existing
manuscript: eleven bibliography entries; validation-based fairness, temporal
client detection and primary attack citations need stronger coverage.

Implementation update: the draft bibliography now has eighteen entries, with
explicit FedVal, FLDetector, FLAME, primary attack and ICLR 2026 comparisons.
The paper-based FedVal adapter and follow-up experiments are implemented; see
[the audit](TRUST_EXTENSION_BASELINE_AUDIT.md). Their scientific results are
pending. The assessment below records why these additions were selected.

| Work and primary source | Verified contribution relevant here | Use in this revision |
|---|---|---|
| [FedVal, USENIX Security 2023](https://www.usenix.org/conference/usenixsecurity23/presentation/valadi) | Server-side validation scores address poisoning and group bias | Highest-priority new validation comparator; disclose trusted data and match its budget |
| [FLDetector, authors' paper, 2022](https://arxiv.org/abs/2207.09209) | Detects persistent inconsistency between predicted and received client updates using history | Distinguish client identification from aggregate-quality trend detection; potential focused comparator |
| [FLAME, USENIX Security 2022](https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen) | Uses clustering, clipping and noise for backdoor defense | Explain why untargeted robustness does not imply backdoor protection; optional targeted comparator |
| [FedRecover, IEEE S&P 2023; authors' version](https://arxiv.org/abs/2210.10936) | Uses stored models/updates to estimate updates during recovery, with exact-update corrections | Compare recovery information and work; do not equate checkpoint restoration with its recovery guarantees |
| [Crab, authors' 2024 preprint](https://arxiv.org/abs/2401.08216) | Selective history and adaptive rollback with a conditional bound relative to retraining | Closest rollback-related prior work; retain the conditional nature of certification |
| [BOBA, AISTATS 2024](https://proceedings.mlr.press/v238/bao24a.html) | Robust aggregation addressing label skewness | Already in core; verify dimensional support and trusted-reference assumptions for each dataset |
| [VerifBFL, authors' 2025 paper](https://arxiv.org/abs/2501.04319) | zk-SNARK/IVC verification of training and aggregation with differential privacy; page states acceptance at NOMS 2025 | Contrast cryptographic computation verification with authenticated committee attestations; no transferred timing claims |
| [FAUN, 2026 preprint](https://arxiv.org/abs/2605.02110) | Poison recovery using a short malicious-update history, proxy data and adversarial unlearning | Retain as recent recovery context, clearly marked preprint; audit suspect-identity/proxy assumptions before comparison |
| [Yu and Fang, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c6d77387a47c2ff1cd47515f400a88e-Abstract-Conference.html) | Cooperative detection for data-poisoned workers, with IID/non-IID attack-loss bounds | Verified recent theoretical context; its data-poisoning model is not automatically the arbitrary-update threat model |
| [A Little Is Enough, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/ec1c59141046cd1866bbbcdfb6ae31d4-Abstract.html) | Primary ALIE attack paper | Add attack citation and full parameter/knowledge audit; disclose current fixed-z omniscient variant |
| [How To Backdoor Federated Learning, AISTATS 2020](https://proceedings.mlr.press/v108/bagdasaryan20a.html) | Model replacement and defense-evasion through constrain-and-scale | Add primary attack citation; distinguish the implemented scaling attack from a fully adaptive evasion attack |

Retain Biscotti as the existing blockchain-system anchor; inspect its full
protocol alongside VerifBFL before constructing a feature table. Use columns for
trust assumptions, validation information, detection target, recovery mechanism,
enforcement and deployment evidence. Mark unexamined features as unverified,
rather than inferring their absence from an abstract. Avoid claiming that this
is the first combination without a broader documented priority search.

The next manuscript revision should add short, explicit comparisons with FedVal
and FLDetector; explain that the bounded reversed-update ramp is a controlled
gradual attack rather than a demonstrated adaptive attack; and sharpen the
recovery distinction around approval attribution and measurable consequences.
Additional paper citations are useful only when connected to those decisions.

Dataset sources checked for the strategy:

- [LEAF official repository](https://github.com/TalwalkarLab/leaf): FEMNIST and
  natural client benchmark tooling. Pin preprocessing and source revisions.
- [UNSW ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets): data types,
  label provenance, download route, academic-use terms and requested citations.
- [TRUST CFP](https://eigtrust.acm.org/trust2027/cfp/): scope, research-quality
  requirements, deadlines and page budget. These requirements do not guarantee
  acceptance and the anticipated objections in the strategy are our inferences.
