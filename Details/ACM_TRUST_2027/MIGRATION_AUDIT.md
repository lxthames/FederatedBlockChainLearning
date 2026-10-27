# GLOBECOM to ACM TRUST migration audit

Compared LaTeX sources on 19 October 2026. This is a content audit, not final
PDF validation. Original main.tex SHA256 remains
`42c189b8d76ad1e92ffa35795083857be4486ca3f8f5e9dce13dab48df2984ab`.

| Original material | ACM TRUST destination | Disposition |
|---|---|---|
| Motivation and two-layer contribution | sections/introduction.tex | Rewritten around observable validation and accountable recovery |
| Identity governance and PoP | sections/threat_model.tex | Corrected to external enrollment assumption, duplicate attempts and collusion |
| Committee voting and trust | sections/framework.tex | Explicit proposer exclusion, minimum size, quorum, local-baseline rule and outcome-based trust |
| Nonlinear trust-weighted aggregation | sections/framework.tex | Replaced with linear sample-count/trust weighting matching current defenses.py; old quadratic/cubic rule is not current implementation |
| Stake/slashing and rollback | sections/framework.tex, implementation.tex | Replaced ledger rollback with append-only model restoration and attributed penalties |
| Adjacent-round monitor | sections/framework.tex, evaluation.tex | Trend monitor plus adjacent-round ablation; final recovery-quorum version still needs reconciliation |
| System diagram | figures/system.tex | Redrawn |
| EMH-DNN architecture | sections/implementation.tex, figures/emh.tex | Retained as supporting case study; 128-dimensional inputs, dual branches, fusion and scalar heads |
| SIFT routing task and labels | sections/evaluation.tex | Replaced unverifiable labels with exact-neighbor-derived SIFT1M labels and matched top-1 routing budget |
| Four/eight-client experiments | sections/evaluation.tex | New case study running; old numeric results excluded |
| Original result and heterogeneous comparison tables | sections/results.tex, generated/ | Replaced by artifact-generated matched comparisons; evidence incomplete |
| Discussion and deployment limitations | sections/limitations.tex, threat_model.tex | Expanded to trusted orchestration, private-validation bias, wrongful penalties and local-EVM limits |
| Related work | sections/related_work.tex, references.bib | Reorganized and expanded; retain only citations supporting revised claims |

## Details still needed before calling migration complete

- Add a compact end-to-end protocol algorithm or equivalent numbered procedure
  after selecting the final recovery rule. The original algorithm cannot be
  copied because its identity, detector and rollback semantics changed.
- Add explicit SIFT target normalization, MSE objective and routing-recall
  definition. State that all output heads can receive gradients from a query;
  one head per shard does not imply only that shard's client trains the head.
- State complete case-study training settings from its frozen manifest, including
  rounds, local epochs, optimizer, batch size and actual attacker fraction.
  Do not inherit the original three-epoch/momentum settings without code evidence.
- Reconcile the main framework description with the extension's local recovery
  attestation quorum. Currently the change is disclosed under the follow-up study,
  but the manuscript does not yet specify one unified final version.
- Check metrics/statistical prose against actual analysis outputs, including the
  planned Holm-adjusted paired tests; planned analysis must not imply execution.
- Populate complete result groups, finish claims/citation audit and render the
  final anonymous PDF. Current draft results do not close the original reviews.

Do not migrate the old 93.8% recall/zero-ASR claims, guaranteed Sybil prevention,
economic irrationality, guaranteed forgetting mitigation, billion-vector scope,
122-dimensional input, or changes to blockchain history. They lack support for
the revised implementation and data. The original source remains preserved.
