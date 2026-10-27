# V7 development outcomes — 19 October 2026

All 13 new runs completed with zero failed or held cells; seven existing rows
were reused. Runtime was eight minutes, ending at 17:47 Taipei time. Source,
data, reused-artifact and policy fingerprints still match the frozen manifest.
Source results: `artifacts/rank-replication-v7/analysis.json` and `comparison.md`.

Means across development seeds 11, 17 and 23:

| Metric | FedAvg | Rank/trust |
|---|---:|---:|
| Clean final utility | 74.63% | 74.14% |
| Gradual final utility | 26.61% | 69.73% |
| Targeted final utility | 64.03% | 72.47% |
| Targeted mean post-onset ASR | 50.76% | 14.11% |

All prespecified development review criteria passed. Every seed had an effective
attack against its FedAvg control and matched participant/data fingerprints.
Clean paired effects were +2.64, -3.19 and -0.92 percentage points. Gradual
utility gains ranged from 25.18 to 57.58 points; targeted mean ASR reductions
ranged from 35.39 to 38.04 points. These are descriptive development results,
not confidence intervals or new confirmatory evidence.

Clean honest-update exclusion ranged from 17.6% to 22.4%. Rank/trust has recovery
disabled; these results do not demonstrate recovery benefit. Targeted ASR for
rank/trust remained 30.5% on seed 11 despite lower values on seeds 17 and 23.

The clean seed 11 median control reached 66.33% utility versus the adapted
detector's 66.43%, with matched participants/data and identical pre-onset model
hashes. Most of the gap to FedAvg is consistent with the aggregation choice.
Correction to the pre-run protocol's historical narrative: inspection of the
original detector events shows client 0 was falsely quarantined at round 22.
The earlier statement that no clean client was quarantined was incorrect;
the frozen protocol is retained unchanged and this erratum records the finding.
Zero measured honest-update exclusion must not be interpreted as zero false
quarantines. The single clean control establishes no attack-detection efficacy.

The candidate merits final-protocol review and evaluation against stronger
matched baselines across datasets. Detection, recovery, final-version EVM
agreement and untouched confirmatory seed groups remain unresolved. No further
experiments were automatically launched.
