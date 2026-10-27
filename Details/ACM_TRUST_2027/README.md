# ACM TRUST 2027 manuscript and evidence

This is a working draft, not a submission-ready paper. The GLOBECOM source is
preserved in the neighboring directory. Its SHA-256 at revision start was
`42c189b8d76ad1e92ffa35795083857be4486ca3f8f5e9dce13dab48df2984ab`.

## Project layout

- `main.tex`: anonymous ACM sigconf entrypoint.
- `sections/`: separate problem, assumptions, protocol, implementation,
  evaluation, results, limitations, conclusion files.
- `figures/`: editable TikZ system and EMH-DNN figures; generated scientific plots.
- `references.bib`: primary-source bibliography, with preprints identified.
- `generated/`: tables and statistics generated from preserved run artifacts.
- `REVIEWER_ISSUES.md`: traceability from prior reviews to changes and evidence.
- `evidence-status.json`: missing experiments and submission status.
- `build-report.json`: layout/build checks; rendered pages are under `build/`.

From the repository root:

```powershell
.\.venv-trust\Scripts\python.exe Details/ACM_TRUST_2027/build.py
```

The build runs pdflatex, BibTeX, and two additional pdflatex passes, then renders
every page. Review the rendered PNGs after material changes. Generated values
must be updated by `python -m scripts.research.paper_artifacts`, not hand-edited.

## Evidence and submission

The official conference guidelines specify anonymous two-column sigconf,
at most nine pages including appendices, with additional pages allowed for
references and GenAI disclosure. Abstract registration is October 24, 2026;
paper submission is October 31, 2026. The current evaluation article fits
Symposium 5, CMT Track 10; a validated defense would fit Symposium 2, CMT Track 7.
The portal cutoff timezone remains unverified; use October 23/30 Taipei as internal
deadlines. Sources: https://eigtrust.acm.org/trust2027/cfp/ and the official
Symposium 5 CFP at https://eigtrust.acm.org/wp-content/uploads/2026/07/CFP_Symposium-5.pdf

The primary audience is secure and resilient AI systems. The original manuscript
was rejected and has no active submission, according to the author.
No external registration or upload is performed by the build or experiment jobs.

Before finalization: complete the matched evidence, inspect held/failed runs,
review the new SIFT1M case-study results, review all claims and citations, complete visual QA,
check anonymity including metadata, and review the GenAI disclosure.
Do not remove the draft banner while the evidence gate is incomplete.

## New SIFT labels selected by the author

The original custom labels are excluded. The replacement SIFT1M dataset is in
`data/sift1m-verified/`; its provenance and checksums are recorded there.
The preparation completed exact top-1000 searches and independent checks before
certifying correspondence. The dataset has 1M database vectors, 100K training
queries and 10K held-out queries. It is not SIFT1B.

The dedicated study is running under `artifacts/sift1m-study/status.json`, with
results in `results/trust-2027/sift1m-case-study/`. It includes 36 development
cells and a 180-cell five-seed matrix; ineffective attacks are held explicitly.
These outputs are separate from the old pilot. The `sift_provenance_available`
field in the older core-study evidence JSON checks only the old data root; use
the new dataset's provenance and dedicated study status for this replacement.

## Literature and natural-client extension

The follow-up under `scripts/trust_extension/` adds a paper-based FedVal adapter,
verified/deduplicated 62-class FEMNIST writer clients, bounded proxy-based adaptive
model replacement, scheduled recovery controls, false recovery reports, vote
withholding, uniform participation and client exclusion/sanction metrics.
It is running under `artifacts/trust-extension-v1/status.json` with 33 development
cells and up to 275 five-seed evaluation cells. See
`docs/research/TRUST_EXTENSION_RUNBOOK.md` and `TRUST_EXTENSION_BASELINE_AUDIT.md`.

Execution update: the follow-up moved to CUDA under
`artifacts/trust-extension-v1-gpu/status.json`; the CPU coordinator is stopped
and its outputs are preserved. GPU groups rerun development/evaluation seeds
and remain separate from CPU measurements. The build selects the GPU evidence.

This extension changes the recovery attestation guard, so its results are kept
apart from the frozen original core. Final-version coverage must be reconciled
before any robustness claim spans the two implementations. FedRecover, Crab,
FLAME and FLDetector are discussed as prior work, not measured implementations.
The build generates follow-up CSV/TeX from complete seed groups and keeps pending
evidence visible. Current background runtimes are exploratory concurrent timings.
