"""Command-line interface for reproducible benchmark execution."""

import argparse
import json
from pathlib import Path

from .config import ExperimentConfig
from .dataio import load_experiment_arrays
from .matrix import ExperimentMatrix
from .models import create_model
from .runner import BenchmarkRunner
from .results import aggregate_summaries


def _run(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_file(args.config)
    _execute_config(config, Path(args.output), args.data_root)
    return 0


def _execute_config(
    config: ExperimentConfig, output: Path, data_root: str | Path
) -> dict[str, object]:
    features, labels = load_experiment_arrays(config, data_root)
    outputs = labels.shape[1] if labels.ndim == 2 else int(labels.max()) + 1
    factory = lambda: create_model(config.dataset, tuple(features.shape[1:]), outputs)
    summary = BenchmarkRunner(config, output).run_arrays(features, labels, factory)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _run_manifest(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    configs = [
        ExperimentConfig(**item)
        for item in json.loads(manifest_path.read_text(encoding="utf-8"))
    ]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    counts = {"completed": 0, "failed": 0, "skipped": 0, "total": len(configs)}
    failures = []
    for config in configs:
        output = output_root / config.experiment_id
        if args.resume and (output / "summary.json").exists():
            counts["skipped"] += 1
            continue
        try:
            _execute_config(config, output, args.data_root)
            counts["completed"] += 1
        except Exception as error:
            counts["failed"] += 1
            failures.append(
                {
                    "experiment_id": config.experiment_id,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
    (output_root / "batch-summary.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True), encoding="utf-8"
    )
    if failures:
        (output_root / "failures.json").write_text(
            json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8"
        )
    return 1 if failures else 0


def _export_matrix(args: argparse.Namespace) -> int:
    matrix = ExperimentMatrix()
    method = getattr(matrix, args.matrix)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps([config.to_dict() for config in method()], indent=2),
        encoding="utf-8",
    )
    return 0


def _analyze(args: argparse.Namespace) -> int:
    paths = sorted(Path(args.results_root).rglob("summary.json"))
    rows = aggregate_summaries(paths, args.require_confirmatory_seeds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--data-root", default="data")
    run.set_defaults(handler=_run)
    run_manifest = commands.add_parser("run-manifest")
    run_manifest.add_argument("--manifest", required=True)
    run_manifest.add_argument("--output-root", required=True)
    run_manifest.add_argument("--data-root", default="data")
    run_manifest.add_argument("--resume", action="store_true")
    run_manifest.set_defaults(handler=_run_manifest)
    export = commands.add_parser("export-matrix")
    export.add_argument(
        "--matrix",
        required=True,
        choices=(
            "screening",
            "core_confirmation",
            "scale_sweep",
            "heterogeneity_sweep",
            "attacker_fraction_sweep",
            "identity_committee_sweep",
            "sift_case_study",
        ),
    )
    export.add_argument("--output", required=True)
    export.set_defaults(handler=_export_matrix)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--results-root", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--require-confirmatory-seeds", action="store_true")
    analyze.set_defaults(handler=_analyze)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
