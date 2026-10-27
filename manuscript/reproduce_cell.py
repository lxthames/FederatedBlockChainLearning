"""Run one frozen trajectory into a new, isolated directory; never overwrite evidence."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("main", "companion"), required=True)
    parser.add_argument("--index", type=int, required=True, help="Zero-based frozen manifest index")
    parser.add_argument("--output-root", type=Path, required=True, help="A new directory outside frozen inputs/results")
    parser.add_argument("--describe", action="store_true", help="Print the selected cell without importing training code or writing files")
    args = parser.parse_args()
    folder = "lifecycle-v25" if args.study == "main" else "lifecycle-v26-coordinate"
    module_name = "scripts.trust_lifecycle_v25.study" if args.study == "main" else "scripts.trust_lifecycle_v26b.study"
    frozen = json.loads((ROOT / "artifacts" / folder / "frozen-inputs.json").read_text())
    if not 0 <= args.index < len(frozen["manifest"]):
        parser.error("index outside the frozen manifest")
    output = args.output_root.resolve()
    for protected in ("artifacts", "results", "data", "scripts"):
        base = (ROOT / protected).resolve()
        if output == base or base in output.parents:
            parser.error("output-root must be outside frozen artifacts, results, data, and source")
    if output.exists():
        parser.error("output-root must not already exist")
    row = frozen["manifest"][args.index]
    print(json.dumps(dict(cell=row, output_root=str(output)), indent=2))
    if args.describe:
        return
    study = importlib.import_module(module_name)
    study.check_frozen(check_data=True)
    assert study.manifest()[args.index]["id"] == row["id"]
    # Redirect only the destination. Frozen implementations and data stay intact.
    study.OUT = output
    output.mkdir(parents=True)
    study.execute(args.index)
    print("Completed isolated replication:", output / row["id"])


if __name__ == "__main__":
    main()
