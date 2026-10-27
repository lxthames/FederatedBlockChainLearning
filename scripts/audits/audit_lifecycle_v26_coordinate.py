"""Independent, read-only audit of the frozen V26 coordinate-median matrix."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "artifacts/lifecycle-v26-coordinate"
OUT = ROOT / "results/lifecycle-v26-coordinate"
DATA = ROOT / "data/lifecycle-v25"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit():
    issues = []
    frozen = read(JOB / "frozen-inputs.json")
    manifest = read(JOB / "manifest.json")
    analysis = read(JOB / "analysis.json")
    status = read(JOB / "status.json")
    if manifest != frozen["manifest"]:
        issues.append("manifest differs from frozen inputs")
    if len(manifest) != 200 or len({r["id"] for r in manifest}) != 200:
        issues.append("manifest does not contain 200 unique runs")
    for name, expected in frozen["source"].items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            issues.append(f"source seal mismatch: {name}")
        snapshot = JOB / "source-snapshot" / name
        if not snapshot.is_file() or sha(snapshot) != expected:
            issues.append(f"source snapshot mismatch: {name}")
    for name, expected in frozen["data"].items():
        path = DATA / name
        if not path.is_file() or sha(path) != expected:
            issues.append(f"data seal mismatch: {name}")
    for name, path in (
        ("data_provenance_sha256", DATA / "provenance.json"),
        ("protocol_sha256", ROOT / "docs/research/LIFECYCLE_V25.md"),
        ("launcher_sha256", ROOT / "scripts/run_lifecycle_v25.ps1"),
        ("tests_sha256", ROOT / "tests/test_lifecycle_v25.py"),
    ):
        if sha(path) != frozen[name]:
            issues.append(f"frozen dependency mismatch: {name}")
    records = []
    total_integrity = 0
    total_repairs = 0
    total_offline = 0
    for row in manifest:
        folder = OUT / row["id"]
        if not folder.is_dir():
            issues.append(f"missing output: {row['id']}")
            continue
        completion = folder / "completion.json"
        if not completion.is_file():
            issues.append(f"missing completion seal: {row['id']}")
            continue
        sealed = read(completion)
        actual = {str(p.relative_to(folder)).replace("/", "\\"): sha(p)
                  for p in folder.rglob("*") if p.is_file() and p != completion}
        if sealed != actual:
            issues.append(f"output seal mismatch: {row['id']}")
        summary = read(folder / "lifecycle-summary.json")
        for key in ("id", "dataset", "seed", "condition", "trigger", "arm"):
            if summary[key] != row[key]:
                issues.append(f"summary/manifest mismatch: {row['id']} {key}")
        if summary["freeze_sha256"] != sha(JOB / "frozen-inputs.json"):
            issues.append(f"freeze digest mismatch: {row['id']}")
        events = [json.loads(line) for line in (folder / "events.jsonl").open(encoding="utf-8")]
        integrity = [e for e in events if e["event_type"] == "model_integrity"]
        repairs = [e for e in events if e["event_type"] == "online_repair"]
        offline = read(folder / "offline-evaluation.json")
        expected_rounds = [25, 50, 75, 100] if row["arm"] in ("periodic", "nad") else []
        if len(integrity) != 100 or not all(e["payload"]["passed"] for e in integrity):
            issues.append(f"model integrity incomplete: {row['id']}")
        if [e["payload"]["round"] for e in repairs] != expected_rounds:
            issues.append(f"repair schedule mismatch: {row['id']}")
        if [o["round"] for o in offline] != expected_rounds:
            issues.append(f"offline evaluation schedule mismatch: {row['id']}")
        if summary["repair_attempts"] != len(repairs) or summary["repair_admissions"] != sum(
            bool(e["payload"]["decision"]["passed"]) for e in repairs
        ):
            issues.append(f"repair count mismatch: {row['id']}")
        for observation, event in zip(offline, repairs):
            if observation["admitted"] != event["payload"]["decision"]["passed"]:
                issues.append(f"offline admission mismatch: {row['id']}")
            committed = observation["models"]["committed"]
            chosen = observation["models"]["candidate" if observation["admitted"] else "source"]
            if committed["model_hash"] != chosen["model_hash"]:
                issues.append(f"commit hash mismatch: {row['id']}")
        if not summary["integrity_passed"]:
            issues.append(f"integrity flag false: {row['id']}")
        total_integrity += len(integrity)
        total_repairs += len(repairs)
        total_offline += len(offline)
        records.append(summary)
    if {p.name for p in OUT.iterdir() if p.is_dir()} != {r["id"] for r in manifest}:
        issues.append("output directory set differs from manifest")
    if status["status"] != "complete" or status["completed"] != 200 or status["failed"] != 0:
        issues.append("coordinator did not complete cleanly")
    if analysis["completed"] != 200 or analysis["total"] != 200:
        issues.append("analysis coverage incomplete")
    if {r["id"] for r in analysis["records"]} != {r["id"] for r in records}:
        issues.append("analysis record set differs from outputs")
    groups = defaultdict(list)
    for r in records:
        groups[(r["dataset"], r["condition"], r["trigger"], r["arm"])].append(r)
    if len(groups) != 40 or any(len(rows) != 5 or len({r["seed"] for r in rows}) != 5 for rows in groups.values()):
        issues.append("incomplete five-seed group")
    aggregate_keys = {(g["dataset"], g["condition"], g["trigger"], g["arm"])
                      for g in analysis["complete_seed_groups"]}
    if aggregate_keys != set(groups):
        issues.append("aggregate groups differ from records")
    for group in analysis["complete_seed_groups"]:
        key = (group["dataset"], group["condition"], group["trigger"], group["arm"])
        for metric in ("final_accuracy", "mean_post_onset_asr", "mean_after_stop_asr", "endpoint_asr", "repair_admissions"):
            vals = [r[metric] for r in groups[key]]
            if all(v is not None for v in vals):
                if abs(group[metric]["mean"] - sum(vals) / len(vals)) > 1e-10:
                    issues.append(f"aggregate mean mismatch: {key} {metric}")
    if len(analysis["paired_comparisons"]) != 30:
        issues.append("paired-comparison count differs from expected 30")
    for comparison in analysis["paired_comparisons"]:
        prefix = (comparison["dataset"], comparison["condition"], comparison["trigger"])
        base = {r["seed"]: r for r in groups[prefix + ("quorum",)]}
        treatment = {r["seed"]: r for r in groups[prefix + (comparison["arm"],)]}
        for metric in ("final_accuracy", "mean_post_onset_asr", "mean_after_stop_asr", "endpoint_asr"):
            key = metric + "_difference"
            if key in comparison:
                vals = [treatment[s][metric] - base[s][metric] for s in base]
                if abs(comparison[key]["mean"] - sum(vals) / len(vals)) > 1e-10:
                    issues.append(f"paired mean mismatch: {prefix} {comparison['arm']} {metric}")
    return dict(passed=not issues, issues=issues, trajectories=len(records), integrity_checks=total_integrity,
                repair_events=total_repairs, offline_observations=total_offline,
                complete_five_seed_groups=len(groups), paired_comparisons=len(analysis["paired_comparisons"]),
                admissions_by_arm=dict(Counter({arm: sum(r["repair_admissions"] for r in records if r["arm"] == arm)
                                                for arm in ("quorum", "periodic", "nad", "coordinate_median")})))


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
