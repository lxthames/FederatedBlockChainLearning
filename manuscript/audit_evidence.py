"""Recompute manuscript provenance diagnostics without changing frozen experiments.

Writes only manuscript/data/evidence_audit.json and sections/sample_counts.tex.
This supplements the full output-seal audit; it does not rerun training or infer
the cause of runtime failures from internally consistent saved files.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SEEDS = (4103, 4211, 4327, 4441, 4549)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def event_payloads(path, kind):
    with path.open(encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["event_type"] == kind:
                yield e


def first_divergence(left, right):
    # Compare scientific payloads, excluding identifiers and wall-clock metadata.
    kinds = {"candidate_clipping", "loss_vote", "model_integrity"}
    def relevant(path):
        with path.open(encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                if e["event_type"] in kinds:
                    yield e
    for a, b in zip(relevant(left), relevant(right)):
        if (a["event_type"], a["round_index"], a["payload"]) != (b["event_type"], b["round_index"], b["payload"]):
            return {"round": a["round_index"], "event_type": a["event_type"],
                    "main": a["payload"], "companion": b["payload"]}
    return None


def main():
    studies = {}
    environments = Counter()
    issues = []
    for name, count in (("lifecycle-v25", 400), ("lifecycle-v26-coordinate", 200)):
        job = ROOT / "artifacts" / name
        a, frozen = read(job / "analysis.json"), read(job / "frozen-inputs.json")
        assert a["completed"] == count == len(a["records"])
        assert {r["id"] for r in a["records"]} == {r["id"] for r in frozen["manifest"]}
        for rel, digest in frozen["source"].items():
            assert sha(ROOT / rel) == digest == sha(job / "source-snapshot" / rel), rel
        for r in a["records"]:
            folder = ROOT / "results" / name / r["id"]
            assert (folder / "completion.json").is_file()
            assert read(folder / "lifecycle-summary.json") == r
            s = read(folder / "summary.json")
            if r["condition"] != "clean":
                h = s["targeted_attack_success_history"]
                assert len(h) == 100
                assert abs(h[-1] - r["endpoint_asr"]) < 1e-12
                assert abs(float(np.mean(h[10:])) - r["mean_post_onset_asr"]) < 1e-12
                if r["condition"] == "attack_stop":
                    assert abs(float(np.mean(h[50:])) - r["mean_after_stop_asr"]) < 1e-12
            env = next(event_payloads(folder / "events.jsonl", "environment"))["payload"]
            environments[tuple((k, str(env[k])) for k in ("python_version", "torch_version", "numpy_version", "device", "gpu"))] += 1
        studies[name] = a
    lookup = {tuple(r[k] for k in ("dataset", "condition", "trigger", "arm", "seed")): r
              for r in studies["lifecycle-v25"]["records"]}
    controls = []
    for b in studies["lifecycle-v26-coordinate"]["records"]:
        if b["arm"] not in ("quorum", "periodic"):
            continue
        a = lookup[tuple(b[k] for k in ("dataset", "condition", "trigger", "arm", "seed"))]
        pa = ROOT / "results/lifecycle-v25" / a["id"]
        pb = ROOT / "results/lifecycle-v26-coordinate" / b["id"]
        ca, cb = read(pa / "config.json"), read(pb / "config.json")
        ca.pop("experiment_id"); cb.pop("experiment_id")
        assert ca == cb and read(pa / "splits.json") == read(pb / "splits.json")
        delta = None if a["endpoint_asr"] is None else b["endpoint_asr"] - a["endpoint_asr"]
        row = {k: b[k] for k in ("dataset", "condition", "trigger", "arm", "seed")}
        row.update(main_id=a["id"], companion_id=b["id"], endpoint_difference=delta,
                   accuracy_difference=b["final_accuracy"] - a["final_accuracy"],
                   admission_difference=b["repair_admissions"] - a["repair_admissions"])
        if delta:
            row["first_recorded_divergence"] = first_divergence(pa / "events.jsonl", pb / "events.jsonl")
        controls.append(row)
    samples = []
    provenance = read(ROOT / "data/lifecycle-v25/provenance.json")
    used = set(provenance["exposure_audit"]["excluded_writer_ids"])
    for seed in SEEDS:
        p = ROOT / "data/lifecycle-v25" / f"femnist-{seed}"
        s = read(p / "natural-splits.json")
        cohort = set(s["writers"]) | set(s["reference_writers"])
        assert not cohort & used
        used.update(cohort)
        y = np.load(p / "test_labels.npy", allow_pickle=False)
        samples.append(dict(seed=seed, train=sum(map(len, s["train"])),
                            validation=sum(map(len, s["validation"])), reference=len(s["reference"]),
                            test=len(y), asr_denominator=int((y != 0).sum())))
    failures = [json.loads(s) for s in (ROOT / "artifacts/lifecycle-v25/failures.jsonl").read_text().splitlines()]
    error_log = ROOT / "artifacts/lifecycle-v25/worker-20261022-190048-494.stderr.log"
    errors = [line for line in error_log.read_text().splitlines()
              if line.startswith(("ValueError:", "scripts.trust_repair_v16_fixed.integrity.IntegrityError:"))]
    assert len(failures) == len(errors) == 3
    for r, error in zip(failures, errors):
        r["error"] = error
        r["archived_attempts"] = [p.name for p in (ROOT / "results/lifecycle-v25").glob(r["id"] + ".interrupted-retry-*")]
        assert len(r["archived_attempts"]) == 1
    fashion = [r for r in controls if r["dataset"] == "fashion_mnist" and r["endpoint_difference"] is not None]
    report = dict(scope="Saved-output consistency and provenance; not independent training replication",
                  completed={k: v["completed"] for k, v in studies.items()},
                  recorded_environments=[dict(values=dict(k), trajectories=v) for k, v in environments.items()],
                  femnist_samples=samples, retries=failures, paired_controls=controls,
                  fashion_attacked_control_pairs=len(fashion),
                  fashion_different_endpoints=sum(bool(r["endpoint_difference"]) for r in fashion),
                  fashion_max_endpoint_difference_pp=100 * max(abs(r["endpoint_difference"]) for r in fashion),
                  unresolved="Cause of numerical/integrity failures and repeated-control divergence is not established by saved evidence.")
    (HERE / "data/evidence_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [r"\begin{table}[t]", r"\caption{FEMNIST examples after deduplication. Each seed also has 620 reserved reference examples, separate from the listed client training and validation examples. ASR excludes label 0.}",
             r"\label{tab:samples}", r"\centering\small\begin{tabular}{rrrrr}", r"\toprule",
             r"Seed & Train & Validation & Test & ASR denominator \\", r"\midrule"]
    lines += [f"{r['seed']} & {r['train']} & {r['validation']} & {r['test']} & {r['asr_denominator']} \\\\" for r in samples]
    lines += [r"\bottomrule\end{tabular}", r"\end{table}"]
    (HERE / "sections/sample_counts.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("completed", "fashion_attacked_control_pairs", "fashion_different_endpoints", "fashion_max_endpoint_difference_pp", "unresolved")}, indent=2))


if __name__ == "__main__":
    main()
