"""Read-only diagnosis of the sealed V13 targeted-attack traces.

This audit does not retrain, replay updates, tune thresholds, or modify V13.
It summarizes the evidence available in the event logs. V13 did not archive
full update tensors, so the report cannot perform a counterfactual trigger-
removed evaluation.

Run from the repository root with::

    python -m scripts.audits.diagnose_security_v13
"""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.trust_security_v13 import study


ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "artifacts/security-v13"
OUT = ROOT / "artifacts/security-v13-diagnosis"
RESULTS = ROOT / "results/security-v13"
DATASETS = ("fashion_mnist", "femnist")
SEEDS = (11, 17, 23)
ARMS = ("fedavg", "clip_only", "fedval", "scaled_renormalized")
REQUIRED = ("config.json", "summary.json", "splits.json", "events.jsonl",
            "chain-trace.json", "diagnostic-inputs.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def mean(values):
    values = [float(x) for x in values if x is not None]
    return float(np.mean(values)) if values else None


def verify_cell(folder):
    seal = read(folder / "completion.json")
    expected = {name: digest(folder / name) for name in REQUIRED}
    if seal != expected:
        raise RuntimeError(f"completion seal mismatch: {folder.name}")
    summary = read(folder / "summary.json")
    if summary.get("status") != "completed":
        raise RuntimeError(f"cell is not completed: {folder.name}")
    return read(folder / "config.json"), summary


def event_rows(folder):
    votes = defaultdict(list)
    diagnostics = {}
    completed = {}
    clipping = {}
    for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        kind = event["event_type"]
        round_index = int(event.get("round_index", 0))
        payload = event["payload"]
        if kind == "loss_vote":
            votes[(round_index, int(payload["candidate"]))].append(payload)
        elif kind == "local_update_diagnostic":
            diagnostics[(round_index, int(payload["client_id"]))] = payload
        elif kind == "round_completed":
            completed[round_index] = payload
        elif kind == "candidate_clipping":
            clipping[round_index] = payload
    return votes, diagnostics, completed, clipping


def average_vote(votes, key, fallback=None):
    values = [vote.get(key, fallback) for vote in votes]
    values = [value for value in values if value is not None]
    return mean(values)


def analyze_candidate(folder, config, summary):
    votes, diagnostics, completed, clipping = event_rows(folder)
    onset = int(config["attack_start_round"])
    malicious = {int(client) for client in summary.get("malicious_client_ids", [])}
    rows = []
    for (round_index, client_id), client_votes in sorted(votes.items()):
        if round_index < onset:
            continue
        if round_index not in completed:
            raise RuntimeError(f"missing round completion: {folder.name} r{round_index}")
        diagnostic = diagnostics.get((round_index, client_id), {})
        loss = average_vote(client_votes, "mean_loss_change")
        if loss is None:
            loss = average_vote(client_votes, "local_mean_loss_change")
        standard_error = average_vote(client_votes, "standard_error")
        if loss is None or standard_error is None:
            raise RuntimeError(f"incomplete paired-loss vote: {folder.name} r{round_index} c{client_id}")
        approval = average_vote(client_votes, "approve")
        decision = completed[round_index]["committee_decisions"][str(client_id)]
        assert len(client_votes) == decision["response_count"] == 15
        assert decision["quorum"] == 10
        assert all(not v["abstain"] for v in client_votes)
        assert all(v["approve"] == (v["mean_loss_change"] <= 2 * v["standard_error"]) for v in client_votes)
        assert decision["accepted"] == (sum(v["approve"] for v in client_votes) >= 10)
        rows.append({
            "round": round_index,
            "client": client_id,
            "malicious": client_id in malicious,
            "accepted": client_id in set(completed[round_index]["accepted_client_ids"]),
            "approval_fraction": approval,
            "loss_change": loss,
            "standard_error": standard_error,
            "two_se_slack": 2.0 * standard_error - loss,
            "locally_harmful_fraction": average_vote(client_votes, "locally_harmful", 0.0),
            "probe_scale": average_vote(client_votes, "probe_scale"),
            "probe_examples": average_vote(client_votes, "n"),
            "norm": diagnostic.get("unclipped_norm"),
            "votes": len(client_votes),
            "positive_loss_approved_fraction": mean([v["approve"] and v["mean_loss_change"] > 0 for v in client_votes]),
            "nonpositive_loss_fraction": mean([v["mean_loss_change"] <= 0 for v in client_votes]),
            "unanimous": all(v["approve"] for v in client_votes),
        })
    if not rows:
        raise RuntimeError(f"no post-onset vote rows: {folder.name}")

    def group_stats(group):
        group = [row for row in rows if row["malicious"] == group]
        return {
            "rows": len(group),
            "accepted_rate": mean([row["accepted"] for row in group]),
            "approval_fraction": mean([row["approval_fraction"] for row in group]),
            "loss_change": mean([row["loss_change"] for row in group]),
            "standard_error": mean([row["standard_error"] for row in group]),
            "two_se_slack": mean([row["two_se_slack"] for row in group]),
            "locally_harmful_fraction": mean([row["locally_harmful_fraction"] for row in group]),
            "norm": mean([row["norm"] for row in group]),
            "probe_examples": mean([row["probe_examples"] for row in group]),
            "positive_loss_approved_fraction": mean([row["positive_loss_approved_fraction"] for row in group]),
            "nonpositive_loss_fraction": mean([row["nonpositive_loss_fraction"] for row in group]),
            "unanimous_fraction": mean([row["unanimous"] for row in group]),
        }

    round_rows = []
    history = summary.get("targeted_attack_success_history", [])
    for round_index in range(onset, int(config["rounds"]) + 1):
        current = [row for row in rows if row["round"] == round_index]
        if not current:
            continue
        round_rows.append({
            "round": round_index,
            "asr": history[round_index - 1] if round_index - 1 < len(history) else None,
            "malicious": group_stats_for_round(current, True),
            "honest": group_stats_for_round(current, False),
            "accepted_count": len(completed[round_index]["accepted_client_ids"]),
            "clipped_count": sum(scale < 1.0 for scale in clipping.get(round_index, {}).get("scales", [])),
            "submitted_count": len(clipping.get(round_index, {}).get("scales", [])),
        })
    active_clip = [row for row in round_rows if row["submitted_count"]]
    return {
        "experiment": folder.name,
        "dataset": config["dataset"],
        "seed": int(config["seed"]),
        "attack": config["attack"],
        "arm": folder.name.split(f"{config['dataset']}-", 1)[1].split("-model_replacement-", 1)[0],
        "rounds": round_rows,
        "post_onset": {"malicious": group_stats(True), "honest": group_stats(False)},
        "active_clipping_fraction": (sum(row["clipped_count"] for row in active_clip) /
                                     sum(row["submitted_count"] for row in active_clip)
                                     if active_clip else None),
        "active_clipping_rounds": len(active_clip),
        "endpoint_asr": summary.get("targeted_attack_success_history", [None])[-1],
        "mean_post_onset_asr": mean(history[onset - 1:]),
        "vote_fields": sorted({key for (round_index, client_id), values in votes.items()
                               if round_index >= onset for key in values[0]}),
    }


def group_stats_for_round(rows, malicious):
    group = [row for row in rows if row["malicious"] == malicious]
    if not group:
        return {"rows": 0}
    return {
        "rows": len(group),
        "accepted_rate": mean([row["accepted"] for row in group]),
        "approval_fraction": mean([row["approval_fraction"] for row in group]),
        "loss_change": mean([row["loss_change"] for row in group]),
        "standard_error": mean([row["standard_error"] for row in group]),
        "two_se_slack": mean([row["two_se_slack"] for row in group]),
        "norm": mean([row["norm"] for row in group]),
    }


def make_report(report):
    by_dataset = defaultdict(list)
    for record in report["records"]:
        by_dataset[record["dataset"]].append(record)
    lines = [
        "# V13 targeted-attack diagnosis", "",
        f"Created {report['created_utc']}. All {report['sealed_cells']} model-replacement cells were sealed and verified.",
        "This is a read-only trace diagnosis. V13 did not archive update tensors, so it cannot replay a trigger-removed counterfactual.",
        "", "## Main finding", "",
        "The paired clean-loss screen admits targeted updates through both clean-loss improvements and its uncertainty allowance. The voting implementation evaluates ordinary private validation examples; it does not evaluate triggered inputs. In FEMNIST, malicious targeted updates receive high approval and acceptance rates while endpoint ASR reaches about 100%. These traces support an objective mismatch but cannot isolate its causal contribution.",
        "", "## Post-onset candidate trace", "",
        "| Dataset | Malicious accepted | Honest accepted | Malicious clean-loss change | Honest clean-loss change | Malicious approval | Honest approval | Endpoint ASR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        records = by_dataset[dataset]
        mal = [r["post_onset"]["malicious"] for r in records]
        hon = [r["post_onset"]["honest"] for r in records]
        endpoint = mean([r["endpoint_asr"] for r in records])
        lines.append("| {} | {:.2f}% | {:.2f}% | {:+.4f} | {:+.4f} | {:.2f}% | {:.2f}% | {:.2f}% |".format(
            dataset, 100 * mean([x["accepted_rate"] for x in mal]),
            100 * mean([x["accepted_rate"] for x in hon]),
            mean([x["loss_change"] for x in mal]), mean([x["loss_change"] for x in hon]),
            100 * mean([x["approval_fraction"] for x in mal]),
            100 * mean([x["approval_fraction"] for x in hon]), 100 * endpoint))
    lines += [
        "", "## What the logs show", "",
        "1. The candidate receives no abstentions in the active targeted traces. A 10-of-15 quorum is applied to paired clean-loss votes, and the malicious proposals usually have negative mean loss change. Negative clean loss change is therefore treated as evidence for acceptance even when the update carries a trigger.",
        "2. FEMNIST targeted malicious updates are accepted at nearly the same rate as honest updates. Their clean-loss changes are negative on average, and their approval fractions remain high. This is consistent with a validation objective mismatch: clean writer-specific examples do not exercise the trigger behavior.",
        "3. Clipping is active but does not provide adequate protection in the measured policy. The local_update_diagnostic norm is recorded BEFORE attack amplification and clipping; it must not be interpreted as the submitted attack norm. The clipping scales describe the later transformed updates. No safe norm threshold can be inferred from these averages.",
        "4. FEMNIST seed 11 already reaches 100% ASR at the first attack round (11), when ordinary accuracy is only 5.29%. That early value can reflect broad target prediction during poor learning. Late ASR remains near 100% after ordinary accuracy improves, so the endpoint failure cannot be dismissed as only an early learning collapse. Fashion has a different, later trajectory. These timings do not establish a counterfactual effect of individual updates.",
        "", "## Limits and next experiment", "",
        "This diagnosis does not justify changing the quorum, standard-error multiplier, or clipping ratio on the exposed seeds. A bounded diagnostic replay should archive checkpoints and updates and compare clean loss, triggered loss against ORIGINAL labels, and trigger-to-target success before and after each update and the combined aggregate. Evaluate the same original images without the trigger to distinguish general target-class collapse from trigger-specific behavior. Trigger-aware measurements are diagnostic oracle information, not an assumed deployable defense or input to training/selection. Any redesigned rule needs a fresh clean screen and untouched-seed attack confirmation.",
        "", "## Reproduction", "",
        "Run `.venv-trust/Scripts/python.exe -W error::RuntimeWarning -m scripts.audits.diagnose_security_v13`.",
        "Machine-readable output: `artifacts/security-v13-diagnosis/diagnosis.json`.",
    ]
    lines += ["", "## Vote decomposition", "",
              "Fractions below use all post-onset malicious votes, averaged over three complete seeds. Repeated votes are dependent observations.", "",
              "| Dataset | Non-increasing clean loss | Increasing loss approved by 2-SE allowance | Unanimously approved malicious proposals |",
              "|---|---:|---:|---:|"]
    for dataset in DATASETS:
        groups = [r["post_onset"]["malicious"] for r in by_dataset[dataset]]
        lines.append("| {} | {:.2f}% | {:.2f}% | {:.2f}% |".format(
            dataset, *(100 * mean([g[k] for g in groups]) for k in
                       ("nonpositive_loss_fraction", "positive_loss_approved_fraction", "unanimous_fraction"))))
    (OUT / "diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if study.inputs() != read(JOB / "frozen-inputs.json"):
        raise RuntimeError("V13 frozen source or data inputs changed")
    records = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for arm in ARMS:
                folder = RESULTS / f"v13-{dataset}-{arm}-model_replacement-s{seed}"
                config, summary = verify_cell(folder)
                if arm == "scaled_renormalized":
                    records.append(analyze_candidate(folder, config, summary))
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sealed_cells": 24,
        "candidate_cells": len(records),
        "frozen_inputs_verified": True,
        "interpretation": "Post hoc descriptive diagnosis of sealed V13 logs; no retraining, replay, threshold tuning, or counterfactual evaluation.",
        "records": records,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnosis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    make_report(report)
    for record in records:
        mal = record["post_onset"]["malicious"]
        hon = record["post_onset"]["honest"]
        print(record["experiment"],
              f"mal_accept={100*mal['accepted_rate']:.2f}%",
              f"hon_accept={100*hon['accepted_rate']:.2f}%",
              f"mal_loss={mal['loss_change']:+.4f}",
              f"endpoint_asr={100*record['endpoint_asr']:.2f}%")
    print("Saved", OUT / "diagnosis.json")


if __name__ == "__main__":
    main()
