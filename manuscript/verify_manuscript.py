"""Check printed results against frozen analyses and inspect the compiled PDF."""
import json
from pathlib import Path
import re
import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(path):
    return json.loads(path.read_text())


def table_rows(path):
    for line in path.read_text().splitlines():
        if line.startswith(("FEMNIST &", "Fashion-MNIST &")):
            yield [v.strip() for v in line.rstrip("\\").split("&")]


def main():
    main_data = read(ROOT / "artifacts/lifecycle-v25/analysis.json")
    companion = read(ROOT / "artifacts/lifecycle-v26-coordinate/analysis.json")
    ds_names = {"FEMNIST": "femnist", "Fashion-MNIST": "fashion_mnist"}
    arms = {"Quorum": "quorum", "Clipping": "clipping", "Periodic": "periodic",
            "Forced periodic": "forced_periodic", "Fine-tune": "finetune",
            "Forced fine-tune": "forced_finetune", "Fine-Pruning": "fineprune",
            "Forced Fine-Pruning": "forced_fineprune", "Median": "coordinate_median"}
    def group(data, dataset, condition, trigger, arm):
        return next(g for g in data["complete_seed_groups"] if
                    (g["dataset"], g["condition"], g["trigger"], g["arm"]) == (dataset, condition, trigger, arm))
    checks = 0
    for c in table_rows(HERE / "sections/stop.tex"):
        ds, arm = ds_names[c[0]], arms[c[1]]
        stop, clean = (group(main_data, ds, cond, "patch", arm) for cond in ("attack_stop", "clean"))
        vals = [100 * x[m]["mean"] for x, m in ((clean, "final_accuracy"), (stop, "final_accuracy"),
                (stop, "endpoint_asr"), (stop, "mean_after_stop_asr"), (stop, "mean_post_onset_asr"))]
        expected = [f"{v:.2f}" for v in vals] + [f"{g['repair_admissions']['mean']:.1f}" for g in (clean, stop)]
        assert c[2:] == expected, c
        checks += 1
    for c in table_rows(HERE / "main.tex"):
        ds, cond, trigger = ds_names[c[0]], {"Attacker-stop": "attack_stop", "Continuous": "continuous"}[c[1]], c[2].lower()
        for arm, value in zip(("quorum", "periodic", "nad", "coordinate_median"), c[3:]):
            g = group(companion, ds, cond, trigger, arm)
            assert value == f"{100*g['endpoint_asr']['mean']:.2f} / {100*g['mean_post_onset_asr']['mean']:.2f}"
            checks += 1
    for c in table_rows(HERE / "sections/paired_results.tex"):
        cond, trigger = c[1].split(" / ")
        key = (ds_names[c[0]], {"Stop": "attack_stop", "Continuous": "continuous"}[cond], trigger, arms[c[2]])
        row = next(r for r in companion["paired_comparisons"] if tuple(r[k] for k in ("dataset", "condition", "trigger", "arm")) == key)
        for metric, actual in zip(("endpoint_asr", "mean_post_onset_asr"), c[3:]):
            v = row[metric + "_difference"]
            assert actual == f"{100*v['mean']:.2f} [{100*v['lower']:.2f}, {100*v['upper']:.2f}]"
            checks += 1
    for data in (main_data, companion):
        lookup = {tuple(r[k] for k in ("dataset", "condition", "trigger", "arm", "seed")): r for r in data["records"]}
        for row in data["paired_comparisons"]:
            prefix = tuple(row[k] for k in ("dataset", "condition", "trigger"))
            for metric, v in row.items():
                if not metric.endswith("_difference"):
                    continue
                m = metric.removesuffix("_difference")
                a = np.array([lookup[prefix + (row["arm"], s)][m] - lookup[prefix + ("quorum", s)][m]
                              for s in (4103, 4211, 4327, 4441, 4549)])
                ids = np.random.default_rng(2501).integers(0, 5, size=(10000, 5))
                lo, hi = np.quantile(a[ids].mean(1), [.025, .975])
                assert max(abs(a.mean() - v["mean"]), abs(lo - v["lower"]), abs(hi - v["upper"])) < 1e-10
                checks += 1
    sources = [HERE / "main.tex", *sorted((HERE / "sections").glob("*.tex")), HERE / "figures/lifecycle.tex"]
    text = "\n".join(p.read_text() for p in sources)
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    assert len(labels) == len(set(labels)), "duplicate labels"
    assert set(re.findall(r"\\ref\{([^}]+)\}", text)) <= set(labels)
    bib = (HERE / "refs.bib").read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    assert {k for cites in re.findall(r"\\cite\{([^}]+)\}", text) for k in cites.split(",")} <= keys
    log = (HERE / "main.log").read_text(errors="replace")
    for bad in ("undefined references", "Citation `", "Overfull", "Label(s) may have changed"):
        assert bad not in log, bad
    pdf = pymupdf.open(HERE / "main.pdf")
    disclosure_page = next(i + 1 for i, page in enumerate(pdf) if "GenAI Usage Disclosure" in page.get_text())
    # The source forces disclosure/references onto a new page after all body floats.
    assert "\\clearpage\n\\balance\n\\section*{GenAI Usage Disclosure}" in (HERE / "main.tex").read_text()
    assert disclosure_page - 1 <= 9, "technical content exceeds nine pages"
    plot_sizes = []
    for page in pdf:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:
                plot_sizes += [s["size"] for line in block["lines"] for s in line["spans"] if "DejaVu" in s["font"]]
        for font in page.get_fonts():
            assert pdf.extract_font(font[0])[3], "font not embedded"
    assert min(plot_sizes) >= 7, "plot labels too small"
    assert not pdf.metadata.get("author"), "author metadata not anonymous"
    result = dict(passed=True, numeric_checks=checks, total_pages=len(pdf),
                  technical_pages=disclosure_page - 1, minimum_plot_font_pt=round(min(plot_sizes), 2),
                  verification_scope="Tables, intervals, references, and PDF presentation checked against saved results; independent training replication was not performed.")
    (HERE / "data/manuscript_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
