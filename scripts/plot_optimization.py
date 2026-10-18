"""Plot vocabulary-stage latency and paired full-request improvements."""

import argparse
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_optimization import analyze


def plot(discovery, confirmation, output):
    reports = [("Discovery", analyze(discovery)), ("Confirmation", analyze(confirmation))]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.5]})
    colors = {"baseline": "#7c8798", "active_head": "#117864"}
    for index, (name, report) in enumerate(reports):
        values = {}
        for offset, policy in ((-.17, "baseline"), (.17, "active_head")):
            samples = [s["vocabulary_ms"] for s in report["device_samples"]
                       if s["policy"] == policy and s["phase"] == "decode"]
            values[policy] = statistics.median(samples)
            axes[0].bar(index+offset, values[policy], width=.3, color=colors[policy],
                        label=("Baseline" if policy == "baseline" else "Active row") if index == 0 else None)
            axes[0].text(index+offset, values[policy]+.012, f"{values[policy]:.3f}", ha="center", fontsize=9)
            axes[0].text(index+offset, .08, "8 rows" if policy == "baseline" else "1 row",
                         ha="center", color="white", fontsize=8, weight="bold")
        reduction = 100*(1-values["active_head"]/values["baseline"])
        axes[0].text(index, .86, f"{reduction:.1f}% lower", ha="center", color=colors["active_head"], weight="bold")
    axes[0].set(xticks=[0, 1], xticklabels=[r[0] for r in reports], ylim=(0, .95),
                ylabel="Vocabulary projection latency (ms)", title="Profiled decode stage")
    labels = []
    position = 0
    for name, report in reports:
        for scenario in ("generation-6", "generation-65"):
            comparison = report["comparisons"][scenario+"/active_head"]
            delta = comparison["paired_request_ms_saved"]
            low, high = delta["bootstrap_95_percent_interval"]
            axes[1].errorbar(delta["median"], position,
                             xerr=[[delta["median"]-low], [high-delta["median"]]], fmt="o",
                             color=colors["active_head"], capsize=4)
            labels.append(f"{name}\n{scenario.split('-')[1]} prompt tokens")
            position += 1
    axes[1].axvline(0, color="#888888", linewidth=.8)
    axes[1].set(yticks=list(range(4)), yticklabels=labels,
                xlabel="Paired request time saved (ms), median and 95% interval",
                title="64-output-token requests (~315 ms total)")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=.2)
    axes[1].set_xlim(left=0)
    fig.suptitle("Active-row vocabulary projection · Llama 3.2 1B · one TPU v5e · BF16\n"
                 "Same model and prompts within each randomized comparison, two model processes",
                 fontsize=12)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, metadata={"Software": "Argus TPU"})
    fig.savefig(output.with_suffix(".svg"), metadata={"Creator": "Argus TPU"})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("discovery", type=Path)
    parser.add_argument("confirmation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.discovery, args.confirmation, args.out)
