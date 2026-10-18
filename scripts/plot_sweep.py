"""Plot the measured bucket shapes and timings, requires Matplotlib."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_sweep import analyze


def plot(directory, output):
    report = analyze(directory)
    groups = report["groups"]
    x = [int(n) for n in groups]
    rows = [g["padded_tokens"] for g in groups.values()]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5), layout="constrained")
    axes[0].plot(x, x, color="#117864", label="Active prompt tokens")
    axes[0].step(x, rows, where="post", color="#bc6c1a", label="Padded backbone rows", linewidth=2)
    axes[0].scatter(x, rows, color="#bc6c1a", s=18)
    axes[0].set(ylabel="Tokens / rows", title="Execution shape", ylim=(0, 140))

    for field, label, color in (("forward_ms", "Forward window", "#245a9b"),
                                ("backbone_ms", "Backbone", "#117864"),
                                ("logits_ms", "Vocabulary", "#757575")):
        med = [g[field]["median"] for g in groups.values()]
        low = [g[field]["min"] for g in groups.values()]
        high = [g[field]["max"] for g in groups.values()]
        axes[1].plot(x, med, "o-", color=color, markersize=3, label=label)
        axes[1].fill_between(x, low, high, color=color, alpha=.15)
    axes[1].set(ylabel="Milliseconds", title="Device medians and min–max", ylim=(0, None))

    med = [g["request_ms"]["median"] for g in groups.values()]
    low = [g["request_ms"]["p10"] for g in groups.values()]
    high = [g["request_ms"]["p90"] for g in groups.values()]
    axes[2].plot(x, med, "o-", color="#74539e", markersize=3, label="One-token request")
    axes[2].fill_between(x, low, high, color="#74539e", alpha=.15, label="10th–90th percentile")
    axes[2].set(ylabel="Milliseconds", title="Unprofiled request latency", ylim=(0, None))
    for ax in axes:
        for edge in (16.5, 32.5, 64.5):
            ax.axvline(edge, color="#999999", linestyle=":", linewidth=.7, zorder=0)
        ax.set(xlabel="Prompt tokens", xlim=(0, 132), xticks=[1, 16, 32, 64, 128])
        ax.grid(axis="y", alpha=.2)
        ax.legend(loc="best", fontsize=8, frameon=False)
    count = next(iter(groups.values()))
    fig.suptitle("Llama 3.2 1B · one TPU v5e · BF16 · one uncached request · one output token\n"
                 f"{count['forward_ms']['count']} device profiles and {count['request_ms']['count']} unprofiled requests per prompt length",
                 fontsize=12)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(output.with_suffix("."+suffix), dpi=180, metadata={"Creator": "Argus TPU"}
                    if suffix == "svg" else {"Software": "Argus TPU"})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(args.directory, args.out)
