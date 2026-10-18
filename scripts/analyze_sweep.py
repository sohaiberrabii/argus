"""Pair prompt lengths with standard profiler events and compare bucket edges."""

import argparse
import gzip
import json
import random
import statistics
from pathlib import Path

from analyze import flop_coefficients, module_events, require, stats, step_flops, verify_manifest


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered)-1) * probability
    low = int(position)
    high = min(low+1, len(ordered)-1)
    return ordered[low] + (ordered[high]-ordered[low]) * (position-low)


def distribution(values):
    return {**stats(values), "p10": quantile(values, .1), "p90": quantile(values, .9)}


def paired_difference(left, right, field, bootstrap=False):
    a = {r["round"]: r[field] for r in left}
    b = {r["round"]: r[field] for r in right}
    require(a.keys() == b.keys(), "Boundary pair has unmatched rounds")
    differences = [b[rep]-a[rep] for rep in sorted(a)]
    result = distribution(differences)
    if bootstrap:
        rng = random.Random(713)
        medians = [statistics.median(rng.choices(differences, k=len(differences)))
                   for _ in range(10000)]
        result["bootstrap_95_percent_interval"] = [quantile(medians, .025), quantile(medians, .975)]
    return result


def analyze(directory):
    directory = Path(directory)
    verify_manifest(directory)
    sweep = json.loads((directory / "sweep.json").read_text())
    require(sweep["completed"], "Sweep did not complete")
    coefficients = flop_coefficients(json.loads((directory / "model-config.json").read_text()))
    trials = sweep["records"]
    require(all(r["generated_tokens"] == 1 and len(r["steps"]) == 1 for r in trials),
            "Sweep must measure single-step prefill with one output token")
    require(all(r["output_matches_warmup"] for r in trials), "An output changed after warmup")
    expected = {}
    for trial in trials:
        n = trial["prompt_tokens"]
        require(trial["steps"][0]["active_tokens"] == n and
                trial["steps"][0]["computed_tokens"] == [0], "Unexpected cached or chunked prefill")
        require(trial["output_token_ids"] == expected.setdefault(n, trial["output_token_ids"]),
                "Inconsistent output tokens across repetitions")
    device = []
    for profile in sorted({r["profile"] for r in trials if r["profiled"]}):
        selected = [r for r in trials if r["profiled"] and r["profile"] == profile]
        paths = list((directory / profile).rglob("*.trace.json.gz"))
        require(len(paths) == 1, f"Expected one trace for {profile}")
        with gzip.open(paths[0], "rt") as stream:
            modules = module_events(json.load(stream)["traceEvents"])
        backbone = [e for e in modules if e["name"].startswith("jit_run_model_impl(")]
        logits = [e for e in modules if e["name"].startswith("jit_run_compute_logits(")]
        require(len(backbone) == len(logits) == len(selected), "Profile does not match the serving log")
        for index, (trial, body, head) in enumerate(zip(selected, backbone, logits)):
            require(body["start_ps"]+body["duration_ps"] <= head["start_ps"], "Reversed module order")
            end = head["start_ps"]+head["duration_ps"]
            require(index+1 == len(backbone) or end <= backbone[index+1]["start_ps"],
                    "Interleaved forward windows")
            step = trial["steps"][0]
            device.append({"prompt_tokens": trial["prompt_tokens"], "round": trial["round"],
                           "trial": trial["trial"], **step_flops(step, coefficients),
                           "padded_tokens": step["padded_tokens"],
                           "padded_logits_rows": step["padded_logits_rows"],
                           "backbone_ms": body["duration_ps"]/1e9,
                           "logits_ms": head["duration_ps"]/1e9,
                           "forward_ms": (end-body["start_ps"])/1e9,
                           "backbone_start_ps": body["start_ps"],
                           "logits_end_ps": end})
    host = [{"prompt_tokens": r["prompt_tokens"], "round": r["round"],
             "request_ms": r["elapsed_s"]*1000} for r in trials if not r["profiled"]]
    groups = {}
    for n in sweep["prompt_lengths"]:
        ds = [r for r in device if r["prompt_tokens"] == n]
        hs = [r for r in host if r["prompt_tokens"] == n]
        require(len(ds) == sweep["profile_rounds"] and len(hs) == sweep["timing_rounds"],
                "Unexpected repetition count")
        require(len({r["round"] for r in ds}) == len(ds) and
                len({r["round"] for r in hs}) == len(hs), "Duplicate condition within a round")
        paddings = {(r["padded_tokens"], r["padded_logits_rows"]) for r in ds}
        require(len(paddings) == 1, "A condition used inconsistent execution shapes")
        padded, vocabulary = next(iter(paddings))
        groups[str(n)] = {"padded_tokens": padded, "padded_logits_rows": vocabulary,
                          "backbone_useful_row_fraction": n/padded,
                          "padded_dense_gflops": ds[0]["padded_dense_flops"]/1e9,
                          **{field: distribution([r[field] for r in ds]) for field in
                             ("backbone_ms", "logits_ms", "forward_ms")},
                          "request_ms": distribution([r["request_ms"] for r in hs])}
    boundaries = {}
    for low in (16, 32, 64):
        high = low+1
        if str(low) not in groups or str(high) not in groups:
            continue
        left, right = groups[str(low)], groups[str(high)]
        delta = paired_difference([r for r in device if r["prompt_tokens"] == low],
                                  [r for r in device if r["prompt_tokens"] == high], "forward_ms")
        boundaries[f"{low}->{high}"] = {
            "padded_rows_ratio": right["padded_tokens"]/left["padded_tokens"],
            "median_forward_ratio": right["forward_ms"]["median"]/left["forward_ms"]["median"],
            "paired_forward_delta_ms": delta,
            "paired_request_delta_ms": paired_difference(
                [r for r in host if r["prompt_tokens"] == low],
                [r for r in host if r["prompt_tokens"] == high], "request_ms", bootstrap=True)}
    return {"groups": groups, "boundaries": boundaries, "device_samples": device,
            "host_samples": host, "output_checks_passed": True,
            "method": sweep["method"], "recorded_utc": sweep["recorded_utc"]}


def markdown(report):
    lines = ["# Prompt-length sweep", "", f"Recorded: {report['recorded_utc']}", "",
             "One uncached request, one generated token. Device profiles and request timings are separate.", "",
             "| Prompt tokens | Backbone rows | Vocabulary rows | Backbone ms | Vocabulary ms | Forward ms | Request ms |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n, g in report["groups"].items():
        lines.append(f"| {n} | {g['padded_tokens']} | {g['padded_logits_rows']} | "
                     f"{g['backbone_ms']['median']:.3f} | {g['logits_ms']['median']:.3f} | "
                     f"{g['forward_ms']['median']:.3f} | {g['request_ms']['median']:.3f} |")
    lines.extend(["", "Device sample count per length: " +
                  str(next(iter(report["groups"].values()))["forward_ms"]["count"]) + ".",
                  "Request sample count per length: " +
                  str(next(iter(report["groups"].values()))["request_ms"]["count"]) + ".", "",
                  "## Bucket edges", "",
                  "| Prompt change | Padded rows ratio | Forward medians ratio | Paired forward delta ms, median [min, max] |",
                  "| --- | ---: | ---: | ---: |"])
    for edge, b in report["boundaries"].items():
        d = b["paired_forward_delta_ms"]
        lines.append(f"| {edge} | {b['padded_rows_ratio']:.2f}x | "
                     f"{b['median_forward_ratio']:.3f}x | {d['median']:.4f} [{d['min']:.4f}, {d['max']:.4f}] |")
    lines.extend(["", "The row ratios describe execution shapes. They do not predict latency ratios.",
                  "Paired deltas compare conditions within the same randomized round.",
                  "Request deltas and paired bootstrap intervals are available in the JSON output.",
                  "All generated token IDs matched the corresponding warmed condition.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze(args.directory)
    print(json.dumps(result, indent=2) if args.json else markdown(result))
