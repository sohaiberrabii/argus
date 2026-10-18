"""Analyze controlled padding-policy comparisons and paired request gains."""

import argparse
import gzip
import json
import statistics
from pathlib import Path

from analyze import module_events, require, verify_manifest
from analyze_sweep import distribution, paired_difference


def analyze(directory):
    directory = Path(directory)
    verify_manifest(directory)
    experiment = json.loads((directory / "optimization.json").read_text())
    validation = json.loads((directory / "validation.json").read_text())
    require(experiment["completed"], "Experiment did not complete")
    records = experiment["records"]
    require(all(r["exact_output_match"] for r in records), "A measured output changed")
    eligible = [name for name, passed in experiment["eligible"].items() if passed]
    require(eligible[0] == "baseline", "Missing baseline")
    require(all(r["accepted"] for r in validation["records"] if r["policy"] in eligible),
            "A measured policy failed validation")
    for row in validation["records"]:
        require(row["exact_tokens_match"] == (row["output_token_ids"] == row["reference_token_ids"]),
                "Validation output flag disagrees with recorded tokens")
    references = {r["scenario"]: r["output_token_ids"] for r in records
                  if not r["profiled"] and r["policy"] == "baseline"}
    require(all(r["output_token_ids"] == references[r["scenario"]][:r["generated_tokens"]] for r in records),
            "Recorded output differs from baseline tokens")
    profiles = sorted({r["profile"] for r in records if r["profiled"]})
    device = []
    for profile in profiles:
        trials = [r for r in records if r["profiled"] and r["profile"] == profile]
        paths = list((directory / profile).rglob("*.trace.json.gz"))
        require(len(paths) == 1, f"Expected one trace in {profile}")
        with gzip.open(paths[0], "rt") as stream:
            modules = module_events(json.load(stream)["traceEvents"])
        body = [e for e in modules if e["name"].startswith("jit_run_model_impl(")]
        heads = [e for e in modules if e["name"].startswith(("jit_run_compute_logits(", "jit_active_compute_logits("))]
        expected = sum(len(r["steps"]) for r in trials)
        require(len(body) == len(heads) == expected,
                f"Profile/serving mismatch in {profile}: {len(body)} backbone, {len(heads)} vocabulary, {expected} steps")
        steps = [(r, s) for r in trials for s in r["steps"]]
        for index, ((trial, step), backbone, head) in enumerate(zip(steps, body, heads)):
            require(backbone["start_ps"]+backbone["duration_ps"] <= head["start_ps"], "Reversed module order")
            end = head["start_ps"]+head["duration_ps"]
            require(index+1 == len(body) or end <= body[index+1]["start_ps"], "Interleaved forward windows")
            require(step["policy"] == trial["policy"], "Policy changed during a request")
            active = experiment["policies"][trial["policy"]]["active_head"]
            require(head["name"].startswith("jit_active_compute_logits(") == active,
                    "Measured vocabulary module does not match selected policy")
            device.append({**step, "scenario": trial["scenario"], "round": trial["round"],
                           "backbone_ms": backbone["duration_ps"]/1e9,
                           "vocabulary_ms": head["duration_ps"]/1e9,
                           "forward_ms": (end-backbone["start_ps"])/1e9})
    host = [{"scenario": r["scenario"], "policy": r["policy"], "round": r["round"],
             "request_ms": r["elapsed_s"]*1000, "generated_tokens": r["generated_tokens"]}
            for r in records if not r["profiled"]]
    groups, comparisons = {}, {}
    for scenario in sorted({r["scenario"] for r in records}):
        groups[scenario] = {}
        baseline = [r for r in host if r["scenario"] == scenario and r["policy"] == "baseline"]
        for policy in eligible:
            hs = [r for r in host if r["scenario"] == scenario and r["policy"] == policy]
            ds = [r for r in device if r["scenario"] == scenario and r["policy"] == policy]
            require(len({r["round"] for r in hs}) == len(hs), "Duplicate host condition within a round")
            first = [r for r in ds if r["phase"] == "prefill"]
            require(len(first) == experiment["profile_rounds"], "Unexpected prefill profile count")
            entry = {"request_ms": distribution([r["request_ms"] for r in hs]),
                     "prefill": {field: distribution([r[field] for r in first])
                                 for field in ("backbone_ms", "vocabulary_ms", "forward_ms")},
                     "prefill_backbone_rows": sorted({r["padded_tokens"] for r in first}),
                     "vocabulary_compute_rows": sorted({r["vocabulary_compute_rows"] for r in ds})}
            decode = [r for r in ds if r["phase"] == "decode"]
            if decode:
                entry["decode"] = {field: distribution([r[field] for r in decode])
                                   for field in ("backbone_ms", "vocabulary_ms", "forward_ms")}
            groups[scenario][policy] = entry
            if policy != "baseline":
                b = statistics.median(r["request_ms"] for r in baseline)
                c = entry["request_ms"]["median"]
                comparisons[f"{scenario}/{policy}"] = {
                    "baseline_median_ms": b, "candidate_median_ms": c,
                    "speedup": b/c, "latency_reduction_percent": 100*(1-c/b),
                    "paired_request_ms_saved": paired_difference(hs, baseline, "request_ms", bootstrap=True)}
    return {"groups": groups, "comparisons": comparisons, "eligible": experiment["eligible"],
            "validation_passed": True, "host_samples": host, "device_samples": device,
            "recorded_utc": experiment["recorded_utc"], "method": experiment["method"]}


def markdown(report):
    lines = ["# Controlled padding optimization", "", f"Recorded: {report['recorded_utc']}", "",
             "Same loaded BF16 Llama 1B model on one v5e, identical prompts, randomized policy order.", "",
             "| Scenario / policy | Baseline request ms | Candidate request ms | Speedup | Latency reduction | Paired ms saved: median [95% interval] |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, comparison in report["comparisons"].items():
        delta = comparison["paired_request_ms_saved"]
        lo, hi = delta["bootstrap_95_percent_interval"]
        lines.append(f"| {name} | {comparison['baseline_median_ms']:.3f} | "
                     f"{comparison['candidate_median_ms']:.3f} | {comparison['speedup']:.3f}x | "
                     f"{comparison['latency_reduction_percent']:.2f}% | {delta['median']:.4f} [{lo:.4f}, {hi:.4f}] |")
    lines.extend(["", "A positive paired difference means the candidate completed faster.",
                  "Intervals describe within-run paired-round variation. Inspect device timings and correctness evidence before selecting a policy.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze(args.directory)
    print(json.dumps(result, indent=2) if args.json else markdown(result))
