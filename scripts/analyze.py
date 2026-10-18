"""Recompute serving and device measurements using only the standard library."""

import argparse
import gzip
import hashlib
import json
import statistics
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def flop_coefficients(config):
    h, i, layers = (config[n] for n in
                    ("hidden_size", "intermediate_size", "num_hidden_layers"))
    q, kv, d, v = (config[n] for n in
                   ("num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size"))
    require(all(isinstance(x, int) and x > 0 for x in (h, i, layers, q, kv, d, v)),
            "Model dimensions must be positive integers")
    return {
        "backbone_dense_per_token": 2 * layers * (2*h*q*d + 2*h*kv*d + 3*h*i),
        "vocabulary_per_row": 2*h*v,
        "attention_per_query_key_pair": 4*layers*q*d,
    }


def step_flops(step, coefficients):
    require(len(step["computed_tokens"]) == 1, "Only one active request is supported")
    n, c = step["active_tokens"], step["computed_tokens"][0]
    require(n > 0 and c >= 0, "Invalid token or cache count")
    require(step["padded_tokens"] >= n and step["padded_logits_rows"] >= 1,
            "Padded row counts must cover useful work")
    attention = (n*c + n*(n+1)//2) * coefficients["attention_per_query_key_pair"]
    dense = n * coefficients["backbone_dense_per_token"] + coefficients["vocabulary_per_row"]
    padded = (step["padded_tokens"] * coefficients["backbone_dense_per_token"]
              + step["padded_logits_rows"] * coefficients["vocabulary_per_row"])
    return {"useful_flops": dense + attention, "padded_dense_flops": padded,
            "useful_attention_flops": attention}


def stats(values):
    require(bool(values), "An expected measurement group is empty")
    return {"count": len(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def module_events(events):
    processes = [e["pid"] for e in events if e.get("name") == "process_name"
                 and e.get("args", {}).get("name") == "/device:TPU:0"]
    require(len(processes) == 1, "Expected one /device:TPU:0 process")
    pid = processes[0]
    lanes = [e["tid"] for e in events if e.get("name") == "thread_name"
             and e.get("pid") == pid and e.get("args", {}).get("name") == "XLA Modules"]
    require(len(lanes) == 1, "Expected one XLA Modules lane")
    modules = []
    for event in events:
        if event.get("ph") != "X" or event.get("pid") != pid or event.get("tid") != lanes[0]:
            continue
        args = event["args"]
        modules.append({"name": event["name"], "start_ps": int(args["device_offset_ps"]),
                        "duration_ps": int(args["device_duration_ps"])})
    require(bool(modules), "No device module intervals")
    modules.sort(key=lambda e: e["start_ps"])
    require(all(e["duration_ps"] > 0 for e in modules), "Nonpositive device interval")
    for left, right in zip(modules, modules[1:]):
        require(left["start_ps"] + left["duration_ps"] <= right["start_ps"],
                "Overlapping device intervals require union accounting")
    return modules


def verify_manifest(directory):
    path = directory / "provenance.json"
    if path.exists():
        provenance = json.loads(path.read_text())
        for name, expected in provenance["public_file_sha256"].items():
            actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            require(actual == expected, f"Evidence hash mismatch: {name}")
        for name, archive in provenance.get("executed_source_archives", {}).items():
            actual = hashlib.sha256(gzip.decompress((directory / name).read_bytes())).hexdigest()
            require(actual == archive["uncompressed_sha256"], f"Executed source hash mismatch: {name}")


def analyze(directory):
    directory = Path(directory)
    verify_manifest(directory)
    bench = json.loads((directory / "benchmark.json").read_text())
    require(bench["completed"], "Benchmark did not complete")
    config = json.loads((directory / "model-config.json").read_text())
    coefficients = flop_coefficients(config)
    peak = bench["peak_bf16_flops_per_s"]
    require(peak > 0, "Peak compute must be positive")
    profiles, samples, host = [], [], []
    for trial in bench["records"]:
        require(bool(trial["steps"]), "Trial has no serving steps")
        require(trial["generated_tokens"] == len(trial["steps"]),
                "Expected one generation token per recorded step")
        if not trial["profiled"]:
            require(trial["elapsed_s"] > 0, "Request duration must be positive")
            flops = [step_flops(s, coefficients) for s in trial["steps"]]
            useful = sum(f["useful_flops"] for f in flops)
            host.append({"trial": trial["trial"], "scenario": trial["scenario"],
                         "elapsed_s": trial["elapsed_s"],
                         "tokens_per_s": trial["generated_tokens"]/trial["elapsed_s"],
                         "useful_flops": useful})
            continue
        paths = list((directory / trial["trial"]).rglob("*.trace.json.gz"))
        require(len(paths) == 1, f"Expected one trace for {trial['trial']}")
        with gzip.open(paths[0], "rt") as stream:
            modules = module_events(json.load(stream)["traceEvents"])
        backbone = [e for e in modules if e["name"].startswith("jit_run_model_impl(")]
        logits = [e for e in modules if e["name"].startswith("jit_run_compute_logits(")]
        require(len(backbone) == len(logits) == len(trial["steps"]),
                "Serving step count does not match backbone and vocabulary events")
        for index, (step, body, head) in enumerate(zip(trial["steps"], backbone, logits)):
            require(body["start_ps"] + body["duration_ps"] <= head["start_ps"],
                    "Vocabulary event precedes its backbone completion")
            end = head["start_ps"] + head["duration_ps"]
            if index + 1 < len(backbone):
                require(end <= backbone[index + 1]["start_ps"],
                        "Forward windows interleave across serving steps")
            window_ps = end - body["start_ps"]
            flops = step_flops(step, coefficients)
            samples.append({**step, **flops,
                            "backbone_duration_ms": body["duration_ps"]/1e9,
                            "logits_duration_ms": head["duration_ps"]/1e9,
                            "forward_window_ms": window_ps/1e9,
                            "useful_peak_percent": 100*flops["useful_flops"]/(window_ps/1e12)/peak,
                            "padded_dense_peak_percent": 100*flops["padded_dense_flops"]/(window_ps/1e12)/peak})
        window_ps = modules[-1]["start_ps"] + modules[-1]["duration_ps"] - modules[0]["start_ps"]
        program_ps = sum(e["duration_ps"] for e in modules)
        profiles.append({"trial": trial["trial"], "window_ms": window_ps/1e9,
                         "all_xla_module_ms": program_ps/1e9,
                         "xla_program_time_percent": 100*program_ps/window_ps})
    groups = {}
    for name, tokens in (("decode", None), ("prefill-short", 6), ("prefill-128", 128)):
        selected = [s for s in samples if
                    (s["phase"] == "decode" if tokens is None else
                     s["phase"] == "prefill" and s["active_tokens"] == tokens)]
        groups[name] = {key: stats([s[key] for s in selected]) for key in
                        ("backbone_duration_ms", "logits_duration_ms", "forward_window_ms",
                         "useful_peak_percent", "padded_dense_peak_percent")}
    host_groups = {
        scenario: {key: stats([row[key] for row in host if row["scenario"] == scenario])
                   for key in ("elapsed_s", "tokens_per_s")}
        for scenario in ("short", "prefill128")
    }
    return {"peak_bf16_flops_per_s": peak, "flop_coefficients": coefficients,
            "device_groups": groups, "host_groups": host_groups,
            "profile_windows": profiles, "device_samples": samples, "host_samples": host}


def print_report(report):
    print("Profiled device forward windows (separate from request timing):")
    print("phase          samples   backbone ms   vocabulary ms   forward ms")
    for name, group in report["device_groups"].items():
        print(f"{name:14} {group['forward_window_ms']['count']:7d}"
              f" {group['backbone_duration_ms']['median']:13.3f}"
              f" {group['logits_duration_ms']['median']:15.3f}"
              f" {group['forward_window_ms']['median']:12.3f}")
    print("\nUnprofiled request measurements (64 generated tokens in this baseline):")
    for name, group in report["host_groups"].items():
        print(f"{name:14} {group['tokens_per_s']['median']:.3f} tokens/s,"
              f" {group['elapsed_s']['median']*1000:.3f} ms median request,"
              f" n={group['elapsed_s']['count']}")
    print("\nFLOP estimates relative to peak are available with --json.")
    print("They describe estimated model work, not hardware busy time.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--json", action="store_true", help="Print the full computed report")
    arguments = parser.parse_args()
    result = analyze(arguments.directory)
    if arguments.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)
