"""Export serving records and a minimal projection of standard device profiles."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from analyze import module_events, require


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def export(source, destination, environment=None):
    source, destination = Path(source), Path(destination)
    benchmark = json.loads((source / "benchmark.json").read_text())
    sweep = json.loads((source / "sweep.json").read_text())
    require(benchmark["completed"] and sweep["completed"], "Incomplete experiment")
    require(digest(source / "benchmark_runner.py") == benchmark["runner_sha256"],
            "Measured runner hash does not match the saved source")
    destination.mkdir(parents=True, exist_ok=False)
    public_benchmark = {key: value for key, value in benchmark.items() if key != "devices"}
    write_json(destination / "benchmark.json", public_benchmark)
    write_json(destination / "sweep.json", sweep)
    for name in ("model-config.json", "model-class.txt", "checkpoint-provenance.json"):
        shutil.copyfile(source / name, destination / name)
    executed = (source / "benchmark_runner.py").read_bytes()
    (destination / "benchmark_runner.py.gz").write_bytes(gzip.compress(executed, mtime=0))
    if environment is not None:
        shutil.copyfile(environment, destination / "environment.freeze.txt")
    profiles = {r["trial"] for r in benchmark["records"] if r["profiled"]}
    profiles.update(r["profile"] for r in sweep["records"] if r["profiled"])
    projected = []
    for profile in sorted(profiles):
        paths = list((source / profile).rglob("*.trace.json.gz"))
        require(len(paths) == 1, f"Expected one standard trace for {profile}")
        with gzip.open(paths[0], "rt") as stream:
            events = json.load(stream)["traceEvents"]
        modules = module_events(events)
        output = [
            {"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "/device:TPU:0"}},
            {"name": "thread_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "XLA Modules"}},
        ]
        for event in modules:
            output.append({"name": event["name"], "ph": "X", "pid": 1, "tid": 1,
                           "ts": event["start_ps"]/1e6, "dur": event["duration_ps"]/1e6,
                           "args": {"device_offset_ps": str(event["start_ps"]),
                                    "device_duration_ps": str(event["duration_ps"])}})
        target = destination / profile / "device.trace.json.gz"
        target.parent.mkdir()
        target.write_bytes(gzip.compress(json.dumps({"traceEvents": output}, separators=(",", ":")).encode(), mtime=0))
        projected.append({"profile": profile, "source_export_sha256": digest(paths[0]),
                          "public_trace": str(target.relative_to(destination)),
                          "module_event_count": len(modules)})
    checkpoint = json.loads((destination / "checkpoint-provenance.json").read_text())
    provenance = {"schema_version": 1, "recorded_utc": sweep["recorded_utc"],
                  "model_id": checkpoint["repository"], "checkpoint_revision": checkpoint["revision"],
                  "runner_status": "The included runner source completed this fresh baseline and sweep on a TPU VM.",
                  "executed_source_archives": {"benchmark_runner.py.gz": {
                      "source_filename": "benchmark_runner.py", "uncompressed_sha256": digest(source / "benchmark_runner.py")}},
                  "trace_projection": "TPU:0 XLA Modules events and exact device timer fields, process and lane identifiers normalized, host and operation lanes omitted.",
                  "traces": projected,
                  "public_file_sha256": {str(path.relative_to(destination)): digest(path)
                                         for path in sorted(destination.rglob("*")) if path.is_file()}}
    write_json(destination / "provenance.json", provenance)
    print(f"Exported {len(provenance['public_file_sha256'])} evidence files and provenance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--environment", type=Path, help="Exact package freeze from the measured environment")
    args = parser.parse_args()
    export(args.source, args.destination, args.environment)
