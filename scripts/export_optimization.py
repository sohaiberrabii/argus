"""Export the optimization evidence without unrelated profiler or host data."""

import argparse
import gzip
import json
import shutil
from pathlib import Path

from analyze import module_events, require
from export_results import digest, write_json


def export(source, destination, environment):
    source, destination = Path(source), Path(destination)
    experiment = json.loads((source / "optimization.json").read_text())
    require(experiment["completed"], "Incomplete experiment")
    require(digest(source / "optimization_runner.py") == experiment["runner_sha256"],
            "Saved runner does not match the measured source")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("optimization.json", "validation.json", "model-config.json", "model-class.txt",
                 "checkpoint-provenance.json"):
        shutil.copyfile(source / name, destination / name)
    executed = (source / "optimization_runner.py").read_bytes()
    (destination / "optimization_runner.py.gz").write_bytes(gzip.compress(executed, mtime=0))
    shutil.copyfile(environment, destination / "environment.freeze.txt")
    profiles = sorted({r["profile"] for r in experiment["records"] if r["profiled"]})
    projected = []
    for profile in profiles:
        paths = list((source / profile).rglob("*.trace.json.gz"))
        require(len(paths) == 1, f"Expected one trace for {profile}")
        with gzip.open(paths[0], "rt") as stream:
            modules = module_events(json.load(stream)["traceEvents"])
        events = [
            {"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "/device:TPU:0"}},
            {"name": "thread_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "XLA Modules"}},
        ]
        for event in modules:
            events.append({"name": event["name"], "ph": "X", "pid": 1, "tid": 1,
                           "ts": event["start_ps"]/1e6, "dur": event["duration_ps"]/1e6,
                           "args": {"device_offset_ps": str(event["start_ps"]),
                                    "device_duration_ps": str(event["duration_ps"])}})
        target = destination / profile / "device.trace.json.gz"
        target.parent.mkdir()
        target.write_bytes(gzip.compress(json.dumps({"traceEvents": events}, separators=(",", ":")).encode(), mtime=0))
        projected.append({"profile": profile, "source_export_sha256": digest(paths[0]),
                          "public_trace": str(target.relative_to(destination)),
                          "module_event_count": len(modules)})
    checkpoint = json.loads((destination / "checkpoint-provenance.json").read_text())
    provenance = {"schema_version": 1, "recorded_utc": experiment["recorded_utc"],
                  "model_id": checkpoint["repository"], "checkpoint_revision": checkpoint["revision"],
                  "runner_status": "Included source completed the controlled optimization experiment on one TPU v5e.",
                  "executed_source_archives": {"optimization_runner.py.gz": {
                      "source_filename": "optimization_runner.py", "uncompressed_sha256": digest(source / "optimization_runner.py")}},
                  "trace_projection": "TPU:0 XLA Modules and exact device timestamps, process/lane identifiers normalized, unrelated events omitted.",
                  "traces": projected,
                  "public_file_sha256": {str(p.relative_to(destination)): digest(p)
                                         for p in sorted(destination.rglob("*")) if p.is_file()}}
    write_json(destination / "provenance.json", provenance)
    print(f"Exported {len(provenance['public_file_sha256'])} evidence files and provenance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--environment", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.destination, args.environment)
