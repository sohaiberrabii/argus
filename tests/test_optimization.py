import hashlib
import gzip
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_optimization import analyze


class ControlledExperimentTests(unittest.TestCase):
    def test_both_recorded_runs_recompute_and_match_executed_sources(self):
        for name, host_count in (("optimization-20261018", 608),
                                 ("optimization-confirm-20261018", 1216)):
            with self.subTest(run=name):
                directory = ROOT / "results" / name
                report = analyze(directory)
                stored = json.loads((directory / "analysis.json").read_text())
                self.assertEqual(report, stored)
                self.assertEqual(len(report["host_samples"]), host_count)
                self.assertEqual(len(report["device_samples"]), 230)
                experiment = json.loads((directory / "optimization.json").read_text())
                executed = gzip.decompress((directory / "optimization_runner.py.gz").read_bytes())
                self.assertEqual(hashlib.sha256(executed).hexdigest(),
                                 experiment["runner_sha256"])

    def test_rejected_bucket_candidates_have_no_performance_claim(self):
        directory = ROOT / "results" / "optimization-20261018"
        report = analyze(directory)
        self.assertEqual(report["eligible"], {"baseline": True, "buckets16": False,
                                            "active_head": True, "combined": False})
        self.assertEqual({row["policy"] for row in report["host_samples"]}, {"baseline", "active_head"})
        self.assertTrue(all(name.endswith("/active_head") for name in report["comparisons"]))


if __name__ == "__main__":
    unittest.main()
