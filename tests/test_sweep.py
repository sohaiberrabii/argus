import sys
import gzip
import hashlib
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_sweep import analyze, paired_difference, quantile


class RandomizedRoundTests(unittest.TestCase):
    def test_pairs_by_round_rather_than_capture_order(self):
        # A large between-round drift should cancel when conditions are paired.
        left = [{"round": 0, "ms": 100}, {"round": 1, "ms": 1000}, {"round": 2, "ms": 10}]
        right = [{"round": 2, "ms": 12}, {"round": 0, "ms": 102}, {"round": 1, "ms": 1002}]
        result = paired_difference(left, right, "ms", bootstrap=True)
        self.assertEqual(result["median"], 2)
        self.assertEqual(result["bootstrap_95_percent_interval"], [2, 2])

    def test_missing_round_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unmatched rounds"):
            paired_difference([{"round": 1, "ms": 2}], [{"round": 2, "ms": 3}], "ms")

    def test_quantiles_interpolate(self):
        self.assertEqual(quantile([0, 10, 20], .25), 5)


class FreshEvidenceTests(unittest.TestCase):
    def test_complete_measured_sweep_and_saved_runner(self):
        directory = Path(__file__).resolve().parents[1] / "results" / "llama-1b-v5e-20261018"
        report = analyze(directory)
        self.assertEqual(len(report["device_samples"]), 65)
        self.assertEqual(len(report["host_samples"]), 390)
        self.assertEqual(len(report["groups"]), 13)
        self.assertTrue(report["output_checks_passed"])
        benchmark = json.loads((directory / "benchmark.json").read_text())
        executed = gzip.decompress((directory / "benchmark_runner.py.gz").read_bytes())
        self.assertEqual(hashlib.sha256(executed).hexdigest(),
                         benchmark["runner_sha256"])
        stored = json.loads((directory / "sweep-analysis.json").read_text())
        self.assertEqual(report, stored)


if __name__ == "__main__":
    unittest.main()
