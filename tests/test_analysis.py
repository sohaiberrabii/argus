import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze", ROOT / "scripts" / "analyze.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class AccountingTests(unittest.TestCase):
    def test_gqa_and_causal_prefix(self):
        config = dict(hidden_size=4, intermediate_size=8, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=2, vocab_size=10)
        coefficients = analysis.flop_coefficients(config)
        # Independently count Q/O, K/V and three MLP matrices: two layers,
        # each with 32 + 16 + 96 weights, at two FLOPs per weight per token.
        self.assertEqual(coefficients["backbone_dense_per_token"], 576)
        step = dict(active_tokens=2, computed_tokens=[3], padded_tokens=16, padded_logits_rows=8)
        result = analysis.step_flops(step, coefficients)
        # Causal queries see 4 and 5 keys, GQA does not reduce query-head work.
        self.assertEqual(result["useful_attention_flops"], 288)
        self.assertEqual(result["useful_flops"], 1520)
        self.assertEqual(result["padded_dense_flops"], 9856)

    def test_rejects_multi_request_accounting(self):
        with self.assertRaisesRegex(ValueError, "one active request"):
            analysis.step_flops(dict(computed_tokens=[3, 3]), {})


class DeviceClockTests(unittest.TestCase):
    def events(self):
        event = dict(ph="X", pid=3, tid=2, name="jit_run_model_impl(123)", ts=0, dur=0,
                     args=dict(device_offset_ps="10000000001", device_duration_ps="2345678901"))
        return [dict(name="process_name", pid=3, args=dict(name="/device:TPU:0")),
                dict(name="thread_name", pid=3, tid=2, args=dict(name="XLA Modules")),
                dict(name="thread_name", pid=3, tid=4, args=dict(name="XLA Ops")),
                {**event, "pid": 701}, {**event, "tid": 4}, event]

    def test_host_and_nested_events_are_excluded(self):
        modules = analysis.module_events(self.events())
        self.assertEqual(len(modules), 1)
        self.assertEqual(modules[0]["start_ps"], 10000000001)
        self.assertEqual(modules[0]["duration_ps"], 2345678901)

    def test_overlaps_are_rejected(self):
        events = self.events()
        events.append(copy.deepcopy(events[-1]))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            analysis.module_events(events)


class RecordedBaselineTests(unittest.TestCase):
    def test_recomputes_observations(self):
        report = analysis.analyze(ROOT / "results" / "llama-1b-v5e")
        self.assertEqual(len(report["host_samples"]), 6)
        self.assertEqual(len(report["device_samples"]), 32)
        decode = report["device_groups"]["decode"]
        self.assertEqual(decode["forward_window_ms"]["count"], 30)
        self.assertAlmostEqual(decode["forward_window_ms"]["median"], 3.368, places=3)
        self.assertAlmostEqual(report["host_groups"]["short"]["tokens_per_s"]["median"],
                               200.895, places=3)
        self.assertTrue(all(s["padded_tokens"] == 16 and s["padded_logits_rows"] == 8
                            for s in report["device_samples"] if s["phase"] == "decode"))


if __name__ == "__main__":
    unittest.main()
