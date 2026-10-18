"""Warm single-request Llama inference and record shapes and JAX device profiles.

Adapted from the runner used for the included historical baseline.
Targets vLLM TPU / tpu-inference 0.30.0 on one TPU v5e chip.
"""

import argparse
import datetime
import functools
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--sweep", action="store_true", help="Also measure prompt-length bucket boundaries")
    parser.add_argument("--sweep-repetitions", type=int, default=30)
    parser.add_argument("--sweep-profile-rounds", type=int, default=5)
    parser.add_argument("--prompt-lengths", type=int, nargs="+",
                        default=[1, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128])
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(MODEL_IMPL_TYPE="flax_nnx", SKIP_JAX_PRECOMPILE="1",
                      VLLM_ENABLE_V1_MULTIPROCESSING="0", JAX_PLATFORMS="tpu,cpu")
    import jax
    for package, expected in (("vllm-tpu", "0.30.0"), ("tpu-inference", "0.30.0")):
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"Expected {package}=={expected}, review runner hooks before upgrading")
    from vllm import LLM, SamplingParams
    from tpu_inference.runner.tpu_runner import TPUModelRunner

    if len(jax.devices()) != 1 or "v5" not in jax.devices()[0].device_kind.lower():
        raise RuntimeError("This runner requires exactly one TPU v5e device")
    if args.repetitions < 1 or args.tokens < 2 or args.tokens > 128:
        raise ValueError("Use positive repetitions and 2..128 generated tokens")
    if args.sweep_repetitions < 1 or args.sweep_profile_rounds < 1:
        raise ValueError("Sweep repetition counts must be positive")
    if not args.prompt_lengths or min(args.prompt_lengths) < 1 or max(args.prompt_lengths) > 128:
        raise ValueError("Prompt lengths must be in 1..128 for this configuration")
    measuring = False
    trial = None
    rows = []
    current = None
    original_prepare = TPUModelRunner._prepare_inputs

    @functools.wraps(original_prepare)
    def prepare(self, scheduler, *a, **kw):
        nonlocal current
        result = original_prepare(self, scheduler, *a, **kw)
        if measuring:
            computed = [int(x) for x in self.input_batch.num_computed_tokens_cpu[:self.input_batch.num_reqs]]
            current = {"trial": trial, "active_tokens": scheduler.total_num_scheduled_tokens,
                       "padded_tokens": int(result[0].shape[0]),
                       "computed_tokens": computed,
                       "phase": "prefill" if all(n == 0 for n in computed) else "decode",
                       "sequence": len(rows)}
            rows.append(current)
        return result

    TPUModelRunner._prepare_inputs = prepare
    original_load = TPUModelRunner.load_model

    @functools.wraps(original_load)
    def load(self, *a, **kw):
        result = original_load(self, *a, **kw)
        config = self.model_config.hf_config.to_dict()
        keys = ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size")
        (out / "model-config.json").write_text(json.dumps({key: config[key] for key in keys}, indent=2))
        (out / "model-class.txt").write_text(f"{type(self.model).__module__}.{type(self.model).__name__}\n")
        for attr in ("model_fn", "compute_logits_fn"):
            fn = getattr(self, attr)
            name = "backbone" if attr == "model_fn" else "logits"

            def wrap(fn, name):
                @functools.wraps(fn)
                def call(*a, **kw):
                    if not measuring or current is None:
                        return fn(*a, **kw)
                    if name == "logits":
                        current["padded_logits_rows"] = int(a[1].shape[0])
                    with jax.profiler.TraceAnnotation("BENCH_"+name, step=current["sequence"],
                             trial=str(trial), phase=current["phase"], active=current["active_tokens"]):
                        return fn(*a, **kw)
                return call

            setattr(self, attr, wrap(fn, name))
        return result

    TPUModelRunner.load_model = load
    llm = LLM(model=str(args.model.expanduser().resolve()), tokenizer=str(args.model.expanduser().resolve()),
              dtype="bfloat16", load_format="safetensors", tensor_parallel_size=1,
              max_model_len=256, max_num_seqs=1, max_num_batched_tokens=128,
              gpu_memory_utilization=0.5, seed=17, enable_prefix_caching=False, async_scheduling=False)
    if (out / "model-class.txt").read_text().strip() != "tpu_inference.models.jax.llama3.LlamaForCausalLM":
        raise RuntimeError("The measured model is not the expected native JAX Llama implementation")
    ids = llm.get_tokenizer().encode("Explain how a computer keeps track of a conversation. " * 32)[:128]
    if len(ids) != 128:
        raise RuntimeError("The shared prompt did not provide 128 tokens")
    prompts = [("short", "The capital of France is"), ("prefill128", {"prompt_token_ids": ids})]

    def generate(prompt, tokens):
        return llm.generate([prompt], SamplingParams(temperature=0, max_tokens=tokens, ignore_eos=True), use_tqdm=False)

    # Trace every variant and run all serving paths before either measurement.
    for _, prompt in prompts:
        generate(prompt, 8)
        generate(prompt, args.tokens)
    records = []
    for scenario, prompt in prompts:
        for rep in range(args.repetitions):
            trial = f"timing-{scenario}-{rep}"
            begin = len(rows)
            measuring = True
            start = time.perf_counter_ns()
            outputs = generate(prompt, args.tokens)
            elapsed = (time.perf_counter_ns() - start)/1e9
            measuring = False
            records.append({"trial": trial, "scenario": scenario, "repetition": rep,
                            "elapsed_s": elapsed, "prompt_tokens": len(outputs[0].prompt_token_ids),
                            "generated_tokens": len(outputs[0].outputs[0].token_ids),
                            "steps": rows[begin:], "profiled": False})
            print(f"TIMING {trial}: {elapsed:.6f}s", flush=True)
    for scenario, prompt in prompts:
        trial = f"profile-{scenario}"
        begin = len(rows)
        options = jax.profiler.ProfileOptions()
        options.host_tracer_level = 2
        options.python_tracer_level = 0
        options.device_tracer_level = 1
        jax.profiler.start_trace(str(out / trial), profiler_options=options)
        measuring = True
        outputs = generate(prompt, 16)
        measuring = False
        jax.profiler.stop_trace()
        records.append({"trial": trial, "scenario": scenario, "prompt_tokens": len(outputs[0].prompt_token_ids),
                        "generated_tokens": len(outputs[0].outputs[0].token_ids), "steps": rows[begin:], "profiled": True})
        print(f"PROFILE {trial} complete", flush=True)
    report = {"completed": True, "records": records, "peak_bf16_flops_per_s": 197e12,
              "peak_source": "https://docs.cloud.google.com/tpu/docs/v5e",
              "versions": {n: importlib.metadata.version(n) for n in ("vllm-tpu", "tpu-inference", "jax", "jaxlib", "libtpu")},
              "devices": [str(d) for d in jax.devices()],
              "hardware": {"accelerator": "Google Cloud TPU v5e", "chips": 1, "precision": "bfloat16"},
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "settings": {"max_model_len": 256, "max_num_seqs": 1, "max_num_batched_tokens": 128,
                           "tensor_parallel_size": 1, "prefix_caching": False, "async_scheduling": False,
                           "sampling_temperature": 0, "ignore_eos": True, "seed": 17}}
    (out / "benchmark.json").write_text(json.dumps(report, indent=2)+"\n")
    (out / "benchmark_runner.py").write_bytes(Path(__file__).read_bytes())
    checkpoint_record = args.model.expanduser().resolve() / "checkpoint-provenance.json"
    if checkpoint_record.exists():
        (out / "checkpoint-provenance.json").write_bytes(checkpoint_record.read_bytes())

    if args.sweep:
        lengths = sorted(set(args.prompt_lengths))
        sweep_prompts = {n: {"prompt_token_ids": ids[:n]} for n in lengths}
        expected = {}
        # Warm the prefill and sampling paths for every execution shape before
        # either measurement. One output token isolates each request's prefill.
        for n in lengths:
            for _ in range(3):
                output = generate(sweep_prompts[n], 1)
                token_ids = output[0].outputs[0].token_ids
                if n in expected and token_ids != expected[n]:
                    raise RuntimeError(f"Nondeterministic warmup for length {n}")
                expected[n] = token_ids
            print(f"SWEEP WARM length={n} output={expected[n]}", flush=True)
        sweep_records = []
        rng = random.Random(20261018)

        def sweep_call(n, rep, profiled, profile_name=None):
            nonlocal trial, measuring
            trial = f"sweep-{'profile' if profiled else 'timing'}-{rep}-{n}"
            begin = len(rows)
            measuring = True
            start = time.perf_counter_ns()
            output = generate(sweep_prompts[n], 1)
            elapsed = (time.perf_counter_ns() - start) / 1e9
            measuring = False
            token_ids = output[0].outputs[0].token_ids
            step = rows[begin:]
            if len(step) != 1 or step[0]["active_tokens"] != n or step[0]["computed_tokens"] != [0]:
                raise RuntimeError(f"Expected one uncached prefill for length {n}: {step}")
            if token_ids != expected[n] or len(token_ids) != 1:
                raise RuntimeError(f"Output changed for length {n}")
            sweep_records.append({"trial": trial, "round": rep, "prompt_tokens": n,
                                  "generated_tokens": 1, "elapsed_s": elapsed, "profiled": profiled,
                                  "profile": profile_name, "steps": step,
                                  "output_token_ids": token_ids, "output_matches_warmup": True})

        for rep in range(args.sweep_repetitions):
            order = lengths.copy()
            rng.shuffle(order)
            for n in order:
                sweep_call(n, rep, False)
            print(f"SWEEP TIMING round={rep+1}/{args.sweep_repetitions}", flush=True)
        for rep in range(args.sweep_profile_rounds):
            profile_name = f"profile-sweep-{rep}"
            order = lengths.copy()
            rng.shuffle(order)
            jax.profiler.start_trace(str(out / profile_name), profiler_options=options)
            for n in order:
                sweep_call(n, rep, True, profile_name)
            jax.profiler.stop_trace()
            print(f"SWEEP PROFILE round={rep+1}/{args.sweep_profile_rounds}", flush=True)
        sweep = {"completed": True, "records": sweep_records, "prompt_lengths": lengths,
                 "timing_rounds": args.sweep_repetitions, "profile_rounds": args.sweep_profile_rounds,
                 "order_seed": 20261018, "prompt_token_ids": ids,
                 "versions": report["versions"], "hardware": report["hardware"],
                 "settings": report["settings"], "runner_sha256": report["runner_sha256"],
                 "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "method": "Shared-prefix pretokenized prompts, one output token, uncached prefill, three warmups per length, randomized complete blocks, timing and profiling separate"}
        (out / "sweep.json").write_text(json.dumps(sweep, indent=2)+"\n")
        print("SWEEP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
