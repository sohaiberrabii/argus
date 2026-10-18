"""Compare prompt buckets and active-row vocabulary projection on one TPU v5e.

Version-specific adapter for vLLM TPU / tpu-inference 0.30.0. Uses the public
JAX model and serving implementation. All policies share one loaded model.
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

POLICIES = {
    "baseline": {"buckets": [16, 32, 64, 128], "active_head": False},
    "buckets16": {"buckets": [16, 32, 48, 64, 80, 96, 112, 128], "active_head": False},
    "active_head": {"buckets": [16, 32, 64, 128], "active_head": True},
    "combined": {"buckets": [16, 32, 48, 64, 80, 96, 112, 128], "active_head": True},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timing-rounds", type=int, default=40)
    parser.add_argument("--generation-rounds", type=int, default=12)
    parser.add_argument("--profile-rounds", type=int, default=5)
    parser.add_argument("--policies", nargs="+", choices=list(POLICIES), default=list(POLICIES))
    parser.add_argument("--performance-corpus", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--order-seed", type=int, default=20261019)
    args = parser.parse_args()
    if min(args.timing_rounds, args.generation_rounds, args.profile_rounds) < 1:
        raise ValueError("All repetition counts must be positive")
    requested = list(dict.fromkeys(args.policies))
    if not requested or requested[0] != "baseline":
        raise ValueError("The baseline policy must be first")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(MODEL_IMPL_TYPE="flax_nnx", SKIP_JAX_PRECOMPILE="1",
                      VLLM_ENABLE_V1_MULTIPROCESSING="0", JAX_PLATFORMS="tpu,cpu")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from vllm import LLM, SamplingParams
    from tpu_inference.runner.tpu_runner import TPUModelRunner

    for package in ("vllm-tpu", "tpu-inference"):
        if importlib.metadata.version(package) != "0.30.0":
            raise RuntimeError("Review the adapter before changing the runtime version")
    if len(jax.devices()) != 1 or "v5" not in jax.devices()[0].device_kind.lower():
        raise RuntimeError("Exactly one TPU v5e device is required")
    policy = "baseline"
    runner = None
    measuring = False
    trial = None
    rows = []
    current = None
    capture_logits = False
    captured_logits = None
    original_prepare = TPUModelRunner._prepare_inputs
    original_load = TPUModelRunner.load_model

    @functools.wraps(original_prepare)
    def prepare(self, scheduler, *a, **kw):
        nonlocal current
        result = original_prepare(self, scheduler, *a, **kw)
        if measuring:
            if self.input_batch.num_reqs != 1:
                raise RuntimeError("This adapter supports exactly one active request")
            computed = [int(x) for x in self.input_batch.num_computed_tokens_cpu[:1]]
            current = {"trial": trial, "policy": policy,
                       "active_tokens": int(scheduler.total_num_scheduled_tokens),
                       "padded_tokens": int(result[0].shape[0]), "computed_tokens": computed,
                       "phase": "prefill" if computed == [0] else "decode"}
            rows.append(current)
        return result

    @functools.wraps(original_load)
    def load(self, *a, **kw):
        nonlocal runner, captured_logits
        result = original_load(self, *a, **kw)
        runner = self
        original_logits = self.compute_logits_fn

        @jax.jit
        def active_compute_logits(state, hidden, lora):
            # The selected hidden states are [padded requests, hidden width].
            # Rows are independent in the vocabulary projection. Preserve the
            # sampler's padded shape, while computing only its active row.
            logits = original_logits(state, hidden[:1], lora)
            return jnp.pad(logits, ((0, hidden.shape[0]-1), (0, 0)))

        def logits_call(*a, **kw):
            nonlocal captured_logits
            if POLICIES[policy]["active_head"] and (self.input_batch.num_reqs != 1 or self.input_batch.num_prompt_logprobs):
                raise RuntimeError("Active-row adapter requires one request without prompt logprobs")
            chosen = active_compute_logits if POLICIES[policy]["active_head"] else original_logits
            if measuring and current is not None:
                current["vocabulary_input_rows"] = int(a[1].shape[0])
                current["vocabulary_compute_rows"] = 1 if POLICIES[policy]["active_head"] else int(a[1].shape[0])
            output = chosen(*a, **kw)
            if capture_logits and captured_logits is None:
                captured_logits = np.asarray(jax.device_get(output[0]), dtype=np.float32)
            return output

        self.compute_logits_fn = logits_call
        config = self.model_config.hf_config.to_dict()
        keys = ("model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size")
        (out / "model-config.json").write_text(json.dumps({k: config[k] for k in keys}, indent=2)+"\n")
        (out / "model-class.txt").write_text(f"{type(self.model).__module__}.{type(self.model).__name__}\n")
        return result

    TPUModelRunner._prepare_inputs = prepare
    TPUModelRunner.load_model = load
    llm = LLM(model=str(args.model.expanduser().resolve()), tokenizer=str(args.model.expanduser().resolve()),
              dtype="bfloat16", load_format="safetensors", tensor_parallel_size=1,
              max_model_len=256, max_num_seqs=1, max_num_batched_tokens=128,
              gpu_memory_utilization=0.5, seed=17, enable_prefix_caching=False, async_scheduling=False,
              additional_config={"compilation_sizes": [48, 80, 96, 112]})
    if runner is None or runner.dp_size != 1 or runner.speculative_config is not None:
        raise RuntimeError("Unsupported runner configuration")
    if (out / "model-class.txt").read_text().strip() != "tpu_inference.models.jax.llama3.LlamaForCausalLM":
        raise RuntimeError("Expected the native JAX Llama model")
    available_buckets = list(runner.num_tokens_paddings)
    for configuration in POLICIES.values():
        if not set(configuration["buckets"]).issubset(available_buckets):
            raise RuntimeError("A requested bucket was not configured at model initialization")

    def choose(name):
        nonlocal policy
        policy = name
        # Select a subset of the configured buckets. Buffer capacity stays 128,
        # no in-flight requests exist when switching this one-request runner.
        runner.num_tokens_paddings = POLICIES[name]["buckets"].copy()
        runner.num_tokens_paddings_per_dp = runner.num_tokens_paddings.copy()

    def generate(prompt, count):
        return llm.generate([prompt], SamplingParams(temperature=0, max_tokens=count,
                            ignore_eos=True), use_tqdm=False)[0].outputs[0].token_ids

    texts = ["Explain how a computer keeps track of a conversation. ",
             "A scientist records careful measurements before drawing a conclusion. ",
             "The garden has red flowers beside a small wooden gate. "]
    sequences = [llm.get_tokenizer().encode(text*64)[:128] for text in texts]
    if any(len(ids) != 128 for ids in sequences):
        raise RuntimeError("Validation prompt is too short")
    validation = []
    eligible = {name: True for name in requested}
    for corpus, ids in enumerate(sequences):
        for n in (17, 33, 65, 97):
            prompt = {"prompt_token_ids": ids[:n]}
            reference = None
            reference_logits = None
            for name in requested:
                choose(name)
                capture_logits = True
                captured_logits = None
                tokens = generate(prompt, 16)
                capture_logits = False
                if name == "baseline":
                    reference, reference_logits = tokens, captured_logits.copy()
                delta = np.abs(captured_logits-reference_logits)
                close = bool(np.allclose(captured_logits, reference_logits, rtol=.02, atol=.0625))
                finite = bool(np.isfinite(captured_logits).all())
                exact = tokens == reference
                accepted = close and finite and exact
                eligible[name] = eligible[name] and accepted
                validation.append({"corpus": corpus, "prompt_tokens": n, "policy": name,
                                   "generated_tokens": 16, "output_token_ids": tokens,
                                   "reference_token_ids": reference, "exact_tokens_match": exact,
                                   "all_logits_finite": finite, "logits_allclose": close,
                                   "logits_rtol": .02, "logits_atol": .0625,
                                   "max_absolute_logit_difference": float(delta.max()),
                                   "mean_absolute_logit_difference": float(delta.mean()),
                                   "accepted": accepted})
            print(f"VALIDATED corpus={corpus} length={n} eligible={eligible}", flush=True)
    policies = [name for name, accepted in eligible.items() if accepted]
    (out / "validation.json").write_text(json.dumps({"eligible": eligible, "records": validation}, indent=2)+"\n")
    if len(policies) == 1:
        raise RuntimeError("All candidate optimizations failed correctness validation")

    lengths = [16, 32, 33, 64, 65, 97, 128]
    performance_ids = sequences[args.performance_corpus]
    scenarios = {f"prefill-{n}": {"prompt": {"prompt_token_ids": performance_ids[:n]}, "tokens": 1}
                 for n in lengths}
    scenarios.update({"generation-6": {"prompt": "The capital of France is", "tokens": 64},
                      "generation-65": {"prompt": {"prompt_token_ids": performance_ids[:65]}, "tokens": 64}})
    expected = {}
    warmup = []
    for scenario, setup in scenarios.items():
        for name in policies:
            choose(name)
            start = time.perf_counter()
            for _ in range(3):
                tokens = generate(setup["prompt"], setup["tokens"])
                if scenario not in expected:
                    expected[scenario] = tokens
                if tokens != expected[scenario]:
                    raise RuntimeError(f"Cross-policy output mismatch: {scenario}, {name}")
            warmup.append({"scenario": scenario, "policy": name,
                           "three_warmups_seconds": time.perf_counter()-start})
            print(f"WARMED {scenario} {name}", flush=True)
    records = []
    rng = random.Random(args.order_seed)

    def measure(scenario, name, rep, profiled, profile=None):
        nonlocal measuring, trial
        choose(name)
        trial = f"{'profile' if profiled else 'timing'}-{scenario}-{name}-{rep}"
        begin = len(rows)
        count = min(scenarios[scenario]["tokens"], 8) if profiled else scenarios[scenario]["tokens"]
        measuring = True
        start = time.perf_counter_ns()
        tokens = generate(scenarios[scenario]["prompt"], count)
        elapsed = (time.perf_counter_ns()-start)/1e9
        measuring = False
        if tokens != expected[scenario][:count]:
            raise RuntimeError(f"Measured output mismatch: {trial}")
        steps = rows[begin:]
        if len(steps) != count or steps[0]["computed_tokens"] != [0]:
            raise RuntimeError("Expected uncached prefill followed by one-token decode steps")
        records.append({"trial": trial, "scenario": scenario, "policy": name, "round": rep,
                        "elapsed_s": elapsed, "generated_tokens": count,
                        "profiled": profiled, "profile": profile, "steps": steps,
                        "output_token_ids": tokens, "exact_output_match": True})

    for family, rounds in (("prefill", args.timing_rounds), ("generation", args.generation_rounds)):
        conditions = [(scenario, name) for scenario in scenarios if scenario.startswith(family) for name in policies]
        for rep in range(rounds):
            order = conditions.copy()
            rng.shuffle(order)
            for scenario, name in order:
                measure(scenario, name, rep, False)
            print(f"TIMING {family} round={rep+1}/{rounds}", flush=True)
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 2
    options.python_tracer_level = 0
    options.device_tracer_level = 1
    for rep in range(args.profile_rounds):
        profile = f"profile-opt-{rep}"
        conditions = [(scenario, name) for scenario in scenarios for name in policies]
        rng.shuffle(conditions)
        jax.profiler.start_trace(str(out / profile), profiler_options=options)
        for scenario, name in conditions:
            measure(scenario, name, rep, True, profile)
        jax.profiler.stop_trace()
        print(f"PROFILE round={rep+1}/{args.profile_rounds}", flush=True)
    report = {"completed": True, "records": records, "policies": POLICIES, "eligible": eligible,
              "available_buckets": available_buckets, "warmups": warmup,
              "timing_rounds": args.timing_rounds, "generation_rounds": args.generation_rounds,
              "profile_rounds": args.profile_rounds, "order_seed": args.order_seed,
              "performance_corpus": args.performance_corpus,
              "prompt_token_ids": sequences, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "versions": {n: importlib.metadata.version(n) for n in
                           ("vllm-tpu", "tpu-inference", "jax", "jaxlib", "libtpu", "flax")},
              "hardware": {"accelerator": "Google Cloud TPU v5e", "chips": 1, "precision": "bfloat16"},
              "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "method": "Same loaded model, identical prompts, randomized complete rounds across policies, timing and profiling separate, three warmups per condition, full-vocabulary first-step logit checks and exact 16/64-token greedy output checks"}
    (out / "optimization.json").write_text(json.dumps(report, indent=2)+"\n")
    (out / "optimization_runner.py").write_bytes(Path(__file__).read_bytes())
    checkpoint = args.model.expanduser().resolve() / "checkpoint-provenance.json"
    (out / "checkpoint-provenance.json").write_bytes(checkpoint.read_bytes())
    print("OPTIMIZATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
