# Measurement method

## Two clocks answer different questions

Unprofiled request timing starts immediately before `LLM.generate` and ends
when it returns. Model loading and warmup are excluded. Input preparation,
the serving loop, sampling, and result delivery are included. A lightweight
shape recorder remains enabled. There are no explicit per-step synchronization
barriers or tensor-to-host copies in the measured serving loop.

Device measurements come from separate JAX profiler trials. The analyzer selects
the `/device:TPU:0` process and its `XLA Modules` lane. It reads integer
`device_offset_ps` and `device_duration_ps` fields, rather than host event times
or nested operation durations. A forward window begins at a backbone module's
start and ends at the corresponding vocabulary module's completion. It includes
the intervening device work and gaps.

The analyzer checks the number and ordering of recorded steps and module events.
It refuses overlapping module intervals, since summing those would double count
time. Whole-profile XLA program time is the sum of nonoverlapping module
durations divided by the first-to-last module window. It is not an arithmetic
unit utilization counter.

## Useful and padded work

For a Llama model with hidden width H, intermediate width I, L layers, Q query
heads, K key/value heads, head dimension D, and vocabulary V, the dominant work is:

```text
backbone dense FLOPs/token = 2 L (2 H Q D + 2 H K D + 3 H I)
vocabulary FLOPs/output row = 2 H V
attention FLOPs/query-key pair = 4 L Q D
```

One multiply-add counts as two FLOPs. For N tokens appended to a C-token cache,
causal attention has `N C + N (N + 1) / 2` query-key pairs. A prefill step
produces one useful vocabulary row for its single request.

The padded dense estimate substitutes the recorded backbone and vocabulary row
counts. It excludes attention and extra implementation-level tile rounding.
Useful FLOPs include dominant dense projections and causal QK/PV attention.
Both estimates omit normalization, rotary embeddings, activations, softmax,
cache maintenance, and sampling. They describe a model-based workload estimate,
not a count of all hardware operations.

For this baseline, H=2048, I=8192, L=16, Q=32, K=8, D=64, and V=128256:

- Backbone: 1,946,157,056 FLOPs per token.
- Vocabulary: 525,336,576 FLOPs per output row.
- Attention: 131,072 FLOPs per query-key pair.
- Padded decode dense projections: 35,341,205,504 FLOPs per step.

For context, `100 * estimated FLOPs / (seconds * peak FLOPs/s)` expresses
estimated work relative to theoretical peak. The v5e denominator is the
[published 197 TFLOP/s BF16 per chip](https://docs.cloud.google.com/tpu/docs/v5e).
The result is not a measurement of hardware busy time. A low value alone does
not diagnose inefficient execution. Single-stream decode can be constrained
by memory bandwidth and serving overhead.

## What the baseline establishes

The model runs through upstream `LlamaForCausalLM` with BF16 safetensors weights,
tensor parallelism 1, maximum sequence length 256, maximum active requests 1,
and maximum batched tokens 128. Prefix caching and asynchronous scheduling are
disabled, and the runtime precompilation pass is skipped. The actual measured
paths are warmed explicitly. Runtime defaults determine execution buckets.
The vocabulary input shape is recorded at the serving function boundary.

There are six unprofiled requests, each with 64 output tokens. Two separate
profiled requests each contain 16 output tokens: one prefill step and 15 decode
steps. Thus the decode group has 30 device samples, while each prefill shape
has one. A token profile is not an independent repeated trial for every step.

The public trace files are projections of the historical standard profiler
exports: only XLA module events, their device timer fields, and the process/lane
metadata needed to select them are retained. Hostnames, filesystem paths, and
unrelated events are omitted. Original-export and public-file hashes are recorded
in `provenance.json`. The included traces permit recalculation of the reported
timings, but they are not complete profiler archives.

The 2026-10-02 baseline predates this standalone repository. The 2026-10-18
baseline and prompt-length sweep were collected with the standalone runner.
Its executed source archive and exact dependency freeze are included. Both recorded
environments emitted an upstream model-cleanup diagnostic after successful
generation. Each exited zero and completed all recorded trials. The
[prompt-length case study](padding-boundaries.md) describes the additional
randomized rounds, output checks, and limitations of the fresh sweep.

Do not infer a speedup from the padding ratio or compare different model sizes,
precisions, batch settings, prompt mixes, or measurement clocks as equivalent.
There is no measured baseline-versus-optimization comparison in this release.
