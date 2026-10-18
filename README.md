# Argus TPU

Reproducible LLM inference measurements on Google Cloud TPUs, using vLLM TPU,
JAX, and the standard JAX profiler.

How much of a serving step is useful model work? How much comes from padded
execution shapes? How does the answer change between prefill and decode?
This project connects serving metadata to device timelines and makes the
accounting reproducible offline.

## Optimization: project only the active vocabulary row

The measurements led to a small JAX adapter: compute the vocabulary projection
for the one active request, then restore the padded output shape expected by
the sampler. The normal serving path projects eight rows.

In a second-process confirmation on the same TPU, the adapter reduced:

- **Vocabulary-stage latency by 5.8%:** 0.74332 → 0.70010 ms.
- **Device forward latency by 1.24%:** 3.36709 → 3.32542 ms.
- **64-output-token request latency by about 0.9%:** 315.608 → 312.722 ms for
  the short prompt, and 316.407 → 313.579 ms for the longer prompt.

All tested greedy outputs match the baseline, and the adapter passes the full
vocabulary numerical checks. The two runs contain 1,824 unprofiled request
measurements. Denser prompt buckets were also tested and excluded after failing
the chosen numerical tolerance.

![Measured optimization effects](assets/optimization.png)

Read [the optimization case study](docs/active-vocabulary.md) for the code,
correctness checks, paired intervals, scope, and rejected candidates. This is a
reference adapter for one active request in the pinned JAX Llama runtime.

## Finding: twice the rows adds 1–5% to the forward window

Our **2026-10-18** sweep measures 13 prompt lengths around the 16-, 32-, and
64-token execution-bucket boundaries. At each boundary, one extra prompt token
doubles the padded backbone rows. Median device forward time increases by
**1.3%, 2.6%, and 5.1%**, respectively.

| Prompt change | Padded backbone rows | Median device forward time | Increase |
| --- | --- | --- | ---: |
| 16 → 17 | 16 → 32 | 3.374 → 3.419 ms | 1.3% |
| 32 → 33 | 32 → 64 | 3.430 → 3.520 ms | 2.6% |
| 64 → 65 | 64 → 128 | 3.527 → 3.708 ms | 5.1% |

Across the sweep, 128 prompt tokens take **11.1% more device forward time** than
one token. The vocabulary projection stays near 0.74 ms, with 8 padded rows
throughout. Padded row counts alone substantially overstate the observed
latency growth for this single-request workload.

![Measured prompt shapes and timings](assets/prompt-sweep.png)

Each prompt length has **30 unprofiled request timings and 5 device-profile
observations**, in randomized complete rounds. Prompts are prefixes of the same
token sequence. Each request has an empty cache and produces exactly one token,
and all measured outputs match their warmed condition. The boundary deltas include
both the extra useful token and the changed execution shape.

Read [the case study](docs/padding-boundaries.md),
[all 13 conditions](results/llama-1b-v5e-20261018/sweep-report.md), and
[fresh-run provenance](results/llama-1b-v5e-20261018/provenance.json).

## Fresh baseline

Llama 3.2 1B Instruct, real pretrained BF16 weights, one Cloud TPU v5e chip,
one active request. Measurements recorded on **2026-10-18** using
vLLM TPU / tpu-inference 0.30.0 and JAX 0.11.0.

| Profiled phase | Active tokens | Backbone rows | Vocabulary rows | Median forward window | Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode | 1 | 16 | 8 | 3.368 ms | 30 |
| Short prefill | 6 | 16 | 8 | 3.366 ms | 1 |
| Longer prefill | 128 | 128 | 8 | 3.738 ms | 1 |

Separate, unprofiled requests generated 64 tokens at a median **203.230 tokens/s**
for the six-token prompt and **203.708 tokens/s** for the 128-token prompt.
Each has three repetitions. These rates include prefill, serving, sampling,
and result delivery. They measure whole requests.

The interesting result is the gap between **active tokens and execution rows**:
single-request decode uses 16 backbone rows and 8 vocabulary rows. For this
model, that corresponds to 35.34 GFLOPs of exposed dense projection work versus
2.47 GFLOPs for one useful token, before attention. This ratio describes tensor
shapes, and it does not establish a proportional latency penalty.

Our contribution is a small JAX optimization, an auditable measurement workflow,
and controlled case studies. The [2026-10-02 baseline](results/llama-1b-v5e/provenance.json) is
retained for comparison, and the fresh run reproduces its device timings closely.

## Reproduce the analysis without a TPU

Python 3.10+ and the standard library are sufficient:

```sh
python3 scripts/analyze.py results/llama-1b-v5e-20261018
python3 scripts/analyze_sweep.py results/llama-1b-v5e-20261018
python3 scripts/analyze_optimization.py results/optimization-20261018
python3 scripts/analyze_optimization.py results/optimization-confirm-20261018
python3 -m unittest discover -s tests -v
```

The analyzer recomputes request throughput, device forward windows, projection
FLOP estimates, and XLA program time from the included observations. For JSON:

```sh
python3 scripts/analyze_sweep.py results/llama-1b-v5e-20261018 --json
```

Read [the measurement method](docs/methodology.md) and
[the fresh-run provenance](results/llama-1b-v5e-20261018/provenance.json) before comparing
these numbers with another system.

## Run a new experiment on a TPU VM

Use an existing single-chip v5e VM and Python 3.12. The recorded environment
used the [official vLLM TPU installation workflow](https://docs.vllm.ai/projects/tpu/en/latest/getting_started/installation/):

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-tpu.txt --torch-backend=cpu
```

The complete dependency snapshot is
[`environment.freeze.txt`](results/llama-1b-v5e-20261018/environment.freeze.txt).
Use it in place of `requirements-tpu.txt` to recreate the recorded package versions.

Obtain the checkpoint through its authorized distribution and comply with its
license. The baseline used
[`unsloth/Llama-3.2-1B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct)
at revision `5a8abab4a5d6f164389b1079fb721cfab8d7126c`. The downloader verifies
the pinned weights' SHA256. Download and run:

```sh
.venv/bin/python benchmarks/download_checkpoint.py /path/to/checkpoint
.venv/bin/python benchmarks/run_llama.py --model /path/to/checkpoint --out runs/new --sweep
python3 scripts/analyze.py runs/new
python3 scripts/analyze_sweep.py runs/new
```

The runner warms both baseline prompt shapes, records six unprofiled requests,
then collects two device profiles. With `--sweep`, it also warms all 13 prompt
lengths, records 390 unprofiled requests, and collects five profiling rounds.
It uses version-specific hooks into
the open-source TPU model runner to record active and padded shapes. Review
these hooks when updating the runtime. It supports one request and
one v5e chip. This exact runner completed the fresh baseline and sweep. Its
compressed source archive, checksum, checkpoint provenance, and full package freeze are
included with the results.

To recreate the figure, install `requirements-plot.txt` and run:

```sh
python3 scripts/plot_sweep.py results/llama-1b-v5e-20261018 --out assets/prompt-sweep
python3 scripts/plot_optimization.py results/optimization-20261018 \
  results/optimization-confirm-20261018 --out assets/optimization
```

## Next experiments

1. Add concurrency sweeps to measure how batching changes useful work per
   padded row and the throughput/latency tradeoff.
2. Repeat on another model size and TPU generation to test how widely the
   measured latency pattern holds.
3. Extend the active-row adapter to additional request counts, preserving
   correctness checks and measuring both component and full-request effects.

These measurements establish the starting point for those experiments.
Results apply to the recorded model, software versions, device, and settings.
