# Controlled padding optimization

Recorded: 2026-10-18T12:10:01.551498+00:00

Same loaded BF16 Llama 1B model on one v5e, identical prompts, randomized policy order.

| Scenario / policy | Baseline request ms | Candidate request ms | Speedup | Latency reduction | Paired ms saved: median [95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| generation-6/active_head | 318.164 | 316.317 | 1.006x | 0.58% | 2.6789 [2.0583, 3.3749] |
| generation-65/active_head | 317.161 | 313.776 | 1.011x | 1.07% | 2.4714 [1.8108, 2.9066] |
| prefill-128/active_head | 5.798 | 5.750 | 1.008x | 0.82% | 0.0414 [0.0324, 0.0483] |
| prefill-16/active_head | 5.297 | 5.259 | 1.007x | 0.72% | 0.0379 [0.0305, 0.0459] |
| prefill-32/active_head | 5.365 | 5.324 | 1.008x | 0.77% | 0.0405 [0.0286, 0.0476] |
| prefill-33/active_head | 5.459 | 5.415 | 1.008x | 0.81% | 0.0450 [0.0356, 0.0544] |
| prefill-64/active_head | 5.484 | 5.435 | 1.009x | 0.90% | 0.0467 [0.0335, 0.0555] |
| prefill-65/active_head | 5.735 | 5.694 | 1.007x | 0.71% | 0.0450 [0.0367, 0.0555] |
| prefill-97/active_head | 5.785 | 5.746 | 1.007x | 0.66% | 0.0427 [0.0302, 0.0531] |

A positive paired difference means the candidate completed faster.
Intervals describe within-run paired-round variation. Inspect device timings and correctness evidence before selecting a policy.
