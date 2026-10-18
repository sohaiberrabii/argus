# Controlled padding optimization

Recorded: 2026-10-18T12:14:11.666128+00:00

Same loaded BF16 Llama 1B model on one v5e, identical prompts, randomized policy order.

| Scenario / policy | Baseline request ms | Candidate request ms | Speedup | Latency reduction | Paired ms saved: median [95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: |
| generation-6/active_head | 315.608 | 312.722 | 1.009x | 0.91% | 2.6963 [2.5687, 2.7988] |
| generation-65/active_head | 316.407 | 313.579 | 1.009x | 0.89% | 2.7135 [2.5333, 2.8301] |
| prefill-128/active_head | 5.728 | 5.689 | 1.007x | 0.67% | 0.0387 [0.0313, 0.0444] |
| prefill-16/active_head | 5.236 | 5.193 | 1.008x | 0.83% | 0.0476 [0.0370, 0.0537] |
| prefill-32/active_head | 5.304 | 5.258 | 1.009x | 0.87% | 0.0428 [0.0376, 0.0478] |
| prefill-33/active_head | 5.395 | 5.351 | 1.008x | 0.82% | 0.0492 [0.0418, 0.0572] |
| prefill-64/active_head | 5.426 | 5.376 | 1.009x | 0.93% | 0.0470 [0.0363, 0.0528] |
| prefill-65/active_head | 5.672 | 5.624 | 1.008x | 0.83% | 0.0423 [0.0359, 0.0489] |
| prefill-97/active_head | 5.714 | 5.670 | 1.008x | 0.76% | 0.0382 [0.0330, 0.0490] |

A positive paired difference means the candidate completed faster.
Intervals describe within-run paired-round variation. Inspect device timings and correctness evidence before selecting a policy.
