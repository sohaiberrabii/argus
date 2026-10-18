# Prompt-length sweep

Recorded: 2026-10-18T11:45:50.085529+00:00

One uncached request, one generated token. Device profiles and request timings are separate.

| Prompt tokens | Backbone rows | Vocabulary rows | Backbone ms | Vocabulary ms | Forward ms | Request ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 8 | 2.614 | 0.740 | 3.363 | 5.222 |
| 8 | 16 | 8 | 2.618 | 0.744 | 3.371 | 5.236 |
| 15 | 16 | 8 | 2.620 | 0.744 | 3.373 | 5.250 |
| 16 | 16 | 8 | 2.621 | 0.744 | 3.374 | 5.251 |
| 17 | 32 | 8 | 2.668 | 0.740 | 3.419 | 5.308 |
| 31 | 32 | 8 | 2.675 | 0.744 | 3.429 | 5.308 |
| 32 | 32 | 8 | 2.676 | 0.744 | 3.430 | 5.307 |
| 33 | 64 | 8 | 2.765 | 0.744 | 3.520 | 5.411 |
| 63 | 64 | 8 | 2.774 | 0.743 | 3.528 | 5.446 |
| 64 | 64 | 8 | 2.772 | 0.743 | 3.527 | 5.440 |
| 65 | 128 | 8 | 2.954 | 0.744 | 3.708 | 5.678 |
| 127 | 128 | 8 | 2.992 | 0.743 | 3.747 | 5.755 |
| 128 | 128 | 8 | 2.983 | 0.743 | 3.737 | 5.744 |

Device sample count per length: 5.
Request sample count per length: 30.

## Bucket edges

| Prompt change | Padded rows ratio | Forward medians ratio | Paired forward delta ms, median [min, max] |
| --- | ---: | ---: | ---: |
| 16->17 | 2.00x | 1.013x | 0.0473 [0.0453, 0.0489] |
| 32->33 | 2.00x | 1.026x | 0.0906 [0.0864, 0.0915] |
| 64->65 | 2.00x | 1.051x | 0.1804 [0.1784, 0.1854] |

The row ratios describe execution shapes. They do not predict latency ratios.
Paired deltas compare conditions within the same randomized round.
Request deltas and paired bootstrap intervals are available in the JSON output.
All generated token IDs matched the corresponding warmed condition.
