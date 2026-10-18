# Abstract

Argus TPU connects serving padding to measured language-model
inference performance and uses those measurements to guide a targeted
optimization. Using pretrained BF16 Llama 3.2 1B through vLLM TPU and JAX on one
Cloud TPU v5e, we first measure 13 prompt lengths around execution-bucket
boundaries. Doubling padded backbone rows increases device forward latency by
only 1–5%, showing that tensor sizes alone overstate latency growth for this
single-request workload. Profiling also identifies an approximately 0.74 ms
vocabulary projection that computes eight rows for one active request. A small
JAX adapter projects only the active row and restores the sampler's padded
output shape. A second-process confirmation reduces vocabulary-stage latency
by 5.8%, device forward latency by 1.24%, and 64-output-token request latency
by approximately 0.9%. The optimization passes full-vocabulary numerical checks
and preserves tested greedy outputs. Two controlled comparisons provide 1,824
unprofiled request measurements, with separate device profiles, exact runtime
versions, executed source, and offline analysis. The project documents both the
successful adapter and rejected bucket configurations to support reproducible
TPU performance engineering.
