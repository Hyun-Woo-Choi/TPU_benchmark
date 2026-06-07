# MLPerf-Style Mini Benchmark for CPU, GPU, and TPU on GCP

An educational mini benchmark that compares CPU, NVIDIA GPU, and Google Cloud TPU runtimes using JAX/XLA on Fashion-MNIST image classification.

> **Not an official MLPerf submission.** Results from this notebook should not be used to rank vendors or hardware platforms.

## Overview

The benchmark is inspired by MLPerf image-classification workloads but uses Fashion-MNIST and a small CNN to stay within a modest GCP credit budget (~$300). It measures:

- **Training throughput and latency** — SGD steps with backpropagation, across multiple batch sizes
- **Inference throughput and latency** — forward-only batched prediction
- **JAX/XLA compile time** — per batch size, separated from warm-up and steady-state
- **Memory usage** — GPU memory (via `nvidia-smi`) or process RSS (CPU)

Each run saves results to CSV so data from different runtimes can be combined for cross-hardware comparison.

## Model and Dataset

| Component | Details |
|---|---|
| Dataset | Fashion-MNIST (28×28 grayscale, 10 classes) |
| Model | `SmallJAXCNN` — 2 conv layers + 2 dense layers, ~207K parameters |
| Framework | JAX/XLA (no Flax, Optax, TensorFlow, or PyTorch) |
| Precision | float32 (default); bfloat16 is a suggested follow-up experiment |

If Fashion-MNIST cannot be downloaded, the notebook automatically falls back to a synthetic image-like dataset and records that in the result notes.

## Benchmark Profiles

Control cost and runtime with `BENCHMARK_PROFILE` in the first code cell:

| Profile | Batch sizes | Train steps | Inference steps | Recommended for |
|---|---|---|---|---|
| `quick` | 1–128 | 10 warm-up + 10 measured | 3 warm-up + 20 measured | First validation run on CPU or a new runtime |
| `expanded` | 1–1024 | 5 warm-up + 30 measured | 5 warm-up + 75 measured | Better accelerator utilization, still budget-safe |
| `accelerator_stress` | 1–4096 | 10 warm-up + 100 measured | 10 warm-up + 250 measured | H100 / TPU v5e after the pipeline is validated |

## Cost Warning and Recommended Run Order

1. Run on **CPU** first — verify correctness and save baseline CSVs.
2. Run on **TPU v5e** next with default (`quick` or `expanded`) iteration counts.
3. Run on **H100** only after the CPU and TPU runs succeed — start with `expanded`, then `accelerator_stress` if budget allows.
4. **Stop or delete** expensive GPU/TPU resources immediately after the experiment.
5. Save the CSV output files so runs do not need to be repeated.

## Output Files

| File | Contents |
|---|---|
| `benchmark_results_training.csv` | Training rows only |
| `benchmark_results_inference.csv` | Inference rows only |
| `benchmark_results_all.csv` | All rows; used by the visualization cells |

Each row records: timestamp, run ID, device type, hardware model, batch size, compile time, warm-up time, average and median latency, throughput, memory usage, precision, and status.

## Results Schema (key columns)

| Column | Description |
|---|---|
| `device_type` | `CPU`, `GPU`, or `TPU` |
| `hardware_model` | Hardware name detected at runtime |
| `batch_size` | Number of samples per step |
| `compile_time_seconds` | JAX/XLA `lower().compile()` time |
| `average_latency_ms` | Mean per-step time over measured iterations |
| `throughput_samples_per_second` | `batch_size / average_step_time` |
| `status` | `success`, `failed`, or `failed_oom` |

## Combining Multi-Runtime Results

Run the same notebook on each runtime (CPU, GPU, TPU). Each run appends rows to `benchmark_results_all.csv`. Copy or concatenate those files, then re-run the **Visualizations** and **Summary Table** cells to produce cross-hardware comparisons.

## Requirements

```
jax[cpu | cuda | tpu]   # install the build matching your runtime
numpy
matplotlib
psutil                  # optional; used for CPU memory reporting
```

JAX must be installed with the build that matches the target runtime. See the [JAX installation guide](https://github.com/google/jax#installation) for CPU, CUDA, and TPU variants.

## Interpretation Notes

- **Batch size changes the story.** Larger batches improve accelerator utilization and throughput but increase per-batch latency.
- **Compile time matters for short jobs.** JAX/XLA recompiles for each new batch size; on TPU the compile cost can be significant.
- **CPU is a correctness/scale baseline,** not an accelerator peer.
- **Hardware generation must match.** H100 is the generation-aligned GPU baseline for TPU v5e. A100 is a practical fallback; P100/V100 are historical.
- **Precision affects results.** This notebook uses float32 for an apples-to-apples baseline. bfloat16 experiments should be labeled separately.
- **Do not overgeneralize.** A small CNN on Fashion-MNIST is not representative of ResNet, LLM, recommender, or diffusion workloads.

## Suggested Next Steps

1. Combine CPU, TPU, and GPU CSVs and regenerate the charts.
2. Add hourly instance prices to the **Cost-Efficiency** section to compare samples-per-dollar.
3. Run an explicitly labeled bfloat16 experiment and compare compile time, throughput, and accuracy.
4. Replace Fashion-MNIST with CIFAR-10 or a small transformer workload (keeping the same logging schema).
5. Increase `train_measure_steps` / `inference_measure_steps` if budget and quota allow for lower variance estimates.
