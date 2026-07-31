# MLPerf-Style Mini Benchmarks for CPU, GPU, and TPU on GCP

Educational mini benchmarks that compare CPU, NVIDIA GPU, and Google Cloud TPU runtimes using JAX/XLA. Each notebook benchmarks the runtime it's currently running on; run the same notebook once per runtime (and per dataset, where applicable) and combine the CSV outputs to compare across hardware.

> **Not an official MLPerf submission.** Results from these notebooks should not be used to rank vendors or hardware platforms, and small mini benchmark results should not be overgeneralized to full-scale MLPerf or production results.

## Notebooks

| Notebook | Domain | Datasets | Model |
|---|---|---|---|
| [`mlperf_style_gcp_mini_benchmark.ipynb`](mlperf_style_gcp_mini_benchmark.ipynb) | Image classification | Fashion-MNIST | `SmallJAXCNN` (fixed 28×28×1 input) |
| [`mlperf_style_vision_benchmark.ipynb`](mlperf_style_vision_benchmark.ipynb) | Image classification | Fashion-MNIST, CIFAR-10, Imagenette (switchable via `DATASET_NAME`) | `SmallJAXCNN`, shape-adapted to the selected dataset |
| [`mlperf_style_lm_benchmark.ipynb`](mlperf_style_lm_benchmark.ipynb) | Language modeling | WikiText-2, WikiText-103, TinyStories (switchable via `DATASET_NAME`) | `SmallJAXGPT` — a small GPT-style decoder-only transformer |

`mlperf_style_vision_benchmark.ipynb` is a generalized superset of the original `mlperf_style_gcp_mini_benchmark.ipynb`: same CNN, same benchmark harness, but with `DATASET_NAME` picking the dataset instead of Fashion-MNIST being hardcoded. The original notebook is kept as-is for a minimal, single-dataset reference. `mlperf_style_lm_benchmark.ipynb` is the language-modeling sibling, following the same structure but with a transformer instead of a CNN.

All three notebooks share the same design:

- **Training throughput and latency** — steps with backpropagation, across multiple batch sizes
- **Inference throughput and latency** — forward-only batched prediction
- **JAX/XLA compile time** — per batch size, separated from warm-up and steady-state
- **Memory usage** — GPU memory (via `nvidia-smi`) or process RSS (CPU)
- **Environment detection** — records the exact CPU/GPU/TPU hardware and GCP metadata for each run
- Failed batch sizes (including OOM) are recorded as rows instead of crashing the notebook

## Model and Dataset Details

### Vision notebooks

| Dataset | Shape | Classes | Source |
|---|---|---|---|
| Fashion-MNIST | 28×28 grayscale | 10 | IDX files from the official `zalandoresearch/fashion-mnist` GitHub repo |
| CIFAR-10 | 32×32 RGB | 10 | Official Python pickle batches from `cs.toronto.edu` |
| Imagenette | 128×128 RGB (resized from `imagenette2-160`) | 10 | fast.ai's `imagenette2-160` release (requires Pillow) |

`SmallJAXCNN` is 2 conv layers + 2 dense layers. In the generalized vision notebook, `conv1`'s input channels and `dense1`'s flattened input size are derived from the selected dataset's image shape instead of being hardcoded.

### Language model notebook

| Dataset | Approx. size | Source | Notes |
|---|---|---|---|
| WikiText-2 | ~2M tokens | `wikitext.smerity.com` zip | Fast pipeline sanity check, not a serious training run |
| WikiText-103 | ~1e8 tokens | `wikitext.smerity.com` zip | Standard LM benchmark scale |
| TinyStories | ~4.7e8 tokens (train split ~2GB) | HuggingFace, downloaded via a capped HTTP Range request | Simple vocabulary; small models show visibly decreasing loss quickly |

`SmallJAXGPT` is a small decoder-only transformer (learned token + position embeddings, pre-norm causal self-attention + GELU MLP blocks, final layer norm, linear head). Tokenization is **byte-level** (vocab size 256) to stay dependency-free — this is a deliberate simplification, not a real BPE tokenizer, so `tokens_per_second` here is not directly comparable to BPE-tokenizer benchmark numbers. `BENCHMARK_PROFILE` also controls model size in this notebook (see below). The inference benchmark measures a full-sequence forward pass, not autoregressive token-by-token generation latency.

If a dataset cannot be downloaded, each notebook falls back to a synthetic dataset shaped like the selected one and records that in the result notes.

## Benchmark Profiles

Control cost and runtime with `BENCHMARK_PROFILE` in the first code cell.

**Vision notebooks** (`quick` / `expanded` / `accelerator_stress`):

| Profile | Batch sizes | Train steps | Inference steps | Recommended for |
|---|---|---|---|---|
| `quick` | 1–128 | 3 warm-up + 10 measured | 3 warm-up + 20 measured | First validation run on CPU or a new runtime |
| `expanded` | 1–1024 | 5 warm-up + 30 measured | 5 warm-up + 75 measured | Better accelerator utilization, still budget-safe |
| `accelerator_stress` | 1–4096 | 10 warm-up + 100 measured | 10 warm-up + 250 measured | H100 / TPU v5e after the pipeline is validated |

**LM notebook** — profiles also scale model size and sequence length:

| Profile | Model size | Sequence length | Batch sizes | Recommended for |
|---|---|---|---|---|
| `quick` | 2 layers, 2 heads, 64 dim | 128 | 1–32 | WikiText-2 pipeline sanity check |
| `expanded` | 6 layers, 8 heads, 512 dim | 512 | 1–128 | Mid-size accelerator comparison |
| `accelerator_stress` | 12 layers, 12 heads, 768 dim (~86M params, GPT-2-small-shaped) | 512 | 1–256 | H100 / TPU v5e; sized to fit ~1 epoch of WikiText-103 on a single TPU v5e chip |

Imagenette (128×128×3) is much more expensive per batch than Fashion-MNIST/CIFAR-10 — consider smaller batch sizes before trying `accelerator_stress`. Likewise, the LM notebook's `accelerator_stress` model is far more memory-hungry than the vision CNN; run `quick` on CPU first.

## Cost Warning and Recommended Run Order

1. Run on **CPU** first — verify correctness (`quick` profile) and save baseline CSVs.
2. Run on **TPU v5e** next with default (`quick` or `expanded`) iteration counts.
3. Run on **H100** only after the CPU and TPU runs succeed — start with `expanded`, then `accelerator_stress` if budget allows.
4. **Stop or delete** expensive GPU/TPU resources immediately after the experiment.
5. Save the CSV output files so runs do not need to be repeated.

## Output Files

Each notebook writes its own CSV files so results from different notebooks don't overwrite each other:

| Notebook | Training CSV | Inference CSV | Combined CSV |
|---|---|---|---|
| Original (Fashion-MNIST only) | `benchmark_results_training.csv` | `benchmark_results_inference.csv` | `benchmark_results_all.csv` |
| Vision (generalized) | `vision_benchmark_results_training.csv` | `vision_benchmark_results_inference.csv` | `vision_benchmark_results_all.csv` |
| Language model | `lm_benchmark_results_training.csv` | `lm_benchmark_results_inference.csv` | `lm_benchmark_results_all.csv` |

Each row records: timestamp, run ID, device type, hardware model, dataset, batch size, compile time, warm-up time, average and median latency, throughput, memory usage, precision, and status. The LM notebook adds `sequence_length`, `vocab_size`, and `tokens_per_second` columns.

## Results Schema (key columns)

| Column | Description |
|---|---|
| `device_type` | `CPU`, `GPU`, or `TPU` |
| `hardware_model` | Hardware name detected at runtime |
| `dataset` | Which dataset (or its synthetic-fallback label) produced the row |
| `batch_size` | Number of samples (vision) or sequences (LM) per step |
| `compile_time_seconds` | JAX/XLA `lower().compile()` time |
| `average_latency_ms` | Mean per-step time over measured iterations |
| `throughput_samples_per_second` | `batch_size / average_step_time` |
| `tokens_per_second` (LM only) | `throughput_samples_per_second * sequence_length` |
| `status` | `success`, `failed`, or `failed_oom` |

## Combining Multi-Runtime Results

Run the same notebook on each runtime (CPU, GPU, TPU) — and, for the vision/LM notebooks, once per `DATASET_NAME`. Each run appends rows to that notebook's `*_all.csv`. Copy or concatenate those files, then re-run the **Visualizations** and **Summary Table** cells; the `dataset` column keeps series separated across datasets.

## Requirements

```
jax[cpu | cuda | tpu]   # install the build matching your runtime
numpy
matplotlib
psutil                  # optional; used for CPU memory reporting
pillow                  # required only for the vision notebook's Imagenette loader
```

JAX must be installed with the build that matches the target runtime. See the [JAX installation guide](https://github.com/google/jax#installation) for CPU, CUDA, and TPU variants.

## Interpretation Notes

- **Batch size changes the story.** Larger batches improve accelerator utilization and throughput but increase per-batch latency (and OOM risk, especially for Imagenette and the LM notebook's larger model sizes).
- **Compile time matters for short jobs.** JAX/XLA recompiles for each new batch size; on TPU the compile cost can be significant, and larger transformer graphs compile more slowly than the small CNN.
- **CPU is a correctness/scale baseline,** not an accelerator peer — don't expect `accelerator_stress`-sized LM runs to finish quickly (or at all) on CPU.
- **Hardware generation must match.** H100 is the generation-aligned GPU baseline for TPU v5e. A100 is a practical fallback; P100/V100 are historical.
- **Precision affects results.** These notebooks use float32 for an apples-to-apples baseline. bfloat16 experiments should be labeled separately.
- **Dataset choice changes what "good" looks like.** WikiText-2 and CIFAR-10 are pipeline sanity checks, not serious training runs — don't draw hardware conclusions from them alone.
- **LM inference ≠ generation latency.** The LM notebook's inference benchmark is a full-sequence forward pass, not autoregressive, KV-cache-based decoding.
- **Do not overgeneralize.** A small CNN or GPT on these datasets is not representative of ResNet, large LLM, recommender, or diffusion workloads.

## Suggested Next Steps

1. Combine CPU, TPU, and GPU CSVs (per notebook) and regenerate the charts.
2. Add hourly instance prices to the **Cost-Efficiency** section to compare samples-per-dollar (or tokens-per-dollar).
3. Run an explicitly labeled bfloat16 experiment and compare compile time, throughput, and accuracy/loss.
4. For the LM notebook, replace the byte-level tokenizer with a real BPE tokenizer (e.g. `tiktoken`), keeping the same packing/benchmark harness.
5. Increase measured iterations if budget and quota allow, for lower-variance estimates.
