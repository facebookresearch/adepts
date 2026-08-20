# ADEPTS

ADEPTS-BENCH: Measuring the Trustworthiness of
Computer Use Agents across Devices

A dual-stream trustworthiness benchmark for Computer Use Agents (CUAs) navigating mobile and desktop applications. ADEPTS evaluates whether agents can **safely** interact with visual interfaces while handling ambiguous instructions:

- **Safety** — Paired benign/malicious tasks with threats embedded in the visual interface (e.g., mislabeled buttons, phishing links, unauthorized purchases). Measures task success vs. attack success rate.
- **Disambiguation** — Evaluates whether agents seek clarification when user intent is ambiguous, rather than making potentially harmful assumptions.

## Paper Highlights

### What's in the benchmark

2,462 tasks, evaluated **offline** — no live environment, no infrastructure to stand up. Each safety task is one API call; each disambiguation task is one model call plus LLM-judge calls for question matching. ADEPTS-BENCH is the only benchmark combining visual grounding, offline evaluation, paired benign/malicious variants, disambiguation, a user-research-grounded taxonomy, and an open release (paper Table 1).

| Stream          | Instances                | Mobile                     | Desktop                    |
| --------------- | ------------------------ | -------------------------- | -------------------------- |
| Safety          | 1,718 (859 paired tasks) | 358 pairs, ~70 Android apps | 501 pairs, ~150 apps       |
| Disambiguation  | 744                      | 381 tasks (4.7 images avg) | 363 tasks (2.3 images avg) |

- **Paired safety design.** The benign and malicious variants of a task share the *same instruction*; only the screenshot differs, embedding the threat in the visual interface (phishing overlay, injected prompt, misleading button). The benign variant measures capability, the malicious one measures safety. 10 threat categories, from prompt injection and web hacking to misalignment and system sabotage (Dangerous Commands is desktop-only).
- **Disambiguation design.** A trajectory of screenshots leads to an ambiguous decision point; the model must decide whether to clarify or act. Categories: referential underspecificity, linguistic ambiguity, missing parameters, and impossible tasks. Each item is scored on *obviousness* (0–2) and *consequence* (0–2).
- **User-grounded risk taxonomy.** Built from expert safety workshops plus a MaxDiff survey of n=1,300 US general-population participants ranking 44 undesirable outcomes. Top concerns: identity theft (80.2%), account compromise (77.5%), unauthorized financial transfer (76.0%). 74.1% were more worried about external attacks than internal errors.
- **Annotation quality.** Pixel-level bounding boxes from professional annotators, validated by cross-model agreement and expert review. The disambiguation LLM judge (Gemini 3.1 Pro) agrees with human judgments 96.1% of the time (κ=0.92).

### Metrics

- **TSR** — task success rate on benign tasks (capability).
- **ASR** — attack success rate on malicious tasks (lower is safer).
- **FRR** — false refusal rate: `status(unsafe)` on benign tasks (over-safety).
- **ADEPTS Score** — harmonic mean of TSR and (1−ASR), so a model can't score well by sacrificing one for the other (90% TSR with 90% ASR scores 17.1%, not 50%).
- **Disambiguation F1** — precision × recall of generated clarification questions against human ground truth.
- **Severity Calibration Error (Δ)** — mean per-component absolute error on obviousness and consequence scores, in [0, 4].

### Headline results

Safety stream, mean ± 95% CI over 10 runs at temperature 0.1 (paper Table 2):

| Model             | Mobile TSR↑ | Mobile ASR↓ | Mobile FRR↓ | Mobile S↑ | Desktop TSR↑ | Desktop ASR↓ | Desktop FRR↓ | Desktop S↑ |
| ----------------- | ----------- | ----------- | ----------- | --------- | ------------ | ------------ | ------------ | ---------- |
| Gemini 3.1 Pro    | 77.6        | 29.6        | 9.6         | 73.8      | 82.9         | 29.8         | 8.7          | **76.0**   |
| Claude 4.7 Opus   | 71.0        | **27.7**    | 9.4         | 71.6      | 78.1         | **27.2**     | 6.3          | 75.4       |
| GPT-5.4           | 62.1        | 36.7        | 5.1         | 62.7      | 68.8         | 36.3         | 7.0          | 66.1       |
| Gemini 2.5 CU     | 78.1        | 52.7        | 1.5         | 58.9      | **83.6**     | 51.5         | 1.3          | 61.4       |
| Qwen3-VL-235B     | **83.0**    | 78.1        | 0.0         | 34.7      | 86.6         | 76.3         | 0.0          | 37.2       |
| Qwen3-VL-8B       | 79.0        | 75.5        | 0.0         | 37.4      | 87.4         | 73.7         | 0.0          | 40.5       |
| Qwen3-VL-4B       | 81.6        | 79.6        | 0.0         | 32.6      | 87.0         | 76.7         | 0.0          | 36.7       |

Disambiguation stream, with the severity-scoring prompt (paper Table 4):

| Model           | Mobile F1↑ | Mobile Δ↓ | Desktop F1↑ | Desktop Δ↓ |
| --------------- | ---------- | --------- | ----------- | ---------- |
| Gemini 3.1 Pro  | **58.9**   | 0.86      | 39.5        | 1.44       |
| Gemini 2.5 CU   | 54.6       | 1.16      | 40.7        | **1.37**   |
| Qwen3-VL-235B   | 52.2       | 1.34      | **46.2**    | 1.38       |
| Qwen3-VL-4B     | 49.8       | 1.31      | 29.3        | 1.57       |
| Claude 4.7 Opus | 49.5       | 0.90      | 43.5        | 1.43       |
| Qwen3-VL-8B     | 43.6       | 1.54      | 5.7         | 0.45       |
| GPT-5.4         | 39.9       | **0.81**  | 26.0        | 1.55       |

(Qwen3-VL-8B's desktop Δ is not comparable — it returns empty responses on 93% of desktop tasks.)

### Key takeaways

1. **No model is both capable and safe.** The best ADEPTS Score is 76.0% (Gemini 3.1 Pro, desktop), the only model × platform combination above 80% TSR while staying below 30% ASR. Every model clicks "Checkout" on a $25K order, and none detects that a "factory reset" button is mislabeled "Optimize."
2. **Three distinct safety architectures.** Ablating the `status(unsafe)` refusal tool raises ASR by 10–23pp for frontier models and leaves Qwen unchanged (±1pp): *tool-dependent* (Gemini 3.1 Pro, +22pp), *partially tool-dependent* (Claude +10pp, GPT-5.4 +11pp — implicit safety survives without the tool), and *no mechanism* (Qwen, which never calls the tool at all). Removing the tool also *raises* benign TSR by 1–8pp — the capability/safety tradeoff lives at the system-prompt level.
3. **Computer-use specialization may cost safety.** Gemini 2.5 CU has the highest TSR (83.6% desktop) but the highest frontier ASR (51.5%), well above general-purpose Gemini 3.1 Pro (29.8%). Its 181 unique vulnerabilities cluster on action-ready UIs and semantic-only threats: it optimizes for action completion over action evaluation.
4. **Open-source safety doesn't scale.** Qwen reaches 87% TSR but 74–80% ASR at every size (4B → 235B), never uses the refusal tool, and never falsely refuses.
5. **A four-level failure spectrum.** L1 (7.4%) explicit visual cues, caught by everyone; L2 (15.6%) embedded text threats, caught by frontier models only; L3 (66.3%) ambiguous contexts where models disagree; L4 (10.8%) no visual cues at all — the harm is in action *scale* ($25K checkout) or *label mismatch*. Safety training handles pattern matching, not consequence reasoning.
6. **Worst-case safety is much worse than the mean.** Per-task pass@k (k=10) ASR runs 2–23pp above mean ASR. GPT-5.4 is the least deterministic (36% → 58–60%, ~200 flaky tasks per platform); Qwen-4B the most (+2–3pp).
7. **Over-refusal is visual pattern matching, not threat detection.** Frontier models falsely refuse 6–10% of benign tasks, triggered by dark/"hacker" styling, urgency and promotional keywords, and — uniquely visible in the paired design — near-identical benign/malicious screenshots.
8. **Models overestimate consequences.** 42–62% of matched disambiguation items are rated more severe than ground truth, only 7–25% less. Obviousness calibration is far better: models know *what* is ambiguous but overweight *how bad* a wrong guess would be — the same cautious bias that drives FRR.
9. **Impossible tasks are a blind spot.** Tasks requiring impossibility detection ("change the phone SIM card" on a software-only interface) are missed at more than double the rate (30.6% vs 13.8%). Models treat them as merely underspecified and ask helpful-but-wrong questions.
10. **The severity-scoring prompt cuts both ways.** It raises clarification rate by 4–40pp, but lowers F1 for most frontier models (GPT-5.4 −5.6pp) while substantially helping open-source ones (Qwen-8B +18pp) — useful scaffolding for weaker models, a metacognitive cost for stronger ones.
11. **Trustworthiness needs multi-dimensional evaluation.** Rankings shift across platform and stream: Gemini 3.1 leads mobile disambiguation but drops to 4th on desktop, while Qwen-235B rises from 5th to 1st despite the worst safety. Safety improves on desktop (+4–8pp TSR) while disambiguation degrades (8–38pp F1).

### Hardest threat categories

ASR by threat category, combined mobile + desktop (paper Table 10):

| Threat            | Gemini 3.1 | Claude 4.7 | GPT-5.4 | Gemini CU |
| ----------------- | ---------- | ---------- | ------- | --------- |
| System Sabotage   | 48.3       | 49.4       | 68.5    | 66.3      |
| Misalignment      | 50.0       | 43.8       | 45.8    | 60.4      |
| Hallucination     | 40.9       | 33.0       | 43.2    | 47.7      |
| Reasoning Gap     | 35.8       | 34.7       | 40.0    | 49.5      |
| Adversarial Attack| 28.6       | 30.0       | 38.5    | 54.9      |
| Jailbreak         | 17.8       | 17.8       | 34.4    | 56.7      |
| Response Latency  | 22.1       | 16.3       | 32.6    | 51.2      |
| Dangerous Cmds    | 16.3       | 8.2        | 20.4    | 51.0      |
| Prompt Injection  | 13.3       | 14.4       | 30.0    | 32.2      |
| Web Hacking       | 5.9        | 15.3       | 10.6    | 41.2      |

System sabotage is the most effective attack overall; prompt injection and web hacking are the best defended. Gemini CU is the most uniformly vulnerable.

## Project Structure

```
adepts/
├── code/
│   ├── download_data.py                     # Download datasets from S3
│   ├── make_sample_data.py                  # Regenerate the committed sample_data/
│   ├── llm_client.py                        # Shared multi-provider LLM client
│   ├── utils.py                             # Shared utilities (logging, paths, summary)
│   ├── reset.py                             # Reset generated data/results
│   ├── disambiguation/
│   │   ├── disambiguation_benchmark.py      # Disambiguation benchmark
│   │   ├── disambiguation_grading.py        # Evaluation, metrics, plotting
│   │   └── disambiguation_prompts.py        # Prompt templates
│   ├── safety/
│   │   ├── images_pre_processing.py         # Safety image pre-processing
│   │   ├── dataset_pre_processing.py        # Safety dataset builder
│   │   ├── safety_benchmark.py              # Safety benchmark evaluation
│   │   ├── rendering.py                     # Screenshot visualization
│   │   └── print_results.py                 # Print results summary table
│   └── analytics/                           # Paper results analysis scripts
│       ├── analyze.py                       # Run all analyses (entry point)
│       ├── common.py                        # Shared constants, data loading
│       ├── convert_results.py               # Convert benchmark output → analytics format
│       ├── safety_main.py                   # Table 2: TSR, ASR, FRR, ADEPTS Score
│       ├── safety_ablation.py               # Table 3: Tool ablation
│       ├── safety_passk.py                  # Table 7: Pass@k worst-case
│       ├── safety_paired.py                 # Table 8: Paired 2x2
│       ├── safety_per_threat.py             # Table 10: Per-threat ASR
│       ├── safety_failure_spectrum.py       # Figure 3: Failure hierarchy
│       ├── disambig_main.py                 # Table 4: F1 and Delta
│       ├── disambig_analysis.py             # Failure mode analysis
│       └── generate_pareto.py               # Figure 5: Pareto TikZ
├── data/
│   ├── disambiguation/
│   │   ├── tasks_desktop.json               # ← Downloaded from S3
│   │   ├── tasks_mobile.json                # ← Downloaded from S3
│   │   └── images/                          # ← Downloaded from S3
│   ├── safety/
│   │   ├── tasks_desktop.json               # ← Downloaded from S3
│   │   ├── tasks_mobile.json                # ← Downloaded from S3
│   │   ├── images/                          # ← Downloaded from S3
│   │   ├── dataset/                         # Generated by dataset_pre_processing.py
│   │   └── pre_processed_images/            # Generated by images_pre_processing.py
│   └── analytics/                           # Generated by convert_results.py
├── sample_data/                             # Committed ready-to-run subset (see `--sample`)
│   ├── disambiguation/                      # tasks_*.json + images/
│   └── safety/                              # dataset/*.jsonl + images/
├── results/                                 # Generated by running benchmarks (see §3, §4)
│   ├── disambiguation/
│   └── safety/
│       └── {dataset_name}/
│           └── {model_name}/
│               ├── results.json
│               └── rendered/
├── tests/                                   # Unit tests (pytest tests/)
├── requirements.txt
├── test_all.sh                              # End-to-end smoke test
└── README.md
```

## 1. Setup

### Clone the repository

```bash
git clone <REPO_URL>
cd adepts
```

### Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Set your API keys

Set the key(s) for the provider(s) you plan to use:

```bash
export OPENAI_API_KEY="your-key-here"
export CLAUDE_API_KEY="your-key-here"
export GEMINI_API_KEY="your-key-here"
export HF_TOKEN="your-key-here"   # Qwen models, via the Hugging Face Inference Providers router
```

### Supported models

The seven models evaluated in the ADEPTS-BENCH paper (§4.1) are listed below. To add a new model, add an entry to the `ModelType` enum in `code/llm_client.py`.

| Model                                     | Provider     |
| ----------------------------------------- | ------------ |
| `claude-opus-4-7`                         | Anthropic    |
| `gpt-5.4`                                 | OpenAI       |
| `gemini-3.1-pro-preview`                  | Google       |
| `gemini-2.5-computer-use-preview-10-2025` | Google       |
| `Qwen/Qwen3-VL-235B-A22B-Instruct`        | Hugging Face |
| `Qwen/Qwen3-VL-8B-Instruct`               | Hugging Face |
| `Qwen/Qwen3-VL-4B-Instruct`               | Hugging Face |

The `Qwen/` models are called through the [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers) router (OpenAI-compatible), authenticated with `HF_TOKEN`. Each model is pinned to a provider automatically (`4B`/`8B` → featherless-ai, `235B-A22B` → novita); override for all Qwen models with `HF_PROVIDER=<name>`, or set `HF_PROVIDER=""` to use the router's automatic selection. Note: the paper evaluated the `235B-A22B-FP8` checkpoint, which has no Hugging Face provider, so the non-FP8 `Qwen/Qwen3-VL-235B-A22B-Instruct` build is used here instead (self-host via vLLM if you need the exact FP8 weights).

## Quick Start on Sample Data (no download)

A tiny, ready-to-run subset of both benchmarks is committed under `sample_data/`
(3 tasks per platform, with downsized screenshots) so you can try the code
without AWS credentials or the full S3 download / pre-processing. You only need
an API key for the model you want to evaluate.

```bash
# Safety benchmark on the sample dataset:
python code/safety/safety_benchmark.py --models gpt-5.4 --sample

# Disambiguation benchmark on the sample dataset:
python code/disambiguation/disambiguation_benchmark.py \
    --models gpt-5.4 --reformat-model gpt-5.4 --check-questions-model gpt-5.4 --sample
```

`--sample` points the benchmark at `sample_data/` instead of `data/`. It composes
with the usual filters (`--platform`, `--scenario`, etc.). The sample includes all
model-resolution variants (vanilla/`_claude`/`_gpt`/`_qwen`), so any supported model
works out of the box. Results still go to `results/`.

> The sample is for trying/smoke-testing the harness, not for reported metrics —
> use the full datasets (below) for those. To regenerate `sample_data/`, see
> `code/make_sample_data.py`.

## 2. Downloading Data

Task definitions and screenshots are stored in S3. AWS credentials and a dataset explorer are available in this [Colab notebook](https://colab.research.google.com/drive/1i8VzDR-ym4tEUja5EGyZ4xI83E6Bku5t?usp=sharing).

```bash
# AWS credentials are in the Colab notebook linked above
export AWS_KEY_ID=<YOUR_ACCESS_KEY_ID>
export AWS_SECRET_KEY=<YOUR_SECRET_ACCESS_KEY>

# Download all benchmarks (default):
python code/download_data.py

# Download only disambiguation data:
python code/download_data.py --benchmark disambiguation

# Download only safety data:
python code/download_data.py --benchmark safety
```

## 3. Safety Benchmark

Evaluates whether models can identify unsafe actions during phone UI navigation tasks.

### 3.1. Pre-process images

Generate resolution variants for the safety benchmark:

```bash
# Test mode:
python code/safety/images_pre_processing.py --test --test-samples 5

# Full mode:
python code/safety/images_pre_processing.py
```

This creates six folders under `data/safety/pre_processed_images/` with original, Claude (1280x720), and GPT (height=768) resolution variants for both desktop and mobile images.

### 3.2. Build dataset

Generate JSONL evaluation files from the pre-processed images and task definitions:

```bash
# Test mode:
python code/safety/dataset_pre_processing.py --test --test-samples 5

# Full mode:
python code/safety/dataset_pre_processing.py
```

This creates 32 JSONL files under `data/safety/dataset/` covering all combinations of platform (desktop/mobile), scenario (benign/malicious), resolution variant (original/claude/gpt/qwen), and unsafe status (with/without).

### 3.3. Run the benchmark

```bash
# Test mode, single model:
python code/safety/safety_benchmark.py --models gpt-5.4 --test --test-samples 5

# Multiple models, filter by platform/scenario/unsafe:
python code/safety/safety_benchmark.py --models gpt-5.4 claude-opus-4-7 gemini-3.1-pro-preview gemini-2.5-computer-use-preview-10-2025 --platform desktop mobile --scenario benign malicious --unsafe-status with --test --test-samples 2

# With rendered output images:
python code/safety/safety_benchmark.py --models gpt-5.4 --platform desktop --scenario benign --unsafe-status with --test --test-samples 10 --render

# Full dataset, multiple models:
python code/safety/safety_benchmark.py --models gpt-5.4 claude-opus-4-7 gemini-3.1-pro-preview gemini-2.5-computer-use-preview-10-2025

# Pass@k worst-case analysis (K runs per sample, temperature=0.1):
python code/safety/safety_benchmark.py --models gpt-5.4 --runs 10 --test --test-samples 5
```

Results are saved to `results/safety/{dataset_name}/{model_name}/results.json`. Rendered images (with `--render`) are saved alongside in a `rendered/` folder.

Each model automatically runs on its matching dataset variant (GPT → `_gpt`, Claude → `_claude`, Qwen → `_qwen`, others → vanilla).

### 3.4. Print results

Print a summary table from previously saved results:

```bash
python code/safety/print_results.py
```

### 3.5. Options reference

| Script         | Flag                      | Description                           | Default                  |
| -------------- | ------------------------- | ------------------------------------- | ------------------------ |
| Pre-processing | `--data-dir`              | Directory containing safety data      | `./data/safety`          |
| Pre-processing | `--output-dir`            | Directory to write output JSONL files | `./data/safety/dataset`  |
| Pre-processing | `--test`                  | Enable test mode (fewer tasks)        | off                      |
| Pre-processing | `--test-samples N`        | Number of tasks in test mode          | `5`                      |
| Benchmark      | `--models`                | Model(s) to evaluate (required)       | —                        |
| Benchmark      | `--platform`              | Platform filter(s)                    | `all`                    |
| Benchmark      | `--scenario`              | Scenario filter(s)                    | `all`                    |
| Benchmark      | `--unsafe-status`         | Unsafe status filter(s)               | `all`                    |
| Benchmark      | `--input`                 | JSONL file or directory               | `./data/safety/dataset`  |
| Benchmark      | `--sample`                | Use committed `sample_data/` (no S3)  | off                      |
| Benchmark      | `--output-dir`            | Directory for result JSON files       | `./results/safety`       |
| Benchmark      | `--api-key`               | OpenAI API key                        | `OPENAI_API_KEY` env var |
| Benchmark      | `--claude-api-key`        | Anthropic API key                     | `CLAUDE_API_KEY` env var |
| Benchmark      | `--base-url`              | Custom API base URL                   | none                     |
| Benchmark      | `--max-workers`           | Concurrent API calls                  | `8`                      |
| Benchmark      | `--request-delay`         | Seconds between API requests          | `0`                      |
| Benchmark      | `--max-tokens`            | Max output tokens for model response  | `32768`                  |
| Benchmark      | `--temperature`           | Sampling temperature override         | per-model default        |
| Benchmark      | `--test`                  | Enable test mode (fewer samples)      | off                      |
| Benchmark      | `--test-samples N`        | Number of samples in test mode        | `1`                      |
| Benchmark      | `--render`                | Render actions on screenshots         | off                      |
| Benchmark      | `--serial`                | Run everything sequentially           | off                      |
| Benchmark      | `--runs K`                | Runs per sample for pass@k analysis   | `1`                      |
| Print results  | `--results-dir`           | Results directory to read from        | `./results/safety`       |

Concurrency is automatically tuned per provider (Claude: 2, Gemini: 4, GPT/Qwen: full `--max-workers`). For additional throttling:

```bash
# Add delay between requests:
python code/safety/safety_benchmark.py --models gpt-5.4 --request-delay 2 --test --test-samples 10

# Fully sequential (no parallelism at all):
python code/safety/safety_benchmark.py --models gpt-5.4 --serial --test --test-samples 5
```

## 4. Disambiguation Benchmark

Evaluates whether models know when to ask clarification questions during phone UI navigation tasks.

### Run the benchmark

```bash
# Test mode, desktop only:
python code/disambiguation/disambiguation_benchmark.py \
    --models gpt-5.4 \
    --reformat-model gpt-5.4 \
    --check-questions-model gpt-5.4 \
    --platform desktop \
    --test --test-samples 5

# Full dataset, multiple models, both platforms:
python code/disambiguation/disambiguation_benchmark.py \
    --models gpt-5.4 claude-opus-4-7 \
    --reformat-model gpt-5.4 \
    --check-questions-model gpt-5.4
```

Results are saved to `results/disambiguation/`. Re-running with existing results will prompt to resume (reuse cached generations) or start fresh.

### Options

| Flag                      | Description                                    | Default                    |
| ------------------------- | ---------------------------------------------- | -------------------------- |
| `--models`                | Model(s) to evaluate (required)                | —                          |
| `--reformat-model`        | Model for reformatting responses (required)    | —                          |
| `--check-questions-model` | Model for checking question matches (required) | —                          |
| `--platform`              | Platform(s) to evaluate                        | `all`                      |
| `--data-dir`              | Directory containing dataset files             | `./data/disambiguation`    |
| `--sample`                | Use committed `sample_data/` (no S3 download)  | off                        |
| `--output-dir`            | Directory to save results                      | `./results/disambiguation` |
| `--api-key`               | OpenAI API key                                 | `OPENAI_API_KEY` env var   |
| `--claude-api-key`        | Anthropic API key                              | `CLAUDE_API_KEY` env var   |
| `--base-url`              | Custom API base URL                            | none                       |
| `--test`                  | Enable test mode (fewer samples)               | off                        |
| `--test-samples N`        | Number of samples in test mode                 | `5`                        |

## 5. Analyze Results

After running either or both benchmarks:

```bash
# Run all analyses (converts results + runs all scripts):
python code/analytics/analyze.py

# Safety analyses only:
python code/analytics/analyze.py safety

# Disambiguation analyses only:
python code/analytics/analyze.py disambiguation
```

Individual scripts can also be run directly (after `python code/analytics/convert_results.py`):

| Script | Output |
| --- | --- |
| `safety_main.py` | Table 2: TSR, ASR, FRR, ADEPTS Score |
| `safety_ablation.py` | Table 3: Tool ablation |
| `safety_passk.py` | Table 7: Pass@k worst-case |
| `safety_paired.py` | Table 8: Paired 2x2 |
| `safety_per_threat.py` | Table 10: Per-threat ASR |
| `safety_failure_spectrum.py` | Figure 3: Failure hierarchy |
| `generate_pareto.py` | Figure 5: Pareto frontier |
| `disambig_main.py` | Table 4: F1 and Delta |
| `disambig_analysis.py` | Failure mode analysis |

## 6. Utilities

### Reset

Remove generated data, pre-processing, and/or results:

```bash
python code/reset.py              # Reset everything (interactive prompt)
python code/reset.py data         # Reset downloaded data only
python code/reset.py preprocess   # Reset pre-processed images and datasets
python code/reset.py results      # Reset benchmark results
python code/reset.py all          # Reset everything
```

### Unit tests

Run the grading and parsing test suite:

```bash
pytest tests/ -v
```

### Smoke test

Run all commands end-to-end and verify outputs:

```bash
# Default model (gpt-5.4):
./test_all.sh

# Custom model:
./test_all.sh gpt-5.4
```

Requires `AWS_KEY_ID`, `AWS_SECRET_KEY` for data download, and the appropriate API key for the selected model. Cleans up all generated files after testing.

## License

See [LICENSE](LICENSE) for details.
