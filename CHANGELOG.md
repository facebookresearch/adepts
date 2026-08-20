# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-20

Initial public release of ADEPTS-BENCH, accompanying the paper *ADEPTS-BENCH:
Measuring the Trustworthiness of Computer Use Agents across Devices*.

### Added

- **Safety benchmark** (`code/safety/`) — evaluation over 1,718 paired
  benign/malicious instances (859 pairs) across 10 threat categories, with
  image pre-processing, dataset construction, screenshot rendering, and a
  results summary table.
- **Disambiguation benchmark** (`code/disambiguation/`) — evaluation over 744
  tasks measuring whether agents seek clarification under ambiguous intent,
  with LLM-judge grading, metrics, prompt templates, and plotting.
- **Analytics suite** (`code/analytics/`) — scripts reproducing the paper's
  results: Table 2 (TSR, ASR, FRR, ADEPTS Score), Table 3 (tool ablation),
  Table 4 (F1 and Delta), Table 7 (pass@k worst case), Table 8 (paired 2x2),
  Table 10 (per-threat ASR), Figure 3 (failure spectrum), and Figure 5 (Pareto).
- **Multi-provider LLM client** (`code/llm_client.py`) — a shared interface
  covering the seven models evaluated in the paper across Anthropic, OpenAI,
  Google, and the Hugging Face Inference Providers router.
- **Data tooling** — S3 dataset download (`code/download_data.py`), sample-data
  regeneration (`code/make_sample_data.py`), and reset of generated artifacts
  (`code/reset.py`).
- **Sample data** (`sample_data/`) — a committed ready-to-run subset of both
  benchmarks (3 tasks per platform, downsized screenshots) usable via
  `--sample` with no download or credentials.
- **Tests** — a 54-case offline `pytest` suite under `tests/`, plus
  `test_all.sh` for an end-to-end smoke test of every CLI entry point.
- **Documentation and project files** — `README.md`, `LICENSE` (CC-BY-NC 4.0),
  `CONTRIBUTING.md`, and `CODE_OF_CONDUCT.md`.
