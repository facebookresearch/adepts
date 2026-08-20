# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Disambiguation Adepts Benchmark — orchestration and CLI.

Evaluates multimodal LLMs on their ability to identify when clarification
questions should be asked during phone UI navigation tasks.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python code/disambiguation/disambiguation_benchmark.py \
        --models gpt-5.4 --reformat-model gpt-5.4 --check-questions-model gpt-5.4 --test
"""

import argparse
import asyncio
import base64
import json
import os
import random
import re
import sys
import time
import traceback
from io import BytesIO

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PIL import Image

from llm_client import ModelType, init_clients, chat_completion, resolve_models, check_models_have_clients
from disambiguation.disambiguation_grading import (
    BenchmarkType, PromptType, BENCHMARK_CONFIGS, METRIC_NAMES,
    get_metrics, analyze_dataset,
    benchmark_evaluation, plot_all, plot_all_histograms, save_figure_to_bytes,
)
from disambiguation.disambiguation_prompts import get_prompt, reformat, check_questions
from utils import log as _log, reset_log


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
TEST_MODE = False
TEST_SAMPLES = 5

_data_dir: str = "."
_output_dir: str = "./results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_duration(latency_ms):
    latency = latency_ms / 1000
    hours = int(latency // 3600)
    minutes = int((latency % 3600) // 60)
    seconds = int(latency % 60)
    milliseconds = int(latency_ms % 1000)
    return f"{hours}h {minutes}m {seconds}s {milliseconds}ms"


def get_key(model, mode, sample_index):
    return f"{model}_{mode.name}_{sample_index}"


def get_benchmark_generations(generations_data_dict, model, mode):
    prefix = f"{model}_{mode.name}"
    return {k: v for k, v in generations_data_dict.items() if k.startswith(prefix)}


def extract_first_curly_block(text):
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
        return json.dumps(obj)
    except json.JSONDecodeError:
        return None


def load_image(path: str) -> Image.Image:
    full_path = os.path.join(_data_dir, path) if not os.path.isabs(path) else path
    return Image.open(full_path)


def image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def detect_mime_type(b64_data: str) -> str:
    header = base64.b64decode(b64_data[:32])
    if header[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def save_to_output(run_dir, name, data, fmt="json"):
    path = os.path.join(run_dir, f"{name}.{fmt}")
    if fmt == "json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        with open(path, "wb") as f:
            f.write(data)
    _log(f"  Saved: {path}")


def get_models_and_modes_from_dict(data_dict_keys):
    model_value_to_enum = {m.value: m for m in ModelType}
    models_to_eval = set()
    modes_to_eval = set()
    for key in data_dict_keys:
        match = re.match(r"^(.*?)_([A-Z][A-Z0-9_]+?)(?:_(\d+))?$", key)
        if match:
            model_name_str = match.group(1)
            mode_name = match.group(2)
            model_enum = model_value_to_enum.get(model_name_str)
            if model_enum is not None:
                models_to_eval.add(model_enum)
            else:
                _log(f"Warning: ModelType not found for '{model_name_str}'")
            try:
                mode_enum = PromptType[mode_name]
                modes_to_eval.add(mode_enum)
            except KeyError:
                _log(f"Warning: PromptType not found for '{mode_name}'")
    return list(models_to_eval), list(modes_to_eval)


def check_missing_generations(generations_dict, dataset):
    models_to_check, modes_to_check = get_models_and_modes_from_dict(generations_dict.keys())
    keys = []
    for mode in modes_to_check:
        for model in models_to_check:
            for sample_index in range(len(dataset)):
                key = get_key(model.value, mode, sample_index)
                if key not in generations_dict:
                    keys.append(key)
    return keys


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _normalize_score(val):
    """Convert score values like 2, '2' to int. Returns None on failure."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


async def generation_process(response_raw, reformat_model):
    response_curly = extract_first_curly_block(response_raw)

    response_json = None
    if response_curly is not None:
        try:
            parsed = json.loads(response_curly)
            # Only accept the direct parse if it has a top-level "clarifications"
            # key. A native tool-call (e.g. Qwen's {"name": ..., "arguments": {...}})
            # is valid JSON but has no "clarifications" key -- accepting it would
            # KeyError / silently score empty. Fall through to reformat instead.
            if isinstance(parsed, dict) and "clarifications" in parsed:
                response_json = parsed
        except json.JSONDecodeError:
            pass

    if response_json is None:
        response_raw = await reformat(response_raw, reformat_model)
        response_curly = extract_first_curly_block(response_raw)
        if response_curly is None:
            raise ValueError("No curly block found after reformatting")
        response_json = json.loads(response_curly)

    for c in response_json.get("clarifications", []):
        if c and "obviousness_score" in c:
            c["obviousness_score"] = _normalize_score(c["obviousness_score"])
        if c and "consequence_score" in c:
            c["consequence_score"] = _normalize_score(c["consequence_score"])

    return response_curly, response_json.get("clarifications", [])


async def model_generation(sample, model, prompt):
    image_paths = sample["image_paths"]
    content = []

    for idx, image_path in enumerate(image_paths, start=1):
        img = load_image(image_path)
        b64 = image_to_base64(img)
        mime = detect_mime_type(b64)
        content.append({"type": "text", "text": f"Screen {idx}"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    return await chat_completion(model, messages, max_tokens=30000)


async def generate_benchmark(benchmark_dataset, model, mode, sample_indices_to_generate, reformat_model):
    _log(f"    Generating {model.value} / {mode.name} ({len(sample_indices_to_generate)} samples)")
    benchmark_generation = {}

    abort = False
    for i in sample_indices_to_generate:
        if abort:
            break
        sample = benchmark_dataset[i]
        for attempt in range(MAX_RETRIES):
            try:
                goal = sample["adg"]
                key = get_key(model.value, mode, i)
                prompt = get_prompt(goal, mode)
                response_raw = await model_generation(sample, model, prompt)
                response_curly, gen_clarifications = await generation_process(response_raw, reformat_model)
                _log(f"    [sample {i} / {model.value} / {mode.name}]: {response_curly}")

                benchmark_generation[key] = {
                    "gen_clarifications": gen_clarifications,
                    "error_type": None,
                    "id": sample["id"],
                }
                break
            except Exception as e:
                error_type = str(type(e).__name__)
                error_msg = str(e).lower()
                _log(f"    Error [{model.value} / {mode.name} / sample {i}] (attempt {attempt + 1}/{MAX_RETRIES}): {error_type}: {e}")
                non_retryable = (
                    "credit balance" in error_msg
                    or "insufficient_quota" in error_msg
                    or "authentication" in error_msg
                    or "invalid_api_key" in error_msg
                    or error_type == "UnidentifiedImageError"
                )
                if non_retryable:
                    if error_type == "UnidentifiedImageError":
                        _log(f"    Failed image paths: {sample.get('image_paths', [])}")
                    else:
                        _log(f"    Non-retryable error, skipping remaining samples for {model.value}")
                        abort = True
                    break
                traceback.print_exc()
                if attempt < MAX_RETRIES - 1:
                    _log(f"    Retrying [{model.value} / {mode.name} / sample {i}] (attempt {attempt + 2}/{MAX_RETRIES})...")
                    await asyncio.sleep(5)
                else:
                    _log(f"    Failed [{model.value} / {mode.name} / sample {i}] after {MAX_RETRIES} retries")

    return benchmark_generation


async def generate_all_benchmarks(benchmark_dataset, reformat_model, selected_models=None, cached_generations=None):
    modes = list(PromptType)
    models = selected_models if selected_models is not None else list(ModelType)
    random.shuffle(models)
    sample_count = TEST_SAMPLES if TEST_MODE else len(benchmark_dataset)

    cached = cached_generations or {}
    tasks = []
    for mode in modes:
        for model in models:
            all_indices = list(range(sample_count))
            if cached:
                all_indices = [i for i in all_indices if get_key(model.value, mode, i) not in cached]
            if not all_indices:
                _log(f"  Skipped: {model.value} / {mode.name} (all {sample_count} cached)")
                continue
            _log(f"  Queued: {model.value} / {mode.name} ({len(all_indices)} to generate, {sample_count - len(all_indices)} cached)")
            tasks.append(generate_benchmark(benchmark_dataset, model, mode, all_indices, reformat_model))

    start_time = time.time()
    benchmark_generations = dict(cached)
    if tasks:
        results = await asyncio.gather(*tasks)
        for result in results:
            benchmark_generations |= result
        _log(f"  Completed {len(tasks)} model/mode combos in {format_duration((time.time() - start_time) * 1000)}")
    else:
        _log("  All generations cached, nothing to generate")
    return benchmark_generations


async def retry_missing_generations(generations_data_dict, benchmark_dataset, reformat_model):
    missing_keys = check_missing_generations(generations_data_dict, benchmark_dataset)
    if len(missing_keys) > 0:
        _log(f"  Found {len(missing_keys)} missing keys")

    tasks = []
    missing_models, missing_modes = get_models_and_modes_from_dict(missing_keys)
    for mode in missing_modes:
        for model in missing_models:
            missing_for_combo = [item for item in missing_keys if model.value in item and mode.name in item]
            missing_indices = [int(m.group(1)) for s in missing_for_combo if (m := re.search(r"_(\d+)$", s))]
            if missing_indices:
                tasks.append(generate_benchmark(benchmark_dataset, model, mode, missing_indices, reformat_model))

    missing_benchmark_generations = {}
    if tasks:
        results = await asyncio.gather(*tasks)
        for result in results:
            missing_benchmark_generations |= result
    _log(f"  Regenerated {len(missing_benchmark_generations)} samples")
    return missing_benchmark_generations


# ---------------------------------------------------------------------------
# Question Matching
# ---------------------------------------------------------------------------

async def _match_single_key(key, value, benchmark_dataset, param_check_model):
    sample_id = value["id"]
    gen_clarifications = value["gen_clarifications"]

    for i in range(len(gen_clarifications)):
        generated_question = gen_clarifications[i]
        if isinstance(generated_question, str):
            generated_question = {"question": generated_question}
            gen_clarifications[i] = generated_question
        elif not isinstance(generated_question, dict):
            continue

        generated_question_text = generated_question.get("question")
        if not generated_question_text:
            generated_question["match"] = False
            generated_question["match_ground_text"] = None
            generated_question["match_ground_id"] = sample_id
            continue

        question_matched = False
        match_ground_text = None
        sample = benchmark_dataset[sample_id]
        goal = sample["adg"]

        for ground_question in sample["clarifications"]:
            ground_question_text = ground_question["question"]
            if ground_question_text is None:
                continue
            match = False
            for attempt in range(MAX_RETRIES):
                try:
                    match, _ = await check_questions(ground_question_text, generated_question_text, param_check_model, goal)
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        _log(f"    Failed [{key} / q{i}] after {MAX_RETRIES} retries: {type(e).__name__}")

            _log(f"    [sample {sample_id} / {key} / q{i}] goal: {goal} | {ground_question_text} vs {generated_question_text} | match: {match}")
            if match:
                question_matched = True
                match_ground_text = ground_question_text

        generated_question["match"] = question_matched
        generated_question["match_ground_text"] = match_ground_text
        generated_question["match_ground_id"] = sample_id

    return key, gen_clarifications


async def match_questions_to_ground_truth(scored_generations_data_dict, benchmark_dataset, param_check_model):
    tasks = [_match_single_key(key, value, benchmark_dataset, param_check_model) for key, value in scored_generations_data_dict.items()]
    results = await asyncio.gather(*tasks)
    for key, gen_clarifications in results:
        scored_generations_data_dict[key]["gen_clarifications"] = gen_clarifications


# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------

async def run_evaluations(benchmark_dataset, scored_generations_data_dict):
    start_time = time.time()
    metrics_by_name = {name: {} for name in METRIC_NAMES if name != "delta"}
    metrics_by_name_no_threshold = {name: {} for name in METRIC_NAMES}

    models_to_eval, modes_to_eval = get_models_and_modes_from_dict(scored_generations_data_dict.keys())
    _log(f"  Models: {[m.value for m in models_to_eval]}")
    _log(f"  Modes: {[m.name for m in modes_to_eval]}")

    for model in models_to_eval:
        for mode in modes_to_eval:
            label = f"{model.value}_{mode.name}"
            _log(f"  {'─' * 60}")
            _log(f"  Evaluating: {label}")

            gens = get_benchmark_generations(scored_generations_data_dict, model.value, mode)
            score_ground_thresholds = [0] if mode == PromptType.WITH_SCORE else [0, 1, 2, 3, 4]

            results_array = []
            for threshold in score_ground_thresholds:
                result = await benchmark_evaluation(benchmark_dataset, gens, model, mode, threshold, get_key)
                results_array.append(result)
                if "NO_SCORE" not in mode.name:
                    _log(f"    threshold={threshold}, errors={result['errors']}")

            metrics = get_metrics(results_array)
            if mode != PromptType.WITH_SCORE:
                _log(f"  Done: {label}")
                for name in METRIC_NAMES:
                    if name == "delta":
                        continue
                    metrics_by_name[name][label] = metrics[name]
            else:
                for name in METRIC_NAMES:
                    metrics_by_name_no_threshold[name][model.value] = metrics[name]

    _log(f"  Latency for evaluation: {format_duration((time.time() - start_time) * 1000)}")
    return metrics_by_name, metrics_by_name_no_threshold


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(dataset_file_name, benchmark_name, run_dir, reformat_model, param_check_model, selected_models=None, cached_generations=None):
    _log(f"Load dataset ({dataset_file_name})")
    dataset_path = os.path.join(_data_dir, dataset_file_name)
    with open(dataset_path, "r") as f:
        benchmark_dataset = json.load(f)
    _log(f"  Loaded {len(benchmark_dataset)} samples")

    _log("Analyze dataset distribution")
    analyze_dataset(benchmark_dataset)

    _log("Generate model responses")
    benchmark_generations = await generate_all_benchmarks(benchmark_dataset, reformat_model, selected_models, cached_generations)

    _log("Save generations")
    save_to_output(run_dir, f"{benchmark_name}_generations", benchmark_generations)

    _log("Retry any failed generations")
    expected_sample_count = TEST_SAMPLES if TEST_MODE else len(benchmark_dataset)
    missing = await retry_missing_generations(benchmark_generations, benchmark_dataset[:expected_sample_count], reformat_model)

    if missing:
        benchmark_generations |= missing
        _log(f"Save patched generations ({len(missing)} retried)")
        save_to_output(run_dir, f"{benchmark_name}_generations_patched", benchmark_generations)
    else:
        _log("No missing generations to patch")

    _log("Match generated questions against ground truth")
    start_time = time.time()
    scored_generations = json.loads(json.dumps(benchmark_generations))
    await match_questions_to_ground_truth(scored_generations, benchmark_dataset, param_check_model)
    _log(f"  Completed in {format_duration((time.time() - start_time) * 1000)}")

    _log("Save scored generations")
    save_to_output(run_dir, f"{benchmark_name}_scored_generations", scored_generations)

    _log("Run evaluations")
    eval_dataset = benchmark_dataset[:TEST_SAMPLES] if TEST_MODE else benchmark_dataset
    metrics_by_name, metrics_by_name_no_threshold = await run_evaluations(eval_dataset, scored_generations)

    _log("Print and save final results")
    label = benchmark_name.upper()

    _log(f"\n{'=' * 80}")
    _log(f"Final results - {label} without score generation")
    _log(f"{'=' * 80}")
    fig = plot_all(metrics_by_name)
    save_to_output(run_dir, f"{benchmark_name}_metrics_without_scores", save_figure_to_bytes(fig), fmt="png")
    lines = []
    for metric_name, series_dict in metrics_by_name.items():
        lines.append(f"{metric_name.replace('_', ' ').title()}:")
        for lbl in sorted(series_dict.keys()):
            vals = ", ".join(f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in series_dict[lbl])
            lines.append(f"  {lbl}: [{vals}]")
        lines.append("")
    save_to_output(run_dir, f"{benchmark_name}_metrics_without_scores", "\n".join(lines).encode("utf-8"), fmt="txt")

    _log(f"\n{'=' * 80}")
    _log(f"Final results - {label} with score generation")
    _log(f"{'=' * 80}")
    fig = plot_all_histograms(metrics_by_name_no_threshold)
    save_to_output(run_dir, f"{benchmark_name}_metrics_with_scores", save_figure_to_bytes(fig), fmt="png")
    lines = []
    for metric_name, model_counts in metrics_by_name_no_threshold.items():
        lines.append(f"{metric_name.replace('_', ' ').title()}:")
        for mdl in sorted(model_counts.keys()):
            val = model_counts[mdl]
            lines.append(f"  {mdl}: {val:.4f}" if isinstance(val, (int, float)) else f"  {mdl}: {val}")
        lines.append("")
    save_to_output(run_dir, f"{benchmark_name}_metrics_with_scores", "\n".join(lines).encode("utf-8"), fmt="txt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main():
    global _data_dir, _output_dir, TEST_MODE, TEST_SAMPLES

    parser = argparse.ArgumentParser(description="Disambiguation Adepts Benchmark")
    parser.add_argument("--data-dir", default="./data/disambiguation", help="Directory containing dataset JSON files and screenshots")
    parser.add_argument("--sample", action="store_true", help="Run on the committed sample dataset (./sample_data/disambiguation); no S3 download needed")
    parser.add_argument("--output-dir", default="./results/disambiguation", help="Directory to save results")
    parser.add_argument("--models", nargs="+", required=True, help="Model(s) to evaluate (e.g. gpt-5.4 claude-opus-4-7)")
    parser.add_argument("--reformat-model", required=True, help="Model for reformatting responses")
    parser.add_argument("--check-questions-model", required=True, help="Model for checking question matches")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--claude-api-key", default=None, help="Anthropic API key (or set CLAUDE_API_KEY env var)")
    parser.add_argument("--base-url", default=None, help="Custom API base URL (for OpenRouter, local models, etc.)")
    parser.add_argument("--platform", nargs="+", default=["all"], choices=["desktop", "mobile", "all"], help="Platform(s) to evaluate")
    parser.add_argument("--test", action="store_true", help="Run in test mode with fewer samples")
    parser.add_argument("--test-samples", type=int, default=1, help="Number of samples in test mode")
    args = parser.parse_args()

    if args.sample and args.data_dir == "./data/disambiguation":
        args.data_dir = "./sample_data/disambiguation"
        _log(f"Sample mode: reading committed sample dataset from {args.data_dir}")

    selected_models = resolve_models(args.models)
    reformat_model = resolve_models([args.reformat_model])[0]
    check_questions_model = resolve_models([args.check_questions_model])[0]

    _data_dir = args.data_dir
    _output_dir = args.output_dir
    TEST_MODE = args.test
    TEST_SAMPLES = args.test_samples
    reset_log()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    claude_api_key = args.claude_api_key or os.environ.get("CLAUDE_API_KEY")
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    hf_api_key = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")

    if not api_key and not claude_api_key and not gemini_api_key and not hf_api_key:
        print("Error: set at least one of OPENAI_API_KEY, CLAUDE_API_KEY, GEMINI_API_KEY, or HF_TOKEN")
        return

    init_clients(
        openai_api_key=api_key,
        claude_api_key=claude_api_key,
        gemini_api_key=gemini_api_key,
        hf_api_key=hf_api_key,
        base_url=args.base_url,
    )

    check_models_have_clients(selected_models + [reformat_model, check_questions_model])

    run_dir = _output_dir
    os.makedirs(run_dir, exist_ok=True)
    _log(f"Results directory: {run_dir}")

    platforms = [bt for bt in BenchmarkType if "all" in args.platform or bt.value in args.platform]
    sample_count = TEST_SAMPLES if TEST_MODE else None

    # Check for cached generations (like safety benchmark)
    cached_by_benchmark: dict[str, dict] = {}
    total_cached = 0
    total_expected = 0
    for benchmark_type in platforms:
        name = benchmark_type.value
        config = BENCHMARK_CONFIGS[benchmark_type]
        dataset_path = os.path.join(_data_dir, config["dataset_file_name"])
        if not os.path.exists(dataset_path):
            continue
        with open(dataset_path) as f:
            n_tasks = len(json.load(f))
        n = sample_count if sample_count else n_tasks
        n_expected = n * len(selected_models) * len(PromptType)
        total_expected += n_expected

        for suffix in ["_scored_generations.json", "_generations_patched.json", "_generations.json"]:
            gen_path = os.path.join(run_dir, f"{name}{suffix}")
            if os.path.exists(gen_path):
                with open(gen_path) as f:
                    data = json.load(f)
                cached_by_benchmark[name] = data
                total_cached += len(data)
                break

    if total_cached > 0 and total_cached < total_expected:
        print(f"Found {total_cached}/{total_expected} cached generations. Resume or start fresh?")
        print("  [r] Resume (skip completed samples)")
        print("  [f] Fresh (delete existing results and re-run)")
        choice = input("Choice [r/f]: ").strip().lower()
        if choice == "f":
            for f in os.listdir(run_dir):
                fpath = os.path.join(run_dir, f)
                if os.path.isfile(fpath):
                    os.remove(fpath)
            cached_by_benchmark.clear()
            _log("Cleared existing results")
        else:
            _log(f"Resuming ({total_expected - total_cached} samples remaining)")
    elif total_cached >= total_expected and total_expected > 0:
        print(f"All {total_expected} generations already cached. Start fresh?")
        print("  [p] Print results only (re-run evaluation)")
        print("  [f] Fresh (delete existing results and re-run)")
        choice = input("Choice [p/f]: ").strip().lower()
        if choice == "f":
            for f in os.listdir(run_dir):
                fpath = os.path.join(run_dir, f)
                if os.path.isfile(fpath):
                    os.remove(fpath)
            cached_by_benchmark.clear()
            _log("Cleared existing results")

    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": [m.value for m in selected_models],
        "reformat_model": reformat_model.value,
        "check_questions_model": check_questions_model.value,
        "test_mode": TEST_MODE,
        "test_samples": TEST_SAMPLES if TEST_MODE else None,
        "score_ground_threshold": 0,
        "data_dir": _data_dir,
    }
    save_to_output(run_dir, "metadata", metadata)

    if TEST_MODE:
        _log(f"Running in TEST MODE ({TEST_SAMPLES} samples per model/mode)")

    for benchmark_type in platforms:
        config = BENCHMARK_CONFIGS[benchmark_type]
        _log(f"\n{'=' * 80}")
        _log(f"Running benchmark: {benchmark_type.value}")
        _log(f"{'=' * 80}\n")

        await run_pipeline(
            config["dataset_file_name"],
            benchmark_type.value,
            run_dir,
            reformat_model,
            check_questions_model,
            selected_models,
            cached_generations=cached_by_benchmark.get(benchmark_type.value),
        )

    _log("All benchmark evaluations complete")


if __name__ == "__main__":
    asyncio.run(main())
