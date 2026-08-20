#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#
# Smoke test every Python command in the ADEPTS codebase.
# Runs each script in test mode, verifies outputs, then cleans up.
#
# Usage:
#   ./test_all.sh              # Uses gpt-5.4
#   ./test_all.sh gpt-5.4
#
# Requires:
#   - AWS_KEY_ID and AWS_SECRET_KEY for data download
#   - Appropriate API key for the selected model (OPENAI_API_KEY, CLAUDE_API_KEY, etc.)

set -euo pipefail

MODEL="${1:-gpt-5.4}"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
PASS=0
FAIL=0

run_test() {
    local name="$1"
    shift
    echo ""
    echo "================================================================"
    echo "TEST: $name"
    echo "CMD:  $*"
    echo "================================================================"
    if "$@"; then
        echo -e "${GREEN}PASS${NC}: $name"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}FAIL${NC}: $name"
        FAIL=$((FAIL + 1))
    fi
}

check_exists() {
    local path="$1"
    local desc="$2"
    if [ -e "$path" ]; then
        echo -e "${GREEN}  OK${NC}: $desc exists ($path)"
    else
        echo -e "${RED}  MISSING${NC}: $desc ($path)"
        FAIL=$((FAIL + 1))
    fi
}

echo "========================================"
echo "  ADEPTS Smoke Test Suite"
echo "  Model: $MODEL"
echo "========================================"

# -------------------------------------------------------------------
# Pre-flight checks
# -------------------------------------------------------------------
if [ -z "${AWS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_KEY:-}" ]; then
    echo -e "${RED}ERROR${NC}: AWS_KEY_ID and AWS_SECRET_KEY must be set"
    exit 1
fi

# -------------------------------------------------------------------
# 1. Download data
# -------------------------------------------------------------------
run_test "Download safety data" \
    python code/download_data.py --benchmark safety

check_exists "data/safety/tasks_desktop.json" "Safety desktop tasks"
check_exists "data/safety/tasks_mobile.json" "Safety mobile tasks"
check_exists "data/safety/images" "Safety images directory"

run_test "Download disambiguation data" \
    python code/download_data.py --benchmark disambiguation

check_exists "data/disambiguation/tasks_desktop.json" "Disambiguation desktop tasks"
check_exists "data/disambiguation/tasks_mobile.json" "Disambiguation mobile tasks"

# -------------------------------------------------------------------
# 2. Image pre-processing
# -------------------------------------------------------------------
rm -rf data/safety/pre_processed_images

run_test "Image pre-processing (test)" \
    python code/safety/images_pre_processing.py --test --test-samples 12  # 6 variants per task × 2 tasks

check_exists "data/safety/pre_processed_images/desktop_original" "Desktop original"
check_exists "data/safety/pre_processed_images/mobile_claude" "Mobile claude"
check_exists "data/safety/pre_processed_images/desktop_gpt" "Desktop GPT"

# -------------------------------------------------------------------
# 3. Dataset pre-processing
# -------------------------------------------------------------------
rm -rf data/safety/dataset

run_test "Dataset pre-processing (test)" \
    python code/safety/dataset_pre_processing.py --test --test-samples 2

check_exists "data/safety/dataset" "Dataset directory"
DATASET_COUNT=$(find data/safety/dataset -name "*.jsonl" 2>/dev/null | wc -l | tr -d ' ')
echo "  Found $DATASET_COUNT JSONL files"

# -------------------------------------------------------------------
# 4. Safety benchmark
# -------------------------------------------------------------------
rm -rf results/safety

run_test "Safety benchmark (test, $MODEL)" \
    python code/safety/safety_benchmark.py \
        --models $MODEL \
        --platform desktop \
        --scenario benign \
        --unsafe-status with \
        --test --test-samples 1 \
        --serial

check_exists "results/safety" "Safety results directory"

# -------------------------------------------------------------------
# 5. Print results
# -------------------------------------------------------------------
run_test "Print safety results" \
    python code/safety/print_results.py

# -------------------------------------------------------------------
# 6. Disambiguation benchmark
# -------------------------------------------------------------------
rm -rf results/disambiguation

run_test "Disambiguation benchmark (test, $MODEL)" \
    python code/disambiguation/disambiguation_benchmark.py \
        --models $MODEL \
        --reformat-model $MODEL \
        --check-questions-model $MODEL \
        --test --test-samples 1

check_exists "results/disambiguation" "Disambiguation results directory"

# -------------------------------------------------------------------
# 7. Analytics
# -------------------------------------------------------------------
rm -rf data/analytics

run_test "Convert results for analytics" \
    python code/analytics/convert_results.py

check_exists "data/analytics/results_all.csv" "Safety aggregate CSV"
check_exists "data/analytics/per_task_first_run.json" "Per-task JSON"
check_exists "data/analytics/tasks_mobile_gt.json" "Mobile ground truth"

run_test "Safety main analytics" \
    python code/analytics/safety_main.py

run_test "Disambiguation main analytics" \
    python code/analytics/disambig_main.py

# -------------------------------------------------------------------
# 8. Unit tests
# -------------------------------------------------------------------
run_test "Unit tests" \
    python -m pytest tests/ --tb=short -q

# -------------------------------------------------------------------
# 9. Reset (clean up everything generated by this test)
# -------------------------------------------------------------------
run_test "Reset pre-processing" \
    bash -c 'echo y | python code/reset.py preprocess'

run_test "Reset results" \
    bash -c 'echo y | python code/reset.py results'

rm -rf data/analytics

# Verify cleanup
if [ -d "results" ]; then
    echo -e "${RED}  FAIL${NC}: Results directory still exists after reset"
    FAIL=$((FAIL + 1))
else
    echo -e "${GREEN}  OK${NC}: Results cleaned up"
fi

if [ -d "data/safety/pre_processed_images" ] || [ -d "data/safety/dataset" ]; then
    echo -e "${RED}  FAIL${NC}: Pre-processed data still exists after reset"
    FAIL=$((FAIL + 1))
else
    echo -e "${GREEN}  OK${NC}: Pre-processed data cleaned up"
fi

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
echo ""
echo "========================================"
printf "  Results: ${GREEN}%d passed${NC}, ${RED}%d failed${NC}\n" "$PASS" "$FAIL"
echo "========================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
