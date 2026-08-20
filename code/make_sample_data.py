# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Generate the committed sample_data/ directory.

sample_data/ is a tiny, ready-to-run subset of both benchmarks (a handful of
tasks per platform, with downsized screenshots) so people can try the code
without downloading the full datasets from S3. It is checked into the repo and
consumed via the `--sample` flag on each benchmark:

    python code/safety/safety_benchmark.py --models gpt-5.4 --sample
    python code/disambiguation/disambiguation_benchmark.py \
        --models gpt-5.4 --reformat-model gpt-5.4 --check-questions-model gpt-5.4 --sample

This script regenerates it end-to-end (maintainer tool; needs AWS creds):

    AWS_KEY_ID=<KEY> AWS_SECRET_KEY=<SECRET> \
        python code/make_sample_data.py --n 3 --bucket submission-2682

Pipeline:
  1. Download the first N tasks/platform + only their referenced images -> data/ (scratch).
  2. Run the existing safety pre-processing to build the JSONL variants.
  3. Downsize images and assemble the committed sample_data/ (image_size in the
     safety JSONL is left untouched, since grading normalizes by it).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import boto3
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import REPO_ROOT, log as _log, reset_log

REGION = "us-west-2"
CAP = 1280  # max image dimension in sample_data

# S3 object keys for the task files (flat names in the bucket).
DISAMBIG_TASKS = {"desktop": "disambiguation_desktop.json", "mobile": "disambiguation_mobile.json"}
SAFETY_TASKS = {"desktop": "safety_desktop.json", "mobile": "safety_mobile.json"}
DISAMBIG_PREFIX = "adepts/disambiguation/"
SAFETY_PREFIX = "adepts/safety/"
SAFETY_IMG_COLS = {
    "desktop": ["screenshot_renamed", "benign_screenshot_renamed"],
    "mobile": ["screenshot_external", "benign_screenshot_external"],
}

DATA_DIR = os.path.join(REPO_ROOT, "data")
SAMPLE_DIR = os.path.join(REPO_ROOT, "sample_data")


# ---------------------------------------------------------------------------
# Step 1: download first N tasks + referenced images into data/ (scratch)
# ---------------------------------------------------------------------------

def _strip_prefix(key: str, prefix: str) -> str:
    return key[len(prefix):] if key.startswith(prefix) else os.path.basename(key)


def _download_image(s3, bucket: str, key: str, local_path: str) -> None:
    if os.path.exists(local_path):
        return
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with open(local_path, "wb") as f:
        f.write(body)


def download(s3, bucket: str, n: int) -> None:
    _log(f"Downloading first {n} tasks/platform from s3://{bucket}", step=True)

    for platform, s3_key in DISAMBIG_TASKS.items():
        tasks = json.loads(s3.get_object(Bucket=bucket, Key=s3_key)["Body"].read())[:n]
        for task in tasks:
            new_paths = []
            for img_key in task.get("image_paths", []):
                rel = os.path.join("images", _strip_prefix(img_key, DISAMBIG_PREFIX))
                _download_image(s3, bucket, img_key, os.path.join(DATA_DIR, "disambiguation", rel))
                new_paths.append(rel)
            task["image_paths"] = new_paths
        out = os.path.join(DATA_DIR, "disambiguation", f"tasks_{platform}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(tasks, open(out, "w"), indent=2)
        _log(f"  disambiguation/{platform}: {len(tasks)} tasks")

    for platform, s3_key in SAFETY_TASKS.items():
        tasks = json.loads(s3.get_object(Bucket=bucket, Key=s3_key)["Body"].read())[:n]
        for task in tasks:
            for col in SAFETY_IMG_COLS[platform]:
                img_key = task.get(col)
                if isinstance(img_key, str) and img_key:
                    local = os.path.join(DATA_DIR, "safety", "images", _strip_prefix(img_key, SAFETY_PREFIX))
                    _download_image(s3, bucket, img_key, local)
        out = os.path.join(DATA_DIR, "safety", f"tasks_{platform}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(tasks, open(out, "w"), indent=2)
        _log(f"  safety/{platform}: {len(tasks)} tasks")


# ---------------------------------------------------------------------------
# Step 2: safety pre-processing (reuse existing scripts)
# ---------------------------------------------------------------------------

def preprocess_safety(n: int) -> None:
    _log("Running safety pre-processing", step=True)
    safety_data = os.path.join(DATA_DIR, "safety")
    code_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(
        [sys.executable, os.path.join(code_dir, "safety", "images_pre_processing.py"),
         "--data-dir", safety_data],
        check=True,
    )
    subprocess.run(
        [sys.executable, os.path.join(code_dir, "safety", "dataset_pre_processing.py"),
         "--data-dir", safety_data, "--test", "--test-samples", str(n)],
        check=True,
    )


# ---------------------------------------------------------------------------
# Step 3: downsize + assemble sample_data/
# ---------------------------------------------------------------------------

def _downsize(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    img = ImageOps.exif_transpose(Image.open(src))
    w, h = img.size
    scale = min(1.0, CAP / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    img = img.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img.save(dst, format="PNG", optimize=True)


def _find_image_content(sample: dict) -> dict | None:
    for msg in sample["messages"]:
        c = msg.get("content", {})
        if isinstance(c, dict) and c.get("content_type") == "multimodal_text_message_content":
            for item in c.get("content", []):
                if isinstance(item, dict) and item.get("content_type") == "image_message_content":
                    return item
    return None


def build_disambiguation(n: int) -> None:
    for platform in ("desktop", "mobile"):
        tasks = json.load(open(os.path.join(DATA_DIR, "disambiguation", f"tasks_{platform}.json")))[:n]
        for task in tasks:
            for rel in task.get("image_paths", []):  # images/<name>
                _downsize(os.path.join(DATA_DIR, "disambiguation", rel),
                          os.path.join(SAMPLE_DIR, "disambiguation", rel))
        out = os.path.join(SAMPLE_DIR, "disambiguation", f"tasks_{platform}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(tasks, open(out, "w"), indent=2)
        _log(f"  disambiguation/{platform}: {len(tasks)} tasks")


def build_safety(n: int) -> None:
    src_dir = os.path.join(DATA_DIR, "safety", "dataset")
    out_dir = os.path.join(SAMPLE_DIR, "safety", "dataset")
    os.makedirs(out_dir, exist_ok=True)
    for fn in sorted(os.listdir(src_dir)):
        if not fn.endswith(".jsonl"):
            continue
        out_samples = []
        with open(os.path.join(src_dir, fn)) as f:
            for line in f:
                if not line.strip() or len(out_samples) >= n:
                    continue
                s = json.loads(line)
                ic = _find_image_content(s)
                if ic:
                    old = ic["image_path"]  # data/safety/pre_processed_images/<variant>/<name>
                    variant, name = os.path.basename(os.path.dirname(old)), os.path.basename(old)
                    rel = os.path.join("sample_data", "safety", "images", variant, name)
                    abs_old = old if os.path.isabs(old) else os.path.join(REPO_ROOT, old)
                    _downsize(abs_old, os.path.join(REPO_ROOT, rel))
                    ic["image_path"] = rel
                out_samples.append(s)
        with open(os.path.join(out_dir, fn), "w") as f:
            for s in out_samples:
                f.write(json.dumps(s) + "\n")
    _log(f"  safety: {len(os.listdir(out_dir))} jsonl files")


def assemble(n: int) -> None:
    _log("Assembling downsized sample_data/", step=True)
    build_disambiguation(n)
    build_safety(n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the committed sample_data/ directory")
    parser.add_argument("--n", type=int, default=3, help="Tasks per platform (default: 3)")
    parser.add_argument("--bucket", default="submission-2682", help="S3 bucket holding the datasets")
    parser.add_argument("--skip-download", action="store_true", help="Reuse existing data/ (no S3 access)")
    args = parser.parse_args()

    reset_log()

    if not args.skip_download:
        key_id, secret = os.environ.get("AWS_KEY_ID"), os.environ.get("AWS_SECRET_KEY")
        if not key_id or not secret:
            raise SystemExit("Error: set AWS_KEY_ID and AWS_SECRET_KEY (or pass --skip-download)")
        s3 = boto3.client("s3", region_name=REGION,
                          aws_access_key_id=key_id, aws_secret_access_key=secret)
        download(s3, args.bucket, args.n)

    preprocess_safety(args.n)
    assemble(args.n)
    _log(f"Done. sample_data/ built with {args.n} tasks/platform.", step=True)


if __name__ == "__main__":
    main()
