"""
Download benchmark datasets from S3: task definitions and images.

Usage:
    AWS_KEY_ID=<KEY> AWS_SECRET_KEY=<SECRET> python code/download_data.py
    AWS_KEY_ID=<KEY> AWS_SECRET_KEY=<SECRET> python code/download_data.py --benchmark safety
"""

import argparse
import json
import os
import sys

import boto3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import log as _log, reset_log

BUCKET = "adepts-test"
REGION = "us-west-2"
LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DISAMBIGUATION_S3_IMAGE_PREFIX = "adepts/disambiguation/"
DISAMBIGUATION_S3_TASK_FILES = {
    "disambiguation_desktop.json": "disambiguation/tasks_desktop.json",
    "disambiguation_mobile.json": "disambiguation/tasks_mobile.json",
}

SAFETY_S3_IMAGE_PREFIX = "adepts/safety/"
SAFETY_S3_TASK_FILES = {
    "safety_desktop.json": "safety/tasks_desktop.json",
    "safety_mobile.json": "safety/tasks_mobile.json",
}


def s3_key_to_local(s3_key, prefix):
    if s3_key.startswith(prefix):
        filename = s3_key[len(prefix):]
        return os.path.join("images", filename)
    return s3_key


def download_images(s3, image_keys, s3_prefix, local_dir):
    _log(f"Downloading {len(image_keys)} images...")
    downloaded = 0
    skipped = 0
    failed = 0

    for s3_key in sorted(image_keys):
        local_rel = s3_key_to_local(s3_key, s3_prefix)
        local_path = os.path.join(local_dir, local_rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if os.path.exists(local_path):
            skipped += 1
            continue

        try:
            s3.download_file(BUCKET, s3_key, local_path)
            downloaded += 1
            if downloaded % 100 == 0:
                _log(f"  Downloaded {downloaded} images...")
        except Exception as e:
            _log(f"  FAILED: {s3_key} ({e})")
            failed += 1

    _log(f"  {downloaded} downloaded, {skipped} already existed, {failed} failed")


def download_disambiguation(s3):
    _log("Downloading DISAMBIGUATION benchmark data", step=True)

    _log("Downloading task JSON files...")
    all_image_keys = set()

    for s3_name, local_name in DISAMBIGUATION_S3_TASK_FILES.items():
        _log(f"  s3://{BUCKET}/{s3_name}")
        obj = s3.get_object(Bucket=BUCKET, Key=s3_name)
        tasks = json.loads(obj["Body"].read())

        for task in tasks:
            new_paths = []
            for img_path in task.get("image_paths", []):
                all_image_keys.add(img_path)
                new_paths.append(s3_key_to_local(img_path, DISAMBIGUATION_S3_IMAGE_PREFIX))
            task["image_paths"] = new_paths

        local_path = os.path.join(LOCAL_DATA_DIR, local_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            json.dump(tasks, f, indent=2)
        _log(f"  Saved {len(tasks)} tasks to {local_path}")

    _log(f"Found {len(all_image_keys)} unique images")
    download_images(s3, all_image_keys, DISAMBIGUATION_S3_IMAGE_PREFIX,
                    os.path.join(LOCAL_DATA_DIR, "disambiguation"))


def download_safety(s3):
    _log("Downloading SAFETY benchmark data", step=True)

    _log("Downloading task JSON files...")
    for s3_name, local_name in SAFETY_S3_TASK_FILES.items():
        _log(f"  s3://{BUCKET}/{s3_name}")
        obj = s3.get_object(Bucket=BUCKET, Key=s3_name)
        tasks = json.loads(obj["Body"].read())

        local_path = os.path.join(LOCAL_DATA_DIR, local_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            json.dump(tasks, f, indent=2)
        _log(f"  Saved {len(tasks)} tasks to {local_path}")

    _log(f"Discovering images under s3://{BUCKET}/{SAFETY_S3_IMAGE_PREFIX}...")
    all_image_keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=SAFETY_S3_IMAGE_PREFIX):
        for obj in page.get("Contents", []):
            all_image_keys.append(obj["Key"])

    _log(f"Found {len(all_image_keys)} images")
    download_images(s3, all_image_keys, SAFETY_S3_IMAGE_PREFIX,
                    os.path.join(LOCAL_DATA_DIR, "safety"))


def main():
    parser = argparse.ArgumentParser(description="Download ADEPTS benchmark data from S3")
    parser.add_argument(
        "--benchmark",
        choices=["disambiguation", "safety", "all"],
        default="all",
        help="Which benchmark to download (default: all)",
    )
    args = parser.parse_args()

    key_id = os.environ.get("AWS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_KEY")
    if not key_id or not secret_key:
        raise SystemExit("Error: set AWS_KEY_ID and AWS_SECRET_KEY environment variables")

    s3 = boto3.client(
        "s3",
        region_name=REGION,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret_key,
    )

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    reset_log()

    if args.benchmark in ("disambiguation", "all"):
        download_disambiguation(s3)
    if args.benchmark in ("safety", "all"):
        download_safety(s3)


if __name__ == "__main__":
    main()
