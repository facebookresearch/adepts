from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import shutil
import time
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import sys

from PIL import Image, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import SAFETY_DATA_DIR, DEFAULT_TEST_SAMPLES, log as _log, reset_log

IMAGES_DIR = os.path.join(SAFETY_DATA_DIR, "images")

MIB = 1024 * 1024

SIZE_LIMIT_MIB = 3.75

_PAREN_NUMBER_RE = re.compile(r"\s*\(\d+\)\s*")


def _normalize_filename(name: str) -> str:
    base, ext = os.path.splitext(name)
    base = _PAREN_NUMBER_RE.sub(" ", base)
    base = re.sub(r"\s+", "_", base.strip().lower())
    if not base:
        base = _stable_short_hash(name)
        _log(f"[WARN] Empty basename for {name!r}, using hash: {base}")
    return base + ext.lower()



def to_png_under_bytes(
    original_bytes: bytes,
    *,
    limit_bytes: int,
    force_resolution: str = "",
) -> bytes:
    """
    Always returns PNG bytes.
    If the PNG is too big, downscales dimensions until it fits (or gives best effort).
    If force_resolution is "claude", resizes every image to 1280x720.
    If force_resolution is "gpt", resizes to height=768 keeping aspect ratio.
    """
    img = Image.open(io.BytesIO(original_bytes))
    img = ImageOps.exif_transpose(img)

    if getattr(img, "is_animated", False):
        img.seek(0)

    if img.mode in ("P", "LA"):
        img = img.convert("RGBA")

    if force_resolution == "claude":
        img = img.resize((1280, 720), resample=Image.Resampling.LANCZOS)
    elif force_resolution == "gpt":
        target_h = 768
        orig_w, orig_h = img.size
        new_w = max(1, int(round(orig_w * target_h / orig_h)))
        _log(f"GPT resize: {orig_w}x{orig_h} -> {new_w}x{target_h}")
        img = img.resize((new_w, target_h), resample=Image.Resampling.LANCZOS)

    b = _encode_bytes(img, fmt="PNG", quality=None)
    if len(b) < limit_bytes:
        _log(
            f"PNG fits at original size: {len(b) / 1024:.1f} KB (limit {limit_bytes / 1024:.1f} KB)"
        )
        return b

    _log(
        f"PNG too large at original size: {len(b) / 1024:.1f} KB, starting downscale (limit {limit_bytes / 1024:.1f} KB)"
    )
    cur = img
    best = b

    for _ in range(30):
        ratio = limit_bytes / len(best)
        linear_scale = ratio**0.5
        linear_scale *= 0.9
        linear_scale = max(linear_scale, 0.5)
        linear_scale = min(linear_scale, 0.95)

        w, h = cur.size
        new_w = max(1, int(w * linear_scale))
        new_h = max(1, int(h * linear_scale))

        if new_w == w and new_h == h:
            break

        cur = cur.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        b = _encode_bytes(cur, fmt="PNG", quality=None)
        if len(b) < len(best):
            best = b

        if len(best) < limit_bytes:
            return best

        if new_w <= 64 or new_h <= 64:
            break

    if len(best) >= limit_bytes:
        _log(
            f"Downscale loop exhausted at {len(best) / 1024:.1f} KB, trying quantization"
        )
        try:
            quantized = cur.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            b = _encode_bytes(quantized, fmt="PNG", quality=None)
            if len(b) < len(best):
                best = b
            if len(best) < limit_bytes:
                return best
        except Exception:
            pass

    if len(best) >= limit_bytes:
        _log(
            f"Quantization insufficient at {len(best) / 1024:.1f} KB, aggressive shrink+quantize"
        )
        for _ in range(10):
            w, h = cur.size
            new_w = max(1, int(w * 0.7))
            new_h = max(1, int(h * 0.7))
            if new_w == w and new_h == h:
                break
            cur = cur.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            try:
                quantized = cur.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                b = _encode_bytes(quantized, fmt="PNG", quality=None)
            except Exception:
                b = _encode_bytes(cur, fmt="PNG", quality=None)
            if len(b) < len(best):
                best = b
            if len(best) < limit_bytes:
                return best
            if new_w <= 32 or new_h <= 32:
                break

    _log(
        f"Best effort result: {len(best) / 1024:.1f} KB ({'over limit' if len(best) >= limit_bytes else 'under limit'})"
    )
    return best


def _encode_bytes(img: Image.Image, *, fmt: str, quality: int | None) -> bytes:
    buf = io.BytesIO()
    save_kwargs = {}
    if fmt == "JPEG":
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        save_kwargs.update(
            dict(
                quality=int(quality) if quality is not None else 85,
                optimize=True,
                progressive=True,
            )
        )
    elif fmt == "WEBP":
        save_kwargs.update(
            dict(
                quality=int(quality) if quality is not None else 85,
                method=6,
            )
        )
    elif fmt == "PNG":
        save_kwargs.update(dict(optimize=True))
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def _best_under_limit_for_quality(
    img: Image.Image,
    *,
    fmt: str,
    limit_bytes: int,
    q_min: int,
    q_max: int,
) -> tuple[bytes | None, int | None]:
    best = None
    best_q = None
    lo, hi = q_min, q_max
    while lo <= hi:
        mid = (lo + hi) // 2
        b = _encode_bytes(img, fmt=fmt, quality=mid)
        if len(b) < limit_bytes:
            best = b
            best_q = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best, best_q


def downsize_to_under_bytes(
    original_bytes: bytes,
    *,
    limit_bytes: int = 5 * MIB,
    original_ext: str,
    allow_format_change: bool = False,
) -> tuple[bytes, str]:
    if len(original_bytes) < limit_bytes:
        _log(
            f"Already under limit: {len(original_bytes) / 1024:.1f} KB < {limit_bytes / 1024:.1f} KB"
        )
        return original_bytes, original_ext
    _log(
        f"Downsizing {original_ext}: {len(original_bytes) / 1024:.1f} KB -> target {limit_bytes / 1024:.1f} KB"
    )
    img = Image.open(io.BytesIO(original_bytes))
    img = ImageOps.exif_transpose(img)
    ext = original_ext.lower()
    if ext in (".jpg", ".jpeg"):
        fmt = "JPEG"
        q_min, q_max = 35, 95
        candidates = [(fmt, False)]
    elif ext == ".webp":
        fmt = "WEBP"
        q_min, q_max = 35, 95
        candidates = [(fmt, False)]
    elif ext == ".png":
        fmt = "PNG"
        q_min, q_max = 0, 0
        candidates = [(fmt, False)]
        if allow_format_change:
            candidates = [("WEBP", True), ("PNG", False), ("JPEG", True)]
    else:
        if not allow_format_change:
            return original_bytes, original_ext
        candidates = [("JPEG", True)]
        q_min, q_max = 35, 95
    cur = img
    for _ in range(10):
        for fmt, _changed in candidates:
            if fmt in ("JPEG", "WEBP"):
                b, _q = _best_under_limit_for_quality(
                    cur, fmt=fmt, limit_bytes=limit_bytes, q_min=q_min, q_max=q_max
                )
                if b is not None:
                    out_ext = ".jpg" if fmt == "JPEG" else ".webp"
                    return b, out_ext
            elif fmt == "PNG":
                b = _encode_bytes(cur, fmt="PNG", quality=None)
                if len(b) < limit_bytes:
                    return b, ".png"
            else:
                b = _encode_bytes(cur, fmt=fmt, quality=85)
                if len(b) < limit_bytes:
                    out_ext = ".jpg" if fmt == "JPEG" else "." + fmt.lower()
                    return b, out_ext
        w, h = cur.size
        if w <= 256 or h <= 256:
            break
        new_w = max(1, int(w * 0.90))
        new_h = max(1, int(h * 0.90))
        cur = cur.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    if allow_format_change:
        b = _encode_bytes(cur, fmt="JPEG", quality=35)
        return b, ".jpg"
    b = _encode_bytes(cur, fmt=fmt, quality=None)
    return b, original_ext


def _with_suffix(name: str, suffix: str) -> str:
    base, ext = os.path.splitext(name)
    return f"{base}{suffix}{ext}"


def _stable_short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).hexdigest()[:n]


def _join(parent: str, child: str) -> str:
    return parent.rstrip("/") + "/" + child.strip("/")


def _relpath(full: str, root: str) -> str:
    root = root.rstrip("/") + "/"
    if not full.startswith(root):
        raise ValueError(f"Path not under root.\nroot={root}\nfull={full}")
    return full[len(root):]


@dataclass(frozen=True)
class PlanItem:
    old_path: str  # full path under SRC
    old_basename: str  # original basename
    renamed_path: str  # full path (same dir) under SRC after rename
    copied_path: str


def _ls_dir(path: str) -> list[str]:
    return [os.path.join(path, f) for f in os.listdir(path)]


def rename_then_copy_filtered(
    src_dir: str,
    dst_dir: str,
    image_list: list[str],
    *,
    dry_run: bool = True,
    extensions: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".gif",
    ),
    max_image_bytes: int = int(SIZE_LIMIT_MIB * 1024 * 1024),
    allow_format_change: bool = False,
    force_png: bool = True,
    force_resolution: str = "",
    verbose: bool = True,
    dir_cache: dict[str, list[str]] | None = None,
    bytes_cache: dict[str, bytes] | None = None,
    max_workers: int = 8,
) -> list[PlanItem]:
    """
    Copy images from src_dir into dst_dir with optional downsizing/PNG conversion.
    Only processes files whose basename is in image_list and has a matching extension.
    """
    _log(
        f"src={src_dir} dst={dst_dir} images={len(image_list)} force_resolution={force_resolution!r}"
    )

    src_dir = src_dir.rstrip("/")
    dst_dir = dst_dir.rstrip("/")
    want = set(image_list)

    if os.path.exists(dst_dir):
        _log(f"Destination already exists, will overwrite: {dst_dir}")

    candidates: list[tuple[str, str]] = []

    if dir_cache is None:
        dir_cache = {}

    def walk(cur: str) -> None:
        if cur in dir_cache:
            children = dir_cache[cur]
            _log(f"Using cached listing for {cur} ({len(children)} children)")
        else:
            _log(f"Listing {cur} ...")
            t0 = time.monotonic()
            children = _ls_dir(cur)
            dir_cache[cur] = children
            _log(f"Listed {len(children)} children in {time.monotonic() - t0:.1f}s")
        image_count = 0
        skipped = 0
        for child in children:
            name = os.path.basename(child)
            ext = os.path.splitext(name)[1].lower()
            if ext not in extensions:
                continue
            image_count += 1
            if name not in want:
                skipped += 1
                continue
            candidates.append((_join(cur, name), name))
        _log(f"{image_count} images, {len(candidates)} matched, {skipped} skipped")

    walk(src_dir)

    unmatched = want - {name for _, name in candidates}
    if unmatched:
        _log(f"[WARN] {len(unmatched)} requested images NOT found in src_dir")
        if verbose:
            for u in sorted(unmatched)[:20]:
                print(f"  {u}")
            if len(unmatched) > 20:
                print(f"  ... and {len(unmatched) - 20} more")

    by_dir: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for old_path, old_name in candidates:
        by_dir[os.path.dirname(old_path)].append((old_path, old_name))

    rename_map: dict[str, str] = {}
    identity_count = 0
    renamed_count = 0
    collision_count = 0

    for dir_path, items in by_dir.items():
        dir_children = dir_cache.get(dir_path) or _ls_dir(dir_path)
        existing_child_names = [os.path.basename(x) for x in dir_children]
        used_basenames = set(existing_child_names)

        desired_to_sources: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for old_path, old_name in items:
            desired = _normalize_filename(old_name)
            if desired != old_name:
                desired_to_sources[desired].append((old_path, old_name))
            else:
                rename_map[old_path] = old_path
                identity_count += 1

        for desired, sources in desired_to_sources.items():
            if len(sources) == 1 and desired not in used_basenames:
                old_path, _old_name = sources[0]
                new_path = _join(dir_path, desired)
                rename_map[old_path] = new_path
                used_basenames.add(desired)
                renamed_count += 1
                continue

            if verbose:
                _log(f"Collision for {desired} ({len(sources)} sources)")
            for old_path, old_name in sources:
                h = _stable_short_hash(old_name, n=8)
                candidate = _with_suffix(_normalize_filename(old_name), f"__{h}")
                k = 2
                while candidate in used_basenames:
                    candidate = _with_suffix(
                        _normalize_filename(old_name), f"__{h}_{k}"
                    )
                    k += 1
                new_path = _join(dir_path, candidate)
                rename_map[old_path] = new_path
                used_basenames.add(candidate)
                collision_count += 1

    _log(
        f"Rename plan: {identity_count} identity, {renamed_count} renamed, {collision_count} collisions"
    )

    plan: list[PlanItem] = []
    for old_path, old_basename in candidates:
        renamed_path = rename_map.get(old_path, old_path)
        rel_dir = os.path.dirname(_relpath(old_path, src_dir))
        dst_subdir = dst_dir if rel_dir in ("", ".") else _join(dst_dir, rel_dir)
        copied_path = _join(dst_subdir, os.path.basename(renamed_path))
        plan.append(PlanItem(old_path, old_basename, renamed_path, copied_path))

    _log(f"Plan built: {len(plan)} items")
    if verbose:
        for i, item in enumerate(plan[:10]):
            print(
                f"  [{i}] {item.old_basename} -> {os.path.basename(item.copied_path)}"
            )
        if len(plan) > 10:
            print(f"  ... and {len(plan) - 10} more")

    if not plan:
        _log("No plan items, nothing to do")
        return plan

    if bytes_cache is None:
        bytes_cache = {}

    created_dirs: set[str] = set()
    _dir_lock = threading.Lock()

    def ensure_dir(path: str) -> None:
        if path in created_dirs:
            return
        with _dir_lock:
            if path not in created_dirs:
                os.makedirs(path, exist_ok=True)
                created_dirs.add(path)

    ensure_dir(dst_dir)

    def _read_bytes(path: str) -> bytes:
        if path in bytes_cache:
            return bytes_cache[path]
        with open(path, "rb") as f:
            data = f.read()
        bytes_cache[path] = data
        return data

    def _process_one(item: PlanItem) -> str:
        if dry_run:
            return "plain"

        ensure_dir(os.path.dirname(item.copied_path))
        src_bytes = _read_bytes(item.old_path)
        ext = os.path.splitext(item.old_basename)[1].lower()

        if force_png and ext in extensions:
            base, _ = os.path.splitext(item.copied_path)
            dst_path = base + ".png"
            png_bytes = to_png_under_bytes(
                src_bytes,
                limit_bytes=max_image_bytes,
                force_resolution=force_resolution,
            )
            with open(dst_path, "wb") as out:
                out.write(png_bytes)
            return "png"

        if ext in extensions and len(src_bytes) >= max_image_bytes:
            new_bytes, out_ext = downsize_to_under_bytes(
                src_bytes,
                limit_bytes=max_image_bytes,
                original_ext=ext,
                allow_format_change=allow_format_change,
            )
            dst_path = item.copied_path
            if allow_format_change and out_ext != os.path.splitext(dst_path)[1].lower():
                base, _ = os.path.splitext(dst_path)
                dst_path = base + out_ext
            with open(dst_path, "wb") as out:
                out.write(new_bytes)
            return "downsized"

        shutil.copy2(item.old_path, item.copied_path)
        return "plain"

    t0 = time.monotonic()
    copy_force_png = 0
    copy_downsized = 0
    copy_plain = 0
    done = 0

    _log(f"Copying {len(plan)} files to {dst_dir} (workers={max_workers})")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, item): item for item in plan}
        for future in as_completed(futures):
            done += 1
            item = futures[future]
            try:
                category = future.result()
            except Exception as e:
                _log(f"[ERROR] Failed {item.old_basename}: {e}")
                continue
            if category == "png":
                copy_force_png += 1
            elif category == "downsized":
                copy_downsized += 1
            else:
                copy_plain += 1
            if done % 50 == 0:
                _log(f"  copied {done}/{len(plan)} ({time.monotonic() - t0:.1f}s elapsed)")

    _log(
        f"Copy done in {time.monotonic() - t0:.1f}s: "
        f"{copy_force_png} png, {copy_downsized} downsized, {copy_plain} plain"
    )

    _log(f"Done: {len(plan)} items processed")
    return plan


PRE_PROCESSED_DIR = os.path.join(SAFETY_DATA_DIR, "pre_processed_images")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-process safety benchmark images into resolution variants",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Enable test mode (process fewer images)",
    )
    parser.add_argument(
        "--test-samples", type=int, default=DEFAULT_TEST_SAMPLES,
        help=f"Number of images to process in test mode (default: {DEFAULT_TEST_SAMPLES})",
    )
    parser.add_argument(
        "--data-dir", type=str, default=SAFETY_DATA_DIR,
        help=f"Directory containing raw safety data (default: {SAFETY_DATA_DIR})",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, "images")
    output_dir = os.path.join(data_dir, "pre_processed_images")

    reset_log()

    if args.test:
        _log(f"Running in TEST MODE ({args.test_samples} images per job)", step=True)
    else:
        _log("Running in FULL MODE", step=True)

    _log("Discovering images", step=True)
    all_files = sorted(f for f in os.listdir(images_dir) if not f.startswith("."))
    desktop_images = [f for f in all_files if f.startswith("desktop_")]
    mobile_images = [f for f in all_files if f.startswith("mobile_")]
    _log(f"Found {len(all_files)} files: {len(desktop_images)} desktop, {len(mobile_images)} mobile")

    if args.test:
        desktop_images = desktop_images[:args.test_samples]
        mobile_images = mobile_images[:args.test_samples]
        _log(f"TEST MODE: trimmed to {len(desktop_images)} desktop, {len(mobile_images)} mobile")

    dir_cache: dict[str, list[str]] = {}
    bytes_cache: dict[str, bytes] = {}

    jobs = [
        (images_dir, os.path.join(output_dir, "mobile_original"), "", mobile_images),
        (images_dir, os.path.join(output_dir, "desktop_original"), "", desktop_images),
        (images_dir, os.path.join(output_dir, "mobile_claude"), "claude", mobile_images),
        (images_dir, os.path.join(output_dir, "desktop_claude"), "claude", desktop_images),
        (images_dir, os.path.join(output_dir, "desktop_gpt"), "gpt", desktop_images),
        (images_dir, os.path.join(output_dir, "mobile_gpt"), "gpt", mobile_images),
    ]
    _log(f"Process {len(jobs)} jobs", step=True)
    for i, (src, dst, resolution, image_list) in enumerate(jobs):
        label = resolution if resolution else "original"
        _log(f"Job {i + 1}/{len(jobs)}: {label} -> {dst}")
        t0 = time.monotonic()
        plan = rename_then_copy_filtered(
            src,
            dst,
            image_list,
            dry_run=False,
            force_resolution=resolution,
            verbose=not args.test,
            dir_cache=dir_cache,
            bytes_cache=bytes_cache,
        )
        _log(
            f"Job {i + 1}/{len(jobs)} done: {len(plan)} files in {time.monotonic() - t0:.1f}s"
        )

    cache_mb = sum(len(v) for v in bytes_cache.values()) / (1024 * 1024)
    _log(f"All jobs complete (bytes_cache: {len(bytes_cache)} files, {cache_mb:.1f} MB)")


if __name__ == "__main__":
    main()
